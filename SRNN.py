import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from optimiser_tools import *
import tqdm

tfd = tfp.distributions

@tf.function
def softplus(x, k=1.0):
    k = tf.cast(k, float)
    return tf.math.divide(tf.math.log(tf.math.add(1.0, tf.math.exp(tf.math.multiply(x, k)))), k)

# initializer that does mean and standard deviation separately - speeds up convergence
def random_gaussian_initializer(shape, dtype):
    n = int(shape / 2)
    loc_norm = tf.random_normal_initializer(mean=0., stddev=0.1)
    loc = tf.Variable(
        initial_value=loc_norm(shape=(n,), dtype=dtype)
    )
    scale_norm = tf.random_normal_initializer(mean=-3., stddev=0.1)
    scale = tf.Variable(
        initial_value=scale_norm(shape=(n,), dtype=dtype)
    )
    return tf.concat([loc, scale], 0)

class SRNN(tf.keras.Model):
     # parameters for data builder
    lag = 14
    model_type = 'FF'
    forecast_type = 'single' # does the model forecast once (single) or make a series or forecasts? (feedback)
    loss = 'NLL'             # 
    query_forecast = 'False' # does the network forecast queries or just ILI?

    # upper and lower limits for optimization, good hyper parameters defined as standard.
    pbounds = {'rnn_units':(25,125),        # units in rnn layer
               'n_queries':(20,100),        # number of queries
               'kl_power':(-3,0),           # KL annealing term = 10^kl_power
               'op_scale':(0.01, 0.1),      # scaling factor for output
               'prior_scale':(1e-4, 1e-2),  # prior stddev
               'epochs':(10,100),           # epochs to train for
               'lr_power':(-4, -2),         # learning rate = 10^lr_power
               'q_scale':(0.001, 0.1)       # posterior scaling factor
               }

    def __init__(self, rnn_units=25, kl_power=-3, op_scale=0.1,
                 prior_scale=1e-4, q_scale=0.1,  gamma=28, n_batches=100, **kwargs):
        super().__init__()

        self.kl_weight = np.power(10.0, kl_power)
        self.units = int(rnn_units)
        self.gamma = gamma
        self.rnn_cell = tf.keras.layers.GRUCell(self.units)
        self.rnn = tf.keras.layers.RNN(self.rnn_cell, return_state=True)

        def posterior(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            c = np.log(np.expm1(1.))
            posterior_model = tf.keras.Sequential([
                tfp.layers.VariableLayer(2 * n, dtype=dtype, trainable=True,
                                         initializer=lambda shape, dtype: random_gaussian_initializer(shape, dtype)
                                         ),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.MultivariateNormalDiag(loc=t[..., :n],
                                               scale_diag=1e-5 + q_scale*tf.nn.softplus(c + t[..., n:])),
                )),
            ])
            return posterior_model

        def prior_trainable(kernel_size, bias_size, dtype=None):
            n = kernel_size + bias_size
            prior_model = tf.keras.Sequential([
                tfp.layers.VariableLayer(n),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.MultivariateNormalDiag(loc = t, scale_diag = prior_scale*tf.ones(n))
                ))
            ])
            return prior_model

        self.dense_var = tfp.layers.DenseVariational(units=2,
                                                      make_posterior_fn=posterior,
                                                      make_prior_fn=prior_trainable,
                                                      kl_weight=self.kl_weight/n_batches,
                                                      kl_use_exact=True)

        self.DistributionLambda = tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :1],
                                 scale=op_scale * softplus(t[..., 1:], k=1.0)
                                 )
        )

    def __call__(self, inputs, training=None):
        x, *states = self.rnn(inputs)
        x = self.dense_var(x, training=training)

        return self.DistributionLambda(x)

    def predict(self, x, n_steps=25, batch_size=None, verbose=False, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        pred = []
        for _ in tqdm.trange(n_steps, disable = np.invert(verbose)):
            pred.append(self(x))

        means = np.asarray([p.mean() for p in pred])
        vars = np.asarray([p.variance() for p in pred])
        mean = means.mean(0)

        std = np.sqrt(means.var(0) + vars.mean(0))

        return mean, std, {'model': np.sqrt(means.var(0)), 'data':np.sqrt(vars.mean(0))}