import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from lib.utils import *
import tqdm

tfd = tfp.distributions


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

class IRNN(tf.keras.Model):

    # parameters for data builder
    lag = 14                # delay between Qs and ILI rates
    model_type = 'feedback'
    forecast_type = 'multi' # does the model forecast once (single) or make a series or forecasts? (feedback)
    loss = 'NLL'             # 
    query_forecast = 'True' # does the network forecast queries or just ILI?

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

    def __init__(self, rnn_units=128, n_queries=50, kl_power=-3, op_scale=0.05,
                 prior_scale=0.005, q_scale=0.02, gamma=28, n_batches=100, **kwargs):
        super().__init__()
        num_features = int(n_queries+1)
        rnn_units = int(rnn_units)
        self.gamma = gamma
        self.kl_weight = np.power(10.0, kl_power)


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

        
        self.gru_cell = tf.keras.layers.GRUCell(rnn_units)
        self.gru = tf.keras.layers.RNN(self.gru_cell, return_state=True)
        
        self.dense_variational = tfp.layers.DenseVariational(units=num_features*2,
                                                             make_posterior_fn=posterior,
                                                             make_prior_fn=prior_trainable,
                                                             kl_weight = self.kl_weight/n_batches,
                                                             kl_use_exact=True
                                                             )
        c = np.log(np.expm1(1.))
        self.distribution_lambda = tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :num_features],
                                 scale=1e-5 + op_scale*tf.nn.softplus(c + t[..., num_features:])
                                 )
        )
    
    def __call__(self, inputs, training=None, look_ahead=False):
        predictions = []
        x, *state = self.gru(inputs[:, :-self.lag, :])
        x = self.dense_variational(x)
        predictions.append(x)

        for i in range(self.gamma-1):
            x = self.distribution_lambda(x).sample()

            if np.logical_and(i < self.lag, look_ahead):
                x = tf.concat([inputs[:, -self.lag+i, :-1], x[:, -1:]], 1)
            x, state = self.gru_cell(x, states=state)
            x = self.dense_variational(x)
            predictions.append(x)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        predictions = self.distribution_lambda(predictions)

        return predictions

    def predict(self, x, n_steps=25, batch_size=None, verbose=False, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        pred = []
        for _ in tqdm.trange(n_steps, disable = np.invert(verbose)):
            pred.append(self(x))

        means = np.asarray([p.mean() for p in pred])
        vars = np.asarray([p.variance() for p in pred])
        mean = means.mean(0)

        std = np.sqrt(means.var(0) + vars.mean(0))


        return mean, std, {'model': np.sqrt(means.var(0)), 'data':np.sqrt(vars.mean(0))}
