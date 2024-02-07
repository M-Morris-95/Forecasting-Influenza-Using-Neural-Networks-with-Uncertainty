import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import tqdm

from scipy.stats import norm

from lib.layers.GRU_cell_variational import *
from lib.layers.Dense_Variational_Reparam import *


tfd = tfp.distributions

def random_gaussian_initializer(shape, dtype):
    if shape[0] == 2:
        n = shape[1:]
        loc_norm = tf.random_normal_initializer(mean=0., stddev=0.1)
        loc = tf.Variable(
            initial_value=loc_norm(shape=n, dtype=dtype)
        )
        scale_norm = tf.random_normal_initializer(mean=-3., stddev=0.1)
        scale = tf.Variable(
            initial_value=scale_norm(shape=n, dtype=dtype)
        )
        init = tf.concat([tf.expand_dims(loc, 0), tf.expand_dims(scale,0)], 0)
    else:
        n = (shape[0], int(shape[-1]/2))
        loc_norm = tf.random_normal_initializer(mean=0., stddev=0.1)
        init = tf.Variable(
            initial_value=loc_norm(shape=shape, dtype=dtype)
        )
    return init

class IRNN_Full_Bayes(tf.keras.Model):
    # parameters for data builder
    lag = 14                # delay between Qs and ILI rates
    model_type = 'feedback'
    forecast_type = 'multi' # does the model forecast once (single) or make a series or forecasts? (feedback)
    loss = 'NLL'             # 
    query_forecast = 'True' # does the network forecast queries or just ILI?


    # upper and lower limits for optimization, good hyper parameters defined as standard.
    pbounds = {'rnn_units':(25,125),        # units in rnn layer
               'n_op':(20,100),             # number of outputs (m+1)
               'kl_power':(-3,0),           # KL annealing term = 10^kl_power
               'p_scale_pwr' :(-3,0),       # prior std = 10^p_scale_pwr
               'q_scale_pwr' :(-3,0),       # post std = 10^p_scale_pwr
               'op_scale_pwr':(-3,0),       # scaling factor for output
               'epochs':(30,200),           # epochs to train for
               'lr_power':(-4, -2),         # learning rate = 10^lr_power
               }

    def __init__(self, 
                 rnn_units=16, 
                 n_op = 1, 
                 kl_power=-2.0,
                 p_scale_pwr = -2.0,
                 q_scale_pwr = -2.0,
                 op_scale_pwr= -2.0,
                 window_size = 47, 
                 gamma=28, 
                 lag = 7, 
                 n_batches=249, 
                 n_regions = 1,
                 n_samples=3,
                 sampling='once',
                 use_bn = False,
                 kl_use_exact=True, **kwargs):
        super().__init__()
        rnn_units = int(rnn_units)
        self.n_op = int(n_op)
        self.kl_weight = np.power(10.0, kl_power)
        p_scale = np.power(10.0, p_scale_pwr)
        q_scale = np.power(10.0, q_scale_pwr)
        op_scale = np.power(10.0, op_scale_pwr)
        self.window_size = int(window_size)
        self.gamma = int(gamma)
        self.lag=lag
        self.n_batches = n_batches
        self.n_samples = n_samples 
        self.sampling=sampling
        self.use_bn=use_bn
        self.kl_use_exact = kl_use_exact
        self.kl_d = 0.
        self.n_regions = n_regions
        

        def rnn_posterior(shape, name, initializer, scale = None,  n_samples = 3, regularizer=None, constraint=None, cashing_device=None, dtype=tf.float32):
            if scale == None:
                # inspired by glorot uniform, doesn't work very well
                scale = tf.math.sqrt(2/(shape[0] + shape[1]))
            
            c = np.log(np.expm1(1.))
            posterior_model = tf.keras.Sequential([
                tfp.layers.VariableLayer((2, ) + shape, dtype=dtype, trainable=True,
                                            initializer=lambda shape, dtype: initializer(shape, dtype),
                                            regularizer = regularizer,
                                            name=name
                                            ),

                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.MultivariateNormalDiag(loc=t[0],
                                                scale_diag=1e-5 + q_scale*tf.nn.softplus(c + t[1])),
                ))
            ])
            return posterior_model
        
        def mean0_prior(shape, name, initializer,  scale = None,   n_samples = 3, regularizer=None, constraint=None, cashing_device=None, dtype=tf.float32):
            if scale == None:
                # inspired by glorot uniform, doesn't work very well
                scale=tf.math.sqrt(2/(shape[0] + shape[1]))
            
            prior_model = tf.keras.Sequential([
                tfp.layers.VariableLayer(shape, dtype=dtype, trainable=True,
                                            initializer=initializer,
                                            regularizer = regularizer,
                                            name=name
                                            ),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.MultivariateNormalDiag(loc = tf.zeros(shape), scale_diag =p_scale*tf.nn.softplus(c + t))
                ))
            ])
            return prior_model
        
        self.rnn_cell = GRU_Cell_Variational(
            int(rnn_units), 
            kernel_prior_fn = mean0_prior, 
            kernel_posterior_fn = rnn_posterior,
            recurrent_kernel_prior_fn = mean0_prior, 
            recurrent_kernel_posterior_fn = rnn_posterior,
            bias_prior_fn = mean0_prior,
            bias_posterior_fn = rnn_posterior, 
            n_samples = n_samples,
            scale = q_scale,
            kl_weight = self.kl_weight,
            sampling=self.sampling
            
        )

        self.rnn = tf.keras.layers.RNN(self.rnn_cell, return_state=True)

        self.dense_var = DenseVariational_repareterisation(
            units=int(n_op*2),
            last_dim=int(rnn_units),
            make_posterior_fn=rnn_posterior,
            make_prior_fn=mean0_prior,
            initializer=random_gaussian_initializer,
            scale = q_scale,
            n_samples=n_samples,
            kl_weight=self.kl_weight
        )            
        self.BN = tf.keras.layers.BatchNormalization()

        c = np.log(np.expm1(1.))
        self.DistributionLambda = tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :self.n_op],
                                scale=1e-5 + op_scale*tf.nn.softplus(c + t[..., self.n_op:])
                                )
            )
        
    def get_kl(self):
        a = self.layers[0]._kernel_posterior(tf.random.normal([5,]))
        b = self.layers[0]._recurrent_kernel_posterior(tf.random.normal([5,]))
        c = self.layers[0]._bias_posterior(tf.random.normal([5,]))
        d = self.layers[2]._posterior(tf.random.normal([5,]))

        e = self.layers[0]._kernel_prior(tf.random.normal([5,]))
        f = self.layers[0]._recurrent_kernel_prior(tf.random.normal([5,]))
        g = self.layers[0]._bias_prior(tf.random.normal([5,]))
        h = self.layers[2]._prior(tf.random.normal([5,]))

        q = tfd.MultivariateNormalDiag(loc = tf.concat([tf.reshape(a.mean(), -1), 
                                                tf.reshape(b.mean(), -1), 
                                                tf.reshape(c.mean(), -1), 
                                                tf.reshape(d.mean(), -1)], axis=0), 
                               scale_diag = tf.concat([tf.reshape(a.stddev(), -1),
                                                       tf.reshape(b.stddev(), -1),
                                                       tf.reshape(c.stddev(), -1),
                                                       tf.reshape(d.stddev(), -1)], axis=0)
                          )

        p = tfd.MultivariateNormalDiag(loc = tf.concat([tf.reshape(e.mean(), -1), 
                                                tf.reshape(f.mean(), -1), 
                                                tf.reshape(g.mean(), -1), 
                                                tf.reshape(h.mean(), -1)], axis=0), 
                               scale_diag = tf.concat([tf.reshape(e.stddev(), -1),
                                                       tf.reshape(f.stddev(), -1),
                                                       tf.reshape(g.stddev(), -1),
                                                       tf.reshape(h.stddev(), -1)], axis=0)
                          )

        self.kl_d = tfp.distributions.kl_divergence(p,q)

    def __call__(self, inputs, training=None):

        predictions = []
        x, state = self.rnn(inputs[:, :self.window_size, :])
        if self.use_bn:
            x = self.BN(x)
        x = self.dense_var(x, first=True)
        predictions.append(x)

        for d in range(self.gamma-1):
            x = self.DistributionLambda(x).mean()
            if np.logical_and(self.n_op > 1, d<self.lag):
                x = tf.concat([inputs[:, self.window_size+d, :-self.n_regions], x[:, -self.n_regions:]], 1)
            
            x, state = self.rnn_cell(x, states=state, training=training)
            if self.use_bn:
                x = self.BN(x)
            x = self.dense_var(x, first=False if self.sampling == 'once' else True)
            predictions.append(x)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        p1 = self.DistributionLambda(predictions)
        
#         if training:
        self.get_kl
        return p1

    def predict(self, x, n_steps=25, batch_size=None, verbose=False, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        pred = [self(x) for _ in range(n_steps)]

        means = tf.convert_to_tensor([p.mean() for p in pred])
        vars = tf.convert_to_tensor([p.variance() for p in pred])

        mean = tf.reduce_mean(means, 0)
        model_var = tf.math.reduce_variance(means, 0)
        data_var = tf.reduce_mean(vars, 0)

        std = tf.math.sqrt(model_var + data_var)
        return mean.numpy(), std.numpy(), {'model':tf.math.sqrt(model_var).numpy(), 'data':tf.math.sqrt(data_var).numpy()}

