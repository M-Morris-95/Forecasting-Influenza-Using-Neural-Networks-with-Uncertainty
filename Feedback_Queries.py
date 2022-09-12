import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import Metrics
from tensorflow_probability.python.distributions import kullback_leibler
import json
from optimizer_tools import *
import tqdm
sigmoid = tf.keras.activations.sigmoid
tanh = tf.keras.activations.tanh
multiply = tf.math.multiply
tfd = tfp.distributions

root = 'Results/feedback_rnn/'

@tf.function
def softplus(x, k=1.0):
    k = tf.cast(k, float)
    return tf.math.divide(tf.math.log(tf.math.add(1.0, tf.math.exp(tf.math.multiply(x, k)))), k)
    #
    # {"epochs": 84.10363144914598,
    #  "l1": 0.00011740328396035743,
    #  "l2": 0.0007437010628971827,
    #  "num_features": 50.308022450707064,
    #  "op_scale": 0.026458969132157326,
    #  "prior_scale": 1.0917798122986158,
    #  "q_scale": 0.019113616299900583,
    #  "rnn_units": 68.90223668025034}

class FeedBack_Forecasting(tf.keras.Model):
    lag = 14
    model_type = 'feedback'
    forecast_type = 'multi'
    loss = 'NLL'
    query_forecast = 'True'

    pbounds = {'rnn_units':(25,125),
               'num_features':(20,100),
               'kl_weight':(0.1, 1.0),
               'op_scale':(0.01, 0.1),
               'prior_scale':(1e-4, 1e-2),
               'dense_post_scale':(1e-2, 1e-1),
               'op_rho':(0.1, 5),
               'posterior_rho':(0.1, 5),
               'epochs':(10,100)
               }

    def __init__(self, rnn_units=99, num_features=50, kl_weight=0.1, op_scale=0.1,
                 prior_scale=1e-4, dense_post_scale=0.1, gamma=28, n_batches=100, op_rho=1.0, posterior_rho=1.0,  **kwargs):
        super().__init__()

        num_features = int(num_features)
        self.gamma = int(gamma)
        self.rnn_cell = tf.keras.layers.LSTMCell(int(rnn_units))
        self.rnn = tf.keras.layers.RNN(self.rnn_cell, return_state=True)


        def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            return tf.keras.Sequential([
                tfp.layers.VariableLayer(2 * n,
                                         dtype=dtype,
                                         regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001),
                                         initializer=tf.keras.initializers.glorot_normal()),

                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.MultivariateNormalDiag(loc=t[..., :n],
                                               scale_diag=1e-5 + dense_post_scale * softplus(t[..., n:], k=posterior_rho)),
                )),
            ])

        def prior(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            prior_model = tf.keras.Sequential([
                tfp.layers.DistributionLambda(lambda t : tfd.Independent(
                    tfd.MultivariateNormalDiag(loc = tf.zeros(n), scale_diag = prior_scale*tf.ones(n))))
            ])
            return prior_model

        self.dense_var = tfp.layers.DenseVariational(units=num_features*2,
                                                     make_posterior_fn=posterior_mean_field,
                                                     make_prior_fn=prior,
                                                     kl_weight=kl_weight/n_batches,
                                                     kl_use_exact=True)

        self.DistributionLambda = tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :int(num_features)],
                                 scale=op_scale * softplus(t[..., int(num_features):], k=op_rho)
                                 )
        )

    def __call__(self, inputs, training=None):
        predictions = []
        x, *state = self.rnn(inputs[:, :-self.lag, :], training=training)
        x = self.dense_var(x, training=training)
        predictions.append(x)

        for n in range(self.gamma-1):
            x = self.DistributionLambda(x).sample()
            if n < self.lag:
                x = tf.concat([inputs[:, -self.lag+n, :-1], x[:, -1:]], 1)
            x, state = self.rnn_cell(x, states=state,training=training)
            x = self.dense_var(x, training=training)
            predictions.append(x)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        predictions = self.DistributionLambda(predictions)

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


if __name__ == "__main__":
    from Eval_Fn import *
    from Test_Fn import *
    from Feedback_Queries import *
    from bayes_opt import BayesianOptimization
    from optimizer_tools import *

    max_iter =250
    model = FeedBack_Forecasting
    gamma = 28
    epochs = 50

    if not os.path.exists(root):
        os.mkdir(root)

    eval = Eval_Fn(model=model,
                   root=root,
                   gamma=gamma,
                   epochs=epochs,
                   n_folds=4,
                   verbose=False)

    pbounds = model.pbounds

    optimizer = BayesianOptimization(
        f=eval,
        pbounds=pbounds,
        random_state=1,
        verbose=2
    )

    optimizer = load_steps(root, optimizer)
    # while len(optimizer.res) < max_iter:
    #     optimizer.maximize(
    #         init_points=10,
    #         n_iter=10)
    #
    #     save_steps(root, optimizer)

    Test_fn(root = root, model = model, gammas=[gamma], epochs=epochs, test_seasons = [2015, 2016,2017,2018], verbose=True)