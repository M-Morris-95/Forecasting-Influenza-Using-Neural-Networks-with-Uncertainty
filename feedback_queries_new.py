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
from Bayes_GRU import Bayes_GRU
from Bayes_Dense import Bayes_Dense

sigmoid = tf.keras.activations.sigmoid
tanh = tf.keras.activations.tanh
multiply = tf.math.multiply
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

class Feedback_Queries(tf.keras.Model):
    lag = 14
    model_type = 'feedback'
    forecast_type = 'multi'
    loss = 'NLL'
    query_forecast = 'True'

    pbounds = {'rnn_units':(25,125),
               'n_queries':(20,100),
               'kl_power':(-4,-1),
               'op_scale':(0.01, 0.1),
               'prior_scale':(1e-4, 1e-2),
               'epochs':(10,100),
               'lr_power':(-4, -2)
               }

    def __init__(self, rnn_units=5, n_queries=5, kl_power=-3, op_scale=0.05,
                 prior_scale=0.005, q_scale=0.02, q_rho=2.5, gamma=28, n_batches=100, full_cov=False, **kwargs):
        super().__init__()
        num_features = int(n_queries+1)
        rnn_units = int(rnn_units)

        self.gamma = gamma

        self.kl_weight = np.power(10.0, kl_power)
        self.gru_cell = tf.keras.layers.GRUCell(rnn_units)
        self.gru = tf.keras.layers.RNN(self.gru_cell, return_state=True)

        def posterior(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            c = np.log(np.expm1(1.))
            posterior_model = tf.keras.Sequential([
                tfp.layers.VariableLayer(2 * n, dtype=dtype, trainable=True,
                                         initializer=lambda shape, dtype: random_gaussian_initializer(shape, dtype)
                                         ),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.MultivariateNormalDiag(loc=t[..., :n],
                                               scale_diag=1e-5 + 0.01*tf.nn.softplus(c + t[..., n:])),
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

def shuffle_batches(x_train, y_train, batch_size):
    x_tr = []
    y_tr = []
    n_batch = int(np.ceil(x_train.shape[0] / batch_size))
    for i in range(n_batch):
        x_tr.append(x_train[i*batch_size : (i+1)*batch_size])
        y_tr.append(y_train[i*batch_size : (i+1)*batch_size])

    shuffled = np.linspace(0, n_batch-2, n_batch-1).astype(int)
    np.random.shuffle(shuffled)

    x_trn = []
    y_trn = []
    for j in shuffled:
        x_trn.append(x_tr[j])
        y_trn.append(y_tr[j])

    x_trn = np.stack(x_trn).reshape((n_batch-1) * batch_size, x_train.shape[-2], x_train.shape[-1])
    y_trn = np.stack(y_trn).reshape((n_batch-1) * batch_size, y_train.shape[-2], y_train.shape[-1])

    x_train = np.concatenate([x_trn, x_tr[-1]], 0)
    y_train = np.concatenate([y_trn, y_tr[-1]], 0)
    return x_train, y_train

if __name__ == "__main__":
    from Eval_Fn import *
    from Test_Fn import *
    from bayes_opt import BayesianOptimization
    from optimizer_tools import *

    batch_size = 32
    gamma = 28
    root = 'Results/feedback_rnn_new14/'
    save_root_new = 'Results/feedback_rnn_shuffle_batch/'

    file = open(os.path.join(root, 'optimiser_max.json'), 'r')
    params = json.load(file)['params']
    model = Feedback_Queries

    def loss(y, p_y):
        return -p_y.log_prob(y)


    for model_num in range(50):
        res = {7: {2015:0, 2016:0, 2017:0, 2018:0},14: {2015:0, 2016:0, 2017:0, 2018:0}, 21: {2015:0, 2016:0, 2017:0, 2018:0}, 28: {2015:0, 2016:0, 2017:0, 2018:0}}
        for season in [2015,2016,2017,2018]:
            # Data
            _data = DataConstructor(test_season=season, gamma=gamma, n_queries=int(params['n_queries']))
            x_train, y_train, x_test, y_test = _data(model.model_type, model.forecast_type, model.query_forecast, bullshitwasteoftime=False)

            shuffle=True
            batch_size = 32
            if shuffle:
                x_train, y_train = shuffle_batches(x_train, y_train, batch_size)

            # Model
            _model = model(n_batches=np.ceil(x_train.shape[0] / batch_size), **params)
            _model.compile(loss=loss, optimizer=tf.optimizers.Adam(learning_rate=10**params['lr_power']))
            _model.fit(x_train, y_train, batch_size=batch_size, epochs=15, verbose=False)
            pred = _model.predict(x_test)
            df = convert_to_df(pred, y_test, _data.test_dates, _data, type=_model.forecast_type)

            # Plot results
            for idx, g, in enumerate([7,14,21,28]):
                plt.subplot(4,1,idx+1)
                plt.plot(df[g]['True'], color='black')
                plt.fill_between(df[g].index, df[g]['Pred']+df[g]['Std'], df[g]['Pred']-df[g]['Std'], color='red', alpha=0.3)
                plt.plot(df[g]['Pred'], color='red')
                res[g][season] = df[g].to_json()
                print(g, season, Metrics.skill(df[g]))


            # Save results
            if save_root_new is not None:
                if not os.path.exists(save_root_new):
                    os.makedirs(save_root_new)
            save_file = open(save_root_new + 'model' + str(model_num) + '.json',
                             "w")
            json.dump(res, save_file)
            save_file.close()
        plt.savefig(save_root_new + 'model' + str(model_num) + '.pdf')
        plt.show()
