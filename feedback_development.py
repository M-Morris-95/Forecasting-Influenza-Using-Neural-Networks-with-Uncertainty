import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import Metrics
from tensorflow_probability.python.distributions import kullback_leibler
import matplotlib
matplotlib.use("TkAgg")

import json
from optimizer_tools import *
import tqdm
from Bayes_GRU import Bayes_GRU
from Bayes_Dense import Bayes_Dense

sigmoid = tf.keras.activations.sigmoid
tanh = tf.keras.activations.tanh
multiply = tf.math.multiply
tfd = tfp.distributions

root = 'Results/feedback_rnn_teach_forced/'
if not os.path.exists(root):
    os.mkdir(root)

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
                 prior_scale=0.005, gamma=28, n_batches=100, bayes=False, full_cov=False, **kwargs):
        super().__init__()
        num_features = int(n_queries+1)
        self.num_features = num_features
        rnn_units = int(rnn_units)
        self.bayes = bayes
        self.gamma = gamma
        self.n_batches = n_batches

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

        self.gru_cell = tf.keras.layers.GRUCell(rnn_units)
        self.gru = tf.keras.layers.RNN(self.gru_cell, return_state=True)

        self.dense_variational = tfp.layers.DenseVariational(units=num_features*2,
                                             make_posterior_fn=posterior,
                                             make_prior_fn=prior_trainable,
                                             kl_weight = self.kl_weight * self.kl_weight/n_batches
                                             )

        c = np.log(np.expm1(1.))
        self.distribution_lambda = tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :num_features],
                                 scale=1e-5 + op_scale*tf.nn.softplus(c + t[..., num_features:])
                                 )
        )

    def __call__(self, inputs, training=None, mean = 0, std=0.5, decay=1.015):
        predictions = []
        x, *state = self.gru(inputs[:, :-self.lag, :])
        x = self.dense_variational(x)

        if training:
            # kl = tf.keras.losses.kl_divergence(self.dense_variational._posterior(x), self.dense_variational._prior(x))/self.n_batches
            kl = kullback_leibler.kl_divergence(self.dense_variational._posterior(x),self.dense_variational._prior(x))/self.n_batches

        predictions.append(x)

        for i in range(self.gamma-1):
            x = self.distribution_lambda(x).sample()
            if i < self.lag:
                x = tf.concat([inputs[:, -self.lag+i, :-1], x[:, -1:]], 1)
            x, state = self.gru_cell(x, states=state)
            x = self.dense_variational(x)
            predictions.append(x)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        predictions = self.distribution_lambda(predictions)

        if not training:
            return predictions
        else:
            return predictions, kl
    def predict(self, x, n_steps=25,  mean = 0, std=0.5, decay=1.015, training = False, batch_size=None, verbose=False, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        pred = []
        for _ in tqdm.trange(n_steps, disable = np.invert(verbose)):
            pred.append(self(x, training=True, mean=mean, std=std, decay=decay))

        means = np.asarray([p.mean() for p in pred])
        vars = np.asarray([p.variance() for p in pred])
        mean = means.mean(0)

        std = np.sqrt(means.var(0) + vars.mean(0))


        return mean, std, {'model': np.sqrt(means.var(0)), 'data':np.sqrt(vars.mean(0))}


if __name__ == "__main__":
    from Eval_Fn import *
    from Test_Fn import *
    from optimizer_tools import *

    gamma=28
    season = 2018
    window_size = 49 + 14
    batch_size = 35

    epochs = 12
    kl_power = -3.3
    lr_power = -2.8
    n_queries = 96
    op_scale = 0.1
    prior_scale = 0.005
    rnn_units = 117

    lr_power = -3.8
    kl_power = -4.8
    rnn_units = 1028
    epochs = 50



    n_steps = 20
    res = {g+1:{season:{'test_prediction':0} for season in [2015,2016,2017,2018]} for g in range(28)}

    for model_num in range(10):
        for season in [2015,2016,2017,2018]:

            model = Feedback_Queries
            _data = DataConstructor(test_season=season, gamma=gamma, n_queries=n_queries, window_size=window_size)
            x_train, y_train, x_test, y_test = _data(model.model_type, model.forecast_type, model.query_forecast, dtype=tf.float32)

            train_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(x_train, tf.float32),tf.cast(y_train, tf.float32)))
            train_dataset = train_dataset.batch(batch_size)

            test_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(x_test, tf.float32),tf.cast(y_test, tf.float32)))
            test_dataset = test_dataset.batch(batch_size)
            n_batches = int(np.ceil(x_train.shape[0]/batch_size))

            _model = Feedback_Queries(kl_power = kl_power,
                                      window_size=window_size,
                                      rnn_units= rnn_units,
                                      n_queries = n_queries,
                                      n_batches= np.ceil(x_train.shape[0] / batch_size))
            def loss_fn(y, p_y):
                return -p_y.log_prob(y)
            optimizer=tf.optimizers.Adam(learning_rate=0.001)

            _model.compile(loss=loss_fn,
                           optimizer=optimizer,
                           )

            _model(x_test)

            class KL_Tracker(tf.keras.metrics.Metric):
                def __init__(self, name='KL_Tracker', **kwargs):
                    super(KL_Tracker, self).__init__(name=name, **kwargs)
                    self.kl = self.add_weight(name='kl', initializer='zeros')
                    self.count = self.add_weight(name='count', initializer='zeros')

                def update_state(self, kl, sample_weight=None):
                    self.kl.assign_add(kl)
                    self.count.assign_add(1.)

                def result(self):
                    return self.kl / self.count

            class loss_tracker(tf.keras.metrics.Metric):
                def __init__(self, name='loss_tracker', **kwargs):
                    super(loss_tracker, self).__init__(name=name, **kwargs)
                    self.total_loss = self.add_weight(name='nll', initializer='zeros')
                    self.count = self.add_weight(name='count', initializer='zeros')


                def update_state(self, nll, sample_weight=None):
                    self.total_loss.assign_add(tf.reduce_mean(nll))
                    self.count.assign_add(1.)

                def result(self):
                    return self.total_loss / self.count

            nll_metric = loss_tracker()
            kl_metric = KL_Tracker()
            val_kl_metric = KL_Tracker()
            val_nll_metric = loss_tracker()

            @tf.function
            def train_step(x, y, teacher_forced=False):
                with tf.GradientTape() as tape:
                    logits, kl = _model(x, training=True)
                    nll = loss_fn(y, logits)
                    loss_value = nll + kl
                grads = tape.gradient(loss_value, _model.trainable_weights)
                optimizer.apply_gradients(zip(grads, _model.trainable_weights))
                nll_metric.update_state(nll)
                kl_metric.update_state(kl)
                return loss_value

            @tf.function
            def evaluation_step(x, y, n_steps=25):
                logits, kl = _model(x, training=True)
                nll = loss_fn(y, logits)
                val_nll_metric.update_state(nll)
                val_kl_metric.update_state(kl)
                return loss_value

            def predict(x, n_steps=25):
                pred = [_model(x, training=False) for _ in range(n_steps)]

                means = np.asarray([p.mean() for p in pred])
                vars = np.asarray([p.variance() for p in pred])
                mean = means.mean(0)

                std = np.sqrt(means.var(0) + vars.mean(0))

                return mean, std

            def test_model(x_test, epochs, n_steps=25):
                mean, std = predict(x_test, n_steps=n_steps)
                skills = []

                for g in range(28):
                    std_g = _data.ili_scaler.inverse_transform(std[:, g, -1:])
                    mean_g = _data.ili_scaler.inverse_transform(mean[:, g, -1:])
                    true_g = _data.ili_scaler.inverse_transform(y_test[:, g, -1:])


                    df = pd.DataFrame(index = _data.test_dates - dt.timedelta(days=13 - g), columns=['True','Pred','Std'], data = np.asarray([true_g, mean_g, std_g]).squeeze().T)
                    res[g+1][season]['test_prediction'] = df.to_json()
                    res[g+1][season]['skill'] = Metrics.skill(df)
                save_file = open(root + 'model' + str(model_num) + '.json',
                                 "w")
                json.dump(res, save_file)
                save_file.close()


                for idx, g in enumerate([0, 6, 13,20,27]):
                    plt.subplot(3,2,idx+1)

                    std_g = _data.ili_scaler.inverse_transform(std[:, g, -1:])
                    mean_g = _data.ili_scaler.inverse_transform(mean[:, g, -1:])
                    true_g = _data.ili_scaler.inverse_transform(y_test[:, g, -1:])

                    plt.plot(np.linspace(0, 1, mean.shape[0]), mean_g, color='red', label='Pred')
                    plt.fill_between(np.linspace(0, 1, mean.shape[0]),
                                     (mean_g + std_g).squeeze(),
                                     (mean_g - std_g).squeeze(),
                                     color='red', alpha=0.3, linewidth=0)
                    plt.plot(np.linspace(0, 1, mean.shape[0]), true_g, '--', color='black', label='True')

                    skills.append(Metrics.skill(pd.DataFrame(columns=['True','Pred','Std'], data = np.asarray([true_g, mean_g, std_g]).squeeze().T)))
                plt.legend()
                plt.subplot(3,2,6)
                plt.text(0.2,0.2, ("1: %.3f\n"
                                   "7: %.3f\n"
                                   "14: %.3f\n"
                                   "21: %.3f\n"
                                   "28: %.3f" % (skills[0], skills[1], skills[2], skills[3], skills[4])))
                plt.suptitle(epochs)
                plt.show()

            history = {'kl_metric':[], 'nll_metric':[], 'val_nll_metric':[], 'val_kl_metric':[]}
            mse_list = []
            probs = tf.sigmoid(np.linspace(-5,5, epochs))
            for epoch in range(epochs):
                prob = epoch/(epochs-1)
                prob = probs[epoch]
                for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                    loss_value = train_step(x_batch_train, y_batch_train)

                    print("epoch: %d/%d, batch: %d/%d, nll: %.3f, kl: %.3f" % (epoch+1, epochs, step+1,n_batches,nll_metric.result(), kl_metric.result()), end='\r')


                for step, (x_batch_test, y_batch_test) in enumerate(test_dataset):
                    evaluation_step(x_batch_test, y_batch_test)
                print("epoch: %d/%d, batch: %d/%d, nll: %.3f, kl: %.3f, val nll: %.3f, val kl: %.3f" % (epoch+1, epochs, step+1,n_batches,nll_metric.result(), kl_metric.result(), val_nll_metric.result(), val_kl_metric.result()))

                history['kl_metric'].append(kl_metric.result())
                history['nll_metric'].append(nll_metric.result())
                history['val_nll_metric'].append(val_nll_metric.result())
                history['val_kl_metric'].append(val_kl_metric.result())

                kl_metric.reset_state()
                nll_metric.reset_state()
                val_nll_metric.reset_state()
                val_kl_metric.reset_state()

            plt.clf()
            plt.subplot(2,1,1)
            plt.plot(history['nll_metric'], 'r-')
            plt.plot(history['val_nll_metric'], 'b--')
            plt.title('nll')
            plt.subplot(2,1,2)
            plt.plot(history['kl_metric'], 'r-', label='train')
            plt.plot(history['val_kl_metric'], 'b--', label = 'val')
            plt.title('KL')
            plt.legend()
            plt.ylim([-1, 10])
            plt.show()

            # test_model(x_test, epoch, n_steps)


            mean, std = predict(x_test, n_steps=n_steps)
            skills = []
            for idx, g in enumerate([0, 6, 13,20,27]):
                plt.subplot(3,2,idx+1)

                std_g = _data.ili_scaler.inverse_transform(std[:, g, -1:])
                mean_g = _data.ili_scaler.inverse_transform(mean[:, g, -1:])
                true_g = _data.ili_scaler.inverse_transform(y_test[:, g, -1:])

                plt.plot(np.linspace(0, 1, mean.shape[0]), mean_g, color='red', label='Pred')
                plt.fill_between(np.linspace(0, 1, mean.shape[0]),
                                 (mean_g + std_g).squeeze(),
                                 (mean_g - std_g).squeeze(),
                                 color='red', alpha=0.3, linewidth=0)
                plt.plot(np.linspace(0, 1, mean.shape[0]), true_g, '--', color='black', label='True')

                skills.append(Metrics.skill(pd.DataFrame(columns=['True','Pred','Std'], data = np.asarray([true_g, mean_g, std_g]).squeeze().T)))
                plt.legend()
            plt.subplot(3,2,6)
            plt.text(0.2,0.2, ("1: %.3f\n"
                               "7: %.3f\n"
                               "14: %.3f\n"
                               "21: %.3f\n"
                               "28: %.3f" % (skills[0], skills[1], skills[2], skills[3], skills[4])))
            plt.suptitle(epochs)
            plt.show()



            mean, std = predict(x_train[-365:], n_steps=n_steps)
            skills = []
            for idx, g in enumerate([0, 6, 13,20,27]):
                plt.subplot(3,2,idx+1)

                std_g = _data.ili_scaler.inverse_transform(std[:, g, -1:])
                mean_g = _data.ili_scaler.inverse_transform(mean[:, g, -1:])
                true_g = _data.ili_scaler.inverse_transform(y_train[-365:, g, -1:])

                plt.plot(np.linspace(0, 1, mean.shape[0]), mean_g, color='red', label='Pred')
                plt.fill_between(np.linspace(0, 1, mean.shape[0]),
                                 (mean_g + std_g).squeeze(),
                                 (mean_g - std_g).squeeze(),
                                 color='red', alpha=0.3, linewidth=0)
                plt.plot(np.linspace(0, 1, mean.shape[0]), true_g, '--', color='black', label='True')

                skills.append(Metrics.skill(pd.DataFrame(columns=['True','Pred','Std'], data = np.asarray([true_g, mean_g, std_g]).squeeze().T)))
                plt.legend()
            plt.subplot(3,2,6)
            plt.text(0.2,0.2, ("1: %.3f\n"
                               "7: %.3f\n"
                               "14: %.3f\n"
                               "21: %.3f\n"
                               "28: %.3f" % (skills[0], skills[1], skills[2], skills[3], skills[4])))
            plt.suptitle(epochs)
            plt.show()
