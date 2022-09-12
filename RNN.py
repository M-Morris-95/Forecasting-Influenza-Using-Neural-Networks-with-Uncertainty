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

@tf.function
def softplus(x, k=1.0):
    k = tf.cast(k, float)
    return tf.math.divide(tf.math.log(tf.math.add(1.0, tf.math.exp(tf.math.multiply(x, k)))), k)

class LSTM_old_style(tf.keras.Model):
    lag = 14
    model_type = 'FF'
    forecast_type = 'single'
    loss = 'NLL'
    query_forecast = 'False'

    def __init__(self, rnn_units=25, kl_weight=1e-5, op_scale=0.1,
                 prior_scale=1e-4, dense_post_scale=0.1,  gamma=28, **kwargs):
        super().__init__()

        self.units = int(rnn_units)
        self.gamma = gamma
        self.rnn_cell = tf.keras.layers.LSTMCell(self.units)
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
                                               scale_diag=1e-5 + dense_post_scale * softplus(t[..., n:], k=1.0)),
                )),
            ])

        def prior(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            prior_model = tf.keras.Sequential([
                tfp.layers.DistributionLambda(lambda t : tfd.Independent(
                    tfd.MultivariateNormalDiag(loc = tf.zeros(n), scale_diag = prior_scale*tf.ones(n))))
            ])
            return prior_model

        self.dense_var = tfp.layers.DenseVariational(units=2,
                                                      make_posterior_fn=posterior_mean_field,
                                                      make_prior_fn=prior,
                                                      kl_weight=kl_weight,
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

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def eval(num_features=50, rnn_units=80, op_scale=0.1, prior_scale=0.1, rnn_post_scale=0.05, dense_post_scale=0.05, kl_weight=0.5,
         epochs=50, n_folds=5, verbose=False):
    try:
        rnn_units=int(rnn_units)
        op_scale=float(op_scale)
        prior_scale=float(prior_scale)
        rnn_post_scale=float(rnn_post_scale)
        dense_post_scale=float(dense_post_scale)
        kl_weight = float(kl_weight)
        num_features=int(num_features)
        epochs = int(epochs)
        window_size=56

        season = 2016
        gamma=28

        _data = DataConstructor(test_season=season, data_season=season-1, n_queries=num_features-1, gamma=gamma,
                                window_size=window_size)
        x_train, y_train, x_test, y_test = _data(LSTM_old_style.model_type, LSTM_old_style.forecast_type, LSTM_old_style.query_forecast)

        nll=0
        plt.figure(figsize=(5,3), dpi=200)
        for fold in range(n_folds):
            x_val = x_train[-(365*(fold+1)): -(365*(fold)+1)]
            x_tr = x_train[:-(365*(fold+1))]

            y_val = y_train[-(365*(fold+1)): -(365*(fold)+1)]
            y_tr = y_train[:-(365*(fold+1))]

            val_dates = _data.train_dates[-365*(fold+1): -(365*fold)-1]
            train_dates = _data.train_dates[:-365*(fold+1)]
            model = LSTM_old_style(rnn_units=rnn_units,
                                         kl_weight=kl_weight,
                                         n_batches=x_train.shape[0]/32,
                                         num_features=num_features,
                                         op_scale=op_scale,
                                         prior_scale=prior_scale,
                                         rnn_post_scale=rnn_post_scale,
                                         dense_post_scale=dense_post_scale,
                                         gamma=gamma)

            model(x_val)

            def loss(y, p_y):
                return -p_y.log_prob(y)
            model.compile(loss=loss,
                          optimizer=tf.optimizers.Adam(learning_rate=0.001),
                          )


            [y_tr[:, :14, -1:], y_tr[:, -1, -1:]]
            model.fit(x_tr, [y_tr[:, :14, -1:], y_tr[:, -1, -1:]], epochs=epochs, batch_size=32, verbose=True)
            mean, std, uncertainties = model.predict(x_val)


            for idx, g in enumerate([7, 14, 21, 28]):
                plt.subplot(4,1,idx+1)
                df = pd.DataFrame(columns=['True', 'Pred', 'Std'],
                                  index=_data.test_dates + dt.timedelta(days=int(28)),
                                  data=np.asarray([y_test[:, 27, -1], mean[:, -1], std[:, -1]]).T)
                df = rescale_df(df, _data.ili_scaler)
                nll -= Metrics.nll(df)

                plt.plot(df['True'], color='black')
                plt.plot(df['Pred'])
                plt.fill_between(df.index, df['Pred']-df['Std'], df['Pred']+df['Std'], linewidth=0, alpha=0.3)

        root = 'Results/feedback_rnn_bayes/'
        figs = os.listdir(root)

        nums=[-1]
        for f in figs:
            if 'fig' in f:
                nums.append(int(f.split('_')[1].split('.')[0]))

        plt.savefig(root+'fig_'+str(max(nums)+1)+'.pdf')
        if not np.isfinite(nll):
            return -25
        else:
            return nll
    except:
        return -25

if __name__ == "__main__":
    from Data_Builder import *
    from bayes_opt import BayesianOptimization
    eval()

    pbounds = {'rnn_units': (25,125),
               'op_scale':(0.01, 0.1),
               'prior_scale':(1e-4, 1e-1),
               'rnn_post_scale':(1e-4, 1e-1),
               'dense_post_scale':(1e-4, 1e-1),
               'kl_weight':(1e-5, 1e-2),
               'num_features':(25, 100),
               'epochs':(10,100),
               }
    optimizer = BayesianOptimization(
        f=eval,
        pbounds=pbounds,
        random_state=1,
        verbose=2
    )
    best = -1000
    num = 200
    init_points=10
    n_iter=20

    save_dir = 'Results/old_lstm/'

    pbounds = {'rnn_units': (25,125),
               'op_scale':(0.01, 0.1),
               'prior_scale':(1e-4, 1e-1),
               'rnn_post_scale':(1e-4, 1e-1),
               'dense_post_scale':(1e-4, 1e-1),
               'kl_weight':(1e-5, 1e-2),
               'num_features':(25, 100),
               'epochs':(10),
               }

    while (len(optimizer.res) <= num):
        if len(optimizer.res) == 0:
            optimizer.probe(
                params={'rnn_units': 25,
                        'op_scale':0.1,
                        'prior_scale':0.1,
                        'rnn_post_scale':1e-2,
                        'dense_post_scale':1e-2,
                        'kl_weight':1e-5,
                        'num_features':25,
                        'epochs':25
                        },
            )

            optimizer.maximize(
                init_points=init_points,
                n_iter=0)
            save_steps(save_dir, optimizer)
            print('initialised')
        else:
            print(len(optimizer.res))
            optimizer.maximize(
                init_points=0,
                n_iter=n_iter)
            save_steps(save_dir, optimizer)






















    # self.f_posterior = posterior_rnn(h * d + h * h + h)
    # self.i_posterior = posterior_rnn(h * d + h * h + h)
    # self.o_posterior = posterior_rnn(h * d + h * h + h)
    # self.c_posterior = posterior_rnn(h * d + h * h + h)

    # self.wf = self.add_weight(shape=(h, d), initializer="uniform", name="wf")
    # self.uf = self.add_weight(shape=(h, h), initializer="uniform", name="uf")
    # self.bf = self.add_weight(shape=(h,), initializer="uniform", name="bf")

    # self.wi = self.add_weight(shape=(h, d), initializer="uniform", name="wi")
    # self.ui = self.add_weight(shape=(h, h), initializer="uniform", name="ui")
    # self.bi = self.add_weight(shape=(h,), initializer="uniform", name="bi")

    # self.wo = self.add_weight(shape=(h, d), initializer="uniform", name="wo")
    # self.uo = self.add_weight(shape=(h, h), initializer="uniform", name="uo")
    # self.bo = self.add_weight(shape=(h,), initializer="uniform", name="bo")

    # self.wc = self.add_weight(shape=(h, d), initializer="uniform", name="wc")
    # self.uc = self.add_weight(shape=(h, h), initializer="uniform", name="uc")
    # self.bc = self.add_weight(shape=(h,), initializer="uniform", name="bc")


    # f_q = self.f_posterior(xt)
    # i_q = self.i_posterior(xt)
    # o_q = self.o_posterior(xt)
    # c_q = self.c_posterior(xt)

    # self.add_loss(1e-5*kullback_leibler.kl_divergence(f_q, p))
    # self.add_loss(1e-5*kullback_leibler.kl_divergence(i_q, p))
    # self.add_loss(1e-5*kullback_leibler.kl_divergence(o_q, p))
    # self.add_loss(1e-5*kullback_leibler.kl_divergence(c_q, p))

    # wi, ui, bi = tf.split(tf.convert_to_tensor(value=i_q), [h * d, h*h, h], axis=-1)
    # wo, uo, bo = tf.split(tf.convert_to_tensor(value=o_q), [h * d, h*h, h], axis=-1)
    # wc, uc, bc = tf.split(tf.convert_to_tensor(value=c_q), [h * d, h*h, h], axis=-1)