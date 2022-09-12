import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

import tqdm
import numpy as np

import Metrics
from optimizer_tools import *

tfd = tfp.distributions

root = 'Results/FF/'
@tf.function
def softplus(x, k=1.0):
    k = tf.cast(k, float)
    return tf.math.divide(tf.math.log(tf.math.add(1.0, tf.math.exp(tf.math.multiply(x, k)))), k)

class FF_Model(tf.keras.Model):
    lag = 14
    model_type = 'FF'
    forecast_type = 'single'
    loss = 'NLL'
    query_forecast = 'False'

    def __init__(self, units1=50, units2=50, kl_weight=0.1, op_scale=0.1,
                 prior_scale=1e-4,dense_post_scale=0.1, n_batches=100, **kwargs):
        super().__init__()

        units1 = int(units1)
        units2 = int(units2)

        self.dense1 = tf.keras.layers.Dense(units = units1, activation='ReLU')
        self.flatten = tf.keras.layers.Flatten()
        self.dense2 = tf.keras.layers.Dense(units = units2, activation='ReLU')

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
                                                     kl_weight=kl_weight/n_batches,
                                                     kl_use_exact=True)

        self.DistributionLambda = tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :-1],
                                 scale=op_scale * softplus(t[..., -1:], k=1.0)
                                 )
        )

    def __call__(self, inputs, training=None):

        x = self.dense1(inputs)
        x = self.flatten(x)
        x = self.dense2(x)

        x = self.dense_var(x)
        predictions = self.DistributionLambda(x)

        return predictions

    def predict(self, x, n_steps=25, batch_size=None, verbose=False, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        pred = []
        for _ in tqdm.trange(n_steps, disable = np.invert(verbose)):
            pred.append(self(x))

        means = np.asarray([p.mean() for p in pred])
        vars = np.asarray([p.variance() for p in pred])
        mean = means.mean(0)

        std = np.sqrt(means.var(0) + vars.mean(0))

        return mean, std, {'Model_Uncertainty': np.sqrt(means.var(0)), 'Data_Uncertainty':np.sqrt(vars.mean(0))}

if __name__ == "__main__":
    from Data_Builder import *
    from bayes_opt import BayesianOptimization
    from Eval_Fn import *
    from Test_Fn import *
    from FF import *

    from bayes_opt import BayesianOptimization
    from Test_Fn import Test_fn
    gamma= 14
    max_iter =500
    model = FF_Model

    def Test(gamma):
        root = 'Results/ff_'+str(gamma)+'/'
        model = FF_Model
        Test_fn(root = root, runs=25, model = model, gammas=[7,14,21,28], test_seasons = [2015, 2016, 2017, 2018], verbose=False)
    [Test(gamma) for gamma in [7,14,21,28]]

    root = 'Results/ff_'+str(gamma)+'/'
    model = FF_Model
    Test_fn(root = root, runs=25, model = model, gammas=[7,14,21,28], test_seasons = [2015, 2016, 2017, 2018], verbose=False)

    epochs = 50

    eval = Eval_Fn(model=model,
                   root=root,
                   gamma=gamma,
                   data_gamma=gamma,
                   epochs=epochs,
                   n_folds=4,
                   verbose=False,
                   plot=False)

    pbounds = {'units1': (25,125),
               'units2': (5,100),
               'op_rho':(0.1, 10),
               'num_features':(20, 100),
               'kl_weight':(1e-2, 1.0),
               'op_scale':(0.01, 0.1),
               'prior_scale':(1e-3, 1e-2),
               'dense_post_scale':(1e-2, 1e-1),
               'q_rho':(1,5),
               'epochs':(10,100)
               }

    optimizer = BayesianOptimization(
        f=eval,
        pbounds=pbounds,
        random_state=1,
        verbose=2
    )

    optimizer = load_steps(root, optimizer)
    while len(optimizer.res) < max_iter:
        optimizer.maximize(
            init_points=10,
            n_iter=10)

        save_steps(root, optimizer)

    root = 'Results/ff_'+str(7)+'/'
    model = FF_Model
    gamma = 7
    Test_fn(root = root, model = model, gammas=[gamma], test_seasons = [2015, 2016,2017,2018], verbose=False)


    if not os.path.exists(root):
        os.mkdir(root)

    pbounds = {'units1': (25,125),
               'units2': (25,125),
               'op_scale':(0.01, 0.1),
               'prior_scale':(1e-4, 1e-1),
               'dense_post_scale':(1e-4, 1e-1),
               'kl_weight':(1e-3, 1.0),
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

    optimizer = load_steps(root, optimizer)

    test(root, **optimizer.max['params'])

    while (len(optimizer.res) <= num):
        if len(optimizer.res) == 0:
            optimizer.probe(
                params={'units1': 25,
                        'units2': 25,
                        'op_scale':0.1,
                        'prior_scale':0.1,
                        'dense_post_scale':1e-2,
                        'kl_weight':1.0,
                        'num_features':25,
                        'epochs':25
                        },
            )

            optimizer.maximize(
                init_points=init_points,
                n_iter=0)
            save_steps(root, optimizer)
            print('initialised')
        else:
            print(len(optimizer.res))
            optimizer.maximize(
                init_points=0,
                n_iter=n_iter)
            save_steps(root, optimizer)


