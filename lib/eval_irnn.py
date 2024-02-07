import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
from sklearn.preprocessing import MinMaxScaler

from scipy import interpolate
import os
import Metrics

from lib.IRNN_Full_Bayes import *
from lib.IRNN import *
from FF import *
from SRNN import *

from scipy.stats import norm

from train_functions import fit
from DataConstructor import *
import numpy as np
import tqdm
import datetime as dt
import pandas as pd

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp
import Metrics
from lib.utils import *
from DataConstructor import *

tfd = tfp.distributions

# Class to evaluate a set of hyper parameters. 
# Using a class allows the __call__ function to be used for different models with different configurations
# class is used wih bayesian optimization to find the best hyper parameters
class Eval_Fn:
    def __init__(self, root='', model=IRNN_Bayes, country='US', n_folds = 5, season = 2017, gamma=28, plot=True, verbose=True, n_queries=49, min_score=-25, **kwargs):
        self.model = model  
        self.n_folds = n_folds      # number of folda (k fold cross validation)
        self.season = season
        self.gamma = gamma      
        self.plot = plot            # save plots validation set forecasts  
        self.verbose = verbose      # print during training
        self.min_score = min_score  # score to give hyper paremeters if they break and get -infinity
        self.root = root            # save directory
        self.country = country
        # get data for training, text data unused.
        
        self._data = DataConstructor(test_season = season, country=country, full_year=False, gamma = 28, window_size = 54, teacher_forcing=True, n_queries = 99)
        self.x_train, self.y_train, self.x_test, self.y_test = self._data()

#         self.x_train, self.y_train, self.x_test, self.y_test = self._data(self.model.model_type, self.model.forecast_type, self.model.query_forecast)
        self.x_train = tf.cast(self.x_train, tf.float32)
        self.y_train = tf.cast(self.y_train, tf.float32)
        self.x_test = tf.cast(self.x_test, tf.float32)
        self.y_test = tf.cast(self.y_test, tf.float32)

    def __call__(self, batch_size = 32, **kwargs):
        tf.keras.backend.clear_session()
        score = {}

        for fold in range(self.n_folds):
#             try:
                if 'n_op' in kwargs:
                    kwargs['n_op'] = int(kwargs['n_op'])
                # split data into train and validation folds
                if isinstance(self.x_train, list):
                    x_val = [d[-(365*(fold+1)): -(365*(fold)+1)] for d in self.x_train]
                    x_tr = [d[:-(365*(fold+1))] for d in self.x_train]

                else:
                    x_val = self.x_train[-(365*(fold+1)): -(365*(fold)+1)]
                    x_tr = self.x_train[:-(365*(fold+1))]

                y_val = self.y_train[-(365*(fold+1)): -(365*(fold)+1)]
                y_tr = self.y_train[:-(365*(fold+1))]

                val_dates = self._data.train_dates[-365*(fold+1): -(365*fold)-1]
                train_dates = self._data.train_dates[:-365*(fold+1)]

                x_val = x_val[:,:,-kwargs['n_op']:]
                y_val = y_val[:,:,-kwargs['n_op']:].numpy()

                train_dataset = tf.data.Dataset.from_tensor_slices((x_tr[:,:,-kwargs['n_op']:], y_tr[:,:,-kwargs['n_op']:]))
                train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)


                _model = self.model(**kwargs)

                # define loss, epochs learning rate
                if _model.loss == 'NLL':
                    def loss(y, p_y):
                        return -p_y.log_prob(y)
                if _model.loss == 'MSE':
                    loss = tf.keras.losses.mean_squared_error

                if 'epochs' in kwargs:
                    epochs = int(kwargs['epochs'])
                else:
                    epochs = self.epochs

                if 'lr_power' in kwargs:
                    lr = np.power(10, kwargs['lr_power'])
                else:
                    lr = 1e-3

                def loss_fn(y, p_y):
                    return -p_y.log_prob(y)

                optimizer = tf.optimizers.Adam(learning_rate=1e-3)

                _model(x_val)
                prediction_steps = 3
                _model, history = fit(_model, 
                        train_dataset,
                        optimizer=optimizer, 
                        epochs = epochs, 
                        loss_fn = loss_fn,  
                        prediction_steps = prediction_steps,
                        speedy_training=False
                        )

                predictions = _model.predict(x_val, 25, verbose=True)

                df = convert_to_df(predictions, y_val, val_dates + dt.timedelta(days = self.gamma), self._data, type=_model.forecast_type)

                # get score for fold, 2 options depending on whether the forecast is a list (IRNN) or a single prediction
                try:
                    score[fold] = Metrics.nll(df[self.gamma])
                except:
                    score[fold] = np.sum(np.asarray([Metrics.nll(d) for d in df.values()]))

#             except Exception as e:
#                 score[fold] = -self.min_score
#                 print(e)
        try:
            # NLL can be give nan values, try to prevent this breaking things
            if np.isfinite(-sum(score.values())):
                return -sum(score.values())
            else:
                return self.min_score
        except:
            return self.min_score