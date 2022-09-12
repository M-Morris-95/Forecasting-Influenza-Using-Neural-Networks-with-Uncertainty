import numpy as np

from Data_Builder import *
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from optimizer_tools import *
from LSTM_old_style import *
from Data_Builder import *

def load_best(save_dir):
    try:
        file = open(save_dir+'optimiser_results.json', 'r')
        res = json.load(file)
        sorted = np.argsort(np.asarray([r['target'] for r in res]))[::-1]
        res = [res[s] for s in sorted]

        max = res[0]

        return {'best':max, 'results':res}
    except Exception as e:
        print(e)
        return 0
        pass

def convert_to_df(op, true, dates, _data, type='FF'):
    try:
        op[2]['Model_Uncertainty'] = op[2].pop('model')
        op[2]['Data_Uncertainty'] = op[2].pop('data')
    except:
        pass

    if type == 'multi':
        if len(op) == 3:
            columns=['True', 'Pred', 'Std', 'Model_Uncertainty', 'Data_Uncertainty']
        else:
            columns=['True', 'Pred', 'Std']

        gammas = list(np.linspace(7, int(7*op[0].shape[1]/7), int(op[0].shape[1]/7)).astype(int))

        res = {}
        for g in gammas:
            mean = op[0][:, g-1, -1].squeeze()
            std = op[1][:, g-1, -1].squeeze()

            data = [true[:, g-1, -1].squeeze(), mean, std]

            try:
                model_uncertainty = op[2]['Model_Uncertainty'][:, g-1, -1].squeeze()
                data_uncertainy = op[2]['Data_Uncertainty'][:, g-1, -1].squeeze()

                data.append(model_uncertainty)
                data.append(data_uncertainy)
            except:
                pass
            df = pd.DataFrame(index = dates+dt.timedelta(days=int(g)), columns = columns, data = np.asarray(data).T)
            df = rescale_df(df, _data.ili_scaler)
            res[g] = df

        return res

    if type == 'single':
        if len(op) == 3:
            columns=['True', 'Pred', 'Std', 'Model_Uncertainty', 'Data_Uncertainty']
        else:
            columns=['True', 'Pred', 'Std']

        mean = op[0].squeeze()
        std = op[1].squeeze()

        data = [true.squeeze(), mean, std]

        try:
            model_uncertainty = op[2]['Model_Uncertainty'].squeeze()
            data_uncertainy = op[2]['Data_Uncertainty'].squeeze()

            data.append(model_uncertainty)
            data.append(data_uncertainy)
        except:
            pass

        df = pd.DataFrame(index = dates, columns = columns, data = np.asarray(data).T)
        df = rescale_df(df, _data.ili_scaler)
        return {'gamma':df}

class Eval_Fn:
    def __init__(self, root='', model=LSTM_old_style, runs=1, n_folds = 5, season = 2015, gamma=14, epochs=5, plot=True, verbose=True, n_queries=49, min_score=-25, data_gamma=None):
        self.model = model
        self.runs = runs
        self.n_folds = n_folds
        self.season = season
        self.gamma = gamma
        self.epochs = epochs
        self.plot = plot
        self.verbose = verbose
        self.min_score = min_score
        self.root = root
        self.n_queries = n_queries
        if data_gamma == None:
            self.data_gamma = gamma
        else:
            self.data_gamma = data_gamma

    def __call__(self, **kwargs):
        if not 'n_queries' in kwargs:
            kwargs['n_queries'] = self.n_queries

        _data = DataConstructor(test_season=self.season, gamma=self.data_gamma, **kwargs)
        x_train, y_train, x_test, y_test = _data(self.model.model_type, self.model.forecast_type, self.model.query_forecast)

        score = {}
        plt.clf()

        for fold in range(self.n_folds):
            try:
                if isinstance(x_train, list):
                    x_val = [d[-(365*(fold+1)): -(365*(fold)+1)] for d in x_train]
                    x_tr = [d[:-(365*(fold+1))] for d in x_train]

                else:
                    x_val = x_train[-(365*(fold+1)): -(365*(fold)+1)]
                    x_tr = x_train[:-(365*(fold+1))]

                y_val = y_train[-(365*(fold+1)): -(365*(fold)+1)]
                y_tr = y_train[:-(365*(fold+1))]

                val_dates = _data.train_dates[-365*(fold+1): -(365*fold)-1]
                train_dates = _data.train_dates[:-365*(fold+1)]

                _model = self.model(**kwargs)

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

                _model.compile(loss=loss,
                               optimizer=tf.optimizers.Adam(learning_rate=lr),
                               )


                _model.fit(x_tr, y_tr, epochs=epochs, batch_size=32, verbose=self.verbose)

                predictions = _model.predict(x_val, verbose=self.verbose)

                df = convert_to_df(predictions, y_val, val_dates + dt.timedelta(days = self.gamma), _data, type=_model.forecast_type)

                try:
                    score[fold] = Metrics.nll(df[self.gamma])
                except:
                    score[fold] = np.sum(np.asarray([Metrics.nll(d) for d in df.values()]))

                if self.plot:
                    for idx, d in enumerate(df.values()):
                        plt.subplot(len(df.keys()), 1, idx+1)
                        plt.plot(d.index, d['True'], color='black')
                        plt.plot(d.index, d['Pred'], color='red')
                        plt.fill_between(d.index, d['Pred']+d['Std'], d['Pred']-d['Std'], color='red', alpha=0.3)
            except Exception as e:
                score[fold] = -self.min_score
                print(e)

        if self.plot:
            if not os.path.exists(self.root):
                os.mkdir(self.root)
            figs = os.listdir(self.root)
            nums=[-1]
            for f in figs:
                if 'fig' in f:
                    nums.append(int(f.split('_')[1].split('.')[0]))

            plt.savefig(self.root+'fig_'+str(max(nums)+1)+'.pdf')

        try:
            if np.isfinite(-sum(score.values())):
                return -sum(score.values())
            else:
                return self.min_score
        except:
            return self.min_score

if __name__ == '__main__':
    # from FF import *
    # from feedback_nowcasting_forecasting import *
    from Feedback_Queries import *
    # from LSTM_old_style import *


    from bayes_opt import BayesianOptimization
    from Test_Fn import Test_fn

    max_iter =250
    model = FeedBack_Forecasting
    gamma = 14
    root = 'Results/feedback_rnn/'
    epochs = 50

    eval = Eval_Fn(model=model, gamma=gamma, data_gamma = 28, epochs=epochs, n_folds=4, verbose=False)

    pbounds = {'rnn_units': (25,125),
               'kl_weight':(1e-3, 1.0),
               'op_scale':(0.01, 0.1),
               'prior_scale':(1e-4, 1e-1),
               'dense_post_scale':(1e-4, 1e-1)
               }

    optimizer = BayesianOptimization(
        f=eval,
        pbounds=pbounds,
        random_state=1,
        verbose=2
    )

    # optimizer = load_steps(root, optimizer)
    # while len(optimizer.res) < max_iter:
    #     optimizer.maximize(
    #         init_points=10,
    #         n_iter=10)
    #     save_steps(root, optimizer)

    Test_fn(root = root, model = model, gammas=[gamma], epochs=epochs, test_seasons = [2015])





