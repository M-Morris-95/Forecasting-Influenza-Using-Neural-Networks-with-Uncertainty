import os
from DataConstructor import *
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import Metrics
import numpy as np



from optimiser_tools import *

def Test_fn(root, model, n_queries=28, batch_size=32, lr_power = -4.0, test_seasons = [2015,2016,2017,2018], gammas = [7,14,21,28], runs=10, epochs=10, plot=True, verbose=True, data_gamma=None, **kwargs):
    'Function to test a model regardless of type'
    plt.clf()
    if not os.path.exists(root):
        os.mkdir(root)
    try:
        hyper_parameters = load_best(root)
        params = hyper_parameters['best']['params']
    except:
        params = kwargs

    skill = pd.DataFrame(index=gammas, columns=test_seasons)
    res = {gamma: {} for gamma in gammas}
    for gamma in gammas:
        res[gamma] = {test_season: {} for test_season in test_seasons}

    for model_num in range(runs):
        for season_idx, season in enumerate(test_seasons):
            if model.model_type == 'feedback':
                gammas = [gammas[-1]]

            for gamma_idx, gamma in enumerate(gammas):
                if not 'n_queries' in params:
                    params['n_queries'] = n_queries

                params['gamma'] = gamma

                _data = DataConstructor(test_season=season, **params)
                x_train, y_train, x_test, y_test = _data(model.model_type, model.forecast_type, model.query_forecast)

                params['n_batches']=np.ceil(x_train.shape[0] / batch_size)
                _model = model( **params)

                if _model.loss == 'NLL':
                    def loss(y, p_y):
                        return -p_y.log_prob(y)
                if _model.loss == 'MSE':
                    loss = tf.keras.losses.mean_squared_error

                if 'epochs' in params:
                    epochs = int(params['epochs'])

                if 'lr_power' in kwargs:
                    lr = np.power(10, kwargs['lr_power'])
                else:
                    lr = 1e-3


                _model.compile(loss=loss,
                              optimizer=tf.optimizers.Adam(learning_rate=lr),
                              )

                pred = _model(x_test)
                _model.fit(x_train, y_train, validation_data = (x_train[-200:], y_train[-200:]), epochs=epochs, batch_size=batch_size, verbose=verbose)

                predictions = _model.predict(x_test, n_steps=50)

                df = convert_to_df(predictions, y_test, _data.test_dates, _data, type=_model.forecast_type, gamma=gamma)

                try:
                    for g in [7,14,21,28]:
                        plt.subplot(2,2, int(g/7))
                        plt.plot(df[g]['True'], color='black')
                        plt.plot(df[g]['Pred'], color='red')
                        plt.fill_between(df[g].index, df[g]['Pred']+ df[g]['Std'],df[g]['Pred']- df[g]['Std'], color='red', alpha=0.3)
                        print(Metrics.skill(df[g]))
                    plt.show()
                except:
                    pass

                for g in df:
                    if len(skill.index) != len(list(df.keys())) and _model.model_type=='feedback':
                        skill = pd.DataFrame(index = list(df.keys()), columns = test_seasons)

                    if len(list(res.keys())) != len(list(df.keys())) and _model.model_type=='feedback':
                        res = {int(gamma): {} for gamma in list(df.keys())}

                        for gamma in list(df.keys()):
                            res[int(gamma)] = {test_season: {} for test_season in test_seasons}

                    skill.loc[g][season] = Metrics.skill(df[g])
                for g in df:
                    res[g][season]['test_prediction'] = df[g].to_json()
        if plot == True:
            print(skill)

            for gamma_idx, gamma in enumerate(res.keys()):
                if len(res.keys()) != 1:
                    plt.subplot(len(res.keys()), 1, gamma_idx+1)
                for season_idx, season in enumerate(res[gamma].keys()):
                    df = pd.read_json(res[gamma][season]['test_prediction'])

                    plt.scatter(df.index, df['True'], color='black')
                    plt.plot(df['Pred'], color='red')
                    plt.fill_between(df.index, df['Pred']-df['Std'], df['Pred']+df['Std'], color='red', alpha=0.3)
            plt.show()

        if root is not None:
            if not os.path.exists(root):
                os.makedirs(root)
        save_file = open(root + 'model' + str(model_num) + '.json',
                         "w")
        json.dump(res, save_file)
        save_file.close()

if __name__ == '__main__':
    from IRNN import *
    from FF import *
    from SRNN import *

    Test_fn(root = 'Results/FF/', model = FF, gammas=[7,14,21,28], epochs =40, test_seasons = [2015])