import os

from Data_Builder import *
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import Metrics

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

def convert_to_df(op, true, dates, _data, type='FF', gamma=7):
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

        gammas = list(np.linspace(1, op[0].shape[1], op[0].shape[1]).astype(int))

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
            df = pd.DataFrame(index = dates+dt.timedelta(days=int(g)-13), columns = columns, data = np.asarray(data).T)
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

        return df

def Test_fn(root, model, n_queries=49, batch_size=32, test_seasons = [2015,2016,2017,2018], gammas = [7,14,21,28], runs=10, epochs=50, plot=True, verbose=True, data_gamma=None, **kwargs):
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

                if _model.model_type != 'feedback':
                    df = {gamma: df}


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
    # from FF import *
    # from feedback_nowcasting_forecasting import *
    from LSTM_old_style import *
    # from Feedback_RNN import *

    Test_fn(root = 'Results/old_lstm/', model = LSTM_old_style, gammas=[7,14,21,28], test_seasons = [2015])




    mean = 0
    std = 1
    decay= 1.1
    iter = 0

    ls = []
    for _ in range(28):
        iter = iter * decay + np.random.normal(mean, std, 1)
        ls.append(iter)
    plt.plot(ls)
    plt.show()