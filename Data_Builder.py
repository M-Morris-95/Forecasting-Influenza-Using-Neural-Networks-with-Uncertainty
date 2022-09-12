import os
import numpy as np
import tqdm
import pandas as pd
import datetime as dt
import copy
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy import interpolate
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from Search_Query_Selection import *

class DataConstructor():
    lag=14
    def __init__(self, test_season, gamma, window_size=49, data_season = None, n_queries=49, num_features = None, selection_method='distance',
                 root=None, rescale=True, type='avg', **kwargs):

        if data_season is None:
            data_season = test_season - 1
        test_season = int(test_season)
        data_season = int(data_season)
        gamma = int(gamma)
        window_size = int(window_size)

        n_queries = int(n_queries)
        if num_features is not None:
            n_queries = int(num_features) - 1

        # type is harmonic_avg or avg. harmonic is a 7 day harmonic average, avg is a 7 day avg. harmonic is not as smooth
        self.test_season = test_season
        self.gamma = gamma
        self.window_size = window_size
        self.data_season = data_season
        self.n_queries = n_queries
        self.selection_method = selection_method
        self.rescale = rescale
        self.type = type

        if root is None:
            for root in ['/home/mimorris/Data/', '/Users/michael/Documents/datasets/Data/']:
                if os.path.exists(root):
                    self.root = root
        else:
            self.root=root
        return

    def get_ili(self):
        self.wILI = pd.read_csv(os.path.join(self.root, 'ILI_rates', 'national_flu.csv'), index_col=-1, parse_dates=True)[
                'weighted_ili']


        # get daily dates
        dates = np.asarray([self.wILI.index[0] + dt.timedelta(days=i) for i in
                            range((self.wILI.index[-1] - self.wILI.index[0]).days + 1)])

        # shift to thursday
        dates = dates + dt.timedelta(days=3)

        # interpolate weekly to daily
        x = np.linspace(0, 1, self.wILI.shape[0])
        x2 = np.linspace(0, 1, dates.shape[0])
        f = interpolate.interp1d(x, self.wILI.values.squeeze(), kind = 'cubic')

        self.daily_wILI = pd.DataFrame(index=dates, columns=['wILI'], data=f(x2)).squeeze()

        if self.rescale:
            self.ili_scaler = MinMaxScaler()
            self.ili_scaler.fit(self.wILI.values.reshape(-1, 1))
            self.wILI = pd.Series(index=self.wILI.index,
                                  data=self.ili_scaler.transform(self.wILI.values.reshape(-1, 1)).squeeze())
            self.daily_wILI = pd.Series(index=self.daily_wILI.index,
                                        data=self.ili_scaler.transform(self.daily_wILI.values.reshape(-1, 1)).squeeze())

    def get_queries(self, smooth_queries=False):
        if smooth_queries:
            queries = pd.read_csv('/Users/michael/Documents/Data/Queries/US_query_data_all.csv', index_col=0)
            queries_smooth = pd.DataFrame(index=queries.index, columns = queries.columns, data= 0 )
            queries_harmonic = pd.DataFrame(index=queries.index, columns = queries.columns, data= 0)

            for idx in tqdm.trange(queries.shape[0] - 7):
                queries_smooth.iloc[idx+7] = queries.iloc[idx:idx+7].mean(0)
            queries_smooth.to_csv('/Users/michael/Documents/Data/Queries/US_query_data_7_day_avg.csv')

            for idx in tqdm.trange(queries.shape[0] - 7):
                queries_harmonic.iloc[idx+7] = (queries.iloc[idx:idx+7] * np.tile(1 / np.linspace(7,1, 7), (queries_harmonic.shape[1],1)).T).sum(0) / 2.5928571428571425
            queries_harmonic.to_csv('/Users/michael/Documents/Data/Queries/US_query_data_7_day_harmonic_avg.csv')

            n = 365
            plt.plot(queries.iloc[-n:, 0])
            plt.plot(queries_smooth.iloc[-n:, 0], label='7 day')
            plt.plot(queries_harmonic.iloc[-n:, 0], label='harmonic')
            plt.legend()
            plt.show()

        self.queries = pd.read_csv(os.path.join(self.root, 'Queries', 'US_query_data_7_day_'+self.type+'.csv'), index_col=0,
                                   parse_dates=True)
        # remove duplicate index
        self.queries = self.queries[~self.queries.index.duplicated(keep='first')]
        self.queries = self.queries.sort_index()

        # remove punctuation
        self.queries = self.queries.rename(
            columns={query: query.replace('+', ' ').replace(',', ' ') for query in self.queries.columns})

        # sort queries alphabetically and remove duplicates
        self.queries = self.queries.rename(
            columns={query: ' '.join(sorted(query.split(' '))) for query in self.queries.columns})
        self.queries = self.queries.loc[:, ~self.queries.columns.duplicated()]

        # remove nan values
        self.queries[np.invert(self.queries.isna().all(1))]

        # scale queries to 0-1
        if self.rescale:
            self.query_scaler = MinMaxScaler()
            self.query_scaler.fit(self.queries)
            self.queries = pd.DataFrame(index=self.queries.index, columns=self.queries.columns,
                                        data=self.query_scaler.transform(self.queries))

    def get_dates(self, leaveoneout=True):
        test_start = {2011:dt.date(2011, 10, 28),
                      2012:dt.date(2012, 11, 4),
                      2013:dt.date(2013, 11, 3),
                      2014:dt.date(2014, 11, 2),
                      2015:dt.date(2015, 11, 1),
                      2016:dt.date(2016, 10, 30),
                      2017:dt.date(2017, 10, 29),
                      2018:dt.date(2018, 10, 28)}
        test_end = {2011:dt.date(2012, 6, 2),
                    2012:dt.date(2013, 6, 8),
                    2013:dt.date(2014, 6, 7),
                    2014:dt.date(2015, 6, 6),
                    2015:dt.date(2016, 6, 5),
                    2016:dt.date(2017, 6, 4),
                    2017:dt.date(2018, 6, 3),
                    2018:dt.date(2019, 6, 2)}

        train_start = {2011:dt.date(2004, 3, 24),
                       2011:dt.date(2004, 3, 24),
                       2012:dt.date(2004, 3, 24),
                       2013:dt.date(2004, 3, 24),
                       2014:dt.date(2004, 3, 24),
                       2015:dt.date(2004, 3, 24),
                       2016:dt.date(2004, 3, 24),
                       2017:dt.date(2004, 3, 24),
                       2018:dt.date(2004, 3, 24)}
        train_end = {2011:dt.date(2011, 8, 19),
                     2012:dt.date(2012, 8, 15),
                     2013:dt.date(2013, 8, 14),
                     2014:dt.date(2014, 8, 13),
                     2015:dt.date(2015, 8, 12),
                     2016:dt.date(2016, 8, 11),
                     2017:dt.date(2017, 8, 10),
                     2018:dt.date(2018, 8, 9)}

        self.test_start_date = test_start[self.test_season]
        self.train_dates =  pd.date_range(train_start[self.test_season], train_end[self.test_season])
        self.test_dates = pd.date_range(test_start[self.test_season], test_end[self.test_season])

        if leaveoneout:
            d1 = pd.date_range(self.train_dates[0], self.test_dates[0]-dt.timedelta(days=1))
            d2 = pd.date_range(self.test_dates[-1]+dt.timedelta(days=1), dt.date(2019, 6, 5))
            self.train_dates = d1.append(d2)



    def __call__(self, model='feedback', forecast_type='multi', query_forecast='True', leaveoneout = True, dtype=None, look_ahead=True):
        self.get_ili()
        self.get_queries()

        selection = query_selection()
        selected_queries = selection(self.queries, self.daily_wILI, n_queries=self.n_queries, data_season=self.data_season)

        self.selected_queries = selected_queries
        self.get_dates(leaveoneout=leaveoneout)

        if model == 'feedback':
            x_train = []
            y_train = []

            x_test = []
            y_test = []
            for date in self.train_dates:
                x = self.queries.loc[date-dt.timedelta(days=self.window_size-1) : date, selected_queries]
                x['wILI'] = self.daily_wILI.loc[date-dt.timedelta(days=self.window_size-1):date-dt.timedelta(days=self.lag)]
                x = x.fillna(0)

                y = self.queries.loc[date-dt.timedelta(days=self.lag-1): date+ dt.timedelta(days=self.gamma - self.lag), selected_queries]
                y['wILI'] = self.daily_wILI

                x_train.append(x)
                y_train.append(y)

            for date in self.test_dates:
                x = self.queries.loc[date-dt.timedelta(days=self.window_size-1) : date, selected_queries]
                x['wILI'] = self.daily_wILI.loc[date-dt.timedelta(days=self.window_size-1):date-dt.timedelta(days=self.lag)]
                x = x.fillna(0)

                y = self.queries.loc[date-dt.timedelta(days=self.lag-1): date+ dt.timedelta(days=self.gamma - self.lag), selected_queries]
                y['wILI'] = self.daily_wILI

                x_test.append(x)
                y_test.append(y)

            x_test = np.stack(x_test)
            x_train = np.stack(x_train)
            y_test = np.stack(y_test)
            y_train = np.stack(y_train)

            if query_forecast == False:
                y_train = y_train[:, :, -1:]
                y_test = y_test[:, :, -1:]

            return x_train, y_train, x_test, y_test

        if model == 'teacher_forced':
            x_train = []
            y_train = []

            x_test = []

            y_test = []
            for date in self.train_dates:
                x = self.queries.loc[date-dt.timedelta(days=self.window_size-1) : date + dt.timedelta(days=self.gamma-self.lag-1), selected_queries]
                x['wILI'] = self.daily_wILI

                y = self.queries.loc[date-dt.timedelta(days=self.lag-1): date+ dt.timedelta(days=self.gamma - self.lag), selected_queries]
                y['wILI'] = self.daily_wILI

                x_train.append(x)
                y_train.append(y)

            for date in self.test_dates:
                x = self.queries.loc[date-dt.timedelta(days=self.window_size-1) : date + dt.timedelta(days=self.gamma-self.lag-1), selected_queries]
                x['wILI'] = self.daily_wILI

                y = self.queries.loc[date-dt.timedelta(days=self.lag-1): date+ dt.timedelta(days=self.gamma - self.lag), selected_queries]
                y['wILI'] = self.daily_wILI

                x_test.append(x)
                y_test.append(y)

            x_test = np.stack(x_test)
            x_train = np.stack(x_train)
            y_test = np.stack(y_test)
            y_train = np.stack(y_train)

            if dtype is not None:
                x_train = tf.cast(x_train, dtype)
                y_train = tf.cast(y_train, dtype)
                x_test = tf.cast(x_test, dtype)
                y_test = tf.cast(y_test, dtype)

            if query_forecast == False:
                y_train = y_train[:, :, -1:]
                y_test = y_test[:, :, -1:]

            return x_train, y_train, x_test, y_test

        if model == 'FF_7':
            x_train = []
            y_train = []

            x_test = []
            y_test = []
            for date in self.train_dates:
                if self.gamma == 7:
                    x = self.queries.loc[date-dt.timedelta(days=self.window_size-1 + 7):date-dt.timedelta(days=7), selected_queries]
                    x['wILI'] = self.daily_wILI.loc[date-dt.timedelta(days=self.window_size-1 + self.lag):date-dt.timedelta(days=self.lag)].values
                    y = self.daily_wILI.loc[date-dt.timedelta(days=self.lag-1):date+dt.timedelta(days=self.gamma-self.lag)]
                else:
                    x = self.queries.loc[date-dt.timedelta(days=self.window_size-1):date, selected_queries]
                    x['wILI'] = self.daily_wILI.loc[date-dt.timedelta(days=self.window_size-1 + self.lag):date-dt.timedelta(days=self.lag)].values
                    y = self.daily_wILI.loc[date-dt.timedelta(days=self.lag-1):date+dt.timedelta(days=self.gamma-self.lag)]

                x_train.append(x)
                y_train.append(y)

            for date in self.test_dates:
                if self.gamma == 7:
                    x = self.queries.loc[date-dt.timedelta(days=self.window_size-1 + 7):date-dt.timedelta(days=7), selected_queries]
                    x['wILI'] = self.daily_wILI.loc[date-dt.timedelta(days=self.window_size-1 + self.lag):date-dt.timedelta(days=self.lag)].values
                    y = self.daily_wILI.loc[date-dt.timedelta(days=self.lag-1):date+dt.timedelta(days=self.gamma-self.lag)]

                else:
                    x = self.queries.loc[date-dt.timedelta(days=self.window_size-1):date, selected_queries]
                    x['wILI'] = self.daily_wILI.loc[date-dt.timedelta(days=self.window_size-1 + self.lag):date-dt.timedelta(days=self.lag)].values
                    y = self.daily_wILI.loc[date-dt.timedelta(days=self.lag-1):date+dt.timedelta(days=self.gamma-self.lag)]

                x_test.append(x)
                y_test.append(y)

            x_test = np.stack(x_test)
            x_train = np.stack(x_train)
            y_test = np.stack(y_test)
            y_train = np.stack(y_train)

            if forecast_type == 'single':
                y_test = y_test[:,-1]
                y_train = y_train[:,-1]

            return x_train, y_train, x_test, y_test

        if model == 'FF':
            x_train = []
            y_train = []

            x_test = []
            y_test = []
            for date in self.train_dates:
                if look_ahead:
                    x = self.queries.loc[date-dt.timedelta(days=self.window_size-1):date, selected_queries]
                else:
                    x = self.queries.loc[date-dt.timedelta(days=self.window_size-1 + self.lag):date-dt.timedelta(days=self.lag), selected_queries]
                x['wILI'] = self.daily_wILI.loc[date-dt.timedelta(days=self.window_size-1 + self.lag):date-dt.timedelta(days=self.lag)].values
                y = self.daily_wILI.loc[date-dt.timedelta(days=self.lag-1):date+dt.timedelta(days=self.gamma-self.lag)]

                x_train.append(x)
                y_train.append(y)

            for date in self.test_dates:
                if look_ahead:
                    x = self.queries.loc[date-dt.timedelta(days=self.window_size-1):date, selected_queries]
                else:
                    x = self.queries.loc[date-dt.timedelta(days=self.window_size-1 + self.lag):date-dt.timedelta(days=self.lag), selected_queries]
                x['wILI'] = self.daily_wILI.loc[date-dt.timedelta(days=self.window_size-1 + self.lag):date-dt.timedelta(days=self.lag)].values
                y = self.daily_wILI.loc[date-dt.timedelta(days=self.lag-1):date+dt.timedelta(days=self.gamma-self.lag)]

                x_test.append(x)
                y_test.append(y)

            x_test = np.stack(x_test)
            x_train = np.stack(x_train)
            y_test = np.stack(y_test)
            y_train = np.stack(y_train)

            if forecast_type == 'single':
                y_test = y_test[:,-1]
                y_train = y_train[:,-1]

            return x_train, y_train, x_test, y_test

        if model == 'Bill':
            x_train = []
            y_train = []

            x_test = []
            y_test = []
            for date in self.train_dates:
                q = self.queries.loc[date-dt.timedelta(days=13):date, selected_queries].values.reshape(-1)
                f = self.daily_wILI.loc[date-dt.timedelta(days=self.window_size-1 + self.lag):date-dt.timedelta(days=self.lag)].values

                x = np.concatenate([q,f], 0)
                y = self.daily_wILI.loc[date-dt.timedelta(days=self.lag)+dt.timedelta(days=self.gamma)]

                x_train.append(x)
                y_train.append(y)

            for date in self.test_dates:
                q = self.queries.loc[date-dt.timedelta(days=13):date, selected_queries].values.reshape(-1)
                f = self.daily_wILI.loc[date-dt.timedelta(days=self.window_size-1 + self.lag):date-dt.timedelta(days=self.lag)].values

                x = np.concatenate([q,f], 0)
                y = self.daily_wILI.loc[date-dt.timedelta(days=self.lag)+dt.timedelta(days=self.gamma)]

                x_test.append(x)
                y_test.append(y)

            x_test = np.stack(x_test)
            x_train = np.stack(x_train)
            y_test = np.stack(y_test)
            y_train = np.stack(y_train)

            y_test = y_test[:, np.newaxis]
            y_train = y_train[:, np.newaxis]

        if forecast_type == 'single':
            y_test = y_test[:,-1:]
            y_train = y_train[:,-1:]

        return x_train, y_train, x_test, y_test

        if model == 'error_FF':

            x_in1 = []
            x_in2 = []
            y_in1 = []
            y_out = []
            for date in self.test_dates:
                x1 = self.queries.loc[date-dt.timedelta(days=self.window_size-1):date, selected_queries]
                x1['wILI'] = self.daily_wILI.loc[date-dt.timedelta(days=self.window_size-1 + self.lag):date-dt.timedelta(days=self.lag)].values
                y1 = self.daily_wILI.loc[date-dt.timedelta(days=self.lag-1):date+dt.timedelta(days=self.gamma-self.lag)]

                x2 = self.queries.loc[date-dt.timedelta(days=self.gamma+self.window_size-1):date-dt.timedelta(days=self.gamma), selected_queries]
                x2['wILI'] = self.daily_wILI.loc[date-dt.timedelta(days=self.window_size-1 + self.lag + self.gamma):date-dt.timedelta(days=self.lag+self.gamma)].values
                y2 = self.daily_wILI.loc[date-dt.timedelta(days=self.lag-1+self.gamma):date-dt.timedelta(days=self.lag)]

                x_in1.append(x1)
                x_in2.append(x2)
                y_in1.append(y2)
                y_out.append(y1)
            y_test = np.stack(y_out)
            x_test = [np.stack(x_in1), np.stack(y_in1), np.stack(x_in2)]

            x_in1 = []
            x_in2 = []
            y_in1 = []
            y_out = []
            for date in self.train_dates[self.gamma:]:
                x1 = self.queries.loc[date-dt.timedelta(days=self.window_size-1):date, selected_queries]
                x1['wILI'] = self.daily_wILI.loc[date-dt.timedelta(days=self.window_size-1 + self.lag):date-dt.timedelta(days=self.lag)].values
                y1 = self.daily_wILI.loc[date-dt.timedelta(days=self.lag-1):date+dt.timedelta(days=self.gamma-self.lag)]

                x2 = self.queries.loc[date-dt.timedelta(days=self.gamma+self.window_size-1):date-dt.timedelta(days=self.gamma), selected_queries]
                x2['wILI'] = self.daily_wILI.loc[date-dt.timedelta(days=self.window_size-1 + self.lag + self.gamma):date-dt.timedelta(days=self.lag+self.gamma)].values
                y2 = self.daily_wILI.loc[date-dt.timedelta(days=self.lag-1+self.gamma):date-dt.timedelta(days=self.lag)]
                x_in1.append(x1)
                x_in2.append(x2)
                y_in1.append(y2)
                y_out.append(y1)
            y_train = np.stack(y_out)
            x_train = [np.stack(x_in1), np.stack(y_in1), np.stack(x_in2)]


            if forecast_type == 'single':
                y_test = y_test[:, -1:]
                y_train = y_train[:, -1:]

            return x_train, y_train, x_test, y_test
                # x_test.append(x)
                # y_test.append(y)

def rescale_df(df, scaler):
    std = scaler.inverse_transform((df['Pred']+df['Std']).values.reshape(-1, 1)) - scaler.inverse_transform(df['Pred'].values.reshape(-1, 1))

    try:
        model = scaler.inverse_transform((df['Pred']+df['Model_Uncertainty']).values.reshape(-1, 1)) - scaler.inverse_transform(df['Pred'].values.reshape(-1, 1))
        data = scaler.inverse_transform((df['Pred']+df['Data_Uncertainty']).values.reshape(-1, 1)) - scaler.inverse_transform(df['Pred'].values.reshape(-1, 1))
    except:
        pass

    mean = scaler.inverse_transform(df['Pred'].values.reshape(-1, 1))
    true = scaler.inverse_transform(df['True'].values.reshape(-1, 1))

    try:
        return pd.DataFrame(index=df.index, columns=df.columns, data=np.asarray([true, mean, std, model, data]).squeeze().T)
    except:
        return pd.DataFrame(index=df.index, columns=df.columns, data=np.asarray([true, mean, std]).squeeze().T)


if __name__ == '__main__':
    selected_queries = pd.DataFrame()

    for test_season in [2015, 2016, 2017, 2018]:
        _data = DataConstructor(test_season=test_season, data_season=test_season-1, n_queries=99, gamma=7,
                                window_size=42, rescale=False)
        x_train, y_train, x_test, y_test = _data('FF_7')

        colors_all = pd.read_csv('/Users/michael/Documents/github/random_bits/sorted_pokemon_colours.csv', index_col=0)
        colors = list(colors_all.loc['Salamence'])
        num = 120
        plt.plot(_data.test_dates[:num], x_test[:num, -14, -1], color = colors[0], label='Observed ILI')
        plt.plot(_data.test_dates[:num+14], x_test[:num+14, -14, 2]/10, color = colors[1], label='Observed Query Data')
        plt.plot(_data.test_dates[num:num+28], x_test[num:num+28, -14, -1], '--', color = colors[2])
        plt.scatter(_data.test_dates[[num+7, num+14, num+21, num+28]], x_test[[num+7, num+14, num+21, num+28], -14, -1], color = colors[2],  label='ILI Targets')
        plt.plot(_data.test_dates[[num+14, num+14]], [0, 4], color = colors[4], alpha=0.5, label='Current Time')
        plt.ylim([0.2, 3.8])
        plt.legend()
        plt.savefig('/Users/michael/Dropbox/Apps/Overleaf/Nature_Paper/figs/forecast_target.pdf')
        plt.show()

        selected_queries[test_season] = _data.selected_queries

        print(pd.DataFrame(index = ['2015/16', '2016/17', '2017/18', '2018/19'],
                           columns = ['Train Start','Train End', 'Test Start', 'Test End'],
                           data = [[dt.date(2004, 2, 24), dt.date(2015, 8, 12), dt.date(2015, 11, 1), dt.date(2016, 6, 5)],
                                   [dt.date(2004, 2, 24), dt.date(2016, 8, 11), dt.date(2016, 10, 30), dt.date(2017, 6, 4)],
                                   [dt.date(2004, 2, 24), dt.date(2017, 8, 10), dt.date(2017, 10, 29), dt.date(2018, 6, 3)],
                                   [dt.date(2004, 2, 24), dt.date(2018, 8, 9), dt.date(2018, 10, 28), dt.date(2019, 6, 4)]]).to_latex())




