# Standard library imports
import os
import datetime as dt

# Data handling and numerical computations
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy import interpolate


def smooth(df, n=7):
    smoothed = pd.DataFrame(index = df.index[n:], 
                            columns = df.columns, 
                            data = np.mean(np.asarray([df[i:-(n-i)] for i in range(n)]), 0))
    return smoothed       

def get_state_query_data(num, root = 'Data/', append = 'Queries/state_queries', ignore = [], return_all = False, smooth_after = False):
    state_codes = {'AK':'Alaska','AL':'Alabama','AR':'Arkansas','AZ':'Arizona','CA':'California','CO':'Colorado','CT':'Connecticut','DE':'Delaware','DC':'District of Columbia','GA':'Georgia','HI':'Hawaii','ID':'Idaho','IL':'Illinois','IN':'Indiana','IA':'Iowa','KS':'Kansas','KY':'Kentucky','LA':'Louisiana','ME':'Maine','MD':'Maryland','MI':'Michigan','MN':'Minnesota','MS':'Mississippi','MO':'Missouri','MT':'Montana','NE':'Nebraska','NV':'Nevada','NH':'New Hampshire','NJ':'New Jersey','NM':'New Mexico','NY':'New York','NC':'North Carolina','ND':'North Dakota','OH':'Ohio','OK':'Oklahoma','OR':'Oregon','PA':'Pennsylvania','RI':'Rhode Island','SC':'South Carolina','SD':'South Dakota','TN':'Tennessee','TX':'Texas','UT':'Utah','VT':'Vermont','VA':'Virginia','WA':'Washington','WV':'West Virginia','WI':'Wisconsin','WY':'Wyoming'}
    
    code = list(state_codes.keys())[num-1]
    
    df = pd.read_csv(root+append +'/'+code+'_query_data.csv', index_col=0, parse_dates=True)
           
    if smooth_after:
        df = smooth(df)
        
    return df    

def get_hhs_query_data(num, root = 'Data/', append = 'Queries/state_queries', ignore = [], return_all = False, smooth_after = False):
    state_pop = pd.read_csv(root + 'state_population_data_2019.csv', index_col = 0)
    state_dict =  {1:['CT', 'ME', 'MT', 'NH', 'RI', 'VT'],
                   2:['NY', 'NJ'],
                   3:['DE', 'MD', 'PA', 'VA', 'WV', 'DC'],
                   4:['AL', 'FL', 'GA', 'KY', 'MS', 'NC', 'SC', 'TN'],
                   5:['IL', 'IN', 'OH', 'MI', 'MN', 'WI'],
                   6:['AR', 'LA', 'NM', 'OK', 'TX'],
                   7:['IA', 'KS', 'MO', 'NE'],
                   8:['CO', 'MT', 'ND', 'SD', 'UT', 'WY'],
                   9:['AZ', 'CA', 'HI', 'NV'],
                  10:['AK', 'ID', 'OR', 'WA']}
    
    total_population = sum([state_pop[state_pop['CODE'] == code]['POP'].values[0] for code in state_dict[num]])
    
    dfs = []
    for code in state_dict[num]:
        if code not in ignore:
            population = state_pop[state_pop['CODE'] == code]['POP'].values[0]/total_population
            new_nf = population*pd.read_csv(root+append +'/'+code+'_query_data.csv', index_col=0, parse_dates=True)
            dfs.append(new_nf)
    
    cols = [d.columns for d in dfs]
    common_cols = cols[0]
    for col_list in cols[1:]:
        common_cols = common_cols.intersection(col_list)
    
    idxs = [d.index for d in dfs]
    common_idxs = idxs[0]
    for idx_list in idxs[1:]:
        common_idxs = common_idxs.intersection(idx_list)
    
    df = pd.DataFrame(index = common_idxs, columns = common_cols, data = 0)
        
    for d in dfs:
        df = df+d.loc[df.index, df.columns]

    if smooth_after:
        df = smooth(df)
        
    if return_all:
        return df, dfs
    return df    

    
def get_nat_query_data(num, root = 'Data/Queries/',  ignore = [], return_all = False, smooth_after = False):
    df = pd.read_csv(root +'US_query_data_all_smoothed.csv', index_col=0, parse_dates=True)
    return df
        
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

def choose_qs(df, daily_ili, region_num, season, n_qs, region='hhs'):
    state_codes = {'AK':'Alaska','AL':'Alabama','AR':'Arkansas','AZ':'Arizona','CA':'California','CO':'Colorado','CT':'Connecticut','DE':'Delaware','DC':'District of Columbia','GA':'Georgia','HI':'Hawaii','ID':'Idaho','IL':'Illinois','IN':'Indiana','IA':'Iowa','KS':'Kansas','KY':'Kentucky','LA':'Louisiana','ME':'Maine','MD':'Maryland','MI':'Michigan','MN':'Minnesota','MS':'Mississippi','MO':'Missouri','MT':'Montana','NE':'Nebraska','NV':'Nevada','NH':'New Hampshire','NJ':'New Jersey','NM':'New Mexico','NY':'New York','NC':'North Carolina','ND':'North Dakota','OH':'Ohio','OK':'Oklahoma','OR':'Oregon','PA':'Pennsylvania','RI':'Rhode Island','SC':'South Carolina','SD':'South Dakota','TN':'Tennessee','TX':'Texas','UT':'Utah','VT':'Vermont','VA':'Virginia','WA':'Washington','WV':'West Virginia','WI':'Wisconsin','WY':'Wyoming'}
    queries = df[region_num]
    
    if region == 'US':
        ili = daily_ili['weighted_ili']
    if region == 'hhs':
        ili = daily_ili['Region '+str(region_num)]
    elif region == 'state':
        ili = daily_ili[state_codes[list(state_codes.keys())[region_num-1]]]
        
    index = daily_ili.index.intersection(queries.index)
    queries = queries.loc[index]
    ili = ili.loc[index]
    
    dates = pd.date_range(dt.date(season-3, 10, 3), dt.date(season,10,1))

    queries_subset = queries.loc[dates].std()
    queries = queries.iloc[:, np.where(queries_subset != 0)[0]]

    corr_df = pd.DataFrame(index=queries.columns,
                 columns=['correlation'],
                 data=[pearsonr(ili.loc[dates].squeeze(), q)[0] for q in
                               queries.loc[dates].values.T])
    scores = pd.read_csv('Data/Similarity_Scores.csv', index_col=0)
    scores['correlation'] = corr_df
    scores = scores.dropna()
    
    for col in scores.columns:
        scores[col] = scores[col] - scores[col].min()
        scores[col] = scores[col] / scores[col].max()
        scores[col] = 1 - scores[col]
    scores['score'] = np.sqrt(np.square(scores).sum(1))
    
    scores = scores.sort_values('score')
    
    query_choice = scores[:n_qs]
    return query_choice.index

def load_ili(location):
    location_dict = {'US':'Data/ILI_rates/national_flu.csv',
                     'England':'Data/ILI_rates/England_ILIrates.csv',
                     'state':'Data/ILI_rates/state_flu.csv',
                     'hhs':'Data/ILI_rates/hhs_flu.csv'}
    
    ili = pd.read_csv(location_dict[location], index_col = -1, parse_dates=True)
    if location == 'state' or location =='hhs':
        new_ili = pd.DataFrame()
        for region in ili['region'].unique():
            new_ili[region] = ili[ili['region'] == region]['unweighted_ili']
        ili = new_ili
        ili /= 13
        ili= ili.fillna(0)
        
    if location == 'US':
        # ili[['weighted_ili']].rename(columns = {'weighted_ili':'National'})
        ili = ili[['weighted_ili']]
        ili /= 13
    
    if location == 'England':
        ili['Date'] = [dt.datetime.strptime(d, '%d/%m/%Y')+dt.timedelta(days=3) for d in ili['ISOWeekStartDate'].values]
        ili = ili[['Date', 'RatePer100000']].set_index('Date')
        ili = ili.rename(columns = {'RatePer100000':'National'})

    return ili

def intepolate_ili(ili, fill_1=False):
    dates = np.asarray([ili.index[0] + dt.timedelta(days=i) for i in
                    range((ili.index[-1] - ili.index[0]).days + 1)])

    x = np.linspace(0, 1, ili.shape[0])
    x2 = np.linspace(0, 1, dates.shape[0])
    f = interpolate.interp1d(x, ili.values, axis = 0, kind = 'cubic')

    if not fill_1:
        return pd.DataFrame(index=dates, columns=ili.columns, data=f(x2))
    else:
        return pd.DataFrame(index=dates, columns=ili.columns, data=ili)

class DataConstructor:
    def __init__(self, test_season, region = 'hhs', window_size = 28, n_queries = 10, gamma = 28, window = 28, lag = 14, n_regions=10, fill_1 = False, root = 'checkpoints/HHS_SIR_Big_new/' ):

        self.lag = lag
        self.window = window
        self.root = root
        self.n_regions = n_regions
        self.test_season = test_season
        self.region = region
        self.window_size = window_size
        self.n_queries = n_queries
        self.gamma = gamma
        self.fill_1 = fill_1

        if region == 'hhs':
            self.n_regions = 10
        elif region == 'state':
            self.n_regions = 49
        else:
            self.n_regions = 1

    def __call__(self, run_backward=False, no_qs_in_output=False):
        state_codes = {'AK':'Alaska','AL':'Alabama','AR':'Arkansas','AZ':'Arizona','CA':'California','CO':'Colorado',
                       'CT':'Connecticut','DE':'Delaware','DC':'District of Columbia','GA':'Georgia',
                       'HI':'Hawaii','ID':'Idaho','IL':'Illinois','IN':'Indiana','IA':'Iowa','KS':'Kansas','KY':'Kentucky',
                       'LA':'Louisiana','ME':'Maine','MD':'Maryland','MI':'Michigan','MN':'Minnesota','MS':'Mississippi','MO':'Missouri','MT':'Montana','NE':'Nebraska','NV':'Nevada','NH':'New Hampshire','NJ':'New Jersey','NM':'New Mexico','NY':'New York','NC':'North Carolina','ND':'North Dakota','OH':'Ohio','OK':'Oklahoma','OR':'Oregon','PA':'Pennsylvania','RI':'Rhode Island','SC':'South Carolina','SD':'South Dakota','TN':'Tennessee','TX':'Texas','UT':'Utah','VT':'Vermont','VA':'Virginia','WA':'Washington','WV':'West Virginia','WI':'Wisconsin','WY':'Wyoming'}
        ili = load_ili(self.region)
        ili = intepolate_ili(ili, fill_1 = False)

        qs_data_dict = {}
        qs_names_dict = {}
        ignore = ['VI', 'PR']

        for i in range(1,1+self.n_regions):
            if self.region == 'US':
                qs_data_dict[i] = get_nat_query_data(i, ignore=ignore, smooth_after = True)
            if self.region == 'hhs':
                qs_data_dict[i] = get_hhs_query_data(i, ignore=ignore, smooth_after = True)
            elif self.region == 'state':
                qs_data_dict[i] = get_state_query_data(i, ignore=ignore, smooth_after = True)

            qs_names_dict[i] = choose_qs(qs_data_dict, ili, i, self.test_season-1, self.n_queries, region = self.region)
            qs_data_dict[i] = qs_data_dict[i].loc[:, list(qs_names_dict[i])]
            qs_data_dict[i] = qs_data_dict[i].div(qs_data_dict[i].max())

        ili = load_ili(self.region)
        ili = intepolate_ili(ili, fill_1 = self.fill_1)
        
        ili = ili.loc[qs_data_dict[i].index[0] : qs_data_dict[i].index[-1]]
        if self.region == 'state':
            ili = ili[list(state_codes.values())]

        scaler = ili.max()*13
        ili = ili.div(np.nanmax(ili, axis=0))

        if self.fill_1:
            ili = ili.fillna(-1)

        inputs = []
        outputs = []
        dates = []
        for batch in range(self.window+1, ili.shape[0] - (self.gamma)):
            batch_inputs = []
            for i in range(1,1+self.n_regions):
                batch_inputs.append(qs_data_dict[i].iloc[batch-self.window-1:batch+self.lag-1])
            
            t_ili = ili.iloc[batch-self.window-1:batch+self.lag-1].copy()
            t_ili.iloc[-self.lag:, :] = -1

            batch_inputs.append(t_ili)
            batch_inputs = np.concatenate(batch_inputs, -1)

            
            batch_outputs = []
            for i in range(1,1+self.n_regions):
                if run_backward:
                    batch_outputs.append(qs_data_dict[i].iloc[batch-self.window-1:batch+self.gamma])     
                    t_ili = ili.iloc[batch-self.window-1:batch+self.gamma].copy()
                else:
                    batch_outputs.append(qs_data_dict[i].iloc[batch:batch+self.gamma])            
                    t_ili = ili.iloc[batch:batch+self.gamma].copy()
            
            batch_outputs.append(t_ili)
            batch_outputs = np.concatenate(batch_outputs, -1) 

            if no_qs_in_output:
                batch_outputs = batch_outputs[..., -self.n_regions:]
                
            dates.append((t_ili.index[0]-dt.timedelta(days=1)).to_pydatetime())
            inputs.append(batch_inputs)
            outputs.append(batch_outputs)

        train_test_dates = pd.read_csv('Data/Dates.csv', index_col=0).loc[self.test_season]

        train_start = dt.datetime.strptime(train_test_dates['train_start'], '%Y-%m-%d')
        train_end = dt.datetime.strptime(train_test_dates['train_end'], '%Y-%m-%d')
        test_start = dt.datetime.strptime(train_test_dates['test_start'], '%Y-%m-%d')
        test_end = dt.datetime.strptime(train_test_dates['test_end'], '%Y-%m-%d')

        try:
            train_start = np.where([train_start == d for d in dates])[0][0]
        except:
            train_start = 0
        
        train_end = np.where([train_end == d for d in dates])[0][0]
        test_start = np.where([test_start == d for d in dates])[0][0]
        test_end = np.where([test_end == d for d in dates])[0][0]

        x_train = np.asarray(inputs[train_start:train_end])
        y_train = np.asarray(outputs[train_start:train_end])
        x_test = np.asarray(inputs[test_start:test_end])
        y_test = np.asarray(outputs[test_start:test_end])

        return x_train, y_train, x_test, y_test, scaler

if __name__ == '__main__':
    _data = DataConstructor(test_season=2016, region = 'US', window_size=28, n_queries=9, gamma=28)
    x_train, y_train, x_test, y_test, _ = _data()