import numpy as np

efficiency = {'car':2000, 'bus':5000, 'cycle':14000, 'walk':19000, 'tube':60000, 'train':90000}
usage = {'car':10.1, 'bus':3.7, 'cycle':0.6, 'walk':6.61, 'tube':2.8, 'train':3}

efficiency = np.asarray(list(efficiency.values()))/np.sum(list(efficiency.values()))
usage = np.asarray(list(usage.values()))/np.sum(list(usage.values()))
