import pandas as pd
import numpy as np
from scipy.stats import pearsonr, norm
from scipy import special


def get_calibration(forecast):
    def calc_in_range(pred, z):
        return(np.sum(np.logical_and((pred['Pred'] - z * pred['Std']) < pred['True'],
                                     (pred['Pred'] + z * pred['Std']) > pred['True'])))

    stds = np.linspace(0, 3, 50)
    freq = np.zeros(stds.shape)
    probs = norm.cdf(stds) - norm.cdf(-stds)
    for idx, std in enumerate(stds):
        freq[idx] = calc_in_range(forecast, std)/forecast.shape[0]

    return {'prob':probs, 'freq':freq}

def _normpdf(x):
    """Probability density function of a univariate standard Gaussian
    distribution with zero mean and unit variance.
    """
    _normconst = 1.0 / np.sqrt(2.0 * np.pi)
    return _normconst * np.exp(-(x * x) / 2.0)

def crps(true, mean=None, std=None, grad=False):
    """
    Computes the CRPS of observations x relative to normally distributed
    forecasts with mean, mu, and standard deviation, sig.
    CRPS(N(mu, sig^2); x)
    Formula taken from Equation (5):
    Calibrated Probablistic Forecasting Using Ensemble Model Output
    Statistics and Minimum CRPS Estimation. Gneiting, Raftery,
    Westveld, Goldman. Monthly Weather Review 2004
    http://journals.ametsoc.org/doi/pdf/10.1175/MWR2904.1
    Parameters
    ----------
    x : scalar or np.ndarray
        The observation or set of observations.
    mu : scalar or np.ndarray
        The mean of the forecast normal distribution
    sig : scalar or np.ndarray
        The standard deviation of the forecast distribution
    grad : boolean
        If True the gradient of the CRPS w.r.t. mu and sig
        is returned along with the CRPS.
    Returns
    -------
    crps : scalar or np.ndarray or tuple of
        The CRPS of each observation x relative to mu and sig.
        The shape of the output array is determined by numpy
        broadcasting rules.
    crps_grad : np.ndarray (optional)
        If grad=True the gradient of the crps is returned as
        a numpy array [grad_wrt_mu, grad_wrt_sig].  The
        same broadcasting rules apply.
    """
    try:
        if isinstance(true, pd.DataFrame):
            mean = true['Pred']
            std = true['Std']
            true = true['True']
        _normcdf = special.ndtr

        true = np.asarray(true)
        mean = np.asarray(mean)
        std = np.asarray(std)
        # standadized x
        sx = (true - mean) / std
        # some precomputations to speed up the gradient
        pdf = _normpdf(sx)
        cdf = _normcdf(sx)
        pi_inv = 1. / np.sqrt(np.pi)
        # the actual crps
        crps = std * (sx * (2 * cdf - 1) + 2 * pdf - pi_inv)
        if grad:
            dmu = 1 - 2 * cdf
            dsig = 2 * pdf - pi_inv
            return crps, np.array([dmu, dsig])
        else:
            return crps

    except Exception as e:
        print(e)
        return np.nan

def nll(true, mean=None, std=None):
    try:
        if isinstance(true, pd.DataFrame):
            mean = true['Pred']
            std = true['Std']
            true = true['True']

        p_y = norm(loc=mean, scale=std)
        return (-np.log(p_y.pdf(true))).mean()

    except Exception as e:
        print(e)
        return 100
def cal(true, mean=None, std=None):
    try:
        if isinstance(true, pd.DataFrame):
            mean = true['Pred']
            std = true['Std']
            true = true['True']

        def calc_in_range(mean, std, true, z):
            return(np.sum(np.logical_and((mean - z * std) < true,
                                         (mean + z * std) > true)) / mean.shape[0])

        stds = np.linspace(0, 3, 50)
        freq = np.zeros(stds.shape)
        probs = norm.cdf(stds) - norm.cdf(-stds)

        for idx, z in enumerate(stds):
            freq[idx] = calc_in_range(mean, std, true, z)

        return np.sum(np.abs(0.5 * (probs[1:] - probs[:-1]) * ((freq[1:] + freq[:-1]) - (probs[1:] + probs[:-1]))))

    except Exception as e:
        print(e)
        return 100

def mae(true, pred=None, bins=False):
    if isinstance(true, pd.DataFrame):
        if bins:
            idx = np.argmin(np.abs(true.cumsum(1).values - 0.5), axis=1)
            true.columns = true.columns.astype(float)
            pred = true.columns[idx].astype(float) + 0.05
            true = true['True'].values
            pred = pred.values

        else:
            pred = true['Pred']
            true = true['True']

    return np.mean(np.abs(true - pred))


def corr(true, pred=None, bins=False):
    if isinstance(true, pd.DataFrame):
        if bins:
            idx = np.argmin(np.abs(true.cumsum(1) - 0.5).values, 1)
            true.columns[idx].astype(float)
            pred = true.columns[idx].astype(float) + 0.05
            true = true['True'].values

            pred = pred.values
        else:
            true = true.replace([np.inf, -np.inf], np.nan)
            true = true.dropna()

            pred = true['Pred'].values
            true = true['True'].values

    if type(pred) != np.ndarray:
        pred = true.numpy()

    pred = pred.astype('float32')
    true = true.astype('float32')
    try:
        corr = pearsonr(true, pred)[0]
        return corr
    except Exception as e:
        print(e)
        return 0

def mb_log(true, mean=None, std=None, bins=False):
    if bins:
        correct_bin = np.floor(true['True']*10)/10
        correct_bin = pd.DataFrame(index=correct_bin.index, data=[float("{:.1f}".format(v)) for v in correct_bin.values])

        cols = [float("{:.1f}".format(v)) for v in true.columns[:-1]]
        cols.append('True')
        true.columns = cols

        mbl = np.asarray([])
        for idx in true.index:
            bin_val = correct_bin.loc[idx][0]
            lower = float("{:.1f}".format(bin_val - 0.5))
            upper = float("{:.1f}".format(bin_val + 0.5))
            mbl = np.append(mbl, np.log(true.loc[idx, lower:upper].sum()))

        return mbl
    try:
        if isinstance(true, pd.DataFrame):
            mean = true['Pred']
            std = true['Std']
            true = true['True']

        dist = norm(loc=mean, scale=std)

        mbl = np.log((dist.cdf(true + 0.6) - dist.cdf(true - 0.5)))
        mbl[np.invert(np.isfinite(mbl))] = -10
        mbl[mbl < -10] = -10

        return mbl

    except Exception as e:
        print(e)
        return -10

def skill(prediction, bins=False):
    return np.exp(mb_log(prediction, bins=bins).mean())
