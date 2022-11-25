# Implementation of Neural network models for influenza forecasting with associated uncertainty using Web search activity trends
<!-- ================================================================================================================================ -->

This repository provides minimal code to recreate the experiments from 'Neural network models for influenza forecasting with associated uncertainty using Web search activity trends'.

## Installation
Code has been tested with `Python 3.8.5`, packages are listed in `requirements.txt`

## Data Availability

The Web search and ILI rates data sets that support this code available from Google and CDC, respectively. As CDC may update ILI rates retrospectively (e.g. by using increased sample rates or applying revisions in the ILI rate computation approach), we have also included the ILI rates used in our study. Restrictions apply to the availability of the Google search data set which was used under license for the current study, and so is not publicly available. This data set is however available upon reasonable request and with the respective permission of Google.

## Basic Usage
`FF`, `SRNN` and `IRNN` are model classes for the three neural networks in the paper. Each model has `pbounds` - the limits for each hyper parameter to be optimized with code from `Optimisation.py`, this requires some functions from `optimiser_tools.py`.

Data is formatted by `DataConstructor.py`, query selection is done with `Search_Query_Selection.py` using word embeddings which can be downloaded from [here](https://figshare.com/articles/dataset/UK_Twitter_word_embeddings/4052331). 

Functions to compute metrics are in `Metrics.py`

`Test_Fn.py` contains a class to forecast ILI rates on the test set. This automatically reads hyper-parameters saved in the Results directory (generated with `optimisation.py`). If none are available then default parameters defined in the model classes will be used.

## References
(will update this in the future)

The paper corresponding with this work is:

Michael Morris, Peter Hayes, Ingemar J. Cox, Vasileios Lampos. "Neural network models for influenza forecasting with associated uncertainty using Web search activity trends."

```
@article{morris2022NN,
  title={Neural network models for influenza forecasting with associated uncertainty using Web search activity trends},
  author={Morris, Michael and Hayes, Peter and Cox, Ingemar J. and Lampos, Vasileios}
}
```




