# Implementation of Neural network models for influenza forecasting with associated uncertainty using Web search activity trends
<!-- ================================================================================================================================ -->

This repository provides minimal code to recreate the experiments from 'Neural network models for influenza forecasting with associated uncertainty using Web search activity trends'.

## Installation
Code has been tested with `Python 3.8.5`, packages are listed in `requirements.txt`


## Data
The Web search and ILI rates data sets that support this code are available from [Google](https://www.google.com) and [CDC](https://www.cdc.gov/), respectively. As CDC may update ILI rates retrospectively (e.g., by using increased sample rates or applying revisions in the ILI rate computation approach), we have also included the ILI rates used in our study.

Restrictions apply to the availability of the Google search data set, which was used under license for the current study, and so is not publicly available. This data set is, however, available upon reasonable request and with the respective permission of Google.

### Pre-smoothed US National Level Searches
Pre-smoothed US national level searches are stored in `/data/Queries/US_query_data_all_smoothed.csv`.

### Update: State and Regional Level Forecasts
State level search queries are stored in the directory `/data/Queries/state_queries`, and are named by `{state_code}_query_data.csv`, where state codes are 'AK', 'AL', 'AR', etc.

HHS level query data is created by aggregating state level queries by population.

## Basic Usage
`FF`, `SRNN` and `IRNN` are model classes for the three neural networks in the paper. Each model has `pbounds` - the limits for each hyper parameter to be optimized with code from `Optimisation.py`, this requires some functions from `optimiser_tools.py`.

Data is formatted by `DataConstructor.py`, query selection is done with `Search_Query_Selection.py` using word embeddings which can be downloaded from [here](https://figshare.com/articles/dataset/UK_Twitter_word_embeddings/4052331). Similarity scores are provided for the top ~3000 queries, some are removed due to being the same query with a different word order.  

Functions to compute metrics are in `Metrics.py`

`Test_Fn.py` contains a class to forecast ILI rates on the test set. This automatically reads hyper-parameters saved in the Results directory (generated with `optimisation.py`). If none are available then default parameters defined in the model classes will be used.

## References
The paper corresponding with this work is:

Michael Morris, Peter Hayes, Ingemar J. Cox, Vasileios Lampos. "Neural network models for influenza forecasting with associated uncertainty using Web search activity trends." [[plos]](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011392)

```
@article{morris2023neural,
  title={Neural network models for influenza forecasting with associated uncertainty using web search activity trends},
  author={Morris, Michael and Hayes, Peter and Cox, Ingemar J and Lampos, Vasileios},
  journal={PLoS Computational Biology},
  volume={19},
  number={8},
  pages={e1011392},
  year={2023},
  publisher={Public Library of Science San Francisco, CA USA}
}
```




