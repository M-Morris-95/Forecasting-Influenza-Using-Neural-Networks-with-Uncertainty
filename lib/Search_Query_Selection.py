import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

# class which embeds search queries, used a class instead of a function so it runs faster without having to load everything
class query_to_embedding:
    # its a class so it can be called multiple
    def __call__(self, query, root='data/embeddings'):
        # root should point to directory with files from here:
        # https://figshare.com/articles/dataset/UK_Twitter_word_embeddings/4052331
        # cannot include these in the repo due to memorey constraints

        self.root = root

        # load vectors for each word if not already done
        if not hasattr(self, 'vectors'):
            self.vectors = pd.read_csv(os.path.join(self.root, 'Twitter_word_embeddings_CBOW.csv'),
                                       header=None)

            f = open(os.path.join(self.root, 'vocabulary.txt'), "r")
            vocab = f.read()
            vocab = vocab.split('\n')[:-1]
            self.vectors.index = vocab

        # embed a multi word query as the average embedding for each word in it
        query = query.split(' ')
        embedding = []
        for word in query:
            try:
                embedding.append(self.vectors.loc[word].values)
            except:
                # if the word isn't in vocabulary then use zeros. - query probably wont get selected if this happens.
                embedding.append(np.zeros(self.vectors.shape[1]))
        embedding = np.asarray(embedding).mean(0)[np.newaxis, :]
        return embedding

# calculate similarity score - preset positive and negative words 
# Lampos, Vasileios, Bin Zou, and Ingemar Johansson Cox. 
# "Enhancing feature selection using word embeddings: The case of flu surveillance." 
# Proceedings of the 26th International Conference on World Wide Web. 2017.
def similarity_score(embed_fn, embedding, pos=['flu', 'fever', 'flu', 'flu medicine', 'gp', 'hospital'],
                     neg=['bieber', 'ebola', 'wikipedia'], gamma=0.001):
    pos = np.asarray([embed_fn(p) for p in pos]).squeeze()
    neg = np.asarray([embed_fn(n) for n in neg]).squeeze()

    pos = cosine_similarity(embedding.reshape(1, -1), pos)
    neg = cosine_similarity(embedding.reshape(1, -1), neg)

    pos = ((pos + 1) / 2).sum()
    neg = ((neg + 1) / 2).sum() + gamma
    return pos / neg

# class which selects queries
# queries is a dataframe where columns are different queries, rows are frequency of that query the day given by the index
# daily_wILI is a dataframe of the ILI rate each day
# n_queries - how many queries to select
# data_season - what season to base selection on. e.g., 2015/16 would be 2015
# root - the place where the data is saved
class query_selection:
    def __call__(self, queries, daily_wILI, n_queries=99, data_season=2015, root='data'):
        self.root=root

        # computing similarity scores can be slow so if they've been saved then use them
        if os.path.exists(os.path.join(self.root, 'embeddings', 'Similarity_Scores.csv')):
            scores = pd.read_csv(os.path.join(self.root, 'embeddings', 'Similarity_Scores.csv'), index_col=0)
        else:
            # calculate similarity scores if they're not already available. Save them at the end
            embed_fn = query_to_embedding()
            query_embeddings = pd.DataFrame(index=queries.columns,
                                            data=np.asarray([embed_fn(query) for query in
                                                             queries.columns]).squeeze())
            query_embeddings.to_csv('data/embeddings/Query_Embeddings.csv')

            scores = pd.DataFrame(index=queries.columns, columns=['similarity'], data=np.asarray(
                [similarity_score(embed_fn, embedding) for embedding in query_embeddings.values]))
            scores.to_csv('data/Similarity_Scores.csv')

        # base query selection on 5 years ILI data
        dates = pd.date_range(str(data_season - 4) + '/8/23', str(data_season + 1) + '/8/23')

        # remove constant frequencies
        queries = queries.loc[:, queries.loc[dates].std() > 0.01]

        scores['correlation'] = pd.DataFrame(index=queries.columns,
                                             columns=['correlation'],
                                             data=[pearsonr(daily_wILI.loc[dates].squeeze(), q)[0] for q in
                                                   queries.loc[dates].values.T])

        # rescale scores and remove nan values
        scores['correlation'] = (scores['correlation'] + 1) / 2
        scores['correlation'] = scores['correlation'].fillna(scores['correlation'].min())
        scores['similarity'] = scores['similarity'].fillna(scores['similarity'].min())

        scores['distance'] = np.sqrt(np.square(1 - scores / np.tile(scores.max(), (scores.shape[0], 1))).sum(1))
        scores = scores.iloc[np.argsort(scores['distance'])]
        selected_queries = scores.index[:n_queries]

        return selected_queries

if __name__ == "__main__":
    Q = query_to_embedding()
    Q('flu')