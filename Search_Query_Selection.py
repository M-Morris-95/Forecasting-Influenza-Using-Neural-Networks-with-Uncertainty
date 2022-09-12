import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

class query_to_embedding:
    def __call__(self, query, root=None):
        if root is None and not hasattr(self, 'root') :
            for root in ['/home/mimorris/Data/', '/Users/michael/Documents/Data/']:
                if os.path.exists(root):
                    self.root = root

        if not hasattr(self, 'vectors'):
            self.vectors = pd.read_csv(os.path.join(self.root, 'vectors_unzipped', 'Twitter_word_embeddings_CBOW.csv'),
                                       header=None)

            f = open(os.path.join(self.root, 'vectors_unzipped', 'vocabulary.txt'), "r")
            vocab = f.read()
            vocab = vocab.split('\n')[:-1]
            self.vectors.index = vocab

        query = query.split(' ')
        embedding = []
        for word in query:
            try:
                embedding.append(self.vectors.loc[word].values)
            except:
                embedding.append(np.zeros(self.vectors.shape[1]))
        embedding = np.asarray(embedding).mean(0)[np.newaxis, :]
        return embedding

def similarity_score(self, embedding, pos=['flu', 'fever', 'flu', 'flu medicine', 'gp', 'hospital'],
                     neg=['bieber', 'ebola', 'wikipedia'], gamma=0.001):
    pos = np.asarray([self.query_to_embedding(p) for p in pos]).squeeze()
    neg = np.asarray([self.query_to_embedding(n) for n in neg]).squeeze()

    pos = cosine_similarity(embedding.reshape(1, -1), pos)
    neg = cosine_similarity(embedding.reshape(1, -1), neg)

    pos = ((pos + 1) / 2).sum()
    neg = ((neg + 1) / 2).sum() + gamma
    return pos / neg

class query_selection:
    def __call__(self, queries, daily_wILI, n_queries=99, data_season=2015, root=None):
        if root is None and not hasattr(self, 'root') :
            for root in ['/home/mimorris/Data/', '/Users/michael/Documents/datasets/Data/']:
                if os.path.exists(root):
                    self.root = root
        else:
            self.root=root

        if os.path.exists(os.path.join(self.root, 'Similarity_Scores.csv')):
            scores = pd.read_csv(os.path.join(self.root, 'Similarity_Scores.csv'), index_col=0)
        else:
            to_embedding = query_to_embedding()
            query_embeddings = pd.DataFrame(index=queries.columns,
                                            data=np.asarray([to_embedding(query) for query in
                                                             queries.columns]).squeeze())
            query_embeddings.to_csv('Data/Query_Embeddings.csv')

            scores = pd.DataFrame(index=queries.columns, columns=['similarity'], data=np.asarray(
                [similarity_score(embedding) for embedding in query_embeddings.values]))
            scores.to_csv('Data/Similarity_Scores.csv')

        dates = pd.date_range(str(data_season - 4) + '/8/23', str(data_season + 1) + '/8/23')

        # remove constant frequencies
        queries = queries.loc[:, queries.loc[dates].std() > 0.01]

        scores['correlation'] = pd.DataFrame(index=queries.columns,
                                             columns=['correlation'],
                                             data=[pearsonr(daily_wILI.loc[dates].squeeze(), q)[0] for q in
                                                   queries.loc[dates].values.T])

        scores['correlation'] = (scores['correlation'] + 1) / 2
        scores['correlation'] = scores['correlation'].fillna(scores['correlation'].min())
        scores['similarity'] = scores['similarity'].fillna(scores['similarity'].min())

        # if selection_method == 'distance':
        scores['distance'] = np.sqrt(np.square(1 - scores / np.tile(scores.max(), (scores.shape[0], 1))).sum(1))
        scores = scores.iloc[np.argsort(scores['distance'])]

        # res[2017] = scores.index[:100]

        selected_queries = scores.index[:n_queries]

        # if selection_method == 'Bill':
        #     scores = scores[scores['similarity'] > self.selection_similarity_threshold]
        #     scores = scores[scores['correlation'] > self.selection_correlation_threshold]
        #     self.selected_queries = scores.index

        return selected_queries

if __name__ == "__main__":
    Q = query_to_embedding()
    Q('flu')

