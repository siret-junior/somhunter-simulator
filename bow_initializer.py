import numpy as np
import re

class BoWInitializer:
    
    def __init__(self, features, keywords_f, kw_feats_f, kw_feats_bias_f, pca_mat_f, pca_mean_f):
        with open(keywords_f, 'r') as f:
            self._keywords = f.readlines()
        self._keywords = [x.strip() for x in self._keywords]
        self._keywords = [(x.split(":")[0], int(x.split(":")[1].strip())) for x in self._keywords]
        self._keywords.sort(key = lambda x: x[1], reverse=False)
        self._keywords = list(map(lambda x: x[0] ,self._keywords))
        
        self._kw_feats = np.fromfile(kw_feats_f, dtype=np.float32)
        self._kw_feats = np.reshape(self._kw_feats, [len(self._keywords), -1])
        self._kw_bias = np.fromfile(kw_feats_bias_f, dtype=np.float32)
        self._pca_mat = np.fromfile(pca_mat_f, dtype=np.float32)
        self._pca_mean = np.fromfile(pca_mean_f, dtype=np.float32)
        self._pca_mat = np.reshape(self._pca_mat, [-1,self._pca_mean.shape[0]])
        self._features = features
        
        
    def score(self, query):
        # Tokenize
        query = re.sub("[\\/?!,.'\"]", " ", query)
        query = query.split(" ")
        # compute embeddings
        query_feats = np.copy(self._kw_bias)
        should_rescore = False
        for word in query:
            if word != "":
                exact = None
                exact_kw = None
                possible = None
                possible_kw = None
                for kw_i, kw in enumerate(self._keywords):
                    if word in kw:
                        index = kw.index(word)
                        if index == 0 and (exact == None or kw < exact_kw ):
                            exact = kw_i
                            exact_kw = kw
                        elif possible == None:
                            possible = kw_i
                            possible_kw = kw
                if exact != None:
                    #print(exact, exact_kw)
                    query_feats += self._kw_feats[exact]
                    should_rescore = True
                elif possible != None:
                    #print(possible, possible_kw) 
                    query_feats += self._kw_feats[possible]
                    should_rescore = True
                
        if should_rescore:
            query_feats = np.tanh(query_feats)
            query_feats /= np.linalg.norm(query_feats)

            # Apply PCA
            query_feats -= self._pca_mean
            query_feats = self._pca_mat @ query_feats
            query_feats /= np.linalg.norm(query_feats)

            # Compute scores
            scores = (1 - np.dot(self._features, query_feats)) / 2
            scores = np.exp(scores * (-50))
            return scores
        else:
            return np.ones(self._features.shape[0], dtype=np.float32)
        
        
