import numpy as np
import math
from matplotlib import pyplot as plt
import scipy.stats as scs

def index_of_max(items):
    maxv = -math.inf
    idx = -1
    for i, value in enumerate(items):
        if value > maxv:
            maxv = value
            idx = i
    return idx

class Model:
    
    def __init__(self, bins = 256):
        self.__bins = bins
    
    def fit(self, train_data, categories):
        cats = list(set(categories))
        cats.sort()
        models  = [self.__hist_of(categories, train_data, cat, self.__bins) for cat in cats]
        self.__models = models
        self.__categories = cats

    def predict(self, test_data):
        hists = [np.histogram(img.flatten(), bins = self.__bins, range = [0,256], density = False)[0] for img in test_data]
        estimated  = [[np.matmul(model[0], hist) + model[1] for model in self.__models] for hist in hists]
        return [self.__categories[index_of_max(est)] for est in estimated]
    
    def score(self, test_data, test_categories):
        predicted = self.predict(test_data)
        hits = np.equal(predicted, test_categories)
        hitCount = np.count_nonzero(hits)
        return hitCount / len(hits)
                
    def __hist_of(self, labels, images, category, bins=256):
        priori = (labels==category).sum()/len(labels)
        h = np.histogram(images[labels==category].flatten(), density=True, bins=bins, range=[0,255])[0]
        return np.log(h), np.log(priori)
    
class GaussModel:
    
    def __init__(self):
        return
    
    def fit(self, train_data, categories):
        cats = list(set(categories))
        cats.sort()
        models = [scs.norm(np.mean(train_data[categories == cat]), np.std(train_data[categories == cat])) for cat in cats]
        self.__models = models
        self.__categories = cats

    def predict(self, test_data):
        estimated = [[np.sum(np.log(model.pdf(item))) for model in self.__models] for item in test_data]
        return [self.__categories[index_of_max(est)] for est in estimated]
    
    def score(self, test_data, test_categories):
        predicted = self.predict(test_data)
        hits = np.equal(predicted, test_categories)
        hitCount = np.count_nonzero(hits)
        return hitCount / len(hits)