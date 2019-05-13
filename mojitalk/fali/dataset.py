import tensorflow as tf
import os
import skimage
import numpy as np
from .archive import archive
import scipy.spatial.distance as sd
import glob
from sklearn.impute import SimpleImputer

class TextDataset(object):
    def __init__(self, text_path, st_embedding_path, label):
        self.embedding = np.load(st_embedding_path)
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        self.embedding = imputer.fit_transform(self.embedding)
        print(np.where(np.isnan(self.embedding)))
        self.sentences = []
        self.label = label
        with open(text_path) as f:
            self.sentences.extend([l[2:] for l in f])
        self._dataset = tf.data.Dataset.from_tensor_slices(self.embedding)
        
    def __getattr__(self, name):
        return getattr(self._dataset, name)


if __name__ == "__main__":
    import sys

    if sys.argv[1] == 'text':
        embedding_path = sys.argv[2]
        dataset = TextDataset(sys.argv[2])


