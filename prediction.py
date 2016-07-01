"""

Neural Embeddings in Keras

this module emulates the functionality of word2vec with negative sampling
written in Keras.

__authors__ = 'Dimitrios Alikaniotis'
__affiliation__ = 'University of Cambridge'
__email__ = 'da352@cam.ac.uk'

"""

import os
import logging
import cPickle as pkl

import numpy as np

from keras.models import Sequential
from keras.layers.core import Merge, Reshape, Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence, text

FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


class CorpusReader(object):
    """ CorpusReader object. Works as a iterator of longer texts

    Parameters:
        path (str): The path to a text file
    """
    def __init__(self, path):
        self._path = path
        self._corpus_size = None

    def __iter__(self):
        with open(self._path) as fin:
            self._corpus_size = 0
            for i, line in enumerate(fin, 1):
                self._corpus_size += 1
                if not i % 1000:
                    logging.info('Done %d sequences', i)
                yield line.strip()

    @property
    def corpus_size(self):
        """ Return the corpus size """
        return self._corpus_size

    @property
    def path(self):
        """ Return the corpus path """
        return self._path


class Model(object):
    """ d"""
    def __init__(self, loss_function='binary_crossentropy',
                 optimizer='rmsprop'):
        self.loss_function = loss_function
        self.optimizer = optimizer

    def prepare_model(self, *args, **kwargs):
        """ Return the instantiated keras model """
        raise NotImplementedError

    def save_model(self, path):
        """ Save the model to the indicated path """
        raise NotImplementedError


class WordContextModel(Model):
    """ d"""
    def __init__(self, corpus_path, embedding_size, *args, **kwargs):
        self.corpus_path = corpus_path
        self.embedding_size = embedding_size
        self.tokenizer = None
        self.corpus = None
        self.model = None
        self._vocab_size = None
        super(WordContextModel, self).__init__(*args, **kwargs)

    @staticmethod
    def _prepare_model(vocab_size, vector_dim, loss_function,
                       optimizer):
        """ d"""
        logging.info('Building word model...')
        word = Sequential()
        word.add(Embedding(vocab_size, vector_dim, input_length=1))
        word.add(Reshape((vector_dim, 1)))

        logging.info('Building context model...')
        context = Sequential()
        context.add(Embedding(vocab_size, vector_dim, input_length=1))
        context.add(Reshape((vector_dim, 1)))

        logging.info('Building composite graph')
        model = Sequential()
        model.add(Merge([word, context], mode='dot', dot_axes=1))
        model.add(Reshape((1, )))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss=loss_function, optimizer=optimizer)
        return model

    def prepare_model(self):
        """ d"""
        self.model = self._prepare_model(self.vocab_size,
                                         self.embedding_size,
                                         self.loss_function,
                                         self.optimizer)

    @property
    def vocab_size(self):
        """ d"""
        try:
            return self._vocab_size
        except AttributeError:
            logging.error('Please tokenize the corpus first')

    def tokenize_corpus(self):
        """ d"""
        self.corpus = CorpusReader(self.corpus_path)
        logging.info('Tokenizing the corpus')
        self.tokenizer = text.Tokenizer()
        self.tokenizer.fit_on_texts(self.corpus)
        self._vocab_size = len(self.tokenizer.word_counts) + 1

    def train_corpus(self, negative_samples=20, window_size=4):
        """ d"""
        logging.info('Initialising sampling table')
        sampling_table = sequence.make_sampling_table(self.vocab_size)
        ans = []
        for i, seq in enumerate(
                self.tokenizer.texts_to_sequences_generator(self.corpus)):
            logging.info(i)
            couples, labels = sequence.skipgrams(
                seq, self.vocab_size, window_size=window_size,
                negative_samples=negative_samples,
                sampling_table=sampling_table)
        if couples:
            word_target, word_context = zip(*couples)
            word_target = np.array(word_target, dtype="int32")
            word_context = np.array(word_context, dtype="int32")
            loss = self.model.train_on_batch([word_target, word_context],
                                             labels)
        ans.append(loss)
        return ans

    def save_model(self, path):
        """ Saves the model in json format and the weights in hdf5"""
        self.model.to_json(os.path.join(path, 'model.json'))
        self.model.save_weights(os.path.join(path, 'model_weights.hdf5'))

    def save_embeddings(self, path):
        """ Saves only the word embeddings along with the dictionary """
        with open(os.path.join(path, 'embeddings.pkl', 'wb')) as fout:
            pkl.dump([self.model.get_weights()[0],
                      self.tokenizer.word_index], fout, -1)
