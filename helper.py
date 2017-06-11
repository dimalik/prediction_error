"""
Helper abstract classes for WordContextModel
"""

import logging


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


class NNModel(object):
    """ Abstract class for Keras models.

    Parameters:
    loss_function (str or func): either a Keras function (see online docs)
                                 or a theano function which given two arrays
                                 returns a scalar
    optimizer (str or class): any of the Keras optimizers
    """
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
