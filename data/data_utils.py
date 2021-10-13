import nltk
# from nltk.tokenize.stanford import StanfordTokenizer
import numpy as np
import gensim.downloader as api
import gensim
import os
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def get_words_and_labels(download=False):
    if download:
        nltk.download('brown')
        nltk.download('universal_tagset')
    dataset = nltk.corpus.brown.tagged_words(tagset='universal')
    return dataset


def get_vocabulary(dataset):
    """
    given a dataset (as from NLTK, above), extract every word in the dataset
    then, return the set of all unique words in the dataset
    :param dataset: NLTK tagging dataset
    :return: list of all unique words in the dataset
    """
    unique_words = None
    return unique_words


def get_unique_labels(dataset):
    """
    Given a dataset (as from NLTK, above) extract every label and return the set of unique labels
    :param dataset: NLTK tagging dataset
    :return: list of all unique labels in the dataset
    """
    unique_labels = None
    return unique_labels


def word_to_index(dataset):
    """
    Given a dataset, map every word to a number/index. Then return that mapping
    Remember, we can't pass words to our network, it needs indices for an embedding matrix.
    This function creates the mapping from word to embedding-matrix index.
    :param dataset: NLTK tagging dataset
    :return: dictionary mapping from word to index (i.e., {'the': 0, 'a': 1, 'scipy': 2, ...}
    """
    mapping = None

    return mapping


def label_to_index(dataset):
    """
    Given a dataset, map every label to a number/index. Then return that mapping
    Remember, we can't pass string labels to a loss function, we need numbered labels.
    :param dataset: NLTK tagging dataset
    :return: dictionary mapping from label to index (i.e., {'DET': 0, 'NOUN': 1, ...}
    """
    mapping = None
    return mapping


def split_dataset(dataset, word_mapping, label_mapping, percent_testing=0.2, shuffle=False):
    """
    Given a dataset, break it up into a training and dev set.
    :param dataset: NLTK tagging dataset
    :param word_mapping: map from word to index
    :param label_mapping: map from label to index
    :param percent_testing: how much data should be held out for testing? float from 0-1, default: 0.2
    :param shuffle: Should we shuffle the data? Boolean: default: False
    :return: training_words (list of all training indices), training_labels (list of all training labels),
    testing_words (list of all testing indices), testing_labels (list of all testing labels)
    """
    training_words, training_labels, testing_words, testing_labels = None, None, None, None
    return training_words, training_labels, testing_words, testing_labels


def word_to_gensim():
    """
    Using gensim and nltk, create a word2vec model of words in the treebank corpus
    Then return that embedding matrix.
    :return: gensim Word2Vec model.
    """
    if os.path.exists('treebank.embeds'):
        w2v = gensim.models.Word2Vec.load('treebank.embeds')
    else:
        w2v = gensim.models.Word2Vec(nltk.corpus.treebank.sents(), min_count=1)
        w2v.save('treebank.embeds')
    return w2v.wv
