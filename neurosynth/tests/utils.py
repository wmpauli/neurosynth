"""Some handy functionality to be used by the Neurosynth test suite"""

__author__ = 'Yaroslav Halchenko'
__copyright__ = 'Copyright (c) 2013 Yaroslav Halchenko'
__license__ = 'MIT'

from os.path import dirname, join, pardir, sep as pathsep
from neurosynth.base.dataset import Dataset


def get_resource_path():
	return join(dirname(__file__), pardir, 'resources') + pathsep

def get_test_data_path():
    """Returns the path to test datasets, terminated with separator (/ vs \)"""
    # TODO: do not rely on __file__
    return join(dirname(__file__), 'data') + pathsep


def get_test_dataset():
    test_data_path = get_test_data_path()
    dataset = Dataset(test_data_path + 'test_dataset.txt')
    dataset.add_features(test_data_path + 'test_features.txt')
    return dataset
