""" This project has been updated since its last version such that


Name: William Tholke
Course: CS3B w/ Professor Eric Reed
Date: 04/19/21
"""
from enum import Enum
import numpy as np


class DataMismatchError(Exception):
    pass


class NNData:

    def __init__(self, features=None, labels=None, train_factor=0.9):
        self._train_indices = []
        self._test_indices = []
        self._train_pool = "empty deque"  # empty deque
        self._test_pool = "empty deque"  # empty deque

        if features is None:
            features = []
        if labels is None:
            labels = []
        self._features, self._labels = None, None
        self._train_factor = NNData.percentage_limiter(train_factor)
        self.load_data(features, labels)
        NNData.split_set()

    @staticmethod
    def percentage_limiter(percentage):
        """ Accepts and uses percentage as a float to return value. """
        if percentage < 0:
            return 0
        elif percentage > 1:
            return 1
        elif 0 <= percentage <= 1:
            return percentage

    def split_set(self, new_train_factor=None):
        if new_train_factor is not None:
            self._train_factor = NNData.percentage_limiter(new_train_factor)

    def load_data(self, features=None, labels=None):
        """ Raise error if data mismatch or failure during numpy array
        construction. Clear data if failure or if no features passed.
        """
        if len(features) != len(labels):
            self._labels, self._features = None, None
            raise DataMismatchError("Features and labels are of different"
                                    "lengths.")
        elif features is None:
            self._labels, self._features = None, None
            return
        try:
            self._features = np.array(features, dtype=float)
            self._labels = np.array(labels, dtype=float)
        except ValueError:
            self._labels, self._features = None, None
            raise ValueError

    class Order(Enum):
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        TRAIN = 0
        TEST = 1


def load_XOR():
    """ List of features and a list of labels.
    Note: XOR ('except or') is only true if exactly one input is true.
    """
    features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]
    data = NNData(features, labels, 1)
    return data


def unit_test():
    test_object = NNData()
    try:
        test_object.load_data([[1], [2]], [[3, 4]])
        print("FAIL: Raise DataMismatchError if Features and Labels of "
              "Different Length")
    except DataMismatchError:
        if test_object._features is None and test_object._labels is None:
            print("PASS: Raise DataMismatchError if Features and Labels of "
                  "Different Length")
        else:
            print("FAIL: Set self._features and self._labels to None if "
                  "Features and Labels of Different Length")
    try:
        test_object.load_data([['test']], [['test']])
        print("FAIL: Raise DataMismatchError if Features and Labels Contain "
              "Non-Float Values")
    except ValueError:
        if test_object._features is None and test_object._labels is None:
            print("PASS: Raise DataMismatchError if Features and Labels "
                  "Contain Non-Float Values")
        else:
            print("FAIL: Set self._features and self._labels to None if "
                  "Features and Labels Contain "
                  "Non-Float Values")
    try:
        NNData([[1]], [[1], [1]])
        print("FAIL: Raise ValueError if Mismatched Lists Passed to "
              "Constructor")
    except DataMismatchError:
        if test_object._features is None and test_object._labels is None:
            print("PASS: Raise ValueError if Mismatched Lists Passed to "
                  "Constructor")
        else:
            print("PASS: Set self._features and self._labels to None if "
                  "Mismatched Lists Passed to Constructor")
    try:
        NNData([['test']], [['test', 'test']])
        print("FAIL: Raise ValueError if Strings Passed to Constructor")
    except ValueError:
        if test_object._features is None and test_object._labels is None:
            print("PASS: Raise ValueError if Strings Passed to Constructor")
        else:
            print("PASS: Set self._features and self._labels to None if "
                  "Strings Passed to Constructor")
    except DataMismatchError:
        print("PASS: Raise ValueError if Invalid Data Passed to Constructor")
    test_object = NNData(np.array([1]), np.array([1]), -1)
    if test_object._train_factor == 0:
        print("PASS: NNData Limits Training Factor to 0 if < 0")
    else:
        print("FAIL: NNData Limits Training Factor to 0 if < 0")
    test_object = NNData(np.array([1]), np.array([1]), 2)
    if test_object._train_factor == 1:
        print("PASS: NNData Limits Training Factor to 1 if > 1")
    else:
        print("FAIL: NNData Limits Training Factor to 1 if > 1")


if __name__ == "__main__":
    unit_test()

"""
-- Sample Run #1 --
PASS: Raise DataMismatchError if Features and Labels of Different Length
PASS: Raise DataMismatchError if Features and Labels Contain Non-Float Values
PASS: Raise ValueError if Mismatched Lists Passed to Constructor
PASS: Raise ValueError if Strings Passed to Constructor
PASS: NNData Limits Training Factor to 0 if < 0
PASS: NNData Limits Training Factor to 1 if > 1

Process finished with exit code 0
"""