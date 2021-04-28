""" This project has been updated since its last version such that
five new methods of class NNData have been added, which either load
deques, return one feature/label pair as a tuple, return
testing examples, testing examples, or both combined, or check
if the data pool is empty.

Name: William Tholke
Course: CS3B w/ Professor Eric Reed
Date: 04/27/21
"""
from collections import deque
from copy import deepcopy
from enum import Enum
import numpy as np
import random
import math


class DataMismatchError(Exception):
    pass


class NNData:

    def __init__(self, features=None, labels=None, train_factor=0.9):
        self._features, self._labels = None, None
        self._train_factor = NNData.percentage_limiter(train_factor)
        self._train_indices, self._test_indices = [], []
        self._train_pool, self._test_pool = deque(), deque()

        if features is None:
            features = []
        if labels is None:
            labels = []

        self.load_data(features, labels)
        self.split_set(self._train_factor)

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
        """ Set up self._train_indices and self._test_indices to be
         used as indirect indices for our example data.
         """
        if new_train_factor is not None:
            self._train_factor = NNData.percentage_limiter(new_train_factor)
        total_examples = len([i for i in self._features])
        training_examples = math.floor(self._train_factor * total_examples)
        testing_examples = math.floor((1 - self._train_factor) *
                                      total_examples)
        fancy_list, n = (list(range(total_examples))), 0
        random.shuffle(fancy_list)
        self._train_indices, self._test_indices = [], []
        for i in range(training_examples):
            self._train_indices.append(fancy_list[n])
            n += 1
        for i in range(testing_examples):
            self._test_indices.append(fancy_list[n])
            n += 1

    def prime_data(self, target_set=None, order=None):
        """ Load one or both deques to be used as indirect indices. """
        if target_set is None:
            self._train_pool = deque(deepcopy(self._train_indices))
            self._test_pool = deque(deepcopy(self._test_indices))
        elif target_set == NNData.Set.TRAIN:
            self._train_pool = deque(deepcopy(self._train_indices))
        elif target_set == NNData.Set.TEST:
            self._test_pool = deque(deepcopy(self._test_indices))
        if order == NNData.Order.RANDOM:
            random.shuffle(self._test_pool), random.shuffle(self._train_pool)
        elif order is None or order is NNData.Order.SEQUENTIAL:
            pass

    def get_one_item(self, target_set=None):
        """ Return exactly one feature/label pair as a tuple. """
        if target_set == NNData.Set.TRAIN or target_set is None:
            if len(self._train_pool) > 0:
                index = self._train_pool.popleft()
                location_1 = self._features[index]
                location_2 = self._labels[index]
                return_val = (location_1, location_2)
                return return_val
            else:
                return None
        elif target_set == NNData.Set.TEST or target_set is None:
            if len(self._test_pool) > 0:
                index = self._test_pool.popleft()
                location_1 = self._features[index]
                location_2 = self._labels[index]
                return_val = (location_1, location_2)
                return return_val
            else:
                return None

    def number_of_samples(self, target_set=None):
        """ Return the total number of testing examples, training
        examples, or both combined.
        """
        if target_set == NNData.Set.TEST:
            return len(self._test_indices)
        if target_set is NNData.Set.TRAIN:
            return len(self._train_indices)

    def pool_is_empty(self, target_set=None):
        """ Return True if target set queue is empty or return
        False if otherwise.
        """
        if target_set is None or target_set is NNData.Set.TRAIN:
            if len(self._train_pool) == 0:
                return True
            else:
                return False
        else:
            if len(self._test_pool) == 0:
                return True
            else:
                return False

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
    errors = False
    try:
        # Create a valid small and large dataset to be used later
        x = [[i] for i in range(10)]
        y = x
        our_data_0 = NNData(x, y)
        x = [[i] for i in range(100)]
        y = x
        our_big_data = NNData(x, y, .5)

        # Try loading lists of different sizes
        y = [[1]]
        try:
            our_bad_data = NNData()
            our_bad_data.load_data(x, y)
            raise Exception
        except DataMismatchError:
            pass
        except:
            raise Exception

        # Create a dataset that can be used to make sure the
        # features and labels are not confused
        x = [[1.0], [2.0], [3.0], [4.0]]
        y = [[.1], [.2], [.3], [.4]]
        our_data_1 = NNData(x, y, .5)

    except:
        print("There are errors that likely come from __init__ or a "
              "method called by __init__")
        errors = True

    # Test split_set to make sure the correct number of examples are in
    # each set, and that the indices do not overlap.
    try:
        our_data_0.split_set(.3)
        print(f"Train Indices:{our_data_0._train_indices}")
        print(f"Test Indices:{our_data_0._test_indices}")

        assert len(our_data_0._train_indices) == 3
        assert len(our_data_0._test_indices) == 7
        assert (list(set(our_data_0._train_indices +
                         our_data_0._test_indices))) == list(range(10))
    except:
        print("There are errors that likely come from split_set")
        errors = True

    # Make sure prime_data sets up the deques correctly, whether
    # sequential or random.
    try:
        our_data_0.prime_data(order=NNData.Order.SEQUENTIAL)
        assert len(our_data_0._train_pool) == 3
        assert len(our_data_0._test_pool) == 7
        assert our_data_0._train_indices == list(our_data_0._train_pool)
        assert our_data_0._test_indices == list(our_data_0._test_pool)
        our_big_data.prime_data(order=NNData.Order.RANDOM)
        assert our_big_data._train_indices != list(our_big_data._train_pool)
        assert our_big_data._test_indices != list(our_big_data._test_pool)
    except:
        print("There are errors that likely come from prime_data")
        errors = True

    # Make sure get_one_item is returning the correct values, and
    # that pool_is_empty functions correctly.
    try:
        our_data_1.prime_data(order=NNData.Order.SEQUENTIAL)
        my_x_list = []
        my_y_list = []
        while not our_data_1.pool_is_empty():
            example = our_data_1.get_one_item()
            my_x_list.append(list(example[0]))
            my_y_list.append(list(example[1]))

        assert len(my_x_list) == 2
        assert my_x_list != my_y_list
        my_matched_x_list = [i[0] * 10 for i in my_y_list]
        assert my_matched_x_list == my_x_list
        while not our_data_1.pool_is_empty(our_data_1.Set.TEST):
            example = our_data_1.get_one_item(our_data_1.Set.TEST)
            my_x_list.append(list(example[0]))
            my_y_list.append(list(example[1]))
        assert my_x_list != my_y_list
        my_matched_x_list = [i[0] * 10 for i in my_y_list]
        assert my_matched_x_list == my_x_list
        assert set(i[0] for i in my_x_list) == set(i[0] for i in x)
        assert set(i[0] for i in my_y_list) == set(i[0] for i in y)
    except:
        print("There are errors that may come from prime_data, but could "
              "be from another method")
        errors = True

    # Summary
    if errors:
        print("You have one or more errors.  Please fix them before "
              "submitting")
    else:
        print("No errors were identified by the unit test.")
        print("You should still double check that your code meets spec.")
        print("You should also check that PyCharm does not identify any "
              "PEP-8 issues.")


if __name__ == "__main__":
    unit_test()

"""
-- Sample Run #1 --

"""