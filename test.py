""" This project has been updated since its last version such that
the constructor for the NNData class includes new instance attributes
and calls a new public method, split_set(). Split_set() sets up 
self._train_indices and self._test_indices as lists to be used
as direct indices for our example data; each list contains unique
data and is of a length determined by a series of calculations
completed within the method.

Name: William Tholke
Course: CS3B w/ Professor Eric Reed
Date: 04/20/21

Notes to self:
1) If the first try block passes with no errors, our_data_0 will be
created as an object within the scope of the unit_test() function.
2) If the first try block throws an error, the second try block will
throw an error immediately because line 169 will be referring to an
object that failed to create in the first try block
3) The except block that has no parameter will except all and every
error, not just a specific error. PyCharm is highlighting that except
block because it's not Pythonic to be unspecific and not include the
specific error we expect.
4) It looks like Professor Reed used the except block without an 
argument is because there are multiple errors that could occur. For 
instance, if the first try block fails, then line 133 in your program 
will fail and raise the following error: "UnboundLocalError: local 
variable 'our_data_0' referenced before assignment." However, if the 
first block passes and it's the one of the three assertions that fails,
then we'll get an AssertionError. So, although not Pythonic, including
the except block without an argument makes it so he doesn't have to
have both an except UnboundLocalError and except AssertionError.
"""
from enum import Enum
import numpy as np
from collections import deque
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
        testing_examples = math.floor((1 - self._train_factor) * total_examples)
        fancy_list, n = (list(range(total_examples))), 0
        random.shuffle(fancy_list)
        self._train_indices, self._test_indices = [], []
        for i in range(training_examples):
            self._train_indices.append(fancy_list[n])
            n += 1
        for i in range(testing_examples):
            self._test_indices.append(fancy_list[n])
            n += 1

    # def prime_data(self, target_set=None, order=None):
    #     """ Load one or both deques to be used as indirect indices. """
    #     if target_set is None:
    #         pass
    #     print(order)
    #
    # def get_one_item(self, target_set=None):
    #     """ Return exactly one feature/label pair as a tuple. """
    #     pass
    #
    # def number_of_samples(self, target_set=None):
    #     """ Return the total number of testing examples, training
    #     examples, or both combined.
    #     """
    #     pass
    #
    # def pool_is_empty(self, target_set=None):
    #     """ Return True if target set queue is empty or return
    #     False if otherwise.
    #     """
    #     if target_set is None:
    #         pass  # use the train pool

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
        x = list(range(10))
        y = x
        our_data_0 = NNData(x, y)
        print(our_data_0._features)
        x = list(range(100))
        y = x
        our_big_data = NNData(x, y, .5)


    except:
        print("There are errors that likely come from __init__ or a "
              "method called by __init__")
        errors = True

    # Test split_set to make sure the correct number of samples are in
    # each set, and that the indices do not overlap.
    try:
        our_data_0.split_set(.3)
        assert len(our_data_0._train_indices) == 3
        assert len(our_data_0._test_indices) == 7
        assert (list(set(our_data_0._train_indices +
                         our_data_0._test_indices))) == list(range(10))
    except:
        print("There are errors that likely come from split_set")
        errors = True  # Summary
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
Text
"""