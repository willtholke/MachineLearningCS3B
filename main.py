""" This project has been updated since its last instance such that
 the new FFNeurode (Feed Forward Neurode) class was created, which
 is inherited from Neurode. The new FFNeurode class includes four
 new methods and uses super() to call __init__() in its constructor.

Name: William Tholke
Course: CS3B w/ Professor Eric Reed
Date: 05/11/21
"""
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy, copy
from enum import Enum
import numpy as np
import random
import math


class DataMismatchError(Exception):
    pass


class LayerType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class MultiLinkNode(ABC):

    class Side(Enum):
        UPSTREAM = 0
        DOWNSTREAM = 1

    def __init__(self):
        self._reporting_nodes = {
            MultiLinkNode.Side.UPSTREAM: 0,
            MultiLinkNode.Side.DOWNSTREAM: 0
        }
        self._reference_value = {
            MultiLinkNode.Side.UPSTREAM: 0,
            MultiLinkNode.Side.DOWNSTREAM: 0
        }
        self._neighbors = {
            MultiLinkNode.Side.UPSTREAM: [],
            MultiLinkNode.Side.DOWNSTREAM: []
        }

    def __str__(self):
        """ Print ID of node and IDs of neighboring nodes upstream
        & downstream.
        """
        upstream, downstream = [], []
        if not self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            upstream = "Empty upstream node"
        else:
            for i in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
                upstream.append(id(i))
        if not self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            downstream = "Empty downstream node"
        else:
            for x in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
                downstream.append(id(x))

        print(f'Upstream Node ID: {upstream}.')
        print(f'Current Node ID: {id(self)}')
        print(f'Downstream Node ID: {downstream}.')

    @abstractmethod
    def _process_new_neighbor(self, node, side):
        """ Take a node and a side enum as parameters. """
        pass

    def reset_neighbors(self, nodes: list, side: Enum):
        """ Reset or set the nodes that link into a node. """
        if side == MultiLinkNode.Side.UPSTREAM:
            self._neighbors[MultiLinkNode.Side.UPSTREAM] = copy(nodes)
            for i in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
                self._process_new_neighbor(i, side)
            self._reference_value[MultiLinkNode.Side.UPSTREAM] = \
                2**len(nodes) - 1

        elif side == MultiLinkNode.Side.DOWNSTREAM:
            self._neighbors[MultiLinkNode.Side.DOWNSTREAM] = copy(nodes)
            for i in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
                self._process_new_neighbor(i, side)
            self._reference_value[MultiLinkNode.Side.DOWNSTREAM] = \
                2**len(nodes) - 1


class Neurode(MultiLinkNode):

    def __init__(self, node_type, learning_rate=0.05):
        super().__init__()
        self._value = 0
        self._node_type = node_type
        self._learning_rate = learning_rate
        self._weights = {}

    @property
    def value(self):
        """ Get self.value. """
        return self._value

    @property
    def node_type(self):
        """ Get self._node_type. """
        return self._node_type

    @property
    def learning_rate(self):
        """ Get self._learning_rate. """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        """ Set self._learning_rate. """
        self._learning_rate = learning_rate

    def _process_new_neighbor(self, node, side):
        """ Execute when any new neighbors are added. """
        if side is MultiLinkNode.Side.UPSTREAM:
            self._weights[node] = random.uniform(0, 1)

    def _check_in(self, node, side: MultiLinkNode.Side):
        """ Execute when node learns that neighboring node has
        available information.
        """
        node_number = self._neighbors[side].index(node)
        self._reporting_nodes[side] = \
            self._reporting_nodes[side] | 1 << node_number
        if self._reporting_nodes[side] == self._reference_value[side]:
            self._reporting_nodes[side] = 0
            return True
        else:
            return False

    def get_weight(self, node):
        """ Get upstream node's associated weight in self._weights
        dictionary.
        """
        return self._weights[node]


class FFNeurode(Neurode):
    def __init__(self, my_type):
        super().__init__(my_type)

    @staticmethod
    def _sigmoid(value):
        """ Keep input values bound to a known range & return the
        result of the sigmoid function at value.
        """
        return 1 / (1 + np.exp(-value))

    def _calculate_value(self):
        """ Calculate weighted sum of upstream nodes' values, pass
        the result through FFNeurode._sigmoid, and store the value.
        """
        products = []
        for i in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            products.append(i._value * self._weights[i])
        self._value = self._sigmoid(sum(products))

    def _fire_downstream(self):
        """ Call self._data_ready_upstream on every downstream
        neighbor of this node. """
        for i in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            i.data_ready_upstream(self)

    def data_ready_upstream(self, node):
        """ Call self._check_in() to check data status of node
        and proceed accordingly.
        """
        if self._check_in(node, MultiLinkNode.Side.UPSTREAM):
            self._calculate_value()
            self._fire_downstream()

    def set_input(self, input_value):
        """ Set value of input layer neurode. """
        for i in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            i.data_ready_upstream(self)
        self._value = input_value


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
        if target_set is NNData.Set.TEST:
            return len(self._test_indices)
        elif target_set is NNData.Set.TRAIN:
            return len(self._train_indices)
        else:
            return len(self._features)

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


def check_point_two_test():
    inodes = []
    hnodes = []
    onodes = []
    for k in range(2):
        inodes.append(FFNeurode(LayerType.INPUT))
    for k in range(2):
        hnodes.append(FFNeurode(LayerType.HIDDEN))
    onodes.append(FFNeurode(LayerType.OUTPUT))
    for node in inodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in hnodes:
        node.reset_neighbors(inodes, MultiLinkNode.Side.UPSTREAM)
        node.reset_neighbors(onodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in onodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.UPSTREAM)
    try:
        inodes[1].set_input(1)
        assert onodes[0].value == 0
    except:
        print("Error: Neurodes may be firing before receiving all input")
    inodes[0].set_input(0)

    # Since input node 0 has value of 0 and input node 1 has value of
    # one, the value of the hidden layers should be the sigmoid of the
    # weight out of input node 1.

    value_0 = (1 / (1 + np.exp(-hnodes[0]._weights[inodes[1]])))
    value_1 = (1 / (1 + np.exp(-hnodes[1]._weights[inodes[1]])))
    inter = onodes[0]._weights[hnodes[0]] * value_0 + \
            onodes[0]._weights[hnodes[1]] * value_1
    final = (1 / (1 + np.exp(-inter)))
    try:
        print(final, onodes[0].value)
        assert final == onodes[0].value
        assert 0 < final < 1
    except:
        print("Error: Calculation of neurode value may be incorrect")


if __name__ == "__main__":
    check_point_two_test()

"""
-- Sample Run #1 --haha
"""