""" This project has been updated since its last instance such that
two new classes, 'BPNeurode' and 'FFBPNeurode,' allow full capability
of training datasets using the neural network. The abstract base
class 'MultiLinkNode' has a child class 'Neurode,' which is the
parent class of the 'FFNeurode' (Feed Forward Neurode) and 'BPNeurode'
(Backpropagation Neurode) classes. The 'FFBPNeurode' (Feed Forward
Back Propagation Neurode) is multiply inherited combination of
'FFNeurode' and 'BPNeurode' and has no new methods or attributes.

Name: William Tholke
Course: CS3B w/ Professor Eric Reed
Date: 05/18/21
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
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            products.append(node._value * self._weights[node])
        self._value = self._sigmoid(sum(products))

    def _fire_downstream(self):
        """ Call self._data_ready_upstream on every downstream
        neighbor of this node. """
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)

    def data_ready_upstream(self, node):
        """ Call self._check_in() to check data status of node
        and proceed accordingly.
        """
        if self._check_in(node, MultiLinkNode.Side.UPSTREAM):
            self._calculate_value()
            self._fire_downstream()

    def set_input(self, input_value):
        """ Set value of input layer neurode. """
        self._value = input_value
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)


class BPNeurode(FFNeurode):
    def __init__(self, my_type):
        super().__init__(my_type)
        self._delta = 0

    @property
    def delta(self):
        """ Get self._delta. """
        return self._delta

    @staticmethod
    def _sigmoid_derivative(value):
        """ Calculate the sigmoid derivative. """
        return value * (1 - value)

    def _calculate_delta(self, expected_value=None):
        """ Calculate the delta of a specific neurode based on
         the layer type and save result to self._delta.
         """
        if self._node_type is LayerType.OUTPUT:
            self._delta = (expected_value - self._value) * \
                          self._sigmoid_derivative(self._value)
        elif self._node_type is LayerType.HIDDEN:
            products = []
            for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
                products.append(node._delta * node.get_weight(self))
            weighted_sum = sum(products)
            self._delta = (weighted_sum *
                           self._sigmoid_derivative(self._value))

    def data_ready_downstream(self, node):
        """ Register that a node has data and proceed accordingly. """
        if self._check_in(node, MultiLinkNode.Side.DOWNSTREAM):
            self._calculate_delta()
            self._fire_upstream()
            self._update_weights()

    def set_expected(self, expected_value):
        """ Calculate delta and call data_ready_downstream() on all
        upstream neighbors.
        """
        self._calculate_delta(expected_value)
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            node.data_ready_downstream(self)

    def adjust_weights(self, node, adjustment):
        """ Use node reference to add adjustment to appropriate
        entry of self._weights.
        """
        self._weights[node] = self.get_weight(node) + adjustment

    def _update_weights(self):
        """ Iterate through downstream neighbors and request adjustment
        to weight of current node's data.
        """
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            adjustment = self._value * node._delta * node._learning_rate
            node.adjust_weights(self, adjustment)

    def _fire_upstream(self):
        """ Call data_ready_downstream() on all of the upstream
        neighbors.
        """
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            node.data_ready_downstream(self)


class FFBPNeurode(BPNeurode, FFNeurode):
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


def main():
    try:
        test_neurode = BPNeurode(LayerType.HIDDEN)
    except:
        print("Error - Cannot instaniate a BPNeurode object")
        return
    print("Testing Sigmoid Derivative")
    try:
        assert BPNeurode._sigmoid_derivative(0) == 0
        if test_neurode._sigmoid_derivative(.4) == .24:
            print("Pass")
        else:
            print("_sigmoid_derivative is not returning the correct "
                  "result")
    except:
        print("Error - Is _sigmoid_derivative named correctly, created "
              "in BPNeurode and decorated as a static method?")
    print("Testing Instance objects")
    try:
        test_neurode.learning_rate
        test_neurode.delta
        print("Pass")
    except:
        print("Error - Are all instance objects created in __init__()?")

    inodes = []
    hnodes = []
    onodes = []
    for k in range(2):
        inodes.append(FFBPNeurode(LayerType.INPUT))
        hnodes.append(FFBPNeurode(LayerType.HIDDEN))
        onodes.append(FFBPNeurode(LayerType.OUTPUT))
    for node in inodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in hnodes:
        node.reset_neighbors(inodes, MultiLinkNode.Side.UPSTREAM)
        node.reset_neighbors(onodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in onodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.UPSTREAM)
    print("testing learning rate values")
    for node in hnodes:
        print(f"my learning rate is {node.learning_rate}")
    print("Testing check-in")
    try:
        hnodes[0]._reporting_nodes[MultiLinkNode.Side.DOWNSTREAM] = 1
        if hnodes[0]._check_in(onodes[1], MultiLinkNode.Side.DOWNSTREAM) and \
                not hnodes[1]._check_in(onodes[1],
                                        MultiLinkNode.Side.DOWNSTREAM):
            print("Pass")
        else:
            print("Error - _check_in is not responding correctly")
    except:
        print("Error - _check_in is raising an error.  Is it named correctly? "
              "Check your syntax")
    print("Testing calculate_delta on output nodes")
    try:
        onodes[0]._value = .2
        onodes[0]._calculate_delta(.5)
        if .0479 < onodes[0].delta < .0481:
            print("Pass")
        else:
            print("Error - calculate delta is not returning the correct value."
                  "Check the math.")
            print("        Hint: do you have a separate process for hidden "
                  "nodes vs output nodes?")
    except:
        print("Error - calculate_delta is raising an error.  Is it named "
              "correctly?  Check your syntax")
    print("Testing calculate_delta on hidden nodes")
    try:
        onodes[0]._delta = .2
        onodes[1]._delta = .1
        onodes[0]._weights[hnodes[0]] = .4
        onodes[1]._weights[hnodes[0]] = .6
        hnodes[0]._value = .3
        hnodes[0]._calculate_delta()
        if .02939 < hnodes[0].delta < .02941:
            print("Pass")
        else:
            print("Error - calculate delta is not returning the correct value.  "
                  "Check the math.")
            print("        Hint: do you have a separate process for hidden "
                  "nodes vs output nodes?")
    except:
        print("Error - calculate_delta is raising an error.  Is it named correctly?  Check your syntax")
    try:
        print("Testing update_weights")
        hnodes[0]._update_weights()
        if onodes[0].learning_rate == .05:
            if .4 + .06 * onodes[0].learning_rate - .001 < \
                    onodes[0]._weights[hnodes[0]] < \
                    .4 + .06 * onodes[0].learning_rate + .001:
                print("Pass")
            else:
                print("Error - weights not updated correctly.  "
                      "If all other methods passed, check update_weights")
        else:
            print("Error - Learning rate should be .05, please verify")
    except:
        print("Error - update_weights is raising an error.  Is it named "
              "correctly?  Check your syntax")
    print("All that looks good.  Trying to train a trivial dataset "
          "on our network")
    inodes = []
    hnodes = []
    onodes = []
    for k in range(2):
        inodes.append(FFBPNeurode(LayerType.INPUT))
        hnodes.append(FFBPNeurode(LayerType.HIDDEN))
        onodes.append(FFBPNeurode(LayerType.OUTPUT))
    for node in inodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in hnodes:
        node.reset_neighbors(inodes, MultiLinkNode.Side.UPSTREAM)
        node.reset_neighbors(onodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in onodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.UPSTREAM)
    inodes[0].set_input(1)
    inodes[1].set_input(0)
    value1 = onodes[0].value
    value2 = onodes[1].value
    onodes[0].set_expected(0)
    onodes[1].set_expected(1)
    inodes[0].set_input(1)
    inodes[1].set_input(0)
    value1a = onodes[0].value
    value2a = onodes[1].value
    if (value1 - value1a > 0) and (value2a - value2 > 0):
        print("Pass - Learning was done!")
    else:
        print("Fail - the network did not make progress.")
        print("If you hit a wall, be sure to seek help in the discussion "
              "forum, from the instructor and from the tutors")


if __name__ == "__main__":
    main()

"""
-- Sample Run #1 --
Testing Sigmoid Derivative
Pass
Testing Instance objects
Pass
testing learning rate values
my learning rate is 0.05
my learning rate is 0.05
Testing check-in
Pass
Testing calculate_delta on output nodes
Pass
Testing calculate_delta on hidden nodes
Pass
Testing update_weights
Pass
All that looks good.  Trying to train a trivial dataset on our network
Pass - Learning was done!

Process finished with exit code 0
"""