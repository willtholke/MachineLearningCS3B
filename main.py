""" This project has been updated since its last instance such that...

Name: William Tholke
Course: CS3B w/ Professor Eric Reed
Date: 06/18/21
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


class DLLNode:

    def __init__(self, data=None):
        self.prev = None
        self.next = None
        self.data = data


class DoublyLinkedList:

    class EmptyListError(Exception):
        pass

    def __init__(self):
        self._head = None
        self._tail = None
        self._current = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._current and self._current.next:
            ret_val = self._current.data
            self._current = self._current.next
            return ret_val
        raise StopIteration

    def move_forward(self):
        """ Return the data from new node if it exists. """
        if not self._current:
            raise DoublyLinkedList.EmptyListError
        if self._current.next:
            self._current = self._current.next
        else:
            raise IndexError

    def move_back(self):
        """ Return the data from new node if it exists. """
        if not self._current:
            raise DoublyLinkedList.EmptyListError
        if self._current.prev:
            self._current = self._current.prev
        else:
            raise IndexError

    def add_to_head(self, data):
        """ Add data to the first node in the list. """
        new_node = DLLNode(data)
        new_node.next = self._head
        if self._head:
            self._head.prev = new_node
        self._head = new_node
        if self._tail is None:
            self._tail = new_node
        self.reset_to_head()

    def remove_from_head(self):
        """ Return data value if there is a node at the head or
        return None otherwise.
        """
        if not self._head:
            raise DoublyLinkedList.EmptyListError
        ret_val = self._head.data
        self._head = self._head.next
        if self._head:
            self._head.prev = None
        else:
            self._tail = None
        self.reset_to_head()
        return ret_val

    def add_after_cur(self, data):
        """ Alter the middle of a list by adding after a node. """
        if not self._current:
            raise DoublyLinkedList.EmptyListError
        new_node = DLLNode(data)
        new_node.prev = self._current
        new_node.next = self._current.next
        if self._current.next:
            self._current.next.prev = new_node
        self._current.next = new_node
        if self._tail == self._current:
            self._tail = new_node

    def remove_after_cur(self):
        """ Alter the middle of a list by removing after a node. """
        if not self._current:
            raise DoublyLinkedList.EmptyListError
        if self._current == self._tail:
            raise IndexError
        ret_val = self._current.next.data
        if self._current.next == self._tail:
            self._tail = self._current
            self._current.next = None
        else:
            self._current.next = self._current.next.next
            self._current.next.prev = self._current
        return ret_val

    def reset_to_head(self):
        """ Reset current node to head. """
        if not self._head:
            raise DoublyLinkedList.EmptyListError
        self._current = self._head

    def reset_to_tail(self):
        """ Reset current node to tail. """
        if not self._tail:
            raise DoublyLinkedList.EmptyListError
        self._current = self._tail

    def get_current_data(self):
        """ Create an iterator for the list. """
        if not self._current:
            raise DoublyLinkedList.EmptyListError
        return self._current.data


class LayerList(DoublyLinkedList):

    def __init__(self, inputs, outputs):
        super().__init__()
        if inputs < 1 or outputs < 1:
            raise ValueError
        input_layer = [FFBPNeurode(LayerType.INPUT) for _ in range(inputs)]
        output_layer = [FFBPNeurode(LayerType.OUTPUT) for _ in range(outputs)]
        self.add_to_head(input_layer)
        self.add_after_cur(output_layer)
        self._link_with_next()

    def _link_with_next(self):
        """ Link with next layer. """
        for node in self._current.data:
            node.reset_neighbors(self._current.next.data,
                                 FFBPNeurode.Side.DOWNSTREAM)
        for node in self._current.next.data:
            node.reset_neighbors(self._current.data,
                                 FFBPNeurode.Side.UPSTREAM)

    def add_layer(self, num_nodes):
        """ Create hidden layer of neurodes after current layer. """
        if self._current == self._tail:
            raise IndexError
        hidden_layer = [FFBPNeurode(LayerType.HIDDEN) for _ in
                        range(num_nodes)]
        self.add_after_cur(hidden_layer)
        self._link_with_next()
        self.move_forward()
        self._link_with_next()
        self.move_back()

    def remove_layer(self):
        """ Remove layer after current layer."""
        if self._current == self._tail or self._current.next == self._tail:
            raise IndexError
        self.remove_after_cur()
        self._link_with_next()

    @property
    def input_nodes(self):
        """ Return self._head.data. """
        return self._head.data

    @property
    def output_nodes(self):
        """ Return self._tail.data. """
        return self._tail.data


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
        if target_set is NNData.Set.TEST:
            return len(self._test_pool) == 0
        else:
            return len(self._train_pool) == 0

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


class FFBPNetwork:

    class EmptySetException(Exception):
        pass

    def __init__(self, num_inputs, num_outputs):
        self.layers = LayerList(num_inputs, num_outputs)
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs

    def add_hidden_layer(self, num_nodes=5, position=0):
        """ Add a hidden layer with the given number of nodes. """
        self.layers.reset_to_head()
        for _ in range(position):
            self.layers.move_forward()
        self.layers.add_layer(num_nodes)

    def train(self, data_set: NNData, epochs=1000, verbosity=2,
              order=NNData.Order.RANDOM):
        """ Train the neural network. """
        if data_set.number_of_samples(NNData.Set.TRAIN) == 0:
            raise FFBPNetwork.EmptySetException
        for epoch in range(0, epochs):
            data_set.prime_data(order=order)
            sum_error = 0
            while not data_set.pool_is_empty(NNData.Set.TRAIN):
                x, y = data_set.get_one_item(NNData.Set.TRAIN)
                for j, node in enumerate(self.layers.input_nodes):
                    node.set_input(x[j])
                produced = []
                for j, node in enumerate(self.layers.output_nodes):
                    node.set_expected(y[j])
                    sum_error += (node.value - y[j]) ** 2 / self._num_outputs
                    produced.append(node.value)

                if epoch % 1000 == 0 and verbosity > 1:
                    # print("Sample", x, "expected", y, "produced", produced)
                    pass
            if epoch % 100 == 0 and verbosity > 0:
                # print("Epoch", epoch, "RMSE = ", math.sqrt(sum_error / data_set.number_of_samples(NNData.Set.TRAIN)))
                # print(math.sqrt(sum_error / data_set.number_of_samples(NNData.Set.TRAIN)))
                pass
        print("Final Epoch RMSE = ", math.sqrt(sum_error / data_set.number_of_samples(NNData.Set.TRAIN)))

    def test(self, data_set: NNData, order=NNData.Order.SEQUENTIAL):
        """ Test the neural network. """
        if data_set.number_of_samples(NNData.Set.TEST) == 0:
            raise FFBPNetwork.EmptySetException
        data_set.prime_data(order=order)
        sum_error, sample, expected, produced, = 0, [], [], []
        while not data_set.pool_is_empty(NNData.Set.TEST):
            x, y = data_set.get_one_item(NNData.Set.TEST)
            for i in x:
                sample.append(i)
            for i in y:
                expected.append(i)
            for j, node in enumerate(self.layers.input_nodes):
                node.set_input(x[j])
            for j, node in enumerate(self.layers.output_nodes):
                node.set_expected(y[j])
                sum_error += (node.value - y[j]) ** 2 / self._num_outputs
                produced.append(node.value)
        produced.sort()
        sample.sort()
        print("\nTest Results:")
        print(f'Input: {sample}')
        print(f'Expected: {expected}')
        print(f'Output: {produced}')
        print("Test RMSE = ", math.sqrt(sum_error / data_set.number_of_samples(NNData.Set.TRAIN)))


def run_iris():
    """ Train and test the Iris Dataset from University of California,
    Irvine.
    """
    network = FFBPNetwork(4, 3)
    network.add_hidden_layer(3)
    Iris_X = [[5.1, 3.5, 1.4, 0.2], [4.9, 3, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2],
              [5, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], [4.6, 3.4, 1.4, 0.3], [5, 3.4, 1.5, 0.2],
              [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
              [4.8, 3, 1.4, 0.1], [4.3, 3, 1.1, 0.1], [5.8, 4, 1.2, 0.2], [5.7, 4.4, 1.5, 0.4],
              [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3], [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3],
              [5.4, 3.4, 1.7, 0.2], [5.1, 3.7, 1.5, 0.4], [4.6, 3.6, 1, 0.2], [5.1, 3.3, 1.7, 0.5],
              [4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 0.2], [5, 3.4, 1.6, 0.4], [5.2, 3.5, 1.5, 0.2],
              [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2], [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4],
              [5.2, 4.1, 1.5, 0.1], [5.5, 4.2, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5, 3.2, 1.2, 0.2],
              [5.5, 3.5, 1.3, 0.2], [4.9, 3.1, 1.5, 0.1], [4.4, 3, 1.3, 0.2], [5.1, 3.4, 1.5, 0.2],
              [5, 3.5, 1.3, 0.3], [4.5, 2.3, 1.3, 0.3], [4.4, 3.2, 1.3, 0.2], [5, 3.5, 1.6, 0.6],
              [5.1, 3.8, 1.9, 0.4], [4.8, 3, 1.4, 0.3], [5.1, 3.8, 1.6, 0.2], [4.6, 3.2, 1.4, 0.2],
              [5.3, 3.7, 1.5, 0.2], [5, 3.3, 1.4, 0.2], [7, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5],
              [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3],
              [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1], [6.6, 2.9, 4.6, 1.3], [5.2, 2.7, 3.9, 1.4], [5, 2, 3.5, 1],
              [5.9, 3, 4.2, 1.5], [6, 2.2, 4, 1], [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3], [6.7, 3.1, 4.4, 1.4],
              [5.6, 3, 4.5, 1.5], [5.8, 2.7, 4.1, 1], [6.2, 2.2, 4.5, 1.5], [5.6, 2.5, 3.9, 1.1],
              [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4, 1.3], [6.3, 2.5, 4.9, 1.5], [6.1, 2.8, 4.7, 1.2],
              [6.4, 2.9, 4.3, 1.3], [6.6, 3, 4.4, 1.4], [6.8, 2.8, 4.8, 1.4], [6.7, 3, 5, 1.7], [6, 2.9, 4.5, 1.5],
              [5.7, 2.6, 3.5, 1], [5.5, 2.4, 3.8, 1.1], [5.5, 2.4, 3.7, 1], [5.8, 2.7, 3.9, 1.2],
              [6, 2.7, 5.1, 1.6], [5.4, 3, 4.5, 1.5], [6, 3.4, 4.5, 1.6], [6.7, 3.1, 4.7, 1.5],
              [6.3, 2.3, 4.4, 1.3], [5.6, 3, 4.1, 1.3], [5.5, 2.5, 4, 1.3], [5.5, 2.6, 4.4, 1.2],
              [6.1, 3, 4.6, 1.4], [5.8, 2.6, 4, 1.2], [5, 2.3, 3.3, 1], [5.6, 2.7, 4.2, 1.3], [5.7, 3, 4.2, 1.2],
              [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3], [5.1, 2.5, 3, 1.1], [5.7, 2.8, 4.1, 1.3],
              [6.3, 3.3, 6, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3, 5.9, 2.1], [6.3, 2.9, 5.6, 1.8],
              [6.5, 3, 5.8, 2.2], [7.6, 3, 6.6, 2.1], [4.9, 2.5, 4.5, 1.7], [7.3, 2.9, 6.3, 1.8],
              [6.7, 2.5, 5.8, 1.8], [7.2, 3.6, 6.1, 2.5], [6.5, 3.2, 5.1, 2], [6.4, 2.7, 5.3, 1.9],
              [6.8, 3, 5.5, 2.1], [5.7, 2.5, 5, 2], [5.8, 2.8, 5.1, 2.4], [6.4, 3.2, 5.3, 2.3], [6.5, 3, 5.5, 1.8],
              [7.7, 3.8, 6.7, 2.2], [7.7, 2.6, 6.9, 2.3], [6, 2.2, 5, 1.5], [6.9, 3.2, 5.7, 2.3],
              [5.6, 2.8, 4.9, 2], [7.7, 2.8, 6.7, 2], [6.3, 2.7, 4.9, 1.8], [6.7, 3.3, 5.7, 2.1],
              [7.2, 3.2, 6, 1.8], [6.2, 2.8, 4.8, 1.8], [6.1, 3, 4.9, 1.8], [6.4, 2.8, 5.6, 2.1],
              [7.2, 3, 5.8, 1.6], [7.4, 2.8, 6.1, 1.9], [7.9, 3.8, 6.4, 2], [6.4, 2.8, 5.6, 2.2],
              [6.3, 2.8, 5.1, 1.5], [6.1, 2.6, 5.6, 1.4], [7.7, 3, 6.1, 2.3], [6.3, 3.4, 5.6, 2.4],
              [6.4, 3.1, 5.5, 1.8], [6, 3, 4.8, 1.8], [6.9, 3.1, 5.4, 2.1], [6.7, 3.1, 5.6, 2.4],
              [6.9, 3.1, 5.1, 2.3], [5.8, 2.7, 5.1, 1.9], [6.8, 3.2, 5.9, 2.3], [6.7, 3.3, 5.7, 2.5],
              [6.7, 3, 5.2, 2.3], [6.3, 2.5, 5, 1.9], [6.5, 3, 5.2, 2], [6.2, 3.4, 5.4, 2.3], [5.9, 3, 5.1, 1.8]]
    Iris_Y = [[1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ]]
    data = NNData(Iris_X, Iris_Y, .9)
    network.train(data, 1001, order=NNData.Order.RANDOM)
    network.test(data)


def run_sin():
    """ Train and test the sine function from 0 to 1.57 radians. """
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    sin_X = [[0], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07], [0.08], [0.09], [0.1], [0.11], [0.12],
             [0.13], [0.14], [0.15], [0.16], [0.17], [0.18], [0.19], [0.2], [0.21], [0.22], [0.23], [0.24], [0.25],
             [0.26], [0.27], [0.28], [0.29], [0.3], [0.31], [0.32], [0.33], [0.34], [0.35], [0.36], [0.37], [0.38],
             [0.39], [0.4], [0.41], [0.42], [0.43], [0.44], [0.45], [0.46], [0.47], [0.48], [0.49], [0.5], [0.51],
             [0.52], [0.53], [0.54], [0.55], [0.56], [0.57], [0.58], [0.59], [0.6], [0.61], [0.62], [0.63], [0.64],
             [0.65], [0.66], [0.67], [0.68], [0.69], [0.7], [0.71], [0.72], [0.73], [0.74], [0.75], [0.76], [0.77],
             [0.78], [0.79], [0.8], [0.81], [0.82], [0.83], [0.84], [0.85], [0.86], [0.87], [0.88], [0.89], [0.9],
             [0.91], [0.92], [0.93], [0.94], [0.95], [0.96], [0.97], [0.98], [0.99], [1], [1.01], [1.02], [1.03],
             [1.04], [1.05], [1.06], [1.07], [1.08], [1.09], [1.1], [1.11], [1.12], [1.13], [1.14], [1.15], [1.16],
             [1.17], [1.18], [1.19], [1.2], [1.21], [1.22], [1.23], [1.24], [1.25], [1.26], [1.27], [1.28], [1.29],
             [1.3], [1.31], [1.32], [1.33], [1.34], [1.35], [1.36], [1.37], [1.38], [1.39], [1.4], [1.41], [1.42],
             [1.43], [1.44], [1.45], [1.46], [1.47], [1.48], [1.49], [1.5], [1.51], [1.52], [1.53], [1.54], [1.55],
             [1.56], [1.57]]
    sin_Y = [[0], [0.00999983333416666], [0.0199986666933331], [0.0299955002024957], [0.0399893341866342],
             [0.0499791692706783], [0.0599640064794446], [0.0699428473375328], [0.0799146939691727],
             [0.089878549198011], [0.0998334166468282], [0.109778300837175], [0.119712207288919],
             [0.129634142619695], [0.139543114644236], [0.149438132473599], [0.159318206614246],
             [0.169182349066996], [0.179029573425824], [0.188858894976501], [0.198669330795061], [0.2084598998461],
             [0.218229623080869], [0.227977523535188], [0.237702626427135], [0.247403959254523],
             [0.257080551892155], [0.266731436688831], [0.276355648564114], [0.285952225104836], [0.29552020666134],
             [0.305058636443443], [0.314566560616118], [0.324043028394868], [0.333487092140814],
             [0.342897807455451], [0.35227423327509], [0.361615431964962], [0.370920469412983], [0.380188415123161],
             [0.389418342308651], [0.398609327984423], [0.40776045305957], [0.416870802429211], [0.425939465066],
             [0.43496553411123], [0.44394810696552], [0.452886285379068], [0.461779175541483], [0.470625888171158],
             [0.479425538604203], [0.488177246882907], [0.496880137843737], [0.505533341204847],
             [0.514135991653113], [0.522687228930659], [0.531186197920883], [0.539632048733969],
             [0.548023936791874], [0.556361022912784], [0.564642473395035], [0.572867460100481],
             [0.581035160537305], [0.58914475794227], [0.597195441362392], [0.60518640573604], [0.613116851973434],
             [0.62098598703656], [0.628793024018469], [0.636537182221968], [0.644217687237691], [0.651833771021537],
             [0.659384671971473], [0.666869635003698], [0.674287911628145], [0.681638760023334],
             [0.688921445110551], [0.696135238627357], [0.70327941920041], [0.710353272417608], [0.717356090899523],
             [0.724287174370143], [0.731145829726896], [0.737931371109963], [0.744643119970859],
             [0.751280405140293], [0.757842562895277], [0.764328937025505], [0.770738878898969],
             [0.777071747526824], [0.783326909627483], [0.78950373968995], [0.795601620036366], [0.801619940883777],
             [0.807558100405114], [0.813415504789374], [0.819191568300998], [0.82488571333845], [0.83049737049197],
             [0.836025978600521], [0.841470984807897], [0.846831844618015], [0.852108021949363],
             [0.857298989188603], [0.862404227243338], [0.867423225594017], [0.872355482344986],
             [0.877200504274682], [0.881957806884948], [0.886626914449487], [0.891207360061435],
             [0.895698685680048], [0.900100442176505], [0.904412189378826], [0.908633496115883],
             [0.912763940260521], [0.916803108771767], [0.920750597736136], [0.92460601240802], [0.928368967249167],
             [0.932039085967226], [0.935616001553386], [0.939099356319068], [0.942488801931697],
             [0.945783999449539], [0.948984619355586], [0.952090341590516], [0.955100855584692],
             [0.958015860289225], [0.960835064206073], [0.963558185417193], [0.966184951612734],
             [0.968715100118265], [0.971148377921045], [0.973484541695319], [0.975723357826659],
             [0.977864602435316], [0.979908061398614], [0.98185353037236], [0.983700814811277], [0.98544972998846],
             [0.98710010101385], [0.98865176285172], [0.990104560337178], [0.991458348191686], [0.992712991037588],
             [0.993868363411645], [0.994924349777581], [0.99588084453764], [0.996737752043143], [0.997494986604054],
             [0.998152472497548], [0.998710143975583], [0.999167945271476], [0.999525830605479],
             [0.999783764189357], [0.999941720229966], [0.999999682931835]]
    data = NNData(sin_X, sin_Y, 0.1)
    network.train(data, 10001, order=NNData.Order.RANDOM)
    network.test(data)


def load_XOR():
    """ List of features and a list of labels.
    Note: XOR ('except or') is only true if exactly one input is true.
    """
    features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]
    data = NNData(features, labels, 1)
    return data


def run_sin_2pi():
    """ Train and test the sine function from 0 to 6.28 radians. """
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    sin_X = [[0], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07], [0.08], [0.09], [0.1], [0.11], [0.12],
             [0.13], [0.14], [0.15], [0.16], [0.17], [0.18], [0.19], [0.2], [0.21], [0.22], [0.23], [0.24], [0.25],
             [0.26], [0.27], [0.28], [0.29], [0.3], [0.31], [0.32], [0.33], [0.34], [0.35], [0.36], [0.37], [0.38],
             [0.39], [0.4], [0.41], [0.42], [0.43], [0.44], [0.45], [0.46], [0.47], [0.48], [0.49], [0.5], [0.51],
             [0.52], [0.53], [0.54], [0.55], [0.56], [0.57], [0.58], [0.59], [0.6], [0.61], [0.62], [0.63], [0.64],
             [0.65], [0.66], [0.67], [0.68], [0.69], [0.7], [0.71], [0.72], [0.73], [0.74], [0.75], [0.76], [0.77],
             [0.78], [0.79], [0.8], [0.81], [0.82], [0.83], [0.84], [0.85], [0.86], [0.87], [0.88], [0.89], [0.9],
             [0.91], [0.92], [0.93], [0.94], [0.95], [0.96], [0.97], [0.98], [0.99], [1], [1.01], [1.02], [1.03],
             [1.04], [1.05], [1.06], [1.07], [1.08], [1.09], [1.1], [1.11], [1.12], [1.13], [1.14], [1.15], [1.16],
             [1.17], [1.18], [1.19], [1.2], [1.21], [1.22], [1.23], [1.24], [1.25], [1.26], [1.27], [1.28], [1.29],
             [1.3], [1.31], [1.32], [1.33], [1.34], [1.35], [1.36], [1.37], [1.38], [1.39], [1.4], [1.41], [1.42],
             [1.43], [1.44], [1.45], [1.46], [1.47], [1.48], [1.49], [1.5], [1.51], [1.52], [1.53], [1.54], [1.55],
             [1.56], [1.57], [1.58], [1.59], [1.6], [1.61], [1.62], [1.63], [1.64], [1.65], [1.66], [1.67], [1.68],
             [1.69], [1.7], [1.71], [1.72], [1.73], [1.74], [1.75], [1.76], [1.77], [1.78], [1.79], [1.8], [1.81],
             [1.82], [1.83], [1.84], [1.85], [1.86], [1.87], [1.88], [1.89], [1.9], [1.91], [1.92], [1.93], [1.94],
             [1.95], [1.96], [1.97], [1.98], [1.99], [2], [2.01], [2.02], [2.03], [2.04], [2.05], [2.06], [2.07],
             [2.08], [2.09], [2.1], [2.11], [2.12], [2.13], [2.14], [2.15], [2.16], [2.17], [2.18], [2.19], [2.2],
             [2.21], [2.22], [2.23], [2.24], [2.25], [2.26], [2.27], [2.28], [2.29], [2.3], [2.31], [2.32], [2.33],
             [2.34], [2.35], [2.36], [2.37], [2.38], [2.39], [2.4], [2.41], [2.42], [2.43], [2.44], [2.45], [2.46],
             [2.47], [2.48], [2.49], [2.5], [2.51], [2.52], [2.53], [2.54], [2.55], [2.56], [2.57], [2.58], [2.59],
             [2.6], [2.61], [2.62], [2.63], [2.64], [2.65], [2.66], [2.67], [2.68], [2.69], [2.7], [2.71], [2.72],
             [2.73], [2.74], [2.75], [2.76], [2.77], [2.78], [2.79], [2.8], [2.81], [2.82], [2.83], [2.84], [2.85],
             [2.86], [2.87], [2.88], [2.89], [2.9], [2.91], [2.92], [2.93], [2.94], [2.95], [2.96], [2.97], [2.98],
             [2.99], [3], [3.01], [3.02], [3.03], [3.04], [3.05], [3.06], [3.07], [3.08], [3.09], [3.1], [3.11],
             [3.12], [3.13], [3.14], [3.15], [3.16], [3.17], [3.18], [3.19], [3.2], [3.21], [3.22], [3.23], [3.24],
             [3.25], [3.26], [3.27], [3.28], [3.29], [3.3], [3.31], [3.32], [3.33], [3.34], [3.35], [3.36], [3.37],
             [3.38], [3.39], [3.4], [3.41], [3.42], [3.43], [3.44], [3.45], [3.46], [3.47], [3.48], [3.49], [3.5],
             [3.51], [3.52], [3.53], [3.54], [3.55], [3.56], [3.57], [3.58], [3.59], [3.6], [3.61], [3.62], [3.63],
             [3.64], [3.65], [3.66], [3.67], [3.68], [3.69], [3.7], [3.71], [3.72], [3.73], [3.74], [3.75], [3.76],
             [3.77], [3.78], [3.79], [3.8], [3.81], [3.82], [3.83], [3.84], [3.85], [3.86], [3.87], [3.88], [3.89],
             [3.9], [3.91], [3.92], [3.93], [3.94], [3.95], [3.96], [3.97], [3.98], [3.99], [4], [4.01], [4.02],
             [4.03], [4.04], [4.05], [4.06], [4.07], [4.08], [4.09], [4.1], [4.11], [4.12], [4.13], [4.14], [4.15],
             [4.16], [4.17], [4.18], [4.19], [4.2], [4.21], [4.22], [4.23], [4.24], [4.25], [4.26], [4.27], [4.28],
             [4.29], [4.3], [4.31], [4.32], [4.33], [4.34], [4.35], [4.36], [4.37], [4.38], [4.39], [4.4], [4.41],
             [4.42], [4.43], [4.44], [4.45], [4.46], [4.47], [4.48], [4.49], [4.5], [4.51], [4.52], [4.53], [4.54],
             [4.55], [4.56], [4.57], [4.58], [4.59], [4.6], [4.61], [4.62], [4.63], [4.64], [4.65], [4.66], [4.67],
             [4.68], [4.69], [4.7], [4.71], [4.72], [4.73], [4.74], [4.75], [4.76], [4.77], [4.78], [4.79], [4.8],
             [4.81], [4.82], [4.83], [4.84], [4.85], [4.86], [4.87], [4.88], [4.89], [4.9], [4.91], [4.92], [4.93],
             [4.94], [4.95], [4.96], [4.97], [4.98], [4.99], [5], [5.01], [5.02], [5.03], [5.04], [5.05], [5.06],
             [5.07], [5.08], [5.09], [5.1], [5.11], [5.12], [5.13], [5.14], [5.15], [5.16], [5.17], [5.18], [5.19],
             [5.2], [5.21], [5.22], [5.23], [5.24], [5.25], [5.26], [5.27], [5.28], [5.29], [5.3], [5.31], [5.32],
             [5.33], [5.34], [5.35], [5.36], [5.37], [5.38], [5.39], [5.4], [5.41], [5.42], [5.43], [5.44], [5.45],
             [5.46], [5.47], [5.48], [5.49], [5.5], [5.51], [5.52], [5.53], [5.54], [5.55], [5.56], [5.57], [5.58],
             [5.59], [5.6], [5.61], [5.62], [5.63], [5.64], [5.65], [5.66], [5.67], [5.68], [5.69], [5.7], [5.71],
             [5.72], [5.73], [5.74], [5.75], [5.76], [5.77], [5.78], [5.79], [5.8], [5.81], [5.82], [5.83], [5.84],
             [5.85], [5.86], [5.87], [5.88], [5.89], [5.9], [5.91], [5.92], [5.93], [5.94], [5.95], [5.96], [5.97],
             [5.98], [5.99], [6], [6.01], [6.02], [6.03], [6.04], [6.05], [6.06], [6.07], [6.08], [6.09], [6.1],
             [6.11], [6.12], [6.13], [6.14], [6.15], [6.16], [6.17], [6.18], [6.19], [6.2], [6.21], [6.22], [6.23],
             [6.24], [6.25], [6.26], [6.27], [6.28]]
    sin_Y = [[0], [0.00999983333416666], [0.0199986666933331], [0.0299955002024957], [0.0399893341866342],
             [0.0499791692706783], [0.0599640064794446], [0.0699428473375328], [0.0799146939691727],
             [0.089878549198011], [0.0998334166468282], [0.109778300837175], [0.119712207288919],
             [0.129634142619695], [0.139543114644236], [0.149438132473599], [0.159318206614246],
             [0.169182349066996], [0.179029573425824], [0.188858894976501], [0.198669330795061], [0.2084598998461],
             [0.218229623080869], [0.227977523535188], [0.237702626427135], [0.247403959254523],
             [0.257080551892155], [0.266731436688831], [0.276355648564114], [0.285952225104836], [0.29552020666134],
             [0.305058636443443], [0.314566560616118], [0.324043028394868], [0.333487092140814],
             [0.342897807455451], [0.35227423327509], [0.361615431964962], [0.370920469412983], [0.380188415123161],
             [0.389418342308651], [0.398609327984423], [0.40776045305957], [0.416870802429211], [0.425939465066],
             [0.43496553411123], [0.44394810696552], [0.452886285379068], [0.461779175541483], [0.470625888171158],
             [0.479425538604203], [0.488177246882907], [0.496880137843737], [0.505533341204847],
             [0.514135991653113], [0.522687228930659], [0.531186197920883], [0.539632048733969],
             [0.548023936791874], [0.556361022912784], [0.564642473395035], [0.572867460100481],
             [0.581035160537305], [0.58914475794227], [0.597195441362392], [0.60518640573604], [0.613116851973434],
             [0.62098598703656], [0.628793024018469], [0.636537182221968], [0.644217687237691], [0.651833771021537],
             [0.659384671971473], [0.666869635003698], [0.674287911628145], [0.681638760023334],
             [0.688921445110551], [0.696135238627357], [0.70327941920041], [0.710353272417608], [0.717356090899523],
             [0.724287174370143], [0.731145829726896], [0.737931371109963], [0.744643119970859],
             [0.751280405140293], [0.757842562895277], [0.764328937025505], [0.770738878898969],
             [0.777071747526824], [0.783326909627483], [0.78950373968995], [0.795601620036366], [0.801619940883777],
             [0.807558100405114], [0.813415504789374], [0.819191568300998], [0.82488571333845], [0.83049737049197],
             [0.836025978600521], [0.841470984807897], [0.846831844618015], [0.852108021949363],
             [0.857298989188603], [0.862404227243338], [0.867423225594017], [0.872355482344986],
             [0.877200504274682], [0.881957806884948], [0.886626914449487], [0.891207360061435],
             [0.895698685680048], [0.900100442176505], [0.904412189378826], [0.908633496115883],
             [0.912763940260521], [0.916803108771767], [0.920750597736136], [0.92460601240802], [0.928368967249167],
             [0.932039085967226], [0.935616001553386], [0.939099356319068], [0.942488801931697],
             [0.945783999449539], [0.948984619355586], [0.952090341590516], [0.955100855584692],
             [0.958015860289225], [0.960835064206073], [0.963558185417193], [0.966184951612734],
             [0.968715100118265], [0.971148377921045], [0.973484541695319], [0.975723357826659],
             [0.977864602435316], [0.979908061398614], [0.98185353037236], [0.983700814811277], [0.98544972998846],
             [0.98710010101385], [0.98865176285172], [0.990104560337178], [0.991458348191686], [0.992712991037588],
             [0.993868363411645], [0.994924349777581], [0.99588084453764], [0.996737752043143], [0.997494986604054],
             [0.998152472497548], [0.998710143975583], [0.999167945271476], [0.999525830605479],
             [0.999783764189357], [0.999941720229966], [0.999999682931835], [0.9999996829], [0.9999417202],
             [0.9997837642], [0.9995258306], [0.9991679453], [0.998710144], [0.9981524725], [0.9974949866],
             [0.996737752], [0.9958808445], [0.9949243498], [0.9938683634], [0.992712991], [0.9914583482],
             [0.9901045603], [0.9886517629], [0.987100101], [0.98544973], [0.9837008148], [0.9818535304],
             [0.9799080614], [0.9778646024], [0.9757233578], [0.9734845417], [0.9711483779], [0.9687151001],
             [0.9661849516], [0.9635581854], [0.9608350642], [0.9580158603], [0.9551008556], [0.9520903416],
             [0.9489846194], [0.9457839994], [0.9424888019], [0.9390993563], [0.9356160016], [0.932039086],
             [0.9283689672], [0.9246060124], [0.9207505977], [0.9168031088], [0.9127639403], [0.9086334961],
             [0.9044121894], [0.9001004422], [0.8956986857], [0.8912073601], [0.8866269144], [0.8819578069],
             [0.8772005043], [0.8723554823], [0.8674232256], [0.8624042272], [0.8572989892], [0.8521080219],
             [0.8468318446], [0.8414709848], [0.8360259786], [0.8304973705], [0.8248857133], [0.8191915683],
             [0.8134155048], [0.8075581004], [0.8016199409], [0.79560162], [0.7895037397], [0.7833269096],
             [0.7770717475], [0.7707388789], [0.764328937], [0.7578425629], [0.7512804051], [0.74464312],
             [0.7379313711], [0.7311458297], [0.7242871744], [0.7173560909], [0.7103532724], [0.7032794192],
             [0.6961352386], [0.6889214451], [0.68163876], [0.6742879116], [0.666869635], [0.659384672],
             [0.651833771], [0.6442176872], [0.6365371822], [0.628793024], [0.620985987], [0.613116852],
             [0.6051864057], [0.5971954414], [0.5891447579], [0.5810351605], [0.5728674601], [0.5646424734],
             [0.5563610229], [0.5480239368], [0.5396320487], [0.5311861979], [0.5226872289], [0.5141359917],
             [0.5055333412], [0.4968801378], [0.4881772469], [0.4794255386], [0.4706258882], [0.4617791755],
             [0.4528862854], [0.443948107], [0.4349655341], [0.4259394651], [0.4168708024], [0.4077604531],
             [0.398609328], [0.3894183423], [0.3801884151], [0.3709204694], [0.361615432], [0.3522742333],
             [0.3428978075], [0.3334870921], [0.3240430284], [0.3145665606], [0.3050586364], [0.2955202067],
             [0.2859522251], [0.2763556486], [0.2667314367], [0.2570805519], [0.2474039593], [0.2377026264],
             [0.2279775235], [0.2182296231], [0.2084598998], [0.1986693308], [0.188858895], [0.1790295734],
             [0.1691823491], [0.1593182066], [0.1494381325], [0.1395431146], [0.1296341426], [0.1197122073],
             [0.1097783008], [0.09983341665], [0.0898785492], [0.07991469397], [0.06994284734], [0.05996400648],
             [0.04997916927], [0.03998933419], [0.0299955002], [0.01999866669], [0.009999833334], [-0.009999833334],
             [-0.01999866669], [-0.0299955002], [-0.03998933419], [-0.04997916927], [-0.05996400648],
             [-0.06994284734], [-0.07991469397], [-0.0898785492], [-0.09983341665], [-0.1097783008],
             [-0.1197122073], [-0.1296341426], [-0.1395431146], [-0.1494381325], [-0.1593182066], [-0.1691823491],
             [-0.1790295734], [-0.188858895], [-0.1986693308], [-0.2084598998], [-0.2182296231], [-0.2279775235],
             [-0.2377026264], [-0.2474039593], [-0.2570805519], [-0.2667314367], [-0.2763556486], [-0.2859522251],
             [-0.2955202067], [-0.3050586364], [-0.3145665606], [-0.3240430284], [-0.3334870921], [-0.3428978075],
             [-0.3522742333], [-0.361615432], [-0.3709204694], [-0.3801884151], [-0.3894183423], [-0.398609328],
             [-0.4077604531], [-0.4168708024], [-0.4259394651], [-0.4349655341], [-0.443948107], [-0.4528862854],
             [-0.4617791755], [-0.4706258882], [-0.4794255386], [-0.4881772469], [-0.4968801378], [-0.5055333412],
             [-0.5141359917], [-0.5226872289], [-0.5311861979], [-0.5396320487], [-0.5480239368], [-0.5563610229],
             [-0.5646424734], [-0.5728674601], [-0.5810351605], [-0.5891447579], [-0.5971954414], [-0.6051864057],
             [-0.613116852], [-0.620985987], [-0.628793024], [-0.6365371822], [-0.6442176872], [-0.651833771],
             [-0.659384672], [-0.666869635], [-0.6742879116], [-0.68163876], [-0.6889214451], [-0.6961352386],
             [-0.7032794192], [-0.7103532724], [-0.7173560909], [-0.7242871744], [-0.7311458297], [-0.7379313711],
             [-0.74464312], [-0.7512804051], [-0.7578425629], [-0.764328937], [-0.7707388789], [-0.7770717475],
             [-0.7833269096], [-0.7895037397], [-0.79560162], [-0.8016199409], [-0.8075581004], [-0.8134155048],
             [-0.8191915683], [-0.8248857133], [-0.8304973705], [-0.8360259786], [-0.8414709848], [-0.8468318446],
             [-0.8521080219], [-0.8572989892], [-0.8624042272], [-0.8674232256], [-0.8723554823], [-0.8772005043],
             [-0.8819578069], [-0.8866269144], [-0.8912073601], [-0.8956986857], [-0.9001004422], [-0.9044121894],
             [-0.9086334961], [-0.9127639403], [-0.9168031088], [-0.9207505977], [-0.9246060124], [-0.9283689672],
             [-0.932039086], [-0.9356160016], [-0.9390993563], [-0.9424888019], [-0.9457839994], [-0.9489846194],
             [-0.9520903416], [-0.9551008556], [-0.9580158603], [-0.9608350642], [-0.9635581854], [-0.9661849516],
             [-0.9687151001], [-0.9711483779], [-0.9734845417], [-0.9757233578], [-0.9778646024], [-0.9799080614],
             [-0.9818535304], [-0.9837008148], [-0.98544973], [-0.987100101], [-0.9886517629], [-0.9901045603],
             [-0.9914583482], [-0.992712991], [-0.9938683634], [-0.9949243498], [-0.9958808445], [-0.996737752],
             [-0.9974949866], [-0.9981524725], [-0.998710144], [-0.9991679453], [-0.9995258306], [-0.9997837642],
             [-0.9999417202], [-0.9999996829], [-0.9999996829], [-0.9999417202], [-0.9997837642], [-0.9995258306],
             [-0.9991679453], [-0.998710144], [-0.9981524725], [-0.9974949866], [-0.996737752], [-0.9958808445],
             [-0.9949243498], [-0.9938683634], [-0.992712991], [-0.9914583482], [-0.9901045603], [-0.9886517629],
             [-0.987100101], [-0.98544973], [-0.9837008148], [-0.9818535304], [-0.9799080614], [-0.9778646024],
             [-0.9757233578], [-0.9734845417], [-0.9711483779], [-0.9687151001], [-0.9661849516], [-0.9635581854],
             [-0.9608350642], [-0.9580158603], [-0.9551008556], [-0.9520903416], [-0.9489846194], [-0.9457839994],
             [-0.9424888019], [-0.9390993563], [-0.9356160016], [-0.932039086], [-0.9283689672], [-0.9246060124],
             [-0.9207505977], [-0.9168031088], [-0.9127639403], [-0.9086334961], [-0.9044121894], [-0.9001004422],
             [-0.8956986857], [-0.8912073601], [-0.8866269144], [-0.8819578069], [-0.8772005043], [-0.8723554823],
             [-0.8674232256], [-0.8624042272], [-0.8572989892], [-0.8521080219], [-0.8468318446], [-0.8414709848],
             [-0.8360259786], [-0.8304973705], [-0.8248857133], [-0.8191915683], [-0.8134155048], [-0.8075581004],
             [-0.8016199409], [-0.79560162], [-0.7895037397], [-0.7833269096], [-0.7770717475], [-0.7707388789],
             [-0.764328937], [-0.7578425629], [-0.7512804051], [-0.74464312], [-0.7379313711], [-0.7311458297],
             [-0.7242871744], [-0.7173560909], [-0.7103532724], [-0.7032794192], [-0.6961352386], [-0.6889214451],
             [-0.68163876], [-0.6742879116], [-0.666869635], [-0.659384672], [-0.651833771], [-0.6442176872],
             [-0.6365371822], [-0.628793024], [-0.620985987], [-0.613116852], [-0.6051864057], [-0.5971954414],
             [-0.5891447579], [-0.5810351605], [-0.5728674601], [-0.5646424734], [-0.5563610229], [-0.5480239368],
             [-0.5396320487], [-0.5311861979], [-0.5226872289], [-0.5141359917], [-0.5055333412], [-0.4968801378],
             [-0.4881772469], [-0.4794255386], [-0.4706258882], [-0.4617791755], [-0.4528862854], [-0.443948107],
             [-0.4349655341], [-0.4259394651], [-0.4168708024], [-0.4077604531], [-0.398609328], [-0.3894183423],
             [-0.3801884151], [-0.3709204694], [-0.361615432], [-0.3522742333], [-0.3428978075], [-0.3334870921],
             [-0.3240430284], [-0.3145665606], [-0.3050586364], [-0.2955202067], [-0.2859522251], [-0.2763556486],
             [-0.2667314367], [-0.2570805519], [-0.2474039593], [-0.2377026264], [-0.2279775235], [-0.2182296231],
             [-0.2084598998], [-0.1986693308], [-0.188858895], [-0.1790295734], [-0.1691823491], [-0.1593182066],
             [-0.1494381325], [-0.1395431146], [-0.1296341426], [-0.1197122073], [-0.1097783008], [-0.09983341665],
             [-0.0898785492], [-0.07991469397], [-0.06994284734], [-0.05996400648], [-0.04997916927],
             [-0.03998933419], [-0.0299955002], [-0.01999866669], [-0.009999833334]]

    x_y_lists = pre_process(sin_X, sin_Y)  # preprocess data
    sin_X_1, sin_X_2, sin_X_3, sin_X_4 = x_y_lists[0][0], x_y_lists[0][1], x_y_lists[0][2], x_y_lists[0][3]
    sin_Y_1, sin_Y_2, sin_Y_3, sin_Y_4 = x_y_lists[1][0], x_y_lists[1][1], x_y_lists[1][2], x_y_lists[1][3]

    data = NNData(sin_X_1, sin_Y_1, 0.1)
    network.train(data, 10001, order=NNData.Order.RANDOM)  # train first 1/4 of data
    network.test(data)
    print('\n\n\n')

    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    data = NNData(sin_X_2, sin_Y_2, 0.1)
    network.train(data, 10001, order=NNData.Order.RANDOM)  # train second 1/4 of data
    network.test(data)

    print('\n\n\n')
    data = NNData(sin_X_3, sin_Y_3, 0.1)
    network.train(data, 10001, order=NNData.Order.RANDOM)  # train second 1/4 of data
    network.test(data)
    print('\n\n\n')
    data = NNData(sin_X_4, sin_Y_4, 0.1)
    network.train(data, 10001, order=NNData.Order.RANDOM)  # train second 1/4 of data
    network.test(data)


def pre_process(sin_X, sin_Y):
    """ Preprocess the data from run_sin_2pi() by splitting it in 4."""
    def split_list(input_list, num_parts=1):
        return [input_list[i * len(input_list) // num_parts:
                           (i + 1) * len(input_list) // num_parts]
                for i in range(num_parts)]
    x_list = split_list(sin_X, num_parts=4)
    y_list = split_list(sin_Y, num_parts=4)
    return x_list, y_list


def run_XOR():
    """ Train the data loaded from load_XOR without bias input. """
    network = FFBPNetwork(2, 1)
    network.add_hidden_layer(3)
    data = load_XOR()
    network.train(data, 20001, order=NNData.Order.RANDOM)


def load_XOR_bias():
    """ List of features and a list of labels with bias input layer.
    Note: XOR ('except or') is only true if exactly one input is true.
    """
    features = [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]]
    labels = [[0], [1], [1], [0]]
    data = NNData(features, labels, 1)
    return data


def run_XOR_bias():
    """ Train the data loaded from load_XOR without bias input. """
    network = FFBPNetwork(3, 1)
    network.add_hidden_layer(3)
    network.add_hidden_layer(3)
    network.add_hidden_layer(3)
    network.add_hidden_layer(3)
    network.add_hidden_layer(3)
    data = load_XOR_bias()
    network.train(data, 20001, order=NNData.Order.RANDOM)


if __name__ == "__main__":
    # run_iris()
    # run_sin()
    # run_XOR()
    # run_XOR_bias()
    run_sin_2pi()

"""
-- run_iris() Sample Run #1: hidden layer of three neurodes, 10001 epochs, 70% training factor, and is randomly ordered --
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.8302900886227069, 0.8709344822968522, 0.7084902377545764]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.8287604160990445, 0.8658352965062974, 0.7017665161964831]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.832975206041737, 0.8709938257169477, 0.7021919320453351]
Sample [5.4 3.4 1.5 0.4] expected [1. 0. 0.] produced [0.8291632779550058, 0.8684548740221945, 0.6959197052331123]
Sample [5.6 3.  4.5 1.5] expected [0. 1. 0.] produced [0.8328085294104992, 0.8727625239558371, 0.6952219883057896]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.8260021356590389, 0.8647171380369416, 0.685591659593524]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [0.8316414279169239, 0.8727561223164833, 0.6868584704790506]
Sample [5.4 3.9 1.7 0.4] expected [1. 0. 0.] produced [0.8262586134919866, 0.865506651219463, 0.68570133509135]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.8240280655599095, 0.858658822095238, 0.6780639555151028]
Sample [6.2 3.4 5.4 2.3] expected [0. 0. 1.] produced [0.8303019010834227, 0.8679414834965673, 0.6798844561994025]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [0.8278258426246055, 0.8662117846791763, 0.682125058471915]
Sample [7.4 2.8 6.1 1.9] expected [0. 0. 1.] produced [0.8250370507306128, 0.8639228180531217, 0.6840551662662703]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.82221439040499, 0.8616204199606506, 0.6859500122897223]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.814851510092596, 0.8502232544341976, 0.6825378819071716]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.819243456167718, 0.8559817680649441, 0.6825390521266148]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [0.8172580634302942, 0.8576161984114019, 0.6785447519203494]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [0.8148309759736792, 0.8584804574302508, 0.6740571795283008]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.8118315549487815, 0.8560602017743544, 0.6761077389156764]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.8073114902653298, 0.8527583418711315, 0.6693808179179478]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [0.8105463306453811, 0.8562765699424375, 0.6674282427939806]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.8028233493328283, 0.843888045020782, 0.6645067984014628]
Sample [4.4 2.9 1.4 0.2] expected [1. 0. 0.] produced [0.8016265181483048, 0.8380240593544256, 0.6578896937526483]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.8065769756210063, 0.8448623123268276, 0.6580574660823097]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.7977472764368129, 0.832569636300902, 0.6468479630550364]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [0.8056004826288982, 0.8456756469893564, 0.6497629716242143]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.799073175261697, 0.8381303394332971, 0.6411713262377183]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.8020196535107906, 0.840782538422957, 0.6388137353970111]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.7953472580946586, 0.8327775028277121, 0.6303962737791912]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [0.801215642626714, 0.8421008269429309, 0.6303590766275015]
Sample [5.  3.4 1.6 0.4] expected [1. 0. 0.] produced [0.7949686599506283, 0.8351419944499757, 0.6224877850732438]
Sample [5.5 2.5 4.  1.3] expected [0. 1. 0.] produced [0.798227819318763, 0.8385591759256327, 0.6199126737253582]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.7919002842020602, 0.8313090116815498, 0.6122161045767396]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [0.7976889122999081, 0.8406740658739567, 0.6112809177902897]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.7936922554360355, 0.8362493268382174, 0.6138681884885377]
Sample [4.9 2.5 4.5 1.7] expected [0. 0. 1.] produced [0.7905461899066671, 0.8366498601402634, 0.6088122615765875]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.7865739523748211, 0.8323352303636801, 0.6114465684139915]
Sample [7.2 3.  5.8 1.6] expected [0. 0. 1.] produced [0.7846515154990323, 0.8359905970488241, 0.6073868302978978]
Sample [6.5 3.  5.5 1.8] expected [0. 0. 1.] produced [0.7814522287212082, 0.8339419842659859, 0.6107243619121268]
Sample [6.7 2.5 5.8 1.8] expected [0. 0. 1.] produced [0.7778787948856174, 0.8310236960494389, 0.6137902025418001]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [0.7743703960900312, 0.828366930969782, 0.6168761678033917]
Sample [6.7 3.3 5.7 2.1] expected [0. 0. 1.] produced [0.7714894784388453, 0.8275403309775421, 0.6205049922521979]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [0.768015317006381, 0.8252825234942794, 0.6236861364351078]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.7624469284390868, 0.8170000142765412, 0.6248551386099069]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [0.7597775866382357, 0.8203129579005375, 0.6208269425264276]
Sample [6.  2.9 4.5 1.5] expected [0. 1. 0.] produced [0.7561357006792463, 0.8210201174405752, 0.6158700208183163]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.749230005709124, 0.8117666023154986, 0.6076789722586771]
Sample [6.  2.2 4.  1. ] expected [0. 1. 0.] produced [0.7520496811649852, 0.8139701616779077, 0.604690080295118]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [0.7499159680939839, 0.8196792099529062, 0.6011307918506127]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [0.7465227653354586, 0.818607767357286, 0.6048558525958969]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.7423673367702438, 0.8151876375298428, 0.6079457039378636]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.734515969301717, 0.8023802183750581, 0.5987820428363815]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.7389755363458895, 0.810234486672447, 0.5973676254524153]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [0.7355253935598369, 0.8127493863019014, 0.5928677984711325]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [0.7318185925531733, 0.8147324685101419, 0.5881430237421409]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [0.7268817464072296, 0.8119741257197305, 0.5824524170804317]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.7229862023087967, 0.8135281352427675, 0.5776941368430102]
Sample [5.5 2.6 4.4 1.2] expected [0. 1. 0.] produced [0.7185254759089377, 0.8128965351392112, 0.5725401337992556]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.7146633683143867, 0.8151956952047658, 0.5678891682510477]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [0.7108330496014397, 0.8183438842236276, 0.5632661075076154]
Sample [7.7 2.6 6.9 2.3] expected [0. 0. 1.] produced [0.7065021862781014, 0.8162170521786769, 0.5672195664373276]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.7013882204743683, 0.8096106521358397, 0.5704615592792255]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.6956307497813935, 0.8019492276963575, 0.5640413055365187]
Sample [5.4 3.4 1.7 0.2] expected [1. 0. 0.] produced [0.6972770132391825, 0.7976959093392559, 0.5592857020686517]
Sample [6.9 3.1 5.1 2.3] expected [0. 0. 1.] produced [0.7012699720896064, 0.8083146944619777, 0.5564567705105551]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.6956131732566406, 0.7975360264077702, 0.5594787912478226]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [0.6920322487042903, 0.804352103226554, 0.5553822405034167]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.6874889893265722, 0.8017882616144275, 0.559352004401133]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.6825667981470623, 0.7958262470364841, 0.5628512213870117]
Sample [6.4 3.1 5.5 1.8] expected [0. 0. 1.] produced [0.678268898245, 0.8000669765870569, 0.5583898056802856]
Sample [5.1 3.5 1.4 0.3] expected [1. 0. 0.] produced [0.672521434690403, 0.7847355423532191, 0.5602397424340771]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.6756361749572575, 0.792413867109506, 0.5573699535572387]
Sample [4.8 3.4 1.6 0.2] expected [1. 0. 0.] produced [0.6700400769009588, 0.7812980484502513, 0.5508553717522664]
Sample [6.1 2.6 5.6 1.4] expected [0. 0. 1.] produced [0.6731142368994144, 0.78926367586857, 0.5477446383048443]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.6679401300692412, 0.7784061404880633, 0.5506740655007393]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [0.6708392765811844, 0.7857385205065532, 0.547206102649616]
Sample [6.4 3.2 5.3 2.3] expected [0. 0. 1.] produced [0.6660588158294651, 0.7830390658974287, 0.551313954082272]
Sample [5.9 3.  4.2 1.5] expected [0. 1. 0.] produced [0.6611354202610071, 0.7761668885292905, 0.5549693680350621]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.6558953351232061, 0.7607103730634907, 0.5479041803015436]
Sample [6.9 3.2 5.7 2.3] expected [0. 0. 1.] produced [0.6587732822228399, 0.7773840216470098, 0.5458307197659534]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [0.6538747822956158, 0.7734567730707951, 0.5498790926862694]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [0.6490381964954173, 0.7671014832952805, 0.5536277340632632]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [0.6442719648525029, 0.7662932900157676, 0.5486230864917464]
Sample [5.1 3.8 1.9 0.4] expected [1. 0. 0.] produced [0.6396626022025135, 0.7570067319277921, 0.5516949329187586]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.6424577282312274, 0.7446675693105999, 0.5459748712014971]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.6448707040505727, 0.7453507815089245, 0.542233280559438]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.6471851083344523, 0.7412210501346689, 0.5374644432766192]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [0.6496303591171962, 0.7512841689381314, 0.534342726453984]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.645033180976605, 0.7408099312268787, 0.5381151845731611]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.6406780667227495, 0.7303610062782327, 0.5324072513762055]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.6430044743266875, 0.7347552772638098, 0.5289993073673098]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [0.6376508417540896, 0.749100169523017, 0.524810568958632]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.633603333094148, 0.7311315715629522, 0.5283267587825894]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.6359490622676746, 0.7340780538886467, 0.5244728034168507]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [0.6307233993001715, 0.7404556409463532, 0.5199139719955467]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.6266909930975179, 0.7273718939010527, 0.5240212139522181]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [0.6217649721497978, 0.7315658674428035, 0.5196944201648271]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [0.6157658336688377, 0.741473115267369, 0.51504503803478]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [0.6111864016178007, 0.73454318825322, 0.5194991021167136]
Sample [6.  3.4 4.5 1.6] expected [0. 1. 0.] produced [0.606252442564531, 0.7302996652473551, 0.5237872048745325]
Sample [5.  3.  1.6 0.2] expected [1. 0. 0.] produced [0.6043358602777698, 0.7145509533711514, 0.5185642869115205]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.6066967747961947, 0.7143721201954721, 0.514572304805534]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [0.6073908912003269, 0.7240739618566822, 0.5103702327535581]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.6053756723430919, 0.7025162637716434, 0.5141728667529651]
Sample [5.1 3.3 1.7 0.5] expected [1. 0. 0.] produced [0.6076403105405144, 0.703176089159536, 0.5103601780131749]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [0.6083381230722225, 0.7127902362810521, 0.5060181267102526]
Epoch 0 RMSE =  0.6132209806031103
Epoch 100 RMSE =  0.3429896454055927
Epoch 200 RMSE =  0.32417096241542065
Epoch 300 RMSE =  0.3142067944565672
Epoch 400 RMSE =  0.31056125695266157
Epoch 500 RMSE =  0.30541247890481776
Epoch 600 RMSE =  0.30397372104571596
Epoch 700 RMSE =  0.3029599559139647
Epoch 800 RMSE =  0.3048361590180093
Epoch 900 RMSE =  0.30221139582902695
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.9225488039992445, 0.2842607653266432, 7.034959876119134e-05]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [0.0017738235532700991, 0.37811982142749306, 0.3696622414827311]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [0.01667071097431171, 0.35658508564632035, 0.0543771148452142]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [0.0001617265882592736, 0.41555191681140113, 0.8710566432415899]
Sample [7.4 2.8 6.1 1.9] expected [0. 0. 1.] produced [0.00021487527771771417, 0.4085318304928669, 0.8349670635440657]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.010452188954103823, 0.3600303330450359, 0.0858369908808217]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.9214799068915306, 0.28821035518336524, 7.129141802159731e-05]
Sample [4.4 2.9 1.4 0.2] expected [1. 0. 0.] produced [0.9185877517799946, 0.2874380045797179, 7.442926630636956e-05]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.9194186225701984, 0.2860939730516863, 7.344850409337756e-05]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [0.0009171085026830974, 0.3882540790014456, 0.5343804523730064]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [0.00015480270965547566, 0.40620284548857705, 0.8784540433491054]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.9219672874105264, 0.280672126979299, 7.16736984742259e-05]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [0.00021490533710118948, 0.39751629700135277, 0.8380771757919097]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.003268029420687505, 0.3632759277092264, 0.24114961633692614]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [0.0014305900305756245, 0.3769921837063133, 0.4247552943575066]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [0.0008288339562854615, 0.3878774648898459, 0.5604939222150741]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.9222922542164593, 0.286486577195545, 6.959904008037556e-05]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.0004406620072327836, 0.3986703988287943, 0.705270553746612]
Sample [5.5 2.6 4.4 1.2] expected [0. 1. 0.] produced [0.026734492043900694, 0.34786975668438225, 0.03382021032301817]
Sample [5.4 3.9 1.7 0.4] expected [1. 0. 0.] produced [0.9233798732084311, 0.28617449892511165, 6.877273791763605e-05]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.15271391732323245, 0.3294004892809056, 0.005070568683391143]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.08998328928696853, 0.33956850225818896, 0.00931680583443968]
Sample [4.8 3.4 1.6 0.2] expected [1. 0. 0.] produced [0.9220894462293315, 0.2912456416243196, 7.004373232967565e-05]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.20002979357358702, 0.33143255299448304, 0.0036221896189691963]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [0.0004269333969992822, 0.4098082408470855, 0.7124569036782582]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [0.0004344857936818901, 0.4062461620793227, 0.7104663056941192]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.9191563595649068, 0.28937401919627703, 7.359671140618323e-05]
Sample [6.9 3.2 5.7 2.3] expected [0. 0. 1.] produced [0.0001598460649253608, 0.4135152436254264, 0.8734358784345241]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.03634646800379017, 0.3467475724473553, 0.02494153011904468]
Sample [5.1 3.8 1.9 0.4] expected [1. 0. 0.] produced [0.920104901746444, 0.28898915888528737, 7.268947357369084e-05]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [0.000258629959812664, 0.4071808864195242, 0.8085897324783663]
Sample [6.  2.2 4.  1. ] expected [0. 1. 0.] produced [0.014488963995014332, 0.3640877598999527, 0.06204826986235509]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.02635503272935385, 0.3609431173846624, 0.03417144170957491]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.9204279116105125, 0.29683790874507643, 7.175766984421189e-05]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [0.00015453260606851425, 0.42608950705609366, 0.875487105468043]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [0.0003239090891754794, 0.4132368274345068, 0.7674689143297955]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [0.0001923598426680134, 0.4161085544749288, 0.8499259576500628]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.9198719888160437, 0.28930155597968144, 7.288333681445924e-05]
Sample [6.4 3.2 5.3 2.3] expected [0. 0. 1.] produced [0.00016621634609307264, 0.41283596101706643, 0.8684294632383055]
Sample [5.4 3.4 1.7 0.2] expected [1. 0. 0.] produced [0.9213376330586531, 0.2857951434774807, 7.131407454306688e-05]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.021925041591255825, 0.35115273419834964, 0.0416722619360693]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.01241355976311519, 0.36161704366079683, 0.0728859214959649]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [0.00014698543259393653, 0.418504077130855, 0.8823658694679923]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [0.00017503572678267157, 0.4127268033533376, 0.8626941921353528]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.9017051708553703, 0.28909472502612904, 9.276900544341919e-05]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.9235810555983899, 0.2849895501983395, 6.926510845383326e-05]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.9213131592450647, 0.28413992433008334, 7.165308013326767e-05]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.002072124889879522, 0.3760797542349265, 0.33349962220388396]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [0.000693007521245895, 0.3933889509205842, 0.6038756392226609]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [0.00047989612329411396, 0.4025252768471117, 0.6853468448275356]
Sample [6.5 3.  5.5 1.8] expected [0. 0. 1.] produced [0.00043937775188381613, 0.400344252906411, 0.7064730886219456]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [0.0002870248071083592, 0.40216607861300957, 0.7897367155437812]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.00020374098593519438, 0.40287021804639417, 0.8428505305143789]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9215211097599556, 0.281038070998015, 7.139238089237764e-05]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [0.0001604334029065497, 0.40089672279840805, 0.8730409016655604]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9215487590156076, 0.2779569283271258, 7.15065148588354e-05]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.9198488294368835, 0.27711608583524044, 7.352791295348282e-05]
Sample [5.9 3.  4.2 1.5] expected [0. 1. 0.] produced [0.007834062353096352, 0.3510339638362411, 0.11288049281980851]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [0.00021598165990779623, 0.3960210430927997, 0.8355064609425992]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.015151880025794576, 0.3449869470719025, 0.060441345593075874]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.06716347916973421, 0.33211903333625203, 0.013059871774228104]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.011437578685301028, 0.3558944991939777, 0.07924273875223957]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [0.00018018399901035162, 0.40828797682369783, 0.8596538487466627]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.0038007898264752946, 0.3694887124294325, 0.21180921708934097]
Sample [6.2 3.4 5.4 2.3] expected [0. 0. 1.] produced [0.00020622224737812394, 0.4079386465598377, 0.8418151209324967]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.03308869278276741, 0.346076896845528, 0.0275371257505998]
Sample [5.1 3.3 1.7 0.5] expected [1. 0. 0.] produced [0.9181099447563796, 0.2882161872623198, 7.493524167646915e-05]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.9213090872709385, 0.2865994805378684, 7.187183973688891e-05]
Sample [5.4 3.4 1.5 0.4] expected [1. 0. 0.] produced [0.921714979662319, 0.28532639452107356, 7.134568630874517e-05]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.03679019832710359, 0.3446808847074834, 0.024717035033041842]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.922509247648993, 0.2870684972562707, 7.065187597152214e-05]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.03918478175920608, 0.3463923711715393, 0.023153786126257223]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.013229462070674852, 0.36287460157458606, 0.0690266815600316]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.9226723221725687, 0.291908491321264, 7.041762049349044e-05]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [0.0012372685731497494, 0.39361160591346134, 0.45986061655991145]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.9241618963549284, 0.2934260063301973, 6.805677295425625e-05]
Sample [6.9 3.1 5.1 2.3] expected [0. 0. 1.] produced [0.00033029040455694357, 0.4128630704801947, 0.7646787106898318]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [0.004366342754605753, 0.378406816050797, 0.18768907236683136]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.9229865961986288, 0.29333072860648884, 6.955770533800136e-05]
Sample [7.7 2.6 6.9 2.3] expected [0. 0. 1.] produced [0.00014235246380249296, 0.42299954445327853, 0.8853303325833933]
Sample [6.7 3.3 5.7 2.1] expected [0. 0. 1.] produced [0.0003522524584624216, 0.40820138717517745, 0.7534328606550771]
Sample [6.  2.9 4.5 1.5] expected [0. 1. 0.] produced [0.006206004174302744, 0.37081776385602927, 0.1391006908601253]
Sample [7.2 3.  5.8 1.6] expected [0. 0. 1.] produced [0.001252603875428073, 0.3941011664969262, 0.45518205958005636]
Sample [5.6 3.  4.5 1.5] expected [0. 1. 0.] produced [0.0008270450015637109, 0.39609478207477433, 0.5656832113175496]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.2191756500343531, 0.33262235713463323, 0.003254603061677044]
Sample [5.  3.4 1.6 0.4] expected [1. 0. 0.] produced [0.9199670542503704, 0.2954432355321381, 7.255175914184241e-05]
Sample [6.1 2.6 5.6 1.4] expected [0. 0. 1.] produced [0.00027621957173513793, 0.41674909607580196, 0.7967369181143084]
Sample [5.  3.  1.6 0.2] expected [1. 0. 0.] produced [0.9192725821293729, 0.2921806537134964, 7.362525684902976e-05]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.9221041764865832, 0.29052534445079853, 7.073488002099784e-05]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [0.0001500450338173956, 0.4177613890021415, 0.8805319815522457]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [0.0004101503114583727, 0.40198558992432437, 0.7248150072212723]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.37440941087953217, 0.32207904105978463, 0.001479963128005754]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.9224590578884941, 0.293069805731372, 6.880149501101511e-05]
Sample [6.7 2.5 5.8 1.8] expected [0. 0. 1.] produced [0.00015838783135349975, 0.4209454591545212, 0.8714107672812802]
Sample [5.1 3.5 1.4 0.3] expected [1. 0. 0.] produced [0.9213258485256033, 0.2898752867587082, 7.013011587061792e-05]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [0.0001531327740055086, 0.4162136653707629, 0.8755303691259645]
Sample [6.4 3.1 5.5 1.8] expected [0. 0. 1.] produced [0.0002420704302701274, 0.4070680996227719, 0.8151164603531947]
Sample [6.  3.4 4.5 1.6] expected [0. 1. 0.] produced [0.003224803038282018, 0.37330005206540334, 0.2371728878217897]
Sample [4.9 2.5 4.5 1.7] expected [0. 0. 1.] produced [0.00022710599739549138, 0.40915179741229657, 0.8245553875731533]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [0.00018929947286239413, 0.40785126870654875, 0.8503712016032158]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9207603931785832, 0.28363737462022554, 7.113917311817355e-05]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [0.0002825776523326539, 0.39828103809143206, 0.7909043455811834]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [0.0001486719768339389, 0.40248388422431214, 0.880112852335332]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.9222680359259089, 0.27836862592178835, 6.991173972457119e-05]
Sample [5.5 2.5 4.  1.3] expected [0. 1. 0.] produced [0.0022932412875613303, 0.36660103666975424, 0.30796698005418627]
Epoch 1000 RMSE =  0.30231860149553336
Epoch 1100 RMSE =  0.29964401678372293
Epoch 1200 RMSE =  0.29993571611924735
Epoch 1300 RMSE =  0.3012083145756997
Epoch 1400 RMSE =  0.29925376744260884
Epoch 1500 RMSE =  0.3013739978769426
Epoch 1600 RMSE =  0.29712669123238844
Epoch 1700 RMSE =  0.297770522351104
Epoch 1800 RMSE =  0.29855812436767315
Epoch 1900 RMSE =  0.29713694848652594
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [3.356338406833291e-05, 0.40495048059149313, 0.8858939572257017]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [7.133595965479018e-05, 0.3922861672369692, 0.769354448633971]
Sample [6.4 3.2 5.3 2.3] expected [0. 0. 1.] produced [3.8671766963510384e-05, 0.396483934489082, 0.8696718595568091]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.9493060248490715, 0.2513727647909234, 2.7649011077596703e-06]
Sample [5.1 3.5 1.4 0.3] expected [1. 0. 0.] produced [0.9481152779347535, 0.25073356478183506, 2.847582817528957e-06]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.9471260593211057, 0.2500560018771233, 2.9168885467935777e-06]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.9460026302392224, 0.24942437816688742, 3.000613549020641e-06]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.005359220080578538, 0.33165544931044205, 0.025532954450565186]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.0026981397677128586, 0.34335738834824364, 0.053729052934672776]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [3.874412811754586e-05, 0.3978924539714666, 0.8697611607457473]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.0082777157218302, 0.33183210335955576, 0.01579348122713216]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.9449180237626565, 0.2553117273600363, 3.069305522549261e-06]
Sample [5.1 3.8 1.9 0.4] expected [1. 0. 0.] produced [0.9468615134360072, 0.253994683331882, 2.932434827592944e-06]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.9341685151635815, 0.25538616864250174, 3.824154447064891e-06]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.9469306449733622, 0.25219832512332957, 2.9406120391361788e-06]
Sample [5.  3.  1.6 0.2] expected [1. 0. 0.] produced [0.9456058763379745, 0.2515401798681978, 3.0243674934667723e-06]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.00586833623743723, 0.3342520364683559, 0.023169746130443772]
Sample [6.5 3.  5.5 1.8] expected [0. 0. 1.] produced [4.749904406293195e-05, 0.3954002203338262, 0.8420851104289341]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.947647807235723, 0.2512572963256548, 2.89284725746091e-06]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [7.560801815448032e-05, 0.38537079489467296, 0.7604571070830914]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [0.0001781578980115676, 0.3798757436340062, 0.5431632224213604]
Sample [6.7 2.5 5.8 1.8] expected [0. 0. 1.] produced [3.153848959087478e-05, 0.397977161537938, 0.8940753951530226]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [6.828761114014064e-05, 0.3852612729139065, 0.7802333151767868]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [7.0553672018055e-05, 0.3898176813363253, 0.7705122766044827]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.948808010990852, 0.2503990572202981, 2.7867509244324175e-06]
Sample [4.4 2.9 1.4 0.2] expected [1. 0. 0.] produced [0.9440348564550268, 0.25047932035131176, 3.114553672010722e-06]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.0016198483625811172, 0.3474354122240771, 0.09094415915698528]
Sample [5.4 3.9 1.7 0.4] expected [1. 0. 0.] produced [0.948233056627978, 0.2515291573268706, 2.821990139772909e-06]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [5.6989652589063164e-05, 0.3905637330322136, 0.8110148921901705]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.009164796686242637, 0.3345761459413303, 0.013753353050185218]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.004239339576687526, 0.3475240563762399, 0.03223516336835915]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9474435994465606, 0.25921175314941364, 2.845456081732081e-06]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.0008359076361306581, 0.3700739751440272, 0.17137518101157415]
Sample [5.4 3.4 1.5 0.4] expected [1. 0. 0.] produced [0.9477954917595828, 0.26099443774588194, 2.8116563641157397e-06]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [4.465143485650908e-05, 0.4101637460873195, 0.8467735004404776]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.03426728681722756, 0.3255529891804711, 0.0030753542025639615]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [0.0005324804185396823, 0.379784311809111, 0.25545033194396943]
Sample [6.  2.9 4.5 1.5] expected [0. 1. 0.] produced [0.001237410303676167, 0.37371706160048346, 0.11680986392270812]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.9485211983758436, 0.2665133524472685, 2.7582056588839247e-06]
Sample [7.4 2.8 6.1 1.9] expected [0. 0. 1.] produced [4.9564967868988225e-05, 0.41807457316338986, 0.8302096259848526]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [3.620320495133864e-05, 0.4185381554337054, 0.874692607525156]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [9.331448383446173e-05, 0.40278720983496374, 0.7073588364266682]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [7.660600127860358e-05, 0.4019885986005429, 0.7526166962292659]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.948267077564985, 0.2578713269281949, 2.814732010246899e-06]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [0.0002465542692469312, 0.3829893157686877, 0.45196196153448887]
Sample [6.2 3.4 5.4 2.3] expected [0. 0. 1.] produced [5.3671952445778105e-05, 0.4068600772109949, 0.8180553450779333]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [5.092887846451735e-05, 0.40408849673011565, 0.8272174441597824]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [0.00018559540101700271, 0.3846694866912875, 0.529504090078257]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [3.42461409058224e-05, 0.4025306276251162, 0.8840853583712729]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.003236022785861001, 0.34462568895956947, 0.044074426265596614]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [3.750097274813268e-05, 0.40263783076843795, 0.8733922769732377]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.01123073472121265, 0.33141343071964635, 0.011198354182533864]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9469345862335717, 0.25647326733341747, 2.927636033747217e-06]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.9455197728317779, 0.2558038173937343, 3.016572495531811e-06]
Sample [7.2 3.  5.8 1.6] expected [0. 0. 1.] produced [6.218342921964341e-05, 0.3951208035237133, 0.7967438988435828]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.9453497747659032, 0.2531324949832702, 3.043005606431699e-06]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.00445914559552884, 0.33995955741602213, 0.03133203581492936]
Sample [4.9 2.5 4.5 1.7] expected [0. 0. 1.] produced [3.881952102478533e-05, 0.40112894947326305, 0.8699275559936279]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.9482677711515087, 0.2526338457892736, 2.8531970273226797e-06]
Sample [4.8 3.4 1.6 0.2] expected [1. 0. 0.] produced [0.9466230000877393, 0.252061483477638, 2.962754249501467e-06]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.0006378886738173836, 0.36171266656143747, 0.2243747489310972]
Sample [6.  3.4 4.5 1.6] expected [0. 1. 0.] produced [0.0007380833906130382, 0.364355936964361, 0.19619876461292016]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.002525631954919375, 0.3539392417991751, 0.05746712494058272]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.9475274310865583, 0.2594274503326504, 2.88847237778218e-06]
Sample [6.  2.2 4.  1. ] expected [0. 1. 0.] produced [0.003762383992977103, 0.35213027107979933, 0.03746267608598289]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.9469028864175247, 0.2614475635632913, 2.929451118212479e-06]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [3.2754337806901834e-05, 0.4142748367551608, 0.8893413705701895]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.05766007433840818, 0.31947815504191535, 0.0017072403559263859]
Sample [6.9 3.1 5.1 2.3] expected [0. 0. 1.] produced [3.968179479803578e-05, 0.41253859357073336, 0.8664352749926691]
Sample [6.7 3.3 5.7 2.1] expected [0. 0. 1.] produced [3.818983631756769e-05, 0.40946121508767563, 0.8715915476448423]
Sample [5.1 3.3 1.7 0.5] expected [1. 0. 0.] produced [0.9431447151583426, 0.2582498575509901, 3.1820233414921516e-06]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [0.00039826987728053416, 0.37576579387734976, 0.32837634623517475]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [3.6121015438038084e-05, 0.41035940734758347, 0.8776143969661337]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [3.655552949277257e-05, 0.40666105895639737, 0.876375458312303]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.2358104215576629, 0.29747218582080304, 0.00027730433930287933]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.947905837423223, 0.25818087566909104, 2.8436420829889395e-06]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.0066331008323771, 0.3428781974549038, 0.020010992550484996]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [3.662678848154665e-05, 0.41056747404834837, 0.8757877265436678]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.9494048143379117, 0.25785033250334854, 2.7477382694203466e-06]
Sample [6.4 3.1 5.5 1.8] expected [0. 0. 1.] produced [4.9283856333776596e-05, 0.4021297273145924, 0.8350836165371185]
Sample [5.6 3.  4.5 1.5] expected [0. 1. 0.] produced [0.0002150680933205878, 0.3807219495030444, 0.49272727064586086]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.9488466218689292, 0.2579775401161241, 2.7593325176043704e-06]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9481033811545558, 0.2572042368357916, 2.812102627144015e-06]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [5.482135028399224e-05, 0.3999825781928561, 0.815953849209128]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [4.779423133810294e-05, 0.39834422739421904, 0.8385094335938676]
Sample [5.4 3.4 1.7 0.2] expected [1. 0. 0.] produced [0.9484915658041219, 0.2525359391648136, 2.796050458128444e-06]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [0.005445938417552171, 0.337614510443048, 0.024794913191823698]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.9490978647042031, 0.2543186906218445, 2.7579161753086726e-06]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [3.48064702017386e-05, 0.401214226745968, 0.8815108679981054]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.002141511862328651, 0.34847510403775234, 0.06795891443296678]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.028406428142121475, 0.32249689535066584, 0.003867305136496533]
Sample [7.7 2.6 6.9 2.3] expected [0. 0. 1.] produced [2.842472756152552e-05, 0.4094582277900017, 0.9033798910744365]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [5.6339224259885493e-05, 0.39741852508688946, 0.812841289552325]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [3.236286533835279e-05, 0.4009154697688779, 0.890428713886006]
Sample [6.9 3.2 5.7 2.3] expected [0. 0. 1.] produced [3.673944515524932e-05, 0.3959400674713128, 0.8759126883390408]
Sample [5.9 3.  4.2 1.5] expected [0. 1. 0.] produced [0.001810390521180446, 0.3468053730288972, 0.08150388857187783]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.033553539392179525, 0.31759428864700767, 0.003212882812371765]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [9.666169608849879e-05, 0.3898632919514591, 0.7045468048875303]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [0.001818403788223244, 0.3591962715620576, 0.07964597541548211]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.9490085235483575, 0.26112918311191885, 2.7414337629462404e-06]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [6.730190114508193e-05, 0.40267961137633274, 0.7780282697944628]
Sample [5.5 2.5 4.  1.3] expected [0. 1. 0.] produced [0.0025744048288011572, 0.3551655637561853, 0.055539723306285836]
Sample [6.1 2.6 5.6 1.4] expected [0. 0. 1.] produced [8.514135610356106e-05, 0.40104635041434294, 0.7301523535904612]
Sample [5.5 2.6 4.4 1.2] expected [0. 1. 0.] produced [0.0010498835535467962, 0.36712755651624135, 0.13963393266154356]
Sample [5.  3.4 1.6 0.4] expected [1. 0. 0.] produced [0.9471495781010706, 0.2624994067968323, 2.880903813092754e-06]
Epoch 2000 RMSE =  0.2970512381260678
Epoch 2100 RMSE =  0.2970555092013455
Epoch 2200 RMSE =  0.296474592604599
Epoch 2300 RMSE =  0.2983920751408094
Epoch 2400 RMSE =  0.29396008633218995
Epoch 2500 RMSE =  0.29294065905079764
Epoch 2600 RMSE =  0.2943294295959975
Epoch 2700 RMSE =  0.29797120730817783
Epoch 2800 RMSE =  0.2961596498799204
Epoch 2900 RMSE =  0.29511794552651066
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [2.2136048084213582e-05, 0.42071856071876246, 0.8415270411586084]
Sample [4.8 3.4 1.6 0.2] expected [1. 0. 0.] produced [0.961589898672246, 0.24611157658355642, 2.4309736125940766e-07]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.961566858764962, 0.24528032695950916, 2.435225983741226e-07]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [1.9803105436329833e-05, 0.4163684764121671, 0.859138816264203]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.054430656221178354, 0.3102714860698754, 0.000385046721095369]
Sample [6.1 2.6 5.6 1.4] expected [0. 0. 1.] produced [4.164067135198864e-05, 0.4069506873384136, 0.7127546533136796]
Sample [7.7 2.6 6.9 2.3] expected [0. 0. 1.] produced [1.4448424617665949e-05, 0.41788393127744833, 0.9003580197598138]
Sample [6.4 3.2 5.3 2.3] expected [0. 0. 1.] produced [2.1154866280048767e-05, 0.40906331513166566, 0.8506890165844916]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [2.3235746945382726e-05, 0.40430643077499806, 0.8360311144931231]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [0.0005845374003729279, 0.3592221755455313, 0.09269060707003615]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.9608429829955364, 0.24092221280960124, 2.5249945949258107e-07]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [1.6812310315743734e-05, 0.40894329662809575, 0.8832900459670371]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [0.00018949764139100363, 0.3737912587837121, 0.2863476652155687]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [3.061971066538173e-05, 0.40236756349064984, 0.7841928269363693]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.9619949131953239, 0.23889600579255862, 2.425470182757752e-07]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [1.8806461225523742e-05, 0.40440726470406024, 0.868421770114092]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [2.4585430348136466e-05, 0.39743124607101055, 0.8269550249069662]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.008151279838268029, 0.32171397827337417, 0.004136543436671964]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [5.6260908867466564e-05, 0.3878059033163598, 0.6372231555499507]
Sample [5.5 2.5 4.  1.3] expected [0. 1. 0.] produced [0.00031490846317388323, 0.3628319159768617, 0.18017321411536447]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.9596687779366734, 0.23895189834309935, 2.6501994776320685e-07]
Sample [6.9 3.1 5.1 2.3] expected [0. 0. 1.] produced [2.4107725962698493e-05, 0.39950673125435676, 0.832040369887366]
Sample [5.6 3.  4.5 1.5] expected [0. 1. 0.] produced [0.00019329168700710109, 0.3693928850881301, 0.2843723017572065]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.0726187324331718, 0.30059489639801756, 0.00027023728158851265]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.03493028695229765, 0.3130086532909036, 0.0006889912455866745]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.961940400620077, 0.24384007381831713, 2.44250676172101e-07]
Sample [7.2 3.  5.8 1.6] expected [0. 0. 1.] produced [6.17143055265714e-05, 0.3960324694733212, 0.6116286169972862]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.9588692760975654, 0.24209074484886575, 2.719674844645757e-07]
Sample [6.4 3.1 5.5 1.8] expected [0. 0. 1.] produced [2.3167824821677844e-05, 0.40471022952384944, 0.839764949392352]
Sample [6.  2.2 4.  1. ] expected [0. 1. 0.] produced [0.0007958993225841176, 0.3558375115594277, 0.0669858595526734]
Sample [5.  3.  1.6 0.2] expected [1. 0. 0.] produced [0.9583171678690109, 0.24234686338938521, 2.774235232577167e-07]
Sample [6.5 3.  5.5 1.8] expected [0. 0. 1.] produced [2.1475622754643237e-05, 0.40599199732801067, 0.8521215429812017]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.9618892696399678, 0.23879279880170226, 2.478458301410933e-07]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [1.841296221441394e-05, 0.40350873565068335, 0.8744639117847115]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [1.5757785204034176e-05, 0.4020881939073926, 0.8939823171753225]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.001203704852290393, 0.3440031267481709, 0.04184139989010547]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [1.661087361230056e-05, 0.4027108091870905, 0.8878878266050813]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [2.326722603453401e-05, 0.3948916116171964, 0.8404984738694783]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.9610965151901687, 0.2382602630994914, 2.5260187610615705e-07]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.9617023644579054, 0.23731678718526678, 2.4769918445311226e-07]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.000217853193452508, 0.3690631177370237, 0.2556011181278167]
Sample [5.5 2.6 4.4 1.2] expected [0. 1. 0.] produced [0.00038588858439123915, 0.3662917619188229, 0.1456336481830544]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [1.7943009021179726e-05, 0.41112968041820636, 0.8756453523305946]
Sample [6.7 2.5 5.8 1.8] expected [0. 0. 1.] produced [1.9160185727742613e-05, 0.40669340705444973, 0.866942642375657]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [6.340150232440282e-05, 0.38756465627821884, 0.6045231223208991]
Sample [6.  3.4 4.5 1.6] expected [0. 1. 0.] produced [0.02302657830809136, 0.31790672865856306, 0.001140760167182146]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [0.0001258131879765186, 0.3876561653513752, 0.3947436934570236]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.959970615728771, 0.24243710767677326, 2.6128394644080557e-07]
Sample [5.1 3.3 1.7 0.5] expected [1. 0. 0.] produced [0.956481990634539, 0.24248965336004058, 2.9002524295275717e-07]
Sample [5.  3.4 1.6 0.4] expected [1. 0. 0.] produced [0.9590279879973391, 0.24104130303476703, 2.6881940118485583e-07]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.0005923569964608283, 0.36161049254424804, 0.09214766639242905]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.002111160645240513, 0.34973804882748655, 0.02119389059907984]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [3.2683607538592966e-05, 0.408644984323307, 0.7733876426545923]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.9609815753570914, 0.24783093300898942, 2.4970675660832807e-07]
Sample [5.1 3.5 1.4 0.3] expected [1. 0. 0.] produced [0.9615853262665951, 0.24678583754523667, 2.443798437513807e-07]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.06789201239041036, 0.311765627721712, 0.00029094252557335955]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [2.7669421284804048e-05, 0.4179463175671502, 0.8037213029106718]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [2.70845807987962e-05, 0.41464975395271947, 0.808478887605637]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [4.6080481708638004e-05, 0.4039421511476961, 0.6899142127722392]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [1.6076028246361447e-05, 0.4147338528558728, 0.8896276058304708]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [3.3921960250500225e-05, 0.4011676518285224, 0.7653864229948537]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.96034404324316, 0.24427449882333732, 2.5481594161217744e-07]
Sample [5.4 3.4 1.5 0.4] expected [1. 0. 0.] produced [0.9608192435565194, 0.24329380451849505, 2.5038390169648937e-07]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [4.3080384306799844e-05, 0.4007884014515891, 0.70539072031096]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.96092744105329, 0.24068616498807602, 2.5138299294015085e-07]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [0.00032128892359961755, 0.37018216319467856, 0.1742217616497935]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.0007725419297825322, 0.36329099499439893, 0.06765082640680244]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.9617606158691218, 0.24505801312775496, 2.438217419337105e-07]
Sample [6.  2.9 4.5 1.5] expected [0. 1. 0.] produced [0.0001717726783732221, 0.3863232917104103, 0.3100653981851536]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.9627941687324253, 0.24660850096128722, 2.3365639628669923e-07]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.025287988545479614, 0.32490950035383914, 0.0010125562450295105]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.00103910808428716, 0.3699787649530483, 0.04775148745320556]
Sample [7.4 2.8 6.1 1.9] expected [0. 0. 1.] produced [2.1117117978509704e-05, 0.42762602369263053, 0.8498736449413402]
Sample [6.2 3.4 5.4 2.3] expected [0. 0. 1.] produced [2.1098292750716308e-05, 0.42387564742622685, 0.8503711971696633]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [2.142089763446388e-05, 0.41995485108139685, 0.8483780544732851]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.1231556894437621, 0.3035280922056872, 0.00013153309635253657]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [0.0002007017646722897, 0.39016778087043036, 0.2706040622470367]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.9623452728677303, 0.2510503811347478, 2.3710384276248827e-07]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.9627925251156839, 0.2500295537677936, 2.3347890801375513e-07]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9612882547844519, 0.24960977985274518, 2.460582823567222e-07]
Sample [5.9 3.  4.2 1.5] expected [0. 1. 0.] produced [0.0021980101432949152, 0.3596154074601854, 0.019767939683699543]
Sample [4.9 2.5 4.5 1.7] expected [0. 0. 1.] produced [2.784657085847932e-05, 0.42307339685680984, 0.8018414241713092]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.005656196826665347, 0.34832501606825794, 0.00636664578033337]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [1.6504075851154564e-05, 0.43131104469931686, 0.8846061320393196]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [2.7186348936071704e-05, 0.42056203385709306, 0.8073869876344034]
Sample [5.4 3.9 1.7 0.4] expected [1. 0. 0.] produced [0.9617722191877945, 0.24827467706425949, 2.4341683989805907e-07]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.014405734539674302, 0.3332504777335828, 0.0020460939741667847]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.0023929463274459098, 0.3601695471087639, 0.018031979312890075]
Sample [5.4 3.4 1.7 0.2] expected [1. 0. 0.] produced [0.9611409379039749, 0.25314769990989394, 2.4865224622787684e-07]
Sample [5.1 3.8 1.9 0.4] expected [1. 0. 0.] produced [0.9603355356990058, 0.25247205089756974, 2.551229240823065e-07]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.005337176464549903, 0.35157662059114647, 0.006870834568018816]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.9478328535766182, 0.2575690163795094, 3.6469159147146004e-07]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [1.826649169762292e-05, 0.43163919970226083, 0.872205716135655]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [1.8497138683252832e-05, 0.42762440682165065, 0.8707410932459655]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9604136170759895, 0.2494672744567937, 2.559286422542539e-07]
Sample [4.4 2.9 1.4 0.2] expected [1. 0. 0.] produced [0.9575014119498334, 0.24940269970200496, 2.8082979598936186e-07]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.0007550855538696178, 0.3715353970220579, 0.06970967334212469]
Sample [6.9 3.2 5.7 2.3] expected [0. 0. 1.] produced [1.6740948124263503e-05, 0.42779354323310526, 0.8839585581369321]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.9610210691563336, 0.24832673395367064, 2.5084052407277123e-07]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.0006061633176482165, 0.3743582479819736, 0.08921851082239608]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [0.00010917045662171358, 0.40190155223353835, 0.43939831329127566]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.9601209804413586, 0.2531938602899132, 2.55864813043712e-07]
Sample [6.7 3.3 5.7 2.1] expected [0. 0. 1.] produced [2.9224424869984924e-05, 0.4237761107925025, 0.7925546030683083]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.960573098786126, 0.2502085630161057, 2.5289850608448516e-07]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.0040012841833624675, 0.35275242349109664, 0.009691095371989767]
Epoch 3000 RMSE =  0.2964335355838148
Epoch 3100 RMSE =  0.29441818750397836
Epoch 3200 RMSE =  0.2980740422635264
Epoch 3300 RMSE =  0.29405391876595033
Epoch 3400 RMSE =  0.2935082542368407
Epoch 3500 RMSE =  0.2972051410769338
Epoch 3600 RMSE =  0.2951240900191327
Epoch 3700 RMSE =  0.29459573731133337
Epoch 3800 RMSE =  0.295857049304094
Epoch 3900 RMSE =  0.29524753522162
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [1.7465521691920454e-05, 0.4386835309978846, 0.8110088677639338]
Sample [5.1 3.5 1.4 0.3] expected [1. 0. 0.] produced [0.9684547355322997, 0.24515650196772426, 3.248050671453844e-08]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.9678039583622158, 0.24456645993317372, 3.345206908522702e-08]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.9674405821519155, 0.24385788425003252, 3.3940536315673936e-08]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.0035194992071705044, 0.3556649272407111, 0.004298217656255458]
Sample [6.  3.4 4.5 1.6] expected [0. 1. 0.] produced [0.00019595546675857552, 0.4007122850992453, 0.15656837165538473]
Sample [6.7 2.5 5.8 1.8] expected [0. 0. 1.] produced [1.1177788874080128e-05, 0.44773273734786717, 0.8848169372640069]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [1.194934264955075e-05, 0.44267434532346855, 0.8758577438220773]
Sample [6.1 2.6 5.6 1.4] expected [0. 0. 1.] produced [1.5790701718518453e-05, 0.4345758183387684, 0.8310627229833121]
Sample [6.  2.9 4.5 1.5] expected [0. 1. 0.] produced [8.324233109675922e-05, 0.4066328411962883, 0.3621633111729826]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [1.582755141075428e-05, 0.43566422400757066, 0.8294629841477851]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.006038930338522418, 0.34764295956174684, 0.0021192683681864996]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.002826343662944377, 0.36204085517562146, 0.005694435567679236]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [1.097561904748373e-05, 0.4465972387939185, 0.8871052329811884]
Sample [5.5 2.6 4.4 1.2] expected [0. 1. 0.] produced [0.0002419656748332712, 0.39746158121384373, 0.12340792813320589]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9683668555652052, 0.2488231441624443, 3.254082699076786e-08]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [1.884186750068165e-05, 0.4383030575438988, 0.7955250869502392]
Sample [5.9 3.  4.2 1.5] expected [0. 1. 0.] produced [0.0006450167426885005, 0.3833314697877277, 0.03791854123658207]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [1.4692831790931738e-05, 0.44295184482888905, 0.8437924330580936]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [3.107701896041594e-05, 0.42797537373056427, 0.6715499402811093]
Sample [4.4 2.9 1.4 0.2] expected [1. 0. 0.] produced [0.9674007079247187, 0.24961000659492436, 3.365497349394101e-08]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [5.035317716324132e-05, 0.42457972699268287, 0.516654360770418]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [0.0007881961992721339, 0.3889943896146567, 0.02851391902402402]
Sample [6.5 3.  5.5 1.8] expected [0. 0. 1.] produced [0.00018323977360528045, 0.41477239525059356, 0.1636909403666541]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9683747359907153, 0.2518405190925725, 3.2178616054034776e-08]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.0075290923781416, 0.35692739182660016, 0.0015665368687993889]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.0013014344205553852, 0.3860148339552035, 0.015268540957076852]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [2.2052259439102856e-05, 0.4512098425758377, 0.7572866657066273]
Sample [6.9 3.1 5.1 2.3] expected [0. 0. 1.] produced [1.3629062341456976e-05, 0.4544953557683545, 0.8544754459062102]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.0027568937468936527, 0.37248965760999486, 0.0058452766116834]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [0.00018998265024755748, 0.41562776286903264, 0.1605477215382794]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.0017489213085369404, 0.38737466296095024, 0.010497844172846844]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [0.00012004050359587576, 0.4315857757368692, 0.257338692802437]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.005745477261059981, 0.3782222646821201, 0.0022285558063301126]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.014232952211265994, 0.3688456459978774, 0.0006785293506699158]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.0031605357221347725, 0.395162107759367, 0.004851708431650901]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.9660059455902497, 0.27274162537157487, 3.5456598378476035e-08]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [1.0421703072147802e-05, 0.486415946959284, 0.8920545420882426]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.967892476203432, 0.26853576153446507, 3.28532857731416e-08]
Sample [6.9 3.2 5.7 2.3] expected [0. 0. 1.] produced [1.1491013168757597e-05, 0.47904945499307683, 0.8793789007742725]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.969075215935789, 0.2646855775855847, 3.125085268325957e-08]
Sample [7.2 3.  5.8 1.6] expected [0. 0. 1.] produced [2.343330225180635e-05, 0.4623210507779966, 0.7430434688721778]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [1.0214668733445197e-05, 0.47096354281024877, 0.8955651057789539]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.0003458869202777004, 0.41340378446294723, 0.08081323387846935]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [1.1400847743747469e-05, 0.46979008553328055, 0.8814909216965234]
Sample [5.4 3.4 1.5 0.4] expected [1. 0. 0.] produced [0.967483913185977, 0.2602884784172101, 3.363502145659839e-08]
Sample [6.4 3.2 5.3 2.3] expected [0. 0. 1.] produced [1.139560557458497e-05, 0.4642220284886388, 0.8817598499311952]
Sample [5.  3.  1.6 0.2] expected [1. 0. 0.] produced [0.9657675934956731, 0.2577368104630932, 3.613793744950972e-08]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.00027460845701705145, 0.4111098177836976, 0.10644945004550974]
Sample [6.4 3.1 5.5 1.8] expected [0. 0. 1.] produced [1.5177647889070486e-05, 0.4592576379300659, 0.8371261784268024]
Sample [5.4 3.9 1.7 0.4] expected [1. 0. 0.] produced [0.9685399729920217, 0.25631292277778905, 3.224302415275521e-08]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.966112735226081, 0.25628249021612953, 3.566741986866772e-08]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [1.8229971264368985e-05, 0.44991505896296297, 0.8025298227532462]
Sample [7.4 2.8 6.1 1.9] expected [0. 0. 1.] produced [1.0963345112247714e-05, 0.4535340325395304, 0.8877766347561978]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [1.1165206028484084e-05, 0.4491209462086288, 0.8855584692040337]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.9698224649745983, 0.24766599261215283, 3.0665790054470906e-08]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.014077751437139845, 0.3421768720019345, 0.0007037076488431362]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.9679441078430316, 0.2502811393890516, 3.329514666332713e-08]
Sample [7.7 2.6 6.9 2.3] expected [0. 0. 1.] produced [9.271756547156424e-06, 0.45008325449192366, 0.9080329626483525]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.0001940914074196976, 0.40161222717638584, 0.15897342368060216]
Sample [6.7 3.3 5.7 2.1] expected [0. 0. 1.] produced [1.2014444624126976e-05, 0.44707247307320314, 0.8755460739672349]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [9.485396397600173e-05, 0.41280488754783384, 0.324038865842763]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.9680306376749755, 0.25058841544660176, 3.300384203296905e-08]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.9691796680311613, 0.24926662054063425, 3.135846732460031e-08]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [3.57402489380569e-05, 0.4295436681278357, 0.6283314127376544]
Sample [5.1 3.3 1.7 0.5] expected [1. 0. 0.] produced [0.9640306288467788, 0.24821483545901474, 3.894518615472848e-08]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [7.873914545839562e-05, 0.4132574951747614, 0.37984727964824055]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.969699314944777, 0.24802434943152266, 3.0637900993503473e-08]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.03031834812587, 0.3326564893382994, 0.0002520542838294043]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.969609086796095, 0.24996559756550193, 3.076334598643885e-08]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [6.238865916326257e-05, 0.42346595244923124, 0.4502765312675904]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.9701458685977714, 0.2515175661160279, 2.9684693296244184e-08]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.9691735970787848, 0.2510285998105893, 3.1095501472246834e-08]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [1.866747182910171e-05, 0.4439578773081662, 0.7948570498907457]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [0.0018454718865055062, 0.3730298899872659, 0.00978192845412914]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [2.651525626712951e-05, 0.4394950945424822, 0.7115432688891609]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [1.320425384961839e-05, 0.4460807982756813, 0.8603533922017849]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.9695616866353091, 0.24659149922498932, 3.085246937653429e-08]
Sample [5.  3.4 1.6 0.4] expected [1. 0. 0.] produced [0.9683702910527521, 0.24618427331527046, 3.248251725768014e-08]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [2.0452274875588047e-05, 0.4333348952826578, 0.7776093801002588]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.9692146684536413, 0.2430597033526866, 3.1437187155977606e-08]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [1.3217199149359456e-05, 0.43486808253196463, 0.8612137707535041]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.9612734548465741, 0.24293203253487275, 4.323987665073253e-08]
Sample [5.4 3.4 1.7 0.2] expected [1. 0. 0.] produced [0.9691823868710108, 0.23947912006510036, 3.153781519801428e-08]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [1.6774538554880974e-05, 0.42535759990750843, 0.8202606517593547]
Sample [6.2 3.4 5.4 2.3] expected [0. 0. 1.] produced [1.5867323970899806e-05, 0.42245247613283937, 0.8312201180814479]
Sample [5.1 3.8 1.9 0.4] expected [1. 0. 0.] produced [0.9686044614839814, 0.23514123790757216, 3.249010320100236e-08]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [1.4253583746760467e-05, 0.4192811700073531, 0.8503613672774296]
Sample [4.8 3.4 1.6 0.2] expected [1. 0. 0.] produced [0.968684865114114, 0.23253499822660603, 3.2487151951842736e-08]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [2.6731598320735573e-05, 0.40572355764020396, 0.7154710977597201]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.0022923232777588586, 0.34190339674510534, 0.007649876183179536]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.0008167877303425031, 0.359715536041903, 0.028729063324807076]
Sample [4.9 2.5 4.5 1.7] expected [0. 0. 1.] produced [1.5205175222992143e-05, 0.4197638624513842, 0.8408686406450915]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.9669940877930675, 0.23407584019570915, 3.516815796219801e-08]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.00022446499770139148, 0.3774263773010344, 0.13753259279894825]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [8.244795129311492e-05, 0.395960555185237, 0.3694647915111678]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [1.4035841993971006e-05, 0.42614293295731764, 0.8528985170028648]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.9691995943129772, 0.23592291352630182, 3.1771100071168104e-08]
Sample [5.6 3.  4.5 1.5] expected [0. 1. 0.] produced [0.00042604739190270715, 0.3732085649610235, 0.06409945803910345]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.9679164573002381, 0.23828731596182087, 3.366250700697084e-08]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.9693756428068713, 0.2369605517995118, 3.151629111596691e-08]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.15792546685919343, 0.2953681473594615, 2.4894782302082634e-05]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [5.649110108046616e-05, 0.40812309249987694, 0.4862572899066259]
Sample [5.5 2.5 4.  1.3] expected [0. 1. 0.] produced [6.666376022351624e-05, 0.4024681617411507, 0.43719324061646]
Sample [6.  2.2 4.  1. ] expected [0. 1. 0.] produced [0.0014387582937331264, 0.36386165042908813, 0.013838067239982196]
Epoch 4000 RMSE =  0.2914347231295721
Epoch 4100 RMSE =  0.29476105444042183
Epoch 4200 RMSE =  0.2954944894466152
Epoch 4300 RMSE =  0.296216279528089
Epoch 4400 RMSE =  0.29188464485579224
Epoch 4500 RMSE =  0.29644092593477156
Epoch 4600 RMSE =  0.2925294845510375
Epoch 4700 RMSE =  0.2922747320927831
Epoch 4800 RMSE =  0.2961006029670823
Epoch 4900 RMSE =  0.2922030839651154
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.009323270168507819, 0.33481310087801314, 0.0004055295895290826]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [3.3017308408225044e-05, 0.42585868706984614, 0.5001543917358109]
Sample [5.4 3.9 1.7 0.4] expected [1. 0. 0.] produced [0.9726118886205819, 0.23222170102555909, 4.578660551643461e-09]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [6.411812516831331e-05, 0.4189586850890784, 0.2821705602223801]
Sample [5.4 3.4 1.5 0.4] expected [1. 0. 0.] produced [0.9725167220578728, 0.23405823501795556, 4.578153172378854e-09]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [7.571226442275326e-05, 0.41994679045869887, 0.2366670303968905]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.0003026147547469093, 0.39450056106116965, 0.04452045872723286]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.00012113812932834583, 0.4136159500642848, 0.14161711098054175]
Sample [5.9 3.  4.2 1.5] expected [0. 1. 0.] produced [0.00020823236408344577, 0.40942480095737666, 0.07223761084772344]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [8.963612443954414e-06, 0.46595977518496806, 0.8572351867570508]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.9718062959914503, 0.23743290986546298, 4.812976288607668e-09]
Sample [5.4 3.4 1.7 0.2] expected [1. 0. 0.] produced [0.971212066303876, 0.23691242029373835, 4.955128044214558e-09]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [1.8509031646818387e-05, 0.447549349775375, 0.6885840354770653]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [1.3820404567826407e-05, 0.4572714544786996, 0.7641850727864496]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9714564288424904, 0.23646247908412052, 4.852538618219511e-09]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [1.2171185713265185e-05, 0.4542306537640642, 0.7953682641137005]
Sample [6.7 3.3 5.7 2.1] expected [0. 0. 1.] produced [1.1657464862688777e-05, 0.45086607541782464, 0.805653587546172]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [8.038855220027081e-06, 0.45291376209039186, 0.8743435615419464]
Sample [6.9 3.2 5.7 2.3] expected [0. 0. 1.] produced [9.054008374515006e-06, 0.44684660926580433, 0.8554098113313581]
Sample [6.7 2.5 5.8 1.8] expected [0. 0. 1.] produced [8.921124698848637e-06, 0.4430508518345501, 0.8582409073210842]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.9699489668690564, 0.2264263920968088, 5.2953331793892324e-09]
Sample [4.4 2.9 1.4 0.2] expected [1. 0. 0.] produced [0.9688930093670407, 0.22614048527859928, 5.562738337022212e-09]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.002119028971798474, 0.35219276254342435, 0.003149647511479224]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.0022629229244791826, 0.3552497832191952, 0.0028760820961937447]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.9689014234818951, 0.23064729174296686, 5.550372706785622e-09]
Sample [5.6 3.  4.5 1.5] expected [0. 1. 0.] produced [6.115322986191995e-05, 0.4146041135855242, 0.29817744637735316]
Sample [5.5 2.5 4.  1.3] expected [0. 1. 0.] produced [0.0004544815921314231, 0.3874090327444452, 0.02568261827915898]
Sample [6.4 3.1 5.5 1.8] expected [0. 0. 1.] produced [1.7204739697209747e-05, 0.4446802352096761, 0.7083569279702392]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.006540094737731542, 0.3472078165346927, 0.0006612462788183863]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [2.3540222849192738e-05, 0.44024389125997926, 0.6137019205180346]
Sample [6.4 3.2 5.3 2.3] expected [0. 0. 1.] produced [1.301790722550996e-05, 0.45487473152540786, 0.7791041437122457]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [2.632533626482594e-05, 0.4392442138513103, 0.57278038201812]
Sample [4.9 2.5 4.5 1.7] expected [0. 0. 1.] produced [1.071781691879397e-05, 0.4501465389451035, 0.8248793726348177]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.9712623585777423, 0.2312935106435099, 4.964989966569099e-09]
Sample [6.  3.4 4.5 1.6] expected [0. 1. 0.] produced [0.00018731615819573741, 0.39919363727731855, 0.08318736345314748]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.9708047417131328, 0.23333562859928322, 5.081582582196947e-09]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.000349436341087149, 0.39283122855466573, 0.03689021697567007]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.9716651229729455, 0.23480979103513733, 4.859768213071093e-09]
Sample [5.1 3.8 1.9 0.4] expected [1. 0. 0.] produced [0.9701370707602873, 0.23469573898044577, 5.233955985923415e-09]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [8.757085009548797e-06, 0.45507692417074663, 0.8619912238940546]
Sample [6.5 3.  5.5 1.8] expected [0. 0. 1.] produced [1.0747704212652647e-05, 0.4475815445050507, 0.8251024626269047]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.017640863719237863, 0.3284841852007023, 0.00016661777027604684]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [4.018997174336187e-05, 0.42666369186045633, 0.43349851441945814]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.9729870909380813, 0.23387053324104595, 4.508844695999091e-09]
Sample [7.2 3.  5.8 1.6] expected [0. 0. 1.] produced [3.052769841832549e-05, 0.43489620276689633, 0.524372651703492]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [9.970262303439087e-06, 0.44944255843886255, 0.8403282477109799]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.9683354654245321, 0.23116608824357807, 5.744511756038983e-09]
Sample [6.  2.2 4.  1. ] expected [0. 1. 0.] produced [0.00021100891854762335, 0.3955080681054567, 0.07213443689838993]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.9709650779074824, 0.23196077647630958, 5.072224348476335e-09]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.969251821023441, 0.23193284277110854, 5.512277817248605e-09]
Sample [6.  2.9 4.5 1.5] expected [0. 1. 0.] produced [3.130457806769722e-05, 0.42862194105823515, 0.5205775729771985]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.014079977711210771, 0.33731931008031624, 0.00022593590798653969]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.00020610994020181219, 0.4072056578382563, 0.07310320802627703]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [8.081070297428922e-06, 0.4652041654616217, 0.8737325641549873]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.9721520437342562, 0.2356265630473554, 4.72194277323683e-09]
Sample [6.2 3.4 5.4 2.3] expected [0. 0. 1.] produced [9.190983234824329e-06, 0.4577384155574042, 0.8530518455151626]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.9723514735381038, 0.23273464321766776, 4.680151912418497e-09]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.9609088074887632, 0.2363130893928139, 7.735721264564846e-09]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [9.299092884524427e-06, 0.45128394471817046, 0.8513981549218558]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.0011448979765441092, 0.3705077455811982, 0.0073650177355142794]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [9.423791770082257e-06, 0.45179766998074555, 0.849410695540452]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [7.580501278875732e-06, 0.45126407585254924, 0.8842781644928602]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [8.661292462690737e-06, 0.4449802539229999, 0.8642646784522607]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.00033002746172923215, 0.38341553268086864, 0.04002690439661181]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.0009608656129132849, 0.37129855308763193, 0.009422642307396414]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9703967281559167, 0.23196613395554638, 5.200865562970858e-09]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.0002235777249574203, 0.39736470063424006, 0.0666595658412324]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.00019439899940910715, 0.40409408270496866, 0.079704605258182]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [1.2609505048046184e-05, 0.45333569109588, 0.7913456791760518]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [7.11847862906346e-06, 0.4586831266395666, 0.8936144775477552]
Sample [5.  3.  1.6 0.2] expected [1. 0. 0.] produced [0.9681600013745018, 0.23321189023595545, 5.788791337099879e-09]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [7.179105820834278e-06, 0.4532832780111246, 0.8926538744005922]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.9707427150223439, 0.2294362382553222, 5.1314081155808915e-09]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.9709338446475416, 0.2286460276548602, 5.0866960360594004e-09]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [7.654644499918878e-06, 0.44608355176160563, 0.8840403189419802]
Sample [5.1 3.3 1.7 0.5] expected [1. 0. 0.] produced [0.9654331867401683, 0.22804657975298553, 6.520797372670026e-09]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [7.279684578704572e-06, 0.4418413315048064, 0.8911562601790074]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [2.705316997916325e-05, 0.4169746827986551, 0.5719782253837002]
Sample [7.4 2.8 6.1 1.9] expected [0. 0. 1.] produced [1.0153769763565234e-05, 0.43754279345900376, 0.8353587333302127]
Sample [4.8 3.4 1.6 0.2] expected [1. 0. 0.] produced [0.971333073579118, 0.22383080889513313, 4.943131485413847e-09]
Sample [5.  3.4 1.6 0.4] expected [1. 0. 0.] produced [0.9704790887351357, 0.22349455456195846, 5.15241219909195e-09]
Sample [7.7 2.6 6.9 2.3] expected [0. 0. 1.] produced [6.914562480430073e-06, 0.4378751810333802, 0.8964647009116424]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.05663859402172637, 0.2989361549797536, 3.13293863408706e-05]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [0.00012764486978753396, 0.3923718795031937, 0.13361140691615692]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.001827773397321998, 0.3562056003071421, 0.003867648008324535]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [3.049982862264716e-05, 0.42415803989946016, 0.5266428000133695]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.9692026594394212, 0.22730566696245405, 5.539857459816593e-09]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.9725281527551417, 0.2252471892613121, 4.697499801915037e-09]
Sample [6.9 3.1 5.1 2.3] expected [0. 0. 1.] produced [8.207440531804105e-06, 0.4396501629216987, 0.8741030070558993]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [1.1605717660722468e-05, 0.43017052806495854, 0.8116796947627992]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.0004071848891626594, 0.3713731722907594, 0.030687522086699087]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [9.385017551461534e-06, 0.43464090196857647, 0.8530338130825543]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [8.53766406135115e-06, 0.4322682410251881, 0.8690014391233892]
Sample [5.5 2.6 4.4 1.2] expected [0. 1. 0.] produced [2.824489882751992e-05, 0.40970167256640944, 0.55985234294264]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.9695873725854083, 0.22336237075454293, 5.401236042042138e-09]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [7.893539665937625e-06, 0.4337617691454759, 0.87899158696696]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [0.00011268742363274202, 0.38855640981722356, 0.15586463365620706]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.007223410747977738, 0.3310062046657409, 0.000580731825797549]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.9723738881678079, 0.2248229805285829, 4.705632725150009e-09]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.0016505245974223017, 0.3556541909044438, 0.004481197417673042]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.9705621616152419, 0.22748872442249288, 5.164148318146408e-09]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [0.0001806007396259349, 0.3927086559669092, 0.08756387718883779]
Sample [6.1 2.6 5.6 1.4] expected [0. 0. 1.] produced [1.14845501184776e-05, 0.4412434642042712, 0.8119986188161415]
Sample [5.1 3.5 1.4 0.3] expected [1. 0. 0.] produced [0.971409426259826, 0.22706270728618025, 4.956027930200099e-09]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9706612362894744, 0.22667751987911108, 5.1474297700923885e-09]
Epoch 5000 RMSE =  0.2932793571720485
Epoch 5100 RMSE =  0.29337918595797774
Epoch 5200 RMSE =  0.2921983538583211
Epoch 5300 RMSE =  0.29497581556085484
Epoch 5400 RMSE =  0.29252691292859023
Epoch 5500 RMSE =  0.2916244440793133
Epoch 5600 RMSE =  0.2952134902291862
Epoch 5700 RMSE =  0.29234428149960684
Epoch 5800 RMSE =  0.29001052192443183
Epoch 5900 RMSE =  0.29113212383023307
Sample [5.9 3.  4.2 1.5] expected [0. 1. 0.] produced [0.0003484643767692603, 0.3837985161353747, 0.02482111331844751]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [1.4680507099801217e-05, 0.4401565390663372, 0.7233368141648101]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.9781518744840538, 0.22069272764966807, 8.649925992002031e-10]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.9767461537089099, 0.2207896972450976, 9.512626309850526e-10]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.0006825312718767935, 0.3722921001949679, 0.009497511259924468]
Sample [5.1 3.3 1.7 0.5] expected [1. 0. 0.] produced [0.9723568750919284, 0.22475116555729796, 1.2321859846661537e-09]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [3.078714982962679e-05, 0.4259663318994872, 0.47157266412889254]
Sample [4.4 2.9 1.4 0.2] expected [1. 0. 0.] produced [0.9753647101563554, 0.22515569541525193, 1.0281775773052745e-09]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [9.695631724537928e-06, 0.4491928523429058, 0.826252939071485]
Sample [5.1 3.8 1.9 0.4] expected [1. 0. 0.] produced [0.9768661224533641, 0.22175317266228994, 9.341707049325817e-10]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.0005011731013638941, 0.3795832389931027, 0.014672128655591338]
Sample [5.6 3.  4.5 1.5] expected [0. 1. 0.] produced [0.00011714010062279648, 0.4075851059239985, 0.11097074365443275]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.0013754565786363029, 0.37202680527010124, 0.0033812232792813208]
Sample [7.7 2.6 6.9 2.3] expected [0. 0. 1.] produced [6.484469342527395e-06, 0.4656971328093552, 0.8956445510250854]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.9779429840580686, 0.22607959900045838, 8.700171351407581e-10]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.0016447453835867717, 0.36882500248573175, 0.002608158478035487]
Sample [5.  3.  1.6 0.2] expected [1. 0. 0.] produced [0.9755834939810116, 0.2292339847936254, 1.0143316374681133e-09]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9767102415792902, 0.22793647882084733, 9.453996714611867e-10]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [0.00017454218213028072, 0.4077043311171411, 0.06512981084707853]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [1.2715283619839924e-05, 0.45662669761786234, 0.7624441580800375]
Sample [6.4 3.1 5.5 1.8] expected [0. 0. 1.] produced [1.1401749250745815e-05, 0.45439502792955283, 0.7911484179958542]
Sample [6.7 2.5 5.8 1.8] expected [0. 0. 1.] produced [7.534117580061752e-06, 0.4573330723762924, 0.8746704826062935]
Sample [6.9 3.1 5.1 2.3] expected [0. 0. 1.] produced [9.073848568183405e-06, 0.4500163030024728, 0.8419695194537845]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [7.107839273954894e-06, 0.4500360207892881, 0.884243920707882]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [6.954072775571073e-06, 0.44629314544241483, 0.8876552095901418]
Sample [6.1 2.6 5.6 1.4] expected [0. 0. 1.] produced [1.015559582044911e-05, 0.43598676632013883, 0.8197623770581806]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.9771159496612508, 0.2160062908537107, 9.345184355817933e-10]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [7.807876687674259e-06, 0.43552001307366306, 0.8702635416988358]
Sample [7.4 2.8 6.1 1.9] expected [0. 0. 1.] produced [7.6187489577048085e-06, 0.43201391679480783, 0.8744985465513448]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [8.023370742027821e-06, 0.42731648235259984, 0.8661922239586891]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [8.334755058281891e-06, 0.4229149417060113, 0.8598799848605884]
Sample [6.7 3.3 5.7 2.1] expected [0. 0. 1.] produced [8.132839692357668e-06, 0.41957848376394674, 0.8644385920329147]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.9768718572479307, 0.20678134295801448, 9.574099052017775e-10]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.00017404114675781095, 0.36770107506381605, 0.0674498915949428]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.9772418595494631, 0.2084681581505477, 9.339863357912993e-10]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.9744756575303978, 0.20919496368956117, 1.1131431308129924e-09]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [4.052755982252087e-05, 0.3930455840032275, 0.3786744359728489]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.00036277708496980013, 0.36387865229968963, 0.023793216211504858]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [2.646790544118803e-05, 0.4093311144720437, 0.5288795368026685]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.9785076801254106, 0.2139213840992341, 8.378862458287783e-10]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [1.2818089079523036e-05, 0.42508353063058424, 0.7608763502489306]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.12219476847385514, 0.2824839598513284, 4.0063703387295275e-06]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.16100816333590506, 0.28129810975577063, 2.5010453466980147e-06]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.9766045425002613, 0.217428599976546, 9.531078639004544e-10]
Sample [6.  2.9 4.5 1.5] expected [0. 1. 0.] produced [0.00018588828737704183, 0.3858204582885092, 0.059889052876570335]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.976638313742209, 0.21926588257309937, 9.502384864540613e-10]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [3.094329004188429e-05, 0.418308861004248, 0.4672135905993918]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9775710214258344, 0.22057504743934084, 8.836393933715436e-10]
Sample [4.8 3.4 1.6 0.2] expected [1. 0. 0.] produced [0.9775726615844641, 0.21991961830401177, 8.830280305890265e-10]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.13249814971697435, 0.29119474835418696, 3.4317943961849623e-06]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [2.758355051931955e-05, 0.42731739994389867, 0.5044825499727655]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.03063724037815386, 0.31421372794097324, 3.4788209714951025e-05]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [1.8813107581604606e-05, 0.4343997810983691, 0.6444872922161178]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [2.6538089173162177e-05, 0.42500390375541225, 0.5258153825252357]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.008560217821120196, 0.3379119971323657, 0.0002305516164665785]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.9754642872676599, 0.22676995705538425, 1.014913426073781e-09]
Sample [7.2 3.  5.8 1.6] expected [0. 0. 1.] produced [1.8620811018482506e-05, 0.43930524644740715, 0.6463430303766536]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [7.491431632675774e-06, 0.45063044165074934, 0.8750468788225984]
Sample [5.4 3.4 1.5 0.4] expected [1. 0. 0.] produced [0.9757101185327096, 0.2221142333618289, 1.0090877952932486e-09]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [6.700458035935432e-06, 0.4474240400285637, 0.8920318080324565]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [9.156876954064752e-06, 0.4382077518948941, 0.8397562994483498]
Sample [5.1 3.5 1.4 0.3] expected [1. 0. 0.] produced [0.9764801680878006, 0.2173237317849789, 9.650976277407651e-10]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [6.9501937337071395e-06, 0.4378829670327046, 0.8872520924261905]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.00039610915463586196, 0.369873024441816, 0.020839989376671277]
Sample [6.  2.2 4.  1. ] expected [0. 1. 0.] produced [0.0001996251692702059, 0.3849466597815235, 0.0548292260091929]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.9771629339270528, 0.2195053992814758, 9.237755355008689e-10]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.9761815506278508, 0.21935784592849544, 9.844784954133543e-10]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [6.646782419988172e-06, 0.44271589436279773, 0.893752168224808]
Sample [6.4 3.2 5.3 2.3] expected [0. 0. 1.] produced [7.2213968504689854e-06, 0.43733704671566087, 0.8818299737981371]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [9.610847934740845e-06, 0.4287674515212477, 0.8311183130733834]
Sample [5.4 3.4 1.7 0.2] expected [1. 0. 0.] produced [0.9763749520906885, 0.2173018296299776, 9.628887473271743e-10]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [8.244171113589187e-06, 0.4354131970393919, 0.8582459835089392]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [9.095167382550755e-05, 0.393068259412639, 0.15341086696991538]
Sample [5.  3.4 1.6 0.4] expected [1. 0. 0.] produced [0.9749489094200673, 0.2180002004956982, 1.0516072445523614e-09]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [7.901761430593823e-06, 0.43629981955301256, 0.8656079662845143]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.0005624878864081315, 0.3648416383512982, 0.012445679433902509]
Sample [5.4 3.9 1.7 0.4] expected [1. 0. 0.] produced [0.9768342890046696, 0.21713382956819843, 9.359915415434514e-10]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [1.378923048547986e-05, 0.4272807379206484, 0.7409112468425523]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [7.797094279822856e-06, 0.432815135588797, 0.8689672610180919]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [6.740726116885837e-06, 0.4312968925106608, 0.8915835195983831]
Sample [6.5 3.  5.5 1.8] expected [0. 0. 1.] produced [8.804054059759198e-06, 0.4231475553397842, 0.8478654984759835]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.00011631672756958576, 0.3790808305893404, 0.11355160095508637]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.975402314625298, 0.21258067092598845, 1.035472668404867e-09]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.00013097542609049448, 0.38098910366898386, 0.09707903367111645]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9756567195176742, 0.21432001992832292, 1.0178648020009384e-09]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.9625558657355378, 0.2187190576805637, 1.960302664014022e-09]
Sample [4.9 2.5 4.5 1.7] expected [0. 0. 1.] produced [8.843912349623101e-06, 0.4266288736106385, 0.8470914816636532]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.9769065818514989, 0.21073984333340454, 9.418536547404546e-10]
Sample [5.5 2.6 4.4 1.2] expected [0. 1. 0.] produced [4.314662341329733e-05, 0.3968593975744055, 0.3535302718363093]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.0006565280536689469, 0.3593015982112048, 0.00997638359984077]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.9748207091962302, 0.21605723744619473, 1.065206243072788e-09]
Sample [6.2 3.4 5.4 2.3] expected [0. 0. 1.] produced [8.433508249346645e-06, 0.43177065842440787, 0.8548149797367358]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [2.2511996933760888e-05, 0.4121363503330304, 0.5840507741238664]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.02109359976698537, 0.3119583385785805, 6.0203682599651356e-05]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.019412325576548595, 0.31673250807186487, 6.814949063055731e-05]
Sample [6.  3.4 4.5 1.6] expected [0. 1. 0.] produced [0.0013585301640313696, 0.3605129210643589, 0.0034077323089940865]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [1.0572129074791855e-05, 0.44305854131069927, 0.8060210927128456]
Sample [6.9 3.2 5.7 2.3] expected [0. 0. 1.] produced [8.99495135845028e-06, 0.4417695296315642, 0.8409135427835395]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [2.2106675080777732e-05, 0.4231405884959047, 0.5873802334288264]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.0024240876185387954, 0.34618548714204056, 0.00149155168897371]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.976378954236273, 0.21989608420366125, 9.686950235152237e-10]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.974586367052282, 0.22011486023918014, 1.0824377860875822e-09]
Sample [5.5 2.5 4.  1.3] expected [0. 1. 0.] produced [6.924386964136181e-05, 0.4039384143905189, 0.2137491773989641]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.976374615237074, 0.22108832741076925, 9.676828415065405e-10]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.0023078697104681246, 0.35247957658136236, 0.001595357293226109]
Epoch 6000 RMSE =  0.291617915974183
Epoch 6100 RMSE =  0.29090087110871554
Epoch 6200 RMSE =  0.29385661626518556
Epoch 6300 RMSE =  0.2899964788289571
Epoch 6400 RMSE =  0.2934789454641101
Epoch 6500 RMSE =  0.2908070833050908
Epoch 6600 RMSE =  0.2909045556731916
Epoch 6700 RMSE =  0.28802179104024955
Epoch 6800 RMSE =  0.29402565559080285
Epoch 6900 RMSE =  0.2939863410650459
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.9800503425632655, 0.21211813049935427, 1.951667200739083e-10]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [7.945721815165498e-06, 0.4491183110934403, 0.8431981454048121]
Sample [6.  2.2 4.  1. ] expected [0. 1. 0.] produced [7.93265239847392e-05, 0.40529806285613196, 0.1355989935312314]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.9767375426706347, 0.21394595376625392, 2.4894199448204423e-10]
Sample [6.9 3.1 5.1 2.3] expected [0. 0. 1.] produced [6.7969803269925e-06, 0.4519021860755521, 0.8724952575250284]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.00014607020566073247, 0.39490262872807175, 0.05779896365229727]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [1.7820232725353558e-05, 0.43584281287185134, 0.6089762835020518]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [1.6428295286724457e-05, 0.4421607102965549, 0.6334988221852089]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [7.628343441067657e-05, 0.41991873664081686, 0.13783988779848375]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [1.1824469988172001e-05, 0.44934720054665284, 0.7399697550617145]
Sample [6.  2.9 4.5 1.5] expected [0. 1. 0.] produced [8.31502013123194e-05, 0.41968391679821526, 0.12218981926551664]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [8.257072898498414e-06, 0.4656160014171219, 0.8284936601536753]
Sample [5.1 3.8 1.9 0.4] expected [1. 0. 0.] produced [0.9798256202125104, 0.2178212696804975, 1.9251267126373432e-10]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [7.857697492562911e-06, 0.4613289279428268, 0.8395391764671621]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.0005750513232114031, 0.38226869258944335, 0.007109075692981532]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.9790138104242121, 0.21822021773453598, 2.0539051991885926e-10]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [1.0149985378784374e-05, 0.45649490790157193, 0.7798380373934766]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.9800538082761316, 0.2150312229293374, 1.9050793369929742e-10]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [7.642363515510977e-06, 0.45653090021661885, 0.8463666547938504]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [6.257377976631005e-06, 0.455911252451794, 0.8825293778538755]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.001568525106344649, 0.3575255958413513, 0.0015419805289985482]
Sample [6.7 3.3 5.7 2.1] expected [0. 0. 1.] produced [7.378601154951325e-06, 0.45364819335935563, 0.8538495891879992]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [6.269350951821837e-06, 0.45238298276230204, 0.8826938510686909]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [4.969269071407517e-05, 0.41241527000461126, 0.23836524346227903]
Sample [6.7 2.5 5.8 1.8] expected [0. 0. 1.] produced [7.05115416073288e-06, 0.4512114076615209, 0.8621458108856321]
Sample [6.  3.4 4.5 1.6] expected [0. 1. 0.] produced [0.0002945145835669582, 0.3831649087589506, 0.01983368324802463]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [7.494076836024179e-06, 0.45097966878340184, 0.8509379605311662]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [6.27853359279072e-06, 0.44998057268823793, 0.8825464940810999]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.9807748543482873, 0.20812745354846626, 1.8084001290228802e-10]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [1.128590384877155e-05, 0.4348623600611279, 0.7535040539208002]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.9802590881376563, 0.21018925379035583, 1.8629776794835609e-10]
Sample [5.6 3.  4.5 1.5] expected [0. 1. 0.] produced [0.00014461383331507572, 0.39514784701683064, 0.05614962660308252]
Sample [4.9 2.5 4.5 1.7] expected [0. 0. 1.] produced [1.1314278187401751e-05, 0.4438610436676797, 0.7489268639651077]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9800587447780963, 0.21030923243112462, 1.8998642841770983e-10]
Sample [5.  3.4 1.6 0.4] expected [1. 0. 0.] produced [0.9795013134538881, 0.2100542158319961, 1.9819637477241036e-10]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.0005764436098494725, 0.37129815268996663, 0.007099116885782146]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9799739039246836, 0.2116465756360427, 1.9127791703377106e-10]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.9797475735358541, 0.21119849768965562, 1.9482258678730364e-10]
Sample [6.5 3.  5.5 1.8] expected [0. 0. 1.] produced [1.1072161657297739e-05, 0.44168557011600607, 0.7564160760049541]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [7.407746366813325e-06, 0.44478558444368405, 0.8528719964761909]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [4.368666764323631e-05, 0.4102270817275489, 0.2754685813026713]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [7.0883958522363285e-06, 0.4465683231793751, 0.8605552902937083]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [1.0138071139068406e-05, 0.436308963264576, 0.7811705611451407]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.9799847491510874, 0.20562175073084205, 1.924856954320443e-10]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.9793483534483378, 0.20545120345276605, 2.025239713550103e-10]
Sample [4.8 3.4 1.6 0.2] expected [1. 0. 0.] produced [0.9801289891527356, 0.20443602866346153, 1.9007713115606583e-10]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.0009831210534853507, 0.3536782339916439, 0.0031620683842344573]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.006956035890040037, 0.3267220620602666, 0.00015536657492240899]
Sample [5.4 3.4 1.7 0.2] expected [1. 0. 0.] produced [0.9802029550034929, 0.20869665256728573, 1.8882360488792417e-10]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.9807887466986538, 0.2077716258507285, 1.8015643259810073e-10]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.000780437017669793, 0.36408869380279113, 0.004505167619842762]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.0009498641962801387, 0.3650780700629035, 0.003333513273123659]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [6.286759946958991e-06, 0.4558259380449387, 0.8821171993004732]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [4.581986947576233e-05, 0.4170078125064801, 0.26149689267818255]
Sample [7.4 2.8 6.1 1.9] expected [0. 0. 1.] produced [7.921735934885136e-06, 0.4525798732326103, 0.8391606166094308]
Sample [5.1 3.5 1.4 0.3] expected [1. 0. 0.] produced [0.9804595596194627, 0.21090547878019236, 1.8481356032251895e-10]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.00021611583130232351, 0.3904912604903763, 0.03149315965295809]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.008634979275049236, 0.3339655616432967, 0.00011095204725159683]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.025023801938005154, 0.3206645950722902, 2.1090462031511798e-05]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [7.434231749608792e-06, 0.4628634828583565, 0.8522845049314939]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [6.797844320698268e-06, 0.46021512436234213, 0.869085721997483]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [1.3571157832331051e-05, 0.4438185924915158, 0.6969062060460878]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.0001117215746221482, 0.40351042436047685, 0.08328393245523351]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.00041069060045643155, 0.3859269100284101, 0.012119307660264889]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [6.042701346971569e-06, 0.46401517532420894, 0.8893960152489883]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [9.355752318189346e-06, 0.45201015541281275, 0.8044435744784275]
Sample [5.9 3.  4.2 1.5] expected [0. 1. 0.] produced [5.3883058109319445e-05, 0.4175064084770908, 0.2188176758458549]
Sample [6.1 2.6 5.6 1.4] expected [0. 0. 1.] produced [8.463798855235494e-06, 0.45467390229936516, 0.8274306354931369]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [1.4173094143535193e-05, 0.4415281273355542, 0.6853943304189659]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [0.0005529579230197043, 0.3831288424751269, 0.007585016277976319]
Sample [5.  3.  1.6 0.2] expected [1. 0. 0.] produced [0.9791450779221449, 0.2191647414343474, 2.0420458561141487e-10]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [8.813161336498233e-06, 0.45875677922102, 0.8157037798442708]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.9804807706294294, 0.2157600444526032, 1.845367738675088e-10]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.9727466479616017, 0.21932301426557482, 3.139035900662322e-10]
Sample [4.4 2.9 1.4 0.2] expected [1. 0. 0.] produced [0.9783554987668845, 0.21580106898009893, 2.1770781493178262e-10]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.980343907175758, 0.2139791976797185, 1.864852619058976e-10]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.9798315731317981, 0.21369775308369798, 1.9444715112059474e-10]
Sample [5.5 2.5 4.  1.3] expected [0. 1. 0.] produced [0.00016547446353055854, 0.39906038567729096, 0.0467531733315244]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.00018556307822403434, 0.4016256298472008, 0.039489534311129636]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.9805614192699865, 0.2175058026146407, 1.8329538752808062e-10]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.9802796386359496, 0.21705655598301246, 1.8748366551165892e-10]
Sample [7.7 2.6 6.9 2.3] expected [0. 0. 1.] produced [5.743938758366929e-06, 0.4657427390902694, 0.8956181254246106]
Sample [5.5 2.6 4.4 1.2] expected [0. 1. 0.] produced [8.832107174202489e-05, 0.4133182756259065, 0.1141504123268606]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [6.8996522243531896e-06, 0.4631436113555277, 0.8661750741057478]
Sample [6.4 3.2 5.3 2.3] expected [0. 0. 1.] produced [7.611517575373493e-06, 0.45714433180187986, 0.8479943002077838]
Sample [6.4 3.1 5.5 1.8] expected [0. 0. 1.] produced [1.0676069316348362e-05, 0.44701153055929893, 0.7688539174405958]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.02578271585502214, 0.3136176495309731, 2.0302151950170186e-05]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [9.578908480469667e-06, 0.44941120609144014, 0.7980928369052074]
Sample [5.4 3.9 1.7 0.4] expected [1. 0. 0.] produced [0.9801933733981465, 0.21181935085025091, 1.906233016199717e-10]
Sample [6.2 3.4 5.4 2.3] expected [0. 0. 1.] produced [6.930448340730525e-06, 0.4501704208010552, 0.8672334500138759]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.9785486704273324, 0.21033891939540153, 2.167607514199422e-10]
Sample [7.2 3.  5.8 1.6] expected [0. 0. 1.] produced [1.058143342477754e-05, 0.4378549492821019, 0.7735650540632081]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.00024127637031984004, 0.3810784117914013, 0.02731168361969786]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.0002563576902029492, 0.3845259880620361, 0.024936792285226272]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [7.37913082903135e-05, 0.4101472094752462, 0.147851265619092]
Sample [5.1 3.3 1.7 0.5] expected [1. 0. 0.] produced [0.975960900249816, 0.21664014174058582, 2.595436601550399e-10]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.9810778794062432, 0.21307843590513187, 1.7811319214559718e-10]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.9809965831503847, 0.21253134860512649, 1.7928248352740963e-10]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [7.732559315208506e-06, 0.45159233932411225, 0.8473494060136102]
Sample [6.9 3.2 5.7 2.3] expected [0. 0. 1.] produced [6.333143606792333e-06, 0.45099768571266063, 0.8832692557729774]
Sample [5.4 3.4 1.5 0.4] expected [1. 0. 0.] produced [0.9791151053500053, 0.20935195706574466, 2.0872038754162797e-10]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.0011358397589016434, 0.358284846766038, 0.002590430720735244]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.00011629125466654358, 0.40052347181390213, 0.07953687579297958]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9789496224734613, 0.21375325740783616, 2.114875147672659e-10]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.979960526547158, 0.21254415495133303, 1.9555970519120448e-10]
Epoch 7000 RMSE =  0.2913372343425888
Epoch 7100 RMSE =  0.2911296289170815
Epoch 7200 RMSE =  0.29445101331608275
Epoch 7300 RMSE =  0.2890916735983615
Epoch 7400 RMSE =  0.29042477384053556
Epoch 7500 RMSE =  0.2909276647136785
Epoch 7600 RMSE =  0.29042260130734715
Epoch 7700 RMSE =  0.29374330074118876
Epoch 7800 RMSE =  0.2905891672954075
Epoch 7900 RMSE =  0.2916272846804148
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [2.75492338448978e-05, 0.42234982519399783, 0.36830664486818343]
Sample [6.1 2.6 5.6 1.4] expected [0. 0. 1.] produced [3.3405768458231215e-05, 0.4237981390589547, 0.29702643523297617]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [5.633242015169319e-06, 0.45144159772260467, 0.8831119064773831]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.980402981691937, 0.2066964719248871, 5.042763614833638e-11]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.981674588957399, 0.20534131297430902, 4.511321887162981e-11]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [5.4472412540028935e-06, 0.44623764851644, 0.8887620658527257]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [5.811071684136121e-06, 0.44104994113701235, 0.8782315890068888]
Sample [7.7 2.6 6.9 2.3] expected [0. 0. 1.] produced [4.949066025153227e-06, 0.43983792096236835, 0.9034384340770394]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.9800672440879108, 0.20047939852360186, 5.206223356342619e-11]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.00046629967238031987, 0.3592063768203184, 0.006206381351619566]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.9811236951763341, 0.20170294469016653, 4.753340725069285e-11]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [8.260148331377516e-06, 0.43034705687108776, 0.8042946721605168]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.005753960470515393, 0.32003286932232505, 0.00010914353584682897]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [5.899894080106881e-06, 0.43696480986569025, 0.8764517482343943]
Sample [6.  3.4 4.5 1.6] expected [0. 1. 0.] produced [4.473557519287572e-05, 0.398704326436687, 0.21449323975909249]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.0008915048941043421, 0.35409358295001037, 0.0021985123723095097]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.00011444145964231108, 0.3921807857370576, 0.056590526755633176]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [1.1561717218513241e-05, 0.43619916926271773, 0.705444639733816]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [7.637037194955884e-05, 0.4082927354380968, 0.10117367548350405]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [7.54497595535742e-06, 0.453620340327799, 0.8230736870901372]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.006682511192226367, 0.3347555072734286, 8.400635509572925e-05]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9808227985089775, 0.21269891064847146, 4.822868824001553e-11]
Sample [6.9 3.1 5.1 2.3] expected [0. 0. 1.] produced [7.780208590870732e-06, 0.45274975567772746, 0.8163422395264858]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.0011469952059467423, 0.36317489248459384, 0.001446097288806532]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.9796619490662727, 0.21340376626219032, 5.3206956103668714e-11]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [1.2883011811789794e-05, 0.4436790307359282, 0.6647880714442946]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.9818790383609692, 0.21369799428120537, 4.33942866270224e-11]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [3.2103646926326996e-05, 0.43151386391692576, 0.308989716528229]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [7.351929251500775e-05, 0.41345016836072224, 0.10744595217378335]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [6.42079195715283e-06, 0.46129688456411305, 0.8584976491466452]
Sample [5.6 3.  4.5 1.5] expected [0. 1. 0.] produced [1.7500345118889272e-05, 0.4393197851112971, 0.548023536053563]
Sample [7.2 3.  5.8 1.6] expected [0. 0. 1.] produced [1.4401131066307533e-05, 0.44763666773805694, 0.6193421922340916]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.9681856929759947, 0.2192782072555497, 1.1162733723082803e-10]
Sample [4.8 3.4 1.6 0.2] expected [1. 0. 0.] produced [0.9801361153838375, 0.21274191076125878, 5.101824020168764e-11]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.9799480578145717, 0.21226885921012467, 5.188473747881876e-11]
Sample [4.4 2.9 1.4 0.2] expected [1. 0. 0.] produced [0.9776424572625856, 0.2130199766291118, 6.214601906019923e-11]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.005313331746435769, 0.3396898407180925, 0.00012164885492050665]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [7.506111477211414e-05, 0.41564524352866444, 0.10382698216707989]
Sample [5.5 2.6 4.4 1.2] expected [0. 1. 0.] produced [3.0921852566052e-05, 0.435990214329333, 0.32513377343131533]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [1.8450148030765134e-05, 0.45001439805941823, 0.5225217099215308]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [5.487462756384291e-06, 0.46787197477551845, 0.8867014637774175]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [5.31952600942602e-05, 0.4232021679813325, 0.16875468424574072]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.00012343204904397996, 0.4130508917858095, 0.049664233492744105]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.9807063813326531, 0.21878040792567152, 4.876272547465076e-11]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [5.208008363562413e-06, 0.47349974351100155, 0.8946936685363542]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [2.2991523523856113e-05, 0.44245299437096935, 0.43848161989820517]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [5.686864281669644e-05, 0.43101271390863366, 0.15201262439442237]
Sample [5.1 3.8 1.9 0.4] expected [1. 0. 0.] produced [0.981033493168174, 0.22064980737425727, 4.681253591378863e-11]
Sample [5.1 3.3 1.7 0.5] expected [1. 0. 0.] produced [0.9793582395893806, 0.22108625768716408, 5.381637255700904e-11]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.9800514460756962, 0.22000273290774103, 5.096978614177531e-11]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [6.282676993480824e-06, 0.47275421593488415, 0.8607057812902249]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.9818217810237253, 0.21615948870499363, 4.3783443550731113e-11]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [8.286673028544969e-06, 0.46247479036358585, 0.7987489206870568]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.9804296440323295, 0.21452178152220855, 4.9612454086994383e-11]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [7.481172383729617e-06, 0.45922230993600743, 0.8245870886723669]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.001351933662075171, 0.36472625116288504, 0.0011028765614575933]
Sample [5.4 3.4 1.7 0.2] expected [1. 0. 0.] produced [0.9812074074505599, 0.2139595421206204, 4.648174637724939e-11]
Sample [6.5 3.  5.5 1.8] expected [0. 0. 1.] produced [9.125992585955723e-06, 0.4554032082911523, 0.7741523787809442]
Sample [5.4 3.9 1.7 0.4] expected [1. 0. 0.] produced [0.9813980926348002, 0.21133524102517023, 4.587956018213227e-11]
Sample [6.9 3.2 5.7 2.3] expected [0. 0. 1.] produced [6.029873156344879e-06, 0.4578383534468991, 0.8703734676200977]
Sample [5.4 3.4 1.5 0.4] expected [1. 0. 0.] produced [0.9806822119936013, 0.20932018638072222, 4.890701652256358e-11]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [6.337993546878916e-06, 0.4519026715796878, 0.8613247574474412]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.981104102307233, 0.20663627690452516, 4.724941598919576e-11]
Sample [6.4 3.2 5.3 2.3] expected [0. 0. 1.] produced [6.316077532088173e-06, 0.44701564587321546, 0.8622881890669785]
Sample [5.1 3.5 1.4 0.3] expected [1. 0. 0.] produced [0.9811978712474685, 0.20421913113302778, 4.696214296608532e-11]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.9804628635980893, 0.20413767375837077, 5.0062297545812995e-11]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.00010038134109833763, 0.39360289147343797, 0.06835783810262513]
Sample [6.  2.2 4.  1. ] expected [0. 1. 0.] produced [0.00036140826010134483, 0.37651213449451926, 0.009250708469390408]
Sample [5.5 2.5 4.  1.3] expected [0. 1. 0.] produced [6.29447116966154e-05, 0.4108511480764643, 0.13447767358857923]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [6.510994113817575e-06, 0.4557131715580769, 0.8563454132872086]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.9795306302012762, 0.20943971351576987, 5.415409081977199e-11]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [1.037663157772477e-05, 0.44243748348334255, 0.7385110733046136]
Sample [6.4 3.1 5.5 1.8] expected [0. 0. 1.] produced [7.52796283867823e-06, 0.444176588971619, 0.8266208455830086]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [5.817212908993772e-05, 0.4047438168283093, 0.15133553983429626]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.9810303430164599, 0.20668644419495372, 4.79161426390885e-11]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.00015816237863975087, 0.3914732089525477, 0.03435710234278406]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [8.452618143421249e-06, 0.447244521385065, 0.7983992057921712]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [5.593712454566524e-06, 0.4595974079671295, 0.883029997320855]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.9811971016807802, 0.2087866720003914, 4.6697903260441586e-11]
Sample [4.9 2.5 4.5 1.7] expected [0. 0. 1.] produced [8.359066065722257e-06, 0.44737095178957265, 0.7985604286807423]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.00031274708711480684, 0.3806819916089972, 0.01160704041825455]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.0029445356567365803, 0.3476562801549236, 0.00031728253327391114]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.9805479736910921, 0.21168252758676703, 4.958693031197705e-11]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.0003784942138236605, 0.3852790236280581, 0.008564702916912775]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.9820736696843754, 0.2125093125798173, 4.3263096421273746e-11]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [7.80573483184204e-06, 0.4573578865781961, 0.8164100590630842]
Sample [6.7 2.5 5.8 1.8] expected [0. 0. 1.] produced [5.5517451399248195e-06, 0.4593147028057749, 0.8853770982004117]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [2.6053390009338667e-05, 0.4277348629457708, 0.3915332865539167]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.9819522791929893, 0.21054173012605348, 4.3534622603306205e-11]
Sample [6.7 3.3 5.7 2.1] expected [0. 0. 1.] produced [8.41918730547922e-06, 0.45179137875616865, 0.7962397249531379]
Sample [6.2 3.4 5.4 2.3] expected [0. 0. 1.] produced [7.144408579975703e-06, 0.45064872496042596, 0.8364280516043027]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [5.822936782453903e-06, 0.45019187063870947, 0.87698367770607]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [7.841590497917985e-06, 0.44086193203336194, 0.8156854909612052]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [8.792738755643229e-06, 0.4349321320164195, 0.7870652223780656]
Sample [5.  3.4 1.6 0.4] expected [1. 0. 0.] produced [0.9800138231128663, 0.2021506058963141, 5.217124353965714e-11]
Sample [6.  2.9 4.5 1.5] expected [0. 1. 0.] produced [5.800135439386074e-05, 0.39801915610985705, 0.15155982253518663]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9809659686309387, 0.2033991468440745, 4.81104891145003e-11]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [5.739875193035934e-06, 0.4419059717192762, 0.8803750460462813]
Sample [7.4 2.8 6.1 1.9] expected [0. 0. 1.] produced [6.716064997598155e-06, 0.4351799094094256, 0.8513441512353992]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.9822831037926916, 0.1985590769890872, 4.284054659739508e-11]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.005993262412232858, 0.31921666511993135, 0.0001019422471062016]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.0001045061089804073, 0.38831847399205544, 0.06489870915269727]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.0012164957227960961, 0.35226766843069024, 0.0013332941918282406]
Sample [5.9 3.  4.2 1.5] expected [0. 1. 0.] produced [0.0001235271570384251, 0.3945463986729116, 0.050337098612322724]
Sample [5.  3.  1.6 0.2] expected [1. 0. 0.] produced [0.979268534467807, 0.2094259198290464, 5.548103548722844e-11]
Epoch 8000 RMSE =  0.2921864417594685
Epoch 8100 RMSE =  0.2886675613281195
Epoch 8200 RMSE =  0.28937064148756936
Epoch 8300 RMSE =  0.28924895283265734
Epoch 8400 RMSE =  0.28954187660286
Epoch 8500 RMSE =  0.28788460575968877
Epoch 8600 RMSE =  0.29334465577170166
Epoch 8700 RMSE =  0.2898059241808345
Epoch 8800 RMSE =  0.2920582366767181
Epoch 8900 RMSE =  0.28835627456281376
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.9836601608847837, 0.20703371553772448, 1.2271099428311848e-11]
Sample [6.4 3.1 5.5 1.8] expected [0. 0. 1.] produced [7.897566409363157e-06, 0.4546627265188636, 0.807963978744142]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.00014505992906223465, 0.398594507658385, 0.03114801721740606]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.0013412021508845526, 0.3644827928444276, 0.0007701265244529736]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [5.27913403395819e-06, 0.4677140831469828, 0.8924745676182538]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [5.5752158817596335e-06, 0.46238666182182175, 0.883534569680937]
Sample [5.6 3.  4.5 1.5] expected [0. 1. 0.] produced [2.1062503033860025e-05, 0.43412608771444205, 0.45017560388240785]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [0.00012021003241432638, 0.4077287711732776, 0.04168293899094829]
Sample [7.2 3.  5.8 1.6] expected [0. 0. 1.] produced [1.572929362284036e-05, 0.4491081887644261, 0.5679182735408024]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.00019583584685127243, 0.3999733256590527, 0.019096468176977268]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.9712812273579545, 0.21801906942902033, 3.26065157136548e-11]
Sample [4.4 2.9 1.4 0.2] expected [1. 0. 0.] produced [0.980188320879054, 0.21262936960744105, 1.7185786998253908e-11]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.9824188983344192, 0.21052412780365595, 1.3985355303271204e-11]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [9.615177096695595e-06, 0.45633061886398907, 0.7528383731886302]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [6.2218228896769875e-06, 0.4692573470565282, 0.8609064730576965]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.9830922010820764, 0.20975962152490316, 1.293048634522344e-11]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.9839528715869474, 0.20852866094856445, 1.1815648152334896e-11]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [6.791071323135818e-06, 0.4616047372022044, 0.8427303927570678]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.017387969192861518, 0.3209649964968716, 1.0123714084582951e-05]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.000835678002903089, 0.3757724255671627, 0.0016830033320142967]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.983109137464033, 0.21158831970934755, 1.2936289516207413e-11]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.00021261505489493935, 0.40334618735069744, 0.016454999100695325]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [5.193098546730905e-06, 0.4758506630596727, 0.8939130174201607]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [5.259979477578749e-06, 0.47118838708872374, 0.8920070500998291]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [3.0833161380333e-05, 0.4345614369080276, 0.2991961426602837]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [1.3871421827963017e-05, 0.4539484175584428, 0.6174643778374215]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [8.244029095555525e-05, 0.41766519036272304, 0.07616469840611467]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [7.332271441247588e-06, 0.46654677761100743, 0.8263064131051363]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.00014502878126478202, 0.4082239663711368, 0.03107950991358236]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [5.357681606115278e-06, 0.47300949265609643, 0.8898422258290859]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.9809148117302431, 0.21226512105939968, 1.6057630122303584e-11]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [7.363039050116909e-06, 0.46189786352207346, 0.8260455452056658]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.9838210570239583, 0.20770353958346868, 1.2128597739818072e-11]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9826437458818963, 0.20800945582033536, 1.368294503500998e-11]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.00012677736009936608, 0.40481805183061503, 0.03882178622139377]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [1.6874497249484536e-05, 0.4457809984515672, 0.5427050884504047]
Sample [6.4 3.2 5.3 2.3] expected [0. 0. 1.] produced [6.739150273071014e-06, 0.4675115429572564, 0.8442617733555116]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.0008388268835470095, 0.37673329731620564, 0.0016701175255084653]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.9841802376726794, 0.21143916134440274, 1.1540051548208563e-11]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.0001101857503734574, 0.416196394347125, 0.04788547897354021]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.9847301857438009, 0.21277208731010522, 1.085237916339366e-11]
Sample [6.7 2.5 5.8 1.8] expected [0. 0. 1.] produced [5.9123114517963984e-06, 0.47368876235547197, 0.8712969900536116]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [6.258358338286463e-06, 0.4682600618771632, 0.8604800223392929]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [6.672509021150035e-06, 0.4627774733402224, 0.8474032312810088]
Sample [5.1 3.3 1.7 0.5] expected [1. 0. 0.] produced [0.9809648173873352, 0.20909837113662627, 1.5897100860123876e-11]
Sample [5.  3.4 1.6 0.4] expected [1. 0. 0.] produced [0.9825424841043205, 0.20746265012292836, 1.371707283958397e-11]
Sample [5.4 3.4 1.7 0.2] expected [1. 0. 0.] produced [0.9836903546106207, 0.2060685646235652, 1.22108918598768e-11]
Sample [6.9 3.2 5.7 2.3] expected [0. 0. 1.] produced [5.8854521539282534e-06, 0.4582786344202825, 0.8730084743699394]
Sample [5.5 2.5 4.  1.3] expected [0. 1. 0.] produced [7.076610592847235e-05, 0.4095320424096997, 0.09633951492955664]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [7.753859442038204e-06, 0.45406324256892827, 0.8125451346786757]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.0001834889156726341, 0.39362353931796107, 0.02118462902539445]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.9842997621286896, 0.20605336549704947, 1.1482745730950252e-11]
Sample [4.8 3.4 1.6 0.2] expected [1. 0. 0.] produced [0.9831360473850367, 0.20638040593552867, 1.2981411064341207e-11]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.00014140660853912651, 0.4011037231622839, 0.032413184193637846]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [5.405631117517117e-06, 0.4647848495036895, 0.8884889493263453]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [3.603574676525749e-05, 0.42612747116213157, 0.24917598722051854]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [0.00018866234442857854, 0.40121117672449147, 0.02014374146640222]
Sample [6.  3.4 4.5 1.6] expected [0. 1. 0.] produced [0.00014576361917504582, 0.41033997280337614, 0.030706799063281186]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.0008553148311187438, 0.3832751842913763, 0.0016272264126729582]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [6.080782980356833e-06, 0.47810229285277195, 0.866793781767056]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [8.556386987750803e-06, 0.4673460186547412, 0.7862894641512141]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.9835410816349199, 0.2157699546658755, 1.2278784914826366e-11]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [4.502013415474303e-05, 0.44046702202803417, 0.18254146264667542]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.9848694566692503, 0.2163928259852088, 1.0586674261048614e-11]
Sample [6.  2.9 4.5 1.5] expected [0. 1. 0.] produced [0.0001249060979372161, 0.42537958260856473, 0.03867612775598353]
Sample [7.4 2.8 6.1 1.9] expected [0. 0. 1.] produced [7.123662613737549e-06, 0.4835510768019906, 0.8301128430293907]
Sample [5.4 3.4 1.5 0.4] expected [1. 0. 0.] produced [0.9837903920769443, 0.21699965175715533, 1.1936483998143562e-11]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [2.327508759168843e-05, 0.45602624436975986, 0.4027320045749685]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.9832379053023699, 0.214889023011409, 1.2824040990116606e-11]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.9828394153288176, 0.2145809326342791, 1.3351717011320497e-11]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [5.620185761818497e-06, 0.4767228046329408, 0.8816543258915561]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9820700782213834, 0.21253152153189156, 1.4419313774702802e-11]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.0001516496432477713, 0.41096383285172233, 0.02892616270342854]
Sample [5.5 2.6 4.4 1.2] expected [0. 1. 0.] produced [1.5351966909616375e-05, 0.4576407316306939, 0.5806884770157799]
Sample [5.1 3.8 1.9 0.4] expected [1. 0. 0.] produced [0.982885033645369, 0.2160148357204148, 1.3110838805816792e-11]
Sample [6.2 3.4 5.4 2.3] expected [0. 0. 1.] produced [6.0514729938251034e-06, 0.47896059166057176, 0.8659305299637009]
Sample [6.5 3.  5.5 1.8] expected [0. 0. 1.] produced [7.381012246129194e-06, 0.4708216409047893, 0.8227241247949556]
Sample [5.4 3.9 1.7 0.4] expected [1. 0. 0.] produced [0.9838141963981816, 0.21069970423158477, 1.197031187107062e-11]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.0001298369183160597, 0.4130609494392895, 0.0367219863773404]
Sample [4.9 2.5 4.5 1.7] expected [0. 0. 1.] produced [6.933586017982847e-06, 0.4717317461852299, 0.8380110499168544]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.984176717807878, 0.2102454160528748, 1.1545649079006861e-11]
Sample [6.1 2.6 5.6 1.4] expected [0. 0. 1.] produced [7.004242862829679e-06, 0.4663432386980531, 0.836128681137828]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.0050693787272213025, 0.3450924626078185, 8.179988800551417e-05]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.0001815845742781307, 0.40726247975885177, 0.021389084696148176]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [7.611321256584211e-06, 0.47016911984338655, 0.81662369599943]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [5.689972818448532e-06, 0.47126495869905566, 0.8792491983133489]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.9830246757046246, 0.20955775452284076, 1.3108324063169774e-11]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [6.211176290141301e-06, 0.4644244614618603, 0.8629920878179963]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.981243989219258, 0.2083080255019406, 1.5572067722140522e-11]
Sample [6.9 3.1 5.1 2.3] expected [0. 0. 1.] produced [5.857616983198313e-06, 0.46037455618055123, 0.8744702252849611]
Sample [6.7 3.3 5.7 2.1] expected [0. 0. 1.] produced [5.668295523337673e-06, 0.4567458216847875, 0.8806078607198315]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [8.400560238409728e-06, 0.44544986486680765, 0.7925920008136472]
Sample [5.  3.  1.6 0.2] expected [1. 0. 0.] produced [0.981745921050663, 0.20590949797211716, 1.471041432312928e-11]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [5.347417397358271e-06, 0.45780258332121665, 0.888798340448398]
Sample [5.1 3.5 1.4 0.3] expected [1. 0. 0.] produced [0.9836051664049213, 0.20220489533175368, 1.2255178017162119e-11]
Sample [5.9 3.  4.2 1.5] expected [0. 1. 0.] produced [6.459887502295109e-05, 0.4081677952164114, 0.10951461558132511]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9830885583621827, 0.20437845517593917, 1.2918103914925679e-11]
Sample [6.  2.2 4.  1. ] expected [0. 1. 0.] produced [0.00020787529463741006, 0.39141673503254204, 0.017022614461419946]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [9.774323831182718e-06, 0.4509228235017662, 0.7441914631613281]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [6.959444416687607e-06, 0.46209339348082934, 0.8342879485308355]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [5.8168461006356295e-06, 0.4611645685979923, 0.872161012998132]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.9827593752093424, 0.20490634186336437, 1.3197896870879243e-11]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.004567202560324285, 0.3393179875679974, 9.566732680493917e-05]
Sample [7.7 2.6 6.9 2.3] expected [0. 0. 1.] produced [5.010372967128697e-06, 0.46356941296228066, 0.897748880056914]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.9825254098400859, 0.20507738707950185, 1.356187284815909e-11]
Epoch 9000 RMSE =  0.29125630617057435
Epoch 9100 RMSE =  0.2896470116020711
Epoch 9200 RMSE =  0.28514388323416284
Epoch 9300 RMSE =  0.2876882230955387
Epoch 9400 RMSE =  0.2892234496724425
Epoch 9500 RMSE =  0.2899735125082878
Epoch 9600 RMSE =  0.2882902624165164
Epoch 9700 RMSE =  0.2882535666138182
Epoch 9800 RMSE =  0.29011198071325034
Epoch 9900 RMSE =  0.28940399187742843
Sample [6.9 3.1 5.1 2.3] expected [0. 0. 1.] produced [8.09253613847059e-06, 0.4429104438545088, 0.7706560830133954]
Sample [7.7 2.6 6.9 2.3] expected [0. 0. 1.] produced [4.621373835025259e-06, 0.4492306050873454, 0.8996013074386873]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.00040803036062549814, 0.3655182370288585, 0.003673166982343272]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [1.3971070841931963e-05, 0.429849215037601, 0.5668910140391502]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.9843448966545517, 0.1969889993146826, 3.4224279362764214e-12]
Sample [6.1 2.6 5.6 1.4] expected [0. 0. 1.] produced [1.731519575092276e-05, 0.4300676416289858, 0.46901890251874284]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [3.2986022586017345e-05, 0.41473025435676775, 0.2267538003655075]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.9848845455483556, 0.19665559304137398, 3.246624797291368e-12]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [5.659293772342326e-06, 0.45110291961045756, 0.862344479350564]
Sample [4.9 2.5 4.5 1.7] expected [0. 0. 1.] produced [8.06766573707863e-06, 0.4404930585480813, 0.7721513606049356]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.984868241252997, 0.19272321672232162, 3.272118393352454e-12]
Sample [4.8 3.4 1.6 0.2] expected [1. 0. 0.] produced [0.9841222547140263, 0.19282188762267932, 3.563553621891054e-12]
Sample [4.4 2.9 1.4 0.2] expected [1. 0. 0.] produced [0.9824001148209235, 0.19357893419161218, 4.2875666581769225e-12]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [9.802097331247513e-06, 0.43081420021040606, 0.7083866579459878]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.9839276077254269, 0.19421708324416384, 3.592180321042701e-12]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.00812175434868314, 0.31819833392491587, 1.964185679204191e-05]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.0004300243121108924, 0.3716696824984949, 0.003293553961579359]
Sample [6.7 2.5 5.8 1.8] expected [0. 0. 1.] produced [6.365200715801188e-06, 0.4525052291738948, 0.8344386433343598]
Sample [5.5 2.5 4.  1.3] expected [0. 1. 0.] produced [0.0002235246874072146, 0.3839426351848023, 0.010245705390682685]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [5.6766048669967e-06, 0.4554859487814529, 0.8605621215359586]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.9835184163033233, 0.19744352108725488, 3.776938393019538e-12]
Sample [5.9 3.  4.2 1.5] expected [0. 1. 0.] produced [0.00031550168525917876, 0.37790399666358787, 0.00566314386142109]
Sample [6.  3.4 4.5 1.6] expected [0. 1. 0.] produced [0.00022002716904827115, 0.3887037885592656, 0.010551116667005719]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9840817171134346, 0.20117465157787223, 3.5439725834248577e-12]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [6.055733172042899e-06, 0.4584478156397055, 0.8468415630454239]
Sample [5.1 3.8 1.9 0.4] expected [1. 0. 0.] produced [0.9837736643764579, 0.19905554276450454, 3.668174981916365e-12]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9840415698982058, 0.19835646856243558, 3.567207984759855e-12]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.9845085019515116, 0.19749245059422174, 3.3822901439104036e-12]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.00026974702547090954, 0.3828093963748042, 0.0074465539738497195]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.9854742207605389, 0.19853016182832323, 3.0165766694655714e-12]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.0024352387094295986, 0.34797259595010394, 0.00016260118095059176]
Sample [5.1 3.3 1.7 0.5] expected [1. 0. 0.] produced [0.9809140472435879, 0.203776582855847, 4.892903891790303e-12]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [5.832673475324651e-06, 0.460744891663774, 0.8555138913994111]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.0002506011838415166, 0.3876220533112267, 0.008477957314526857]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.9848273751167359, 0.2008923217677582, 3.2650205870816705e-12]
Sample [6.  2.2 4.  1. ] expected [0. 1. 0.] produced [0.00018315322529740242, 0.39695901634967506, 0.014540669688305388]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.0001804815774317267, 0.40173015739729706, 0.014911554338174603]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [8.499266269961202e-06, 0.46343495081084823, 0.7551110436868563]
Sample [5.  3.4 1.6 0.4] expected [1. 0. 0.] produced [0.983204446763507, 0.20853206101308144, 3.85463264183796e-12]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [5.0716116152346495e-06, 0.47728890578572564, 0.8811976060481602]
Sample [6.  2.9 4.5 1.5] expected [0. 1. 0.] produced [4.7116804252424594e-05, 0.43076823826206107, 0.13334419024650868]
Sample [5.  3.  1.6 0.2] expected [1. 0. 0.] produced [0.9831722837581621, 0.20833093997377922, 3.869085308052845e-12]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.0003911052649684294, 0.3952441710737574, 0.0038504074619771195]
Sample [6.9 3.2 5.7 2.3] expected [0. 0. 1.] produced [5.571797389685869e-06, 0.4800641978955934, 0.8629384365785551]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.9838264062989314, 0.2076673559643436, 3.61649794282754e-12]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [5.149998406806663e-06, 0.47627006957277435, 0.8786073070374857]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [9.305746052191253e-05, 0.41741302264291036, 0.04506497664193755]
Sample [7.4 2.8 6.1 1.9] expected [0. 0. 1.] produced [5.6406749303860915e-06, 0.4750760185912698, 0.8608837663579327]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.9850036281817504, 0.20456866125412174, 3.167061867662299e-12]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [9.189303449440295e-05, 0.41738881064505134, 0.04610757077299872]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [5.859802033142847e-06, 0.4740836350827572, 0.8530337917105222]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [2.295008958558276e-05, 0.44389083391128187, 0.35122241404168514]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [5.645406642198314e-06, 0.4753623869515418, 0.859917064707524]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.985525056045176, 0.20426644212250894, 2.9601489108961362e-12]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [7.021146225536153e-06, 0.4660040220892637, 0.8081118827045733]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [7.568636253340911e-06, 0.4603407735640433, 0.7877742033037773]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.984317036218857, 0.2010130796902485, 3.441003130276861e-12]
Sample [6.4 3.2 5.3 2.3] expected [0. 0. 1.] produced [6.405674542508328e-06, 0.45849931486007844, 0.8329989886919321]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.0017508227530521876, 0.3530655133396977, 0.00028758139699874597]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.003543905659259054, 0.34497540947304306, 8.409050505215577e-05]
Sample [5.4 3.4 1.7 0.2] expected [1. 0. 0.] produced [0.9842226796690634, 0.2034964531693245, 3.4793290531727604e-12]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.9845083144479595, 0.2027382873538951, 3.371357819167114e-12]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.00016724623125476386, 0.4016454501902395, 0.016868601821768962]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.0003043442765453564, 0.39521329952229767, 0.006016905872239518]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.0006512900693242877, 0.38575523513987464, 0.001607784643287076]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [2.5932961286661832e-05, 0.45027879729266207, 0.30528122542760355]
Sample [5.4 3.4 1.5 0.4] expected [1. 0. 0.] produced [0.9841720972587974, 0.21185757252408643, 3.4770144115407707e-12]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.03371991358047118, 0.3230366157651174, 1.570497323947798e-06]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.0007566071689548628, 0.3947923350092523, 0.0012282789422380944]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [7.046226789295822e-06, 0.4884694552077608, 0.8077165053041204]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [5.5068578846694735e-06, 0.48873953690951316, 0.8662767649163844]
Sample [5.1 3.5 1.4 0.3] expected [1. 0. 0.] produced [0.9842619989303796, 0.211998363422174, 3.4585380080423313e-12]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [6.887411964813932e-06, 0.47902041357926256, 0.8148142812928665]
Sample [5.5 2.6 4.4 1.2] expected [0. 1. 0.] produced [3.414333207462044e-05, 0.4441899553417312, 0.214259297833898]
Sample [7.2 3.  5.8 1.6] expected [0. 0. 1.] produced [1.0686457235352728e-05, 0.47108059264819435, 0.6718028917363885]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.9830172150219331, 0.21073675569130823, 3.9884106128325205e-12]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.9722239626904601, 0.21664622193179858, 9.611792948847748e-12]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.0004687101551718266, 0.3941935827989486, 0.002865365840735713]
Sample [6.2 3.4 5.4 2.3] expected [0. 0. 1.] produced [5.25184430626676e-06, 0.48349607588146737, 0.8768622300034294]
Sample [6.4 3.1 5.5 1.8] expected [0. 0. 1.] produced [6.2607515405029066e-06, 0.47564947688081166, 0.8401415668219864]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [4.790792221532812e-06, 0.47635860719645, 0.8936038914338013]
Sample [6.5 3.  5.5 1.8] expected [0. 0. 1.] produced [6.018693713914863e-06, 0.4676180579956273, 0.8497411330763358]
Sample [5.6 3.  4.5 1.5] expected [0. 1. 0.] produced [1.632576170606148e-05, 0.44465349934318965, 0.4997587455075923]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9842494160227189, 0.20539668687975735, 3.4681581376075575e-12]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [1.1767932943425626e-05, 0.45480264777094587, 0.6343195238885664]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [5.956466798693536e-06, 0.4635387527487087, 0.8516749562412439]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.00040665992126577365, 0.3820383886580223, 0.0036874935238409975]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.9823718241102001, 0.204915951150407, 4.277070199233373e-12]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.9831752353919084, 0.20379091403175326, 3.93893484865925e-12]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [6.9329577383383255e-06, 0.45970326322335725, 0.8155565877891859]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [8.492651132131317e-06, 0.4517459128617181, 0.7572238790596885]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.9843677866517694, 0.19870394419263138, 3.4809847074773543e-12]
Sample [6.7 3.3 5.7 2.1] expected [0. 0. 1.] produced [5.344460005318803e-06, 0.4555022514289949, 0.8754528452797852]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [6.405992471011217e-05, 0.4060308170878505, 0.08549071125823456]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [4.988188933976653e-06, 0.45763905185531856, 0.8880713554408661]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [1.7462646295371234e-05, 0.4303965141295527, 0.47308910649296276]
Sample [5.4 3.9 1.7 0.4] expected [1. 0. 0.] produced [0.9848648625359767, 0.1987027632025617, 3.2525224140967528e-12]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.036548013020847266, 0.30165888957328135, 1.3802625703515105e-06]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [6.726113092716619e-06, 0.4565827443145375, 0.8227856487739428]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.984444114983758, 0.19907298854049074, 3.4263079129725812e-12]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [1.6399290391282736e-05, 0.4351971560593728, 0.4970925470909452]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [7.346468382357349e-06, 0.45499240257958334, 0.7970074432345736]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [3.022006715946548e-05, 0.42482578946435245, 0.25197855311564016]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.9836597733348575, 0.1979166523229795, 3.753617042577307e-12]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [8.77869345460622e-06, 0.4431542574537639, 0.7470190969254518]
Epoch 10000 RMSE =  0.29097837974132806
Final Epoch RMSE =  0.29097837974132806

Test Results:
Input: [6.1, 2.8, 4.7, 1.2, 5.5, 3.5, 1.3, 0.2, 4.3, 3.0, 1.1, 0.1, 7.9, 3.8, 6.4, 2.0, 5.0, 3.2, 1.2, 0.2, 4.8, 3.1, 1.6, 0.2, 6.4, 2.9, 4.3, 1.3, 5.8, 4.0, 1.2, 0.2, 6.3, 2.7, 4.9, 1.8, 5.8, 2.8, 5.1, 2.4, 6.4, 2.8, 5.6, 2.1, 6.2, 2.9, 4.3, 1.3, 5.6, 2.8, 4.9, 2.0, 6.8, 3.0, 5.5, 2.1, 5.2, 2.7, 3.9, 1.4, 6.7, 3.0, 5.0, 1.7, 4.9, 3.0, 1.4, 0.2, 6.9, 3.1, 4.9, 1.5, 5.7, 3.0, 4.2, 1.2, 4.4, 3.2, 1.3, 0.2, 7.2, 3.6, 6.1, 2.5, 6.5, 3.0, 5.2, 2.0, 6.9, 3.1, 5.4, 2.1, 6.3, 3.3, 6.0, 2.5, 5.7, 3.8, 1.7, 0.3, 5.8, 2.6, 4.0, 1.2, 5.2, 3.4, 1.4, 0.2, 5.0, 3.5, 1.6, 0.6, 7.7, 2.8, 6.7, 2.0, 4.6, 3.6, 1.0, 0.2, 5.5, 2.3, 4.0, 1.3, 5.1, 3.8, 1.6, 0.2, 5.7, 2.8, 4.5, 1.3, 5.0, 3.4, 1.5, 0.2, 6.1, 3.0, 4.9, 1.8, 5.7, 2.5, 5.0, 2.0, 6.1, 2.9, 4.7, 1.4, 5.6, 3.0, 4.1, 1.3, 6.7, 3.1, 5.6, 2.4, 6.1, 3.0, 4.6, 1.4, 5.2, 4.1, 1.5, 0.1, 5.0, 2.3, 3.3, 1.0, 4.6, 3.1, 1.5, 0.2, 5.4, 3.9, 1.3, 0.4, 5.1, 3.8, 1.5, 0.3]
Expected: [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
Output: [3.060012269326321e-12, 3.0731922515109183e-12, 3.191689919264339e-12, 3.206070430227265e-12, 3.221367502491394e-12, 3.2237193491971767e-12, 3.2509617066399544e-12, 3.3186042053378506e-12, 3.3726962353553786e-12, 3.4187767126698114e-12, 3.4314817916133835e-12, 3.487041874658193e-12, 3.5124639538276272e-12, 3.637356952770311e-12, 4.095273622992135e-12, 4.303040739561229e-12, 4.5325440463285285e-12, 4.775609629771913e-06, 4.811070489904519e-06, 4.92059995230596e-06, 4.931416710738387e-06, 5.124355244918531e-06, 5.299420757301267e-06, 5.340000642010402e-06, 5.432926403881734e-06, 6.057766156676608e-06, 7.114146069683144e-06, 7.560282878495872e-06, 8.59420017847215e-06, 9.915528300173757e-06, 1.2114225836927937e-05, 1.4373804990351737e-05, 2.2685164005179118e-05, 2.3032792859236794e-05, 2.3567852115832156e-05, 3.0141144199799218e-05, 3.895923179803153e-05, 8.181682968750242e-05, 0.00010600497779061666, 0.00011988995843190557, 0.00016068823552631045, 0.0002444909942199148, 0.0004165298700417892, 0.0006349224062419415, 0.0006511476422560624, 0.0010988504081746748, 0.001125916693156445, 0.002479978100628907, 0.0035759257031604407, 0.00752021577139956, 0.008968948308773272, 0.030476950606443724, 0.037562256683610455, 0.057373180448680894, 0.1818262873897445, 0.18922321110697554, 0.19007405512178321, 0.1910217734654506, 0.19214338476103207, 0.19323174441880347, 0.19362974383522508, 0.19364173737619628, 0.19375491248680227, 0.19474882643624705, 0.19514535294202343, 0.1951615230319728, 0.1953958746454252, 0.19563317392884041, 0.1968428517394404, 0.19723146969664274, 0.19754334863288722, 0.19796450415204547, 0.25906698969045083, 0.32248335344606427, 0.33836071301285164, 0.3473499394162242, 0.3499161437938141, 0.35052185395187235, 0.35687432868731045, 0.36034592552791744, 0.3775774703949945, 0.3851145536804704, 0.3865793275779838, 0.3910579035672823, 0.40011697507108046, 0.4045075016669342, 0.40978945577569487, 0.42165605381895077, 0.42259268420361173, 0.42927786071783325, 0.4381090103632876, 0.4391982982923428, 0.43954409395236216, 0.44117338251065974, 0.441795509472138, 0.4430480340826679, 0.44310290896520804, 0.44350235198422505, 0.4438748995755671, 0.44523815148303353, 0.4463646823567864, 0.4485623655778134, 0.45124695982543894, 0.5574563561270286, 0.6282448059786959, 0.7026980239893575, 0.753071027312627, 0.7926749285875211, 0.8101234268791948, 0.8474397579886986, 0.8729380082145209, 0.8757965113615049, 0.8778865739854081, 0.8843944928118134, 0.8903499414408307, 0.8904331901473374, 0.8940426105996404, 0.895551166419678, 0.9818557802564746, 0.9823729919082314, 0.9828951337573275, 0.9839628941482169, 0.9842633833963627, 0.9842854531262133, 0.9844279961847997, 0.9844828690925619, 0.9846483142982327, 0.9847980121700408, 0.9849567324927021, 0.985019096554075, 0.9850642126166311, 0.9850648343714051, 0.9850903964004852, 0.9854421502242846, 0.9854519561931968]
Test RMSE =  0.17769261138394016





-- run_iris() Sample Run #2: hidden layer of three neurodes, 1001 epochs, 90% training factor, and is randomly ordered --
Sample [5.5 2.5 4.  1.3] expected [0. 1. 0.] produced [0.854929989709539, 0.7282250591941475, 0.7469204849362673]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.8459704178051306, 0.7267205413782565, 0.735137500705031]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [0.8538765912383722, 0.7259343908378675, 0.7395000485163953]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [0.8518341157445594, 0.7215815466314232, 0.7409078356001628]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.8422332531368938, 0.7138803484847905, 0.7340730396192068]
Sample [7.9 3.8 6.4 2. ] expected [0. 0. 1.] produced [0.8502172313025771, 0.7128712244438997, 0.7383468660762893]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [0.8481096723760978, 0.7083579325554845, 0.7397788534762841]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [0.8459444304865233, 0.7037842104558975, 0.7411762043288388]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.8367943852040617, 0.6964738145860145, 0.7350331991167254]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [0.844254044004297, 0.6946394235820126, 0.7386209222024253]
Sample [5.5 2.3 4.  1.3] expected [0. 1. 0.] produced [0.8414642993276867, 0.6896820554676609, 0.7394549763482633]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [0.839669313356796, 0.6919925264481988, 0.7358150340009867]
Sample [6.5 3.  5.5 1.8] expected [0. 0. 1.] produced [0.8374621929867732, 0.6941259308914884, 0.7317166474473035]
Sample [4.4 2.9 1.4 0.2] expected [1. 0. 0.] produced [0.8256247446044839, 0.6859077563883259, 0.7232569896420102]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [0.835524103855561, 0.684751973795467, 0.7290240911364997]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [0.8330876165165718, 0.6869421530070726, 0.724672459638676]
Sample [7.2 3.6 6.1 2.5] expected [0. 0. 1.] produced [0.8308925577827397, 0.6891893170906883, 0.7205746118013359]
Sample [4.4 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.8185018997018182, 0.6809530018382974, 0.7119950220912652]
Sample [6.9 3.1 5.1 2.3] expected [0. 0. 1.] produced [0.8289031223940194, 0.6797420567493515, 0.7179197285202982]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.8188512650814826, 0.6727766532078472, 0.7117020306442109]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.8184464772506355, 0.6676773336270859, 0.7064531008795975]
Sample [5.6 2.8 4.9 2. ] expected [0. 0. 1.] produced [0.8274117793764795, 0.6652661731316092, 0.710843660660916]
Sample [6.9 3.1 5.4 2.1] expected [0. 0. 1.] produced [0.8249874983033119, 0.6603622616882299, 0.7127818051676115]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [0.8223785948446904, 0.6553544353980846, 0.7145697398425039]
Sample [6.2 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.8194352934307587, 0.6502561530446067, 0.7160357028859012]
Sample [5.5 3.5 1.3 0.2] expected [1. 0. 0.] produced [0.8092038674476284, 0.6515167576908932, 0.7036863808626087]
Sample [6.5 3.  5.2 2. ] expected [0. 0. 1.] produced [0.8176572176656318, 0.6481044015438503, 0.7074458595123272]
Sample [5.2 4.1 1.5 0.1] expected [1. 0. 0.] produced [0.8083468825511878, 0.6418040946427653, 0.7025103475995378]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [0.8156554135466565, 0.6380664462030793, 0.7049335786846446]
Sample [4.8 3.4 1.6 0.2] expected [1. 0. 0.] produced [0.8055432763564316, 0.6315854680750875, 0.6993786001465125]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.8121581251028046, 0.6275944137353623, 0.7009895040466468]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.8049031214008554, 0.6298569026808027, 0.6919049644570284]
Sample [6.1 2.9 4.7 1.4] expected [0. 1. 0.] produced [0.8111727662118379, 0.6258568451641122, 0.6929071978746962]
Sample [5.  3.4 1.5 0.2] expected [1. 0. 0.] produced [0.800761580251687, 0.6277536582933995, 0.6806313666730366]
Sample [5.1 3.8 1.5 0.3] expected [1. 0. 0.] produced [0.802470076650839, 0.6229983067078855, 0.6769636988589919]
Sample [5.4 3.4 1.5 0.4] expected [1. 0. 0.] produced [0.8032717076109884, 0.6181655378952291, 0.6723147539785486]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.803721950072525, 0.6131803497652225, 0.6673319815053407]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [0.8110400139490102, 0.608673175702381, 0.6691737339186152]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [0.8083691721662719, 0.6120430641555822, 0.6644206760635316]
Sample [5.2 3.4 1.4 0.2] expected [1. 0. 0.] produced [0.7972214334264359, 0.6062257200909955, 0.6588687779243022]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.8056679070946372, 0.6017464652051668, 0.6615882925917903]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [0.8031667815848812, 0.6052136488866899, 0.657054622488408]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [0.8003860355450709, 0.6086178950684308, 0.652263010614243]
Sample [7.7 2.8 6.7 2. ] expected [0. 0. 1.] produced [0.7972952139925779, 0.6034260780562655, 0.6549327341803847]
Sample [6.4 3.2 5.3 2.3] expected [0. 0. 1.] produced [0.7940751318985411, 0.598219073675948, 0.6574997287563904]
Sample [6.4 3.1 5.5 1.8] expected [0. 0. 1.] produced [0.7908636600393133, 0.593027675024399, 0.6600888010952168]
Sample [6.7 3.1 5.6 2.4] expected [0. 0. 1.] produced [0.7876453735601952, 0.5878451701771408, 0.6626876560927838]
Sample [5.7 3.  4.2 1.2] expected [0. 1. 0.] produced [0.7838477623597325, 0.5826266738496616, 0.6647682572458774]
Sample [5.6 3.  4.1 1.3] expected [0. 1. 0.] produced [0.7804328498565452, 0.5863043812925205, 0.6597872959710571]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.7769755712463806, 0.5899413786319285, 0.6547869317186804]
Sample [6.4 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.773632664158525, 0.5935578402648222, 0.6499067049748677]
Sample [5.5 2.6 4.4 1.2] expected [0. 1. 0.] produced [0.7699649738410398, 0.5970490201813814, 0.644789782107025]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [0.7667208389429162, 0.6005928941271177, 0.6400477538171345]
Sample [5.8 2.6 4.  1.2] expected [0. 1. 0.] produced [0.7624745532081592, 0.5953520429016229, 0.6424119345064879]
Sample [7.2 3.  5.8 1.6] expected [0. 0. 1.] produced [0.7593275733420135, 0.5989212219981518, 0.6378716389362216]
Sample [5.9 3.  4.2 1.5] expected [0. 1. 0.] produced [0.7551292473670088, 0.5936884598048935, 0.6404290106524461]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [0.7516688475665355, 0.5972685690410854, 0.6357053787644893]
Sample [5.  3.2 1.2 0.2] expected [1. 0. 0.] produced [0.7368969009175644, 0.5916496288162604, 0.6287107320914257]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.7420089366708852, 0.5869984807582678, 0.6271518380711725]
Sample [7.4 2.8 6.1 1.9] expected [0. 0. 1.] produced [0.7504593533705988, 0.5819769849774933, 0.6286697143341881]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.746220520466146, 0.5768014696434902, 0.6314539854730341]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [0.7423110210516481, 0.5805654053744198, 0.6264302309345254]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.727844441729288, 0.5753078975449851, 0.6202035880400861]
Sample [6.  3.4 4.5 1.6] expected [0. 1. 0.] produced [0.7396276171983563, 0.570402851392821, 0.6244412459742882]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.7278923110804627, 0.5743982046505629, 0.6124995960450511]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.7370257236636394, 0.5692389652076032, 0.6142466276347541]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.73298151315416, 0.5730979924975718, 0.6091858741466771]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [0.7288879015338825, 0.5679659869334643, 0.612603942772955]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [0.7242599775728139, 0.5628371045124793, 0.6155770167482615]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [0.7202560015029853, 0.5667886856957042, 0.61069747084286]
Sample [7.7 2.6 6.9 2.3] expected [0. 0. 1.] produced [0.7158938385560945, 0.5616605690008483, 0.6140275748576598]
Sample [4.6 3.1 1.5 0.2] expected [1. 0. 0.] produced [0.7022186003411596, 0.5569010770952381, 0.609119320487553]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [0.713215282411167, 0.5516015425427279, 0.6122083317435506]
Sample [4.9 3.  1.4 0.2] expected [1. 0. 0.] produced [0.6991233110039669, 0.5474024426391932, 0.6068768669734684]
Sample [6.  2.2 4.  1. ] expected [0. 1. 0.] produced [0.7099186827631836, 0.541718230173758, 0.6098677547372808]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.6959078477713365, 0.5467656671712863, 0.5964061907592975]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.7006837760179134, 0.5418770677191512, 0.5938739347866694]
Sample [5.8 2.8 5.1 2.4] expected [0. 0. 1.] produced [0.7098793518508609, 0.5360812155043948, 0.5952291014334072]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.6962730813481568, 0.53235968673796, 0.5909115948446154]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.7017872829202593, 0.5272866404033402, 0.5889182788624595]
Sample [5.6 3.  4.5 1.5] expected [0. 1. 0.] produced [0.7089563236993695, 0.5214959567097481, 0.5884712771894974]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.7046057707823377, 0.5259331644916165, 0.5834452751960627]
Sample [6.1 3.  4.9 1.8] expected [0. 0. 1.] produced [0.6999926252555911, 0.5210415916939733, 0.5871041662091077]
Sample [5.1 3.3 1.7 0.5] expected [1. 0. 0.] produced [0.6894498644784203, 0.5172717705364669, 0.5856650565869757]
Sample [5.4 3.9 1.3 0.4] expected [1. 0. 0.] produced [0.6900817586775381, 0.5130294580875264, 0.5793643073400693]
Sample [6.9 3.2 5.7 2.3] expected [0. 0. 1.] produced [0.6996057357330636, 0.5067201057024301, 0.5807576513519135]
Sample [6.  2.9 4.5 1.5] expected [0. 1. 0.] produced [0.694663776293164, 0.5020305786329128, 0.5842445626026115]
Sample [4.9 2.5 4.5 1.7] expected [0. 0. 1.] produced [0.6898929719660497, 0.5066647090901623, 0.5790509054293851]
Sample [6.1 2.6 5.6 1.4] expected [0. 0. 1.] produced [0.6854262430660265, 0.5019102017380227, 0.5829830941840625]
Sample [5.7 3.8 1.7 0.3] expected [1. 0. 0.] produced [0.6753081998701365, 0.49866629115570776, 0.5819438006114062]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.6824696486686069, 0.49273422301855424, 0.5812258129845086]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.6776979341445767, 0.49744417037551475, 0.5761187334972637]
Sample [5.  3.4 1.6 0.4] expected [1. 0. 0.] produced [0.6666839892727475, 0.5035906266397061, 0.5658045822657131]
Sample [6.4 2.8 5.6 2.1] expected [0. 0. 1.] produced [0.6756820742733232, 0.4974305754693463, 0.5663324318979487]
Sample [4.6 3.6 1.  0.2] expected [1. 0. 0.] produced [0.6589910618043081, 0.49577880871726154, 0.5604377167147823]
Sample [5.1 3.5 1.4 0.3] expected [1. 0. 0.] produced [0.6653862247965643, 0.49053873503150736, 0.5588640617881204]
Sample [6.1 3.  4.6 1.4] expected [0. 1. 0.] produced [0.675398899257291, 0.48389340797578567, 0.5602314881369215]
Sample [6.7 3.  5.  1.7] expected [0. 1. 0.] produced [0.6706631990735016, 0.4886788891234196, 0.5552411744192932]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.6643691218593791, 0.4937608416050617, 0.5491676122261148]
Sample [6.7 3.3 5.7 2.1] expected [0. 0. 1.] produced [0.660913795387412, 0.4981499877100666, 0.5452226442083423]
Sample [5.4 3.4 1.7 0.2] expected [1. 0. 0.] produced [0.649776482323123, 0.49524747463226926, 0.5443998477958739]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.6579877308661126, 0.4890944385816571, 0.5440658515531193]
Sample [4.3 3.  1.1 0.1] expected [1. 0. 0.] produced [0.6406932998239405, 0.49690456482888873, 0.529814767327245]
Sample [5.4 3.9 1.7 0.4] expected [1. 0. 0.] produced [0.6511283146853603, 0.4907640456411543, 0.5308343077944094]
Sample [6.8 3.  5.5 2.1] expected [0. 0. 1.] produced [0.6588496897602278, 0.4847518110318608, 0.5299159241759082]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.6536946702376033, 0.48027254937591934, 0.5341684141661235]
Sample [5.1 3.8 1.6 0.2] expected [1. 0. 0.] produced [0.6424735886334307, 0.48699598524322774, 0.5245240016419411]
Sample [6.7 2.5 5.8 1.8] expected [0. 0. 1.] produced [0.651548662958051, 0.48061679131215485, 0.5245450216333559]
Sample [6.1 2.8 4.7 1.2] expected [0. 1. 0.] produced [0.6462663797702289, 0.4762040522988481, 0.5288084530342796]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.6407006877405396, 0.48122644415533505, 0.5235350692873533]
Sample [5.7 2.5 5.  2. ] expected [0. 0. 1.] produced [0.6362368118894893, 0.485866931891104, 0.5191231696644857]
Sample [5.1 3.8 1.9 0.4] expected [1. 0. 0.] produced [0.6268590886089854, 0.48270151461310074, 0.5203414132751015]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.633969703102569, 0.4769515141351167, 0.5187059197741098]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [0.6289377846773444, 0.48179547046146054, 0.5139341777739505]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.6169462254972484, 0.4888247693676209, 0.5040877510846217]
Sample [6.9 3.1 4.9 1.5] expected [0. 1. 0.] produced [0.6269366509633489, 0.4821608841842778, 0.5045587223638847]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [0.621904020395657, 0.4869537361166383, 0.49992384926417516]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.6165837640887795, 0.4824498308084026, 0.5045000992064614]
Sample [4.8 3.1 1.6 0.2] expected [1. 0. 0.] produced [0.6046902813752958, 0.489409303791105, 0.49500453423794616]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.6139118491784538, 0.4830850455401364, 0.4947438964536355]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [0.6093525784422517, 0.4877024060946708, 0.4905329874969814]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [0.6044043923231113, 0.4924022192170567, 0.4861123423686413]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.5990761026967835, 0.48785411535368195, 0.49080273279351194]
Sample [6.2 3.4 5.4 2.3] expected [0. 0. 1.] produced [0.5941001001535813, 0.4925643887730522, 0.4863694429147396]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.5881681151609209, 0.488205023620606, 0.49065821327727477]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.5771515790039968, 0.4949117259867022, 0.4817300364433253]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [0.5872954712405816, 0.48825292189236097, 0.48213950627374746]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.5815999730516389, 0.48387580453674256, 0.4866201635710143]
Sample [5.  3.  1.6 0.2] expected [1. 0. 0.] produced [0.5709686522520832, 0.49065744167796765, 0.47803875332395435]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [0.5806897800319216, 0.48403670947776006, 0.4780856583331202]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.5668296688368317, 0.4828773368321176, 0.4767300824101839]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.5710421401980157, 0.4783356156313623, 0.4730458669135202]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.5817406327714882, 0.4713634020203462, 0.47365013634074116]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.5711851926075215, 0.47845637747562564, 0.465464103417923]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [0.5816451455492873, 0.4714017145847414, 0.4658936951592105]
Epoch 0 RMSE =  0.5719589260654753
Epoch 100 RMSE =  0.3324275923761881
Epoch 200 RMSE =  0.31361423887133005
Epoch 300 RMSE =  0.30535262495276827
Epoch 400 RMSE =  0.30253834558594384
Epoch 500 RMSE =  0.29811941723815466
Epoch 600 RMSE =  0.3002913279502846
Epoch 700 RMSE =  0.29641426154038836
Epoch 800 RMSE =  0.29488499320148537
Epoch 900 RMSE =  0.2957230322295651
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [0.0022111058009760165, 0.35071792940311247, 0.18085243221722577]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.9288027081214872, 0.2741170499667209, 2.514691237969447e-05]
Sample [6.9 3.2 5.7 2.3] expected [0. 0. 1.] produced [9.641953517894024e-05, 0.3853465676611288, 0.853771091499177]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [0.00011943339678054771, 0.3800207999932439, 0.823940866544219]
Sample [5.1 3.8 1.9 0.4] expected [1. 0. 0.] produced [0.9264850180968652, 0.26976459303170425, 2.621530056353191e-05]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.9282832155283213, 0.2685041044840969, 2.551321955277381e-05]
Sample [6.8 3.  5.5 2.1] expected [0. 0. 1.] produced [0.00011302454545253514, 0.3750538414211753, 0.8328125659155508]
Sample [5.6 3.  4.1 1.3] expected [0. 1. 0.] produced [0.02144378734259896, 0.32163513483880424, 0.019838695718059376]
Sample [4.4 2.9 1.4 0.2] expected [1. 0. 0.] produced [0.9234855456865689, 0.2693302528303619, 2.7656691655280586e-05]
Sample [4.6 3.6 1.  0.2] expected [1. 0. 0.] produced [0.9281645448876518, 0.26767946532935977, 2.5720924466547372e-05]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.9266856846579742, 0.2668418867126965, 2.6324185805694934e-05]
Sample [6.7 2.5 5.8 1.8] expected [0. 0. 1.] produced [8.868071551124689e-05, 0.37513091818240196, 0.8657973272451227]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.1670773895948938, 0.29948620060262615, 0.0020012393199115913]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [7.352616121972266e-05, 0.377959677529368, 0.8869319795321282]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [0.00013136134431280668, 0.36915526901189577, 0.8106776130185849]
Sample [5.4 3.4 1.7 0.2] expected [1. 0. 0.] produced [0.9267695039851168, 0.263489658344722, 2.6315755784573117e-05]
Sample [7.4 2.8 6.1 1.9] expected [0. 0. 1.] produced [8.584580496873788e-05, 0.36923138767916924, 0.8703597472985978]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [7.999382196913594e-05, 0.3669715572058328, 0.8786991153410488]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [0.00011221518454295265, 0.36083954264638934, 0.8358992813574488]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.006319826400055655, 0.3211409610083841, 0.06960222962463423]
Sample [6.7 3.1 5.6 2.4] expected [0. 0. 1.] produced [6.991957684737176e-05, 0.3669872927772089, 0.8933697494446969]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [0.00013314873872528093, 0.3579783799893812, 0.8104735857836916]
Sample [6.7 3.3 5.7 2.1] expected [0. 0. 1.] produced [8.562021635164493e-05, 0.35937713422269146, 0.8720697807408994]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [6.854213386904261e-05, 0.35863611492138425, 0.896073252755651]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [0.0010543706700037482, 0.33123626673474305, 0.33052034839665834]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.92554361851194, 0.2570534045976634, 2.694831477570681e-05]
Sample [6.5 3.  5.5 1.8] expected [0. 0. 1.] produced [0.00016614002455812198, 0.35120210569497873, 0.7720117781560947]
Sample [6.1 2.9 4.7 1.4] expected [0. 1. 0.] produced [0.0017619713594147865, 0.3274530207086094, 0.22314987308984965]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [0.00012983376507938932, 0.35539301986931104, 0.814312527963209]
Sample [5.5 2.3 4.  1.3] expected [0. 1. 0.] produced [0.001955355501534628, 0.32825511764725557, 0.20465844734111863]
Sample [6.9 3.1 5.1 2.3] expected [0. 0. 1.] produced [0.00016276834800423547, 0.35513818520141555, 0.7757875643914549]
Sample [5.7 2.5 5.  2. ] expected [0. 0. 1.] produced [8.749633507148726e-05, 0.35822817141811797, 0.8695090630500253]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.9259574197412477, 0.25554349468269955, 2.6958402678147535e-05]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.12943744781206276, 0.2896101805809622, 0.002772764962694078]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.039293693074397534, 0.3036676409076222, 0.010602931450885908]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.01475407364998709, 0.31584753369388496, 0.02974927757759337]
Sample [5.6 2.8 4.9 2. ] expected [0. 0. 1.] produced [9.251925984990011e-05, 0.3662399080601112, 0.862802596202241]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [9.226235362655701e-05, 0.3633810522932044, 0.8634253254632945]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [0.00027329973042980644, 0.3505504027192759, 0.6702997378557585]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [0.00017056043791783006, 0.35962451945989715, 0.7653773555312877]
Sample [6.4 3.2 5.3 2.3] expected [0. 0. 1.] produced [0.00011713869209690472, 0.360411876010123, 0.8294206087902551]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [0.0001181701718326568, 0.35756637238504524, 0.8286143276697376]
Sample [7.7 2.6 6.9 2.3] expected [0. 0. 1.] produced [6.586142259777071e-05, 0.36017690324522444, 0.8994534317729598]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.00011586809370522622, 0.352238235788923, 0.8322064051587987]
Sample [4.9 3.  1.4 0.2] expected [1. 0. 0.] produced [0.9263392521422633, 0.2547279287629884, 2.6789549079508244e-05]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.9279983372922165, 0.25359825750806364, 2.6079027155365974e-05]
Sample [5.1 3.8 1.6 0.2] expected [1. 0. 0.] produced [0.9282871214139955, 0.25265779081529216, 2.5968609646798423e-05]
Sample [5.7 3.  4.2 1.2] expected [0. 1. 0.] produced [0.06811205600174697, 0.29170879063411537, 0.005816072822742744]
Sample [6.9 3.1 5.4 2.1] expected [0. 0. 1.] produced [0.00015137756944949432, 0.3479201607499086, 0.7901483534470426]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.2920468804290345, 0.2792610142159297, 0.0009586936153727318]
Sample [5.4 3.4 1.5 0.4] expected [1. 0. 0.] produced [0.9253180183577697, 0.25606025271572874, 2.7105005783857892e-05]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.9224030159001451, 0.25549137337346217, 2.838690800497782e-05]
Sample [5.1 3.3 1.7 0.5] expected [1. 0. 0.] produced [0.9203609862463976, 0.2547427690627787, 2.9196156415264763e-05]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [0.00039993063170373327, 0.3371665483660275, 0.5764938076524748]
Sample [7.2 3.  5.8 1.6] expected [0. 0. 1.] produced [0.0005953130091442308, 0.33812275182276197, 0.4682296537867762]
Sample [5.  3.2 1.2 0.2] expected [1. 0. 0.] produced [0.9259185712909873, 0.25465180680868815, 2.6930506044340774e-05]
Sample [7.9 3.8 6.4 2. ] expected [0. 0. 1.] produced [0.00010914715064365326, 0.34986780565870657, 0.8411659363272752]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [7.691347349676935e-05, 0.35035840343018904, 0.884572051220676]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.9264763253772228, 0.2506504645386825, 2.6776215862496107e-05]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.921785583019976, 0.2503423936777848, 2.885237660458779e-05]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.0006755553151861054, 0.32673537681086784, 0.4413619263212932]
Sample [6.1 3.  4.6 1.4] expected [0. 1. 0.] produced [0.00293853751706376, 0.3183637316816884, 0.14295825909275947]
Sample [5.4 3.9 1.7 0.4] expected [1. 0. 0.] produced [0.9275017702509402, 0.25433967757416936, 2.602138603247992e-05]
Sample [5.8 2.8 5.1 2.4] expected [0. 0. 1.] produced [6.96243266401283e-05, 0.3545398846892497, 0.8934571079660212]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [7.879664900293414e-05, 0.3506724710613242, 0.8806400977736498]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.9143013356300908, 0.2518822568915004, 3.1779612622789264e-05]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.04936991045299238, 0.2919077204637065, 0.008236435631347553]
Sample [6.1 3.  4.9 1.8] expected [0. 0. 1.] produced [0.00022063679271566117, 0.3417975513776169, 0.7156850446262429]
Sample [6.  2.9 4.5 1.5] expected [0. 1. 0.] produced [0.0008088699622348097, 0.32809751748322447, 0.39434387518195546]
Sample [6.7 3.  5.  1.7] expected [0. 1. 0.] produced [0.0012374151191444015, 0.3286649241912165, 0.2917689363978094]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.9274633710959258, 0.2564911247160205, 2.5951690287882274e-05]
Sample [4.8 3.4 1.6 0.2] expected [1. 0. 0.] produced [0.9276531325782283, 0.2555135421211518, 2.585350607756731e-05]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.24761616441410467, 0.28331164005697573, 0.0011890439035010625]
Sample [5.4 3.9 1.3 0.4] expected [1. 0. 0.] produced [0.9281982255564044, 0.2572359978686888, 2.548445202756e-05]
Sample [5.1 3.5 1.4 0.3] expected [1. 0. 0.] produced [0.9275020315591772, 0.2563983695008018, 2.5797554559521768e-05]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [0.000677256701380023, 0.3374927100893587, 0.4337113079768178]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.375964440560296, 0.28275437100141165, 0.0006220873573749625]
Sample [6.4 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.08020576544753592, 0.3020858786015728, 0.004632000245298598]
Sample [5.8 2.6 4.  1.2] expected [0. 1. 0.] produced [0.027784507947694687, 0.31524709964430664, 0.014714027853070402]
Sample [5.9 3.  4.2 1.5] expected [0. 1. 0.] produced [0.0076316088860076845, 0.33073293472597703, 0.05563572973366675]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.9253088403682579, 0.26995124932016695, 2.6183630847497636e-05]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.927461677872655, 0.2686101123649033, 2.528782898656826e-05]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.9280726012282827, 0.2674969934891716, 2.5051634763486244e-05]
Sample [5.1 3.8 1.5 0.3] expected [1. 0. 0.] produced [0.9271357040192085, 0.2666012192940069, 2.5449710531471635e-05]
Sample [6.2 3.4 5.4 2.3] expected [0. 0. 1.] produced [9.620308853140579e-05, 0.3715133163294414, 0.852104881649445]
Sample [4.6 3.1 1.5 0.2] expected [1. 0. 0.] produced [0.9243390295749779, 0.2642778228928422, 2.668950036802482e-05]
Sample [4.3 3.  1.1 0.1] expected [1. 0. 0.] produced [0.9256236352536701, 0.2632087243597308, 2.6296593648904144e-05]
Sample [7.2 3.6 6.1 2.5] expected [0. 0. 1.] produced [8.240834889017465e-05, 0.367673189590835, 0.8717888060369285]
Sample [6.1 2.6 5.6 1.4] expected [0. 0. 1.] produced [0.00016935659950669687, 0.3579367888487874, 0.76235509395179]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.03468984634332527, 0.3070309392429926, 0.011752385152939145]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [0.001185492151056238, 0.3412298437642495, 0.2963424096115654]
Sample [4.4 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.9256568344321504, 0.26464896123895776, 2.621301433040527e-05]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.10410479672720059, 0.3028229531532796, 0.0034493710152765367]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.008918028121530286, 0.32929070200708105, 0.047847935244600945]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [8.350222407672618e-05, 0.3781340259090973, 0.8700982693076597]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.08379009330958499, 0.30955844036374147, 0.004429757187514822]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.92701967464312, 0.27034225733573497, 2.5587567578813566e-05]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.002895563829044439, 0.3436641800890283, 0.14091921829977694]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [7.469479950116574e-05, 0.3837091063458073, 0.8826761916482873]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [0.00020008980958987024, 0.3708405654988269, 0.7289406422953258]
Sample [6.2 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.11591866594970023, 0.3135464289730202, 0.0029901163158420657]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [0.0003489398890965322, 0.3741237580842501, 0.595328560250574]
Sample [6.4 3.1 5.5 1.8] expected [0. 0. 1.] produced [0.0001350931738214194, 0.3807745886363117, 0.8008784810262792]
Sample [5.  3.  1.6 0.2] expected [1. 0. 0.] produced [0.9233137493746574, 0.2731945467903526, 2.7002892441962666e-05]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.0046299747722505944, 0.342215093957915, 0.09083097672239364]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9252483683909017, 0.2748454339899824, 2.6241127663133786e-05]
Sample [5.5 2.6 4.4 1.2] expected [0. 1. 0.] produced [0.0012223311178480932, 0.3579410415999304, 0.2874866678985419]
Sample [6.  3.4 4.5 1.6] expected [0. 1. 0.] produced [0.007987704040988653, 0.34383545118960485, 0.052897044978674555]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [0.0005185076079316157, 0.375167694956281, 0.4954018386936211]
Sample [5.2 3.4 1.4 0.2] expected [1. 0. 0.] produced [0.9258235121410491, 0.277814535902655, 2.616346399601743e-05]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.00148455730571898, 0.3606155903926318, 0.2493233405374375]
Sample [5.5 2.5 4.  1.3] expected [0. 1. 0.] produced [0.001255976400229223, 0.3666429806249898, 0.28235948013379963]
Sample [4.9 2.5 4.5 1.7] expected [0. 0. 1.] produced [0.000115747538695841, 0.3957654010301578, 0.8255097708269313]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [7.325699894524208e-05, 0.3972445146035396, 0.884504922750097]
Sample [4.8 3.1 1.6 0.2] expected [1. 0. 0.] produced [0.9242138245789043, 0.2790295327168618, 2.6673868439001277e-05]
Sample [5.5 3.5 1.3 0.2] expected [1. 0. 0.] produced [0.9275854685181288, 0.27744830905858425, 2.5306906466840328e-05]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [0.00021155399727804748, 0.38037989685716017, 0.7169711771343106]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.9262737448801694, 0.27472638020394774, 2.600330060267076e-05]
Sample [6.4 2.8 5.6 2.1] expected [0. 0. 1.] produced [7.289748837510189e-05, 0.3869026578436317, 0.8862072489558881]
Sample [6.  2.2 4.  1. ] expected [0. 1. 0.] produced [0.010291392067798142, 0.3352773000370947, 0.04172637250684338]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9253442850001536, 0.2749014435297467, 2.644380713616174e-05]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [9.352855376274354e-05, 0.3843666311584902, 0.8574036640366873]
Sample [5.  3.4 1.6 0.4] expected [1. 0. 0.] produced [0.9236379521143969, 0.2721878507047882, 2.7156489647158132e-05]
Sample [5.  3.4 1.5 0.2] expected [1. 0. 0.] produced [0.9260744485740868, 0.27083893534094444, 2.6202447597160688e-05]
Sample [6.1 2.8 4.7 1.2] expected [0. 1. 0.] produced [0.0021961945796525216, 0.3477240275759364, 0.18149588983663537]
Sample [6.5 3.  5.2 2. ] expected [0. 0. 1.] produced [0.0001206871446319967, 0.3807924646186894, 0.8216003597804435]
Sample [5.6 3.  4.5 1.5] expected [0. 1. 0.] produced [0.0007655068060629128, 0.359466753286152, 0.4008179508797709]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [9.26931940296649e-05, 0.3851935939031489, 0.8573907348303864]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [0.0006361575719752672, 0.3628496986513913, 0.4455694360731376]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.38964898032655315, 0.30126540568884114, 0.000576902430788666]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.9268011270513974, 0.27774703641809756, 2.5194065053095123e-05]
Sample [6.9 3.1 4.9 1.5] expected [0. 1. 0.] produced [0.02646215060948076, 0.33276520400198134, 0.015246169456267583]
Sample [5.2 4.1 1.5 0.1] expected [1. 0. 0.] produced [0.927786640226926, 0.2794641041594094, 2.479625041801524e-05]
Sample [7.7 2.8 6.7 2. ] expected [0. 0. 1.] produced [8.898111008337179e-05, 0.39282040484235037, 0.8598287633172788]
Sample [5.7 3.8 1.7 0.3] expected [1. 0. 0.] produced [0.927339072118612, 0.2765073391242043, 2.500910110298326e-05]
Epoch 1000 RMSE =  0.2944339805214937
Final Epoch RMSE =  0.2944339805214937

Test Results:
Input: [5.0, 3.3, 1.4, 0.2, 6.7, 3.1, 4.7, 1.5, 5.8, 4.0, 1.2, 0.2, 6.3, 2.7, 4.9, 1.8, 5.2, 2.7, 3.9, 1.4, 6.3, 3.3, 6.0, 2.5, 5.0, 2.3, 3.3, 1.0, 5.7, 2.8, 4.5, 1.3, 5.6, 2.5, 3.9, 1.1, 5.7, 2.9, 4.2, 1.3, 5.1, 3.7, 1.5, 0.4, 4.9, 3.1, 1.5, 0.1, 4.6, 3.2, 1.4, 0.2, 6.0, 3.0, 4.8, 1.8]
Expected: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
Output: [2.4776019155536388e-05, 2.537166561124758e-05, 2.5613447604648384e-05, 2.5872252064370826e-05, 2.6152795314712687e-05, 6.705611581019218e-05, 0.0002631180179701485, 0.00027398868461316224, 0.0021503849985933018, 0.0027059808943399916, 0.005826815919843812, 0.012416311924513644, 0.012901844554181748, 0.016573289905290146, 0.02453157639790754, 0.032367471925675566, 0.03248457368975764, 0.06509288278087567, 0.1483505881826575, 0.18183101302784255, 0.27553863383062027, 0.27718798039985615, 0.28538791222637516, 0.2864838666049742, 0.2875840759996845, 0.3225906643138982, 0.33094606490621603, 0.33698043591059484, 0.35006592511892165, 0.3549104975687809, 0.35782319838945625, 0.378847077344246, 0.39003530944119147, 0.3946402266219371, 0.6566708455428445, 0.6646694290251864, 0.89289188209593, 0.9250240740439082, 0.9256352363104356, 0.9261521916075618, 0.9265643970689571, 0.9280086982453776]
Test RMSE =  0.09659712960363329





-- run_sin() Sample Run #1: hidden layer of three neurodes, 10001 epochs, 10% training factor, and is randomly ordered --
Sample [1.1] expected [0.89120736] produced [0.5938982789803587]
Sample [0.32] expected [0.31456656] produced [0.5861311375635908]
Sample [0.49] expected [0.47062589] produced [0.5873851795098698]
Sample [0.5] expected [0.47942554] produced [0.5871889157611648]
Sample [1.55] expected [0.99978376] produced [0.5980242257180931]
Sample [1.13] expected [0.90441219] produced [0.5951426527698868]
Sample [1.27] expected [0.95510086] produced [0.5976546446374545]
Sample [0.86] expected [0.75784256] produced [0.5943019209201831]
Sample [0.67] expected [0.62098599] produced [0.5925848291966141]
Sample [0.44] expected [0.42593947] produced [0.5899437800489853]
Sample [0.16] expected [0.15931821] produced [0.5861780969722582]
Sample [1.36] expected [0.9778646] produced [0.5987531555215347]
Sample [0.46] expected [0.44394811] produced [0.5898379999086146]
Sample [0.08] expected [0.07991469] produced [0.5849298289826578]
Sample [0.02] expected [0.01999867] produced [0.5830713982877956]
Epoch 0 RMSE =  0.3210370009660702
Epoch 100 RMSE =  0.3176358546632305
Epoch 200 RMSE =  0.3150198185790094
Epoch 300 RMSE =  0.31209939186060226
Epoch 400 RMSE =  0.307870874408097
Epoch 500 RMSE =  0.300087840899046
Epoch 600 RMSE =  0.28382378789741713
Epoch 700 RMSE =  0.2559031805100752
Epoch 800 RMSE =  0.2238898990767785
Epoch 900 RMSE =  0.19579019094358918
Sample [1.36] expected [0.9778646] produced [0.7629691908258639]
Sample [0.02] expected [0.01999867] produced [0.3152216927856604]
Sample [1.1] expected [0.89120736] produced [0.7239057535914061]
Sample [0.46] expected [0.44394811] produced [0.5239571909029975]
Sample [0.44] expected [0.42593947] produced [0.5147015217822872]
Sample [0.49] expected [0.47062589] produced [0.5366105070835913]
Sample [0.5] expected [0.47942554] produced [0.5407179737428526]
Sample [0.32] expected [0.31456656] produced [0.45781436934184977]
Sample [0.08] expected [0.07991469] produced [0.3416370017786081]
Sample [1.55] expected [0.99978376] produced [0.7809361928930587]
Sample [0.16] expected [0.15931821] produced [0.3796605359441828]
Sample [1.13] expected [0.90441219] produced [0.7280487621214429]
Sample [0.86] expected [0.75784256] produced [0.6679660106196305]
Sample [1.27] expected [0.95510086] produced [0.7509762128278938]
Sample [0.67] expected [0.62098599] produced [0.6084974294157542]
Epoch 1000 RMSE =  0.17324079177034504
Epoch 1100 RMSE =  0.15535888291652689
Epoch 1200 RMSE =  0.14099694206165128
Epoch 1300 RMSE =  0.12924777540395752
Epoch 1400 RMSE =  0.11945583424229511
Epoch 1500 RMSE =  0.11116444234616954
Epoch 1600 RMSE =  0.10404160895475668
Epoch 1700 RMSE =  0.09784977215431727
Epoch 1800 RMSE =  0.09241467577541314
Epoch 1900 RMSE =  0.08760319866158126
Sample [0.32] expected [0.31456656] produced [0.3648472018551476]
Sample [1.27] expected [0.95510086] produced [0.8388386383822258]
Sample [0.44] expected [0.42593947] produced [0.46920074951041685]
Sample [0.5] expected [0.47942554] produced [0.5188417440551507]
Sample [0.49] expected [0.47062589] produced [0.5105666692495334]
Sample [0.16] expected [0.15931821] produced [0.2345041405625056]
Sample [1.1] expected [0.89120736] produced [0.8082702380611859]
Sample [1.13] expected [0.90441219] produced [0.8147801825841666]
Sample [0.67] expected [0.62098599] produced [0.6404619713920465]
Sample [0.02] expected [0.01999867] produced [0.1466519433500784]
Sample [0.86] expected [0.75784256] produced [0.7360338668722147]
Sample [0.08] expected [0.07991469] produced [0.18069461653755434]
Sample [1.36] expected [0.9778646] produced [0.8502785408320964]
Sample [1.55] expected [0.99978376] produced [0.8679523942591576]
Sample [0.46] expected [0.44394811] produced [0.4861365808581857]
Epoch 2000 RMSE =  0.08331507848035857
Epoch 2100 RMSE =  0.07946878427161831
Epoch 2200 RMSE =  0.07600141954926327
Epoch 2300 RMSE =  0.0728616278954446
Epoch 2400 RMSE =  0.07000653194912705
Epoch 2500 RMSE =  0.06740275796806743
Epoch 2600 RMSE =  0.06501997101527106
Epoch 2700 RMSE =  0.06283336446459256
Epoch 2800 RMSE =  0.06082207983397633
Epoch 2900 RMSE =  0.058967569205653096
Sample [0.86] expected [0.75784256] produced [0.7591067918937077]
Sample [0.46] expected [0.44394811] produced [0.46803600477698953]
Sample [0.67] expected [0.62098599] produced [0.6490580174228628]
Sample [0.02] expected [0.01999867] produced [0.10856918499966584]
Sample [1.1] expected [0.89120736] produced [0.8380494959257961]
Sample [1.27] expected [0.95510086] produced [0.8697423330999292]
Sample [0.44] expected [0.42593947] produced [0.44842971833878875]
Sample [1.36] expected [0.9778646] produced [0.8815305514425252]
Sample [0.08] expected [0.07991469] produced [0.13993306683310908]
Sample [1.13] expected [0.90441219] produced [0.8449739626752906]
Sample [0.5] expected [0.47942554] produced [0.5066786965105418]
Sample [0.32] expected [0.31456656] produced [0.33001026467492167]
Sample [0.16] expected [0.15931821] produced [0.19219996303969433]
Sample [0.49] expected [0.47062589] produced [0.49693007118574545]
Sample [1.55] expected [0.99978376] produced [0.8989412087179195]
Epoch 3000 RMSE =  0.05725451299924766
Epoch 3100 RMSE =  0.055668693980218405
Epoch 3200 RMSE =  0.05419855406637255
Epoch 3300 RMSE =  0.05283320385180187
Epoch 3400 RMSE =  0.051563892104284324
Epoch 3500 RMSE =  0.05038166045698134
Epoch 3600 RMSE =  0.04927932402938598
Epoch 3700 RMSE =  0.048250386329277625
Epoch 3800 RMSE =  0.04728848963915622
Epoch 3900 RMSE =  0.046388684144752654
Sample [1.27] expected [0.95510086] produced [0.886355309260565]
Sample [0.86] expected [0.75784256] produced [0.7716356279157117]
Sample [0.44] expected [0.42593947] produced [0.4371074369944694]
Sample [0.67] expected [0.62098599] produced [0.6536376846225714]
Sample [1.1] expected [0.89120736] produced [0.8541813929150667]
Sample [1.36] expected [0.9778646] produced [0.8981040357646636]
Sample [0.49] expected [0.47062589] produced [0.48924453552518415]
Sample [1.55] expected [0.99978376] produced [0.9153890039451892]
Sample [0.46] expected [0.44394811] produced [0.4581907210360395]
Sample [0.08] expected [0.07991469] produced [0.12240345546045879]
Sample [0.16] expected [0.15931821] produced [0.17285974990267983]
Sample [1.13] expected [0.90441219] produced [0.8611301103659753]
Sample [0.32] expected [0.31456656] produced [0.3122389929151117]
Sample [0.02] expected [0.01999867] produced [0.0930600945898265]
Sample [0.5] expected [0.47942554] produced [0.4994105048373188]
Epoch 4000 RMSE =  0.045545949632551756
Epoch 4100 RMSE =  0.044755920769309086
Epoch 4200 RMSE =  0.044014574934214915
Epoch 4300 RMSE =  0.043318156373913716
Epoch 4400 RMSE =  0.04266411823264981
Epoch 4500 RMSE =  0.04204847938541597
Epoch 4600 RMSE =  0.04146886643823504
Epoch 4700 RMSE =  0.0409225800066788
Epoch 4800 RMSE =  0.04040753949097107
Epoch 4900 RMSE =  0.039921811586134476
Sample [1.55] expected [0.99978376] produced [0.925424321771928]
Sample [0.08] expected [0.07991469] produced [0.11325480690850133]
Sample [0.02] expected [0.01999867] produced [0.08513325425661555]
Sample [1.1] expected [0.89120736] produced [0.8642500443341069]
Sample [0.86] expected [0.75784256] produced [0.7793420558220849]
Sample [0.46] expected [0.44394811] produced [0.4517902921382971]
Sample [0.67] expected [0.62098599] produced [0.6562043263508966]
Sample [0.16] expected [0.15931821] produced [0.16225714955145126]
Sample [0.44] expected [0.42593947] produced [0.42978541003390947]
Sample [1.36] expected [0.9778646] produced [0.908254444300128]
Sample [0.49] expected [0.47062589] produced [0.4840181282525701]
Sample [1.27] expected [0.95510086] produced [0.896544594317596]
Sample [0.5] expected [0.47942554] produced [0.4947229257113594]
Sample [1.13] expected [0.90441219] produced [0.8711198659091568]
Sample [0.32] expected [0.31456656] produced [0.30167654108460995]
Epoch 5000 RMSE =  0.03946297230219772
Epoch 5100 RMSE =  0.03902884877025144
Epoch 5200 RMSE =  0.038618874684784546
Epoch 5300 RMSE =  0.038231037561381014
Epoch 5400 RMSE =  0.03786312703054895
Epoch 5500 RMSE =  0.03751477159548247
Epoch 5600 RMSE =  0.037184783181875666
Epoch 5700 RMSE =  0.03687124928443106
Epoch 5800 RMSE =  0.03657345502067836
Epoch 5900 RMSE =  0.036290465682715094
Sample [0.08] expected [0.07991469] produced [0.10804561497037543]
Sample [1.13] expected [0.90441219] produced [0.8779530491276258]
Sample [0.16] expected [0.15931821] produced [0.15621862333426198]
Sample [0.46] expected [0.44394811] produced [0.4476234355733135]
Sample [0.49] expected [0.47062589] produced [0.48072249388421434]
Sample [0.02] expected [0.01999867] produced [0.08073170665458533]
Sample [0.86] expected [0.75784256] produced [0.7843036503909351]
Sample [1.1] expected [0.89120736] produced [0.870853511260299]
Sample [0.5] expected [0.47942554] produced [0.49147654421006054]
Sample [0.32] expected [0.31456656] produced [0.2950252650827075]
Sample [1.27] expected [0.95510086] produced [0.9034867742260732]
Sample [1.36] expected [0.9778646] produced [0.9152461255744556]
Sample [1.55] expected [0.99978376] produced [0.9322382505180108]
Sample [0.44] expected [0.42593947] produced [0.4255137223582886]
Sample [0.67] expected [0.62098599] produced [0.657990541688245]
Epoch 6000 RMSE =  0.0360218016609194
Epoch 6100 RMSE =  0.03576600969823486
Epoch 6200 RMSE =  0.035522438822922296
Epoch 6300 RMSE =  0.03529036139037706
Epoch 6400 RMSE =  0.03506921187963983
Epoch 6500 RMSE =  0.03485774633110683
Epoch 6600 RMSE =  0.03465609471411541
Epoch 6700 RMSE =  0.03446446279189683
Epoch 6800 RMSE =  0.03428070911855418
Epoch 6900 RMSE =  0.034105064349008625
Sample [1.55] expected [0.99978376] produced [0.9371001871969705]
Sample [1.1] expected [0.89120736] produced [0.8757888991397069]
Sample [1.13] expected [0.90441219] produced [0.8828826813916707]
Sample [0.02] expected [0.01999867] produced [0.0782802215597024]
Sample [0.16] expected [0.15931821] produced [0.15257335565266567]
Sample [0.67] expected [0.62098599] produced [0.6588776994073128]
Sample [0.32] expected [0.31456656] produced [0.2909176441027797]
Sample [0.44] expected [0.42593947] produced [0.42215043420671094]
Sample [0.5] expected [0.47942554] produced [0.4894111598688111]
Sample [1.36] expected [0.9778646] produced [0.9202237254337844]
Sample [1.27] expected [0.95510086] produced [0.9085520341472142]
Sample [0.49] expected [0.47062589] produced [0.4784709755390404]
Sample [0.46] expected [0.44394811] produced [0.44481866332357195]
Sample [0.86] expected [0.75784256] produced [0.7880202647886988]
Sample [0.08] expected [0.07991469] produced [0.10504684976444269]
Epoch 7000 RMSE =  0.03393692219691062
Epoch 7100 RMSE =  0.03377584345647846
Epoch 7200 RMSE =  0.03362192266699938
Epoch 7300 RMSE =  0.033474174520012515
Epoch 7400 RMSE =  0.03333253974182826
Epoch 7500 RMSE =  0.033196507207251646
Epoch 7600 RMSE =  0.0330660691497394
Epoch 7700 RMSE =  0.03294069401080388
Epoch 7800 RMSE =  0.03282026589911121
Epoch 7900 RMSE =  0.03270442348166225
Sample [0.02] expected [0.01999867] produced [0.07685767067405341]
Sample [0.32] expected [0.31456656] produced [0.28810872663862785]
Sample [0.49] expected [0.47062589] produced [0.47647017691648497]
Sample [1.36] expected [0.9778646] produced [0.9239444145941078]
Sample [0.67] expected [0.62098599] produced [0.6593149585318467]
Sample [1.27] expected [0.95510086] produced [0.9121694413035722]
Sample [0.5] expected [0.47942554] produced [0.4876023033513225]
Sample [1.1] expected [0.89120736] produced [0.8792281495922665]
Sample [0.46] expected [0.44394811] produced [0.4424861874490735]
Sample [0.16] expected [0.15931821] produced [0.15029170667164082]
Sample [0.08] expected [0.07991469] produced [0.10330050568122144]
Sample [0.44] expected [0.42593947] produced [0.4197658514087868]
Sample [0.86] expected [0.75784256] produced [0.7903621575113485]
Sample [1.55] expected [0.99978376] produced [0.9407461142953264]
Sample [1.13] expected [0.90441219] produced [0.8863597771073201]
Epoch 8000 RMSE =  0.03259301087269934
Epoch 8100 RMSE =  0.03248546068832052
Epoch 8200 RMSE =  0.03238229805534632
Epoch 8300 RMSE =  0.032282616820748726
Epoch 8400 RMSE =  0.03218671701932394
Epoch 8500 RMSE =  0.03209388775306149
Epoch 8600 RMSE =  0.032004677122751526
Epoch 8700 RMSE =  0.03191840970288743
Epoch 8800 RMSE =  0.03183513407195444
Epoch 8900 RMSE =  0.03175456442569074
Sample [1.55] expected [0.99978376] produced [0.943627799690892]
Sample [0.67] expected [0.62098599] produced [0.6594039456043714]
Sample [0.5] expected [0.47942554] produced [0.4860327358787584]
Sample [1.36] expected [0.9778646] produced [0.9267337764721014]
Sample [1.1] expected [0.89120736] produced [0.8818381376052312]
Sample [0.44] expected [0.42593947] produced [0.41795057998357765]
Sample [0.16] expected [0.15931821] produced [0.14894668597382685]
Sample [0.02] expected [0.01999867] produced [0.07611832576453031]
Sample [0.32] expected [0.31456656] produced [0.28617471180466475]
Sample [0.46] expected [0.44394811] produced [0.44089904485879927]
Sample [1.13] expected [0.90441219] produced [0.8891090274810983]
Sample [1.27] expected [0.95510086] produced [0.9150797383183712]
Sample [0.08] expected [0.07991469] produced [0.10236499936567517]
Sample [0.49] expected [0.47062589] produced [0.47510696007514236]
Sample [0.86] expected [0.75784256] produced [0.7921897601153657]
Epoch 9000 RMSE =  0.03167663084187742
Epoch 9100 RMSE =  0.03160128685601634
Epoch 9200 RMSE =  0.03152841051110542
Epoch 9300 RMSE =  0.031457795509788056
Epoch 9400 RMSE =  0.03138929009257179
Epoch 9500 RMSE =  0.03132329067237128
Epoch 9600 RMSE =  0.03125876222157173
Epoch 9700 RMSE =  0.031196849268610664
Epoch 9800 RMSE =  0.031136334898400588
Epoch 9900 RMSE =  0.03107775967350073
Sample [0.5] expected [0.47942554] produced [0.4850834681477599]
Sample [1.36] expected [0.9778646] produced [0.9290614300385365]
Sample [0.16] expected [0.15931821] produced [0.1482093106892515]
Sample [0.44] expected [0.42593947] produced [0.41677925487631245]
Sample [0.02] expected [0.01999867] produced [0.07580071608424223]
Sample [0.49] expected [0.47062589] produced [0.47385627168163624]
Sample [1.27] expected [0.95510086] produced [0.9173008108559174]
Sample [1.13] expected [0.90441219] produced [0.8912477789897671]
Sample [0.67] expected [0.62098599] produced [0.6595255688396663]
Sample [1.55] expected [0.99978376] produced [0.9459044862217094]
Sample [1.1] expected [0.89120736] produced [0.8839602310272718]
Sample [0.08] expected [0.07991469] produced [0.10183810022649659]
Sample [0.46] expected [0.44394811] produced [0.4395917756410006]
Sample [0.86] expected [0.75784256] produced [0.7933638099830812]
Sample [0.32] expected [0.31456656] produced [0.2848212680445571]
Epoch 10000 RMSE =  0.031020938763814374
Final Epoch RMSE =  0.031020938763814374

Test Results:
Input: [1.48, 1.2, 0.82, 0.61, 0.51, 0.1, 0.2, 0.35, 0.62, 0.48, 1.0, 0.76, 0.54, 0.78, 1.52, 0.6, 0.84, 0.13, 0.58, 0.06, 0.55, 0.53, 0.79, 0.05, 1.51, 1.06, 1.04, 0.21, 1.01, 0.9, 0.93, 0.8, 1.45, 0.41, 0.89, 0.07, 0.96, 1.12, 0.17, 0.71, 1.25, 0.85, 0.97, 0.95, 0.19, 1.02, 0.18, 1.11, 1.4, 0.74, 1.14, 1.39, 0.98, 0.73, 1.08, 1.43, 0.57, 1.05, 1.57, 0.94, 0.63, 0.29, 0.59, 1.32, 0.14, 1.16, 0.15, 0.03, 1.41, 0.26, 1.07, 1.53, 1.34, 1.19, 0.38, 0.11, 1.44, 1.37, 0.37, 0.91, 1.33, 0.23, 0.77, 1.5, 0.04, 0.56, 0.66, 1.46, 0.69, 0.4, 0.39, 1.18, 1.47, 0.75, 1.49, 1.09, 0.0, 0.27, 1.15, 1.29, 0.42, 0.68, 0.47, 0.88, 0.3, 1.35, 0.25, 1.38, 1.26, 0.83, 1.17, 0.99, 0.01, 0.28, 0.92, 0.31, 1.21, 1.31, 0.7, 1.3, 0.45, 0.36, 1.22, 0.81, 0.34, 1.54, 1.28, 1.03, 1.42, 0.24, 0.09, 1.24, 0.52, 0.65, 0.43, 0.72, 0.87, 1.23, 0.22, 0.33, 0.64, 1.56]
Expected: [0.99588084453764, 0.932039085967226, 0.731145829726896, 0.572867460100481, 0.488177246882907, 0.0998334166468282, 0.198669330795061, 0.342897807455451, 0.581035160537305, 0.461779175541483, 0.841470984807897, 0.688921445110551, 0.514135991653113, 0.70327941920041, 0.998710143975583, 0.564642473395035, 0.744643119970859, 0.129634142619695, 0.548023936791874, 0.0599640064794446, 0.522687228930659, 0.505533341204847, 0.710353272417608, 0.0499791692706783, 0.998152472497548, 0.872355482344986, 0.862404227243338, 0.2084598998461, 0.846831844618015, 0.783326909627483, 0.801619940883777, 0.717356090899523, 0.992712991037588, 0.398609327984423, 0.777071747526824, 0.0699428473375328, 0.819191568300998, 0.900100442176505, 0.169182349066996, 0.651833771021537, 0.948984619355586, 0.751280405140293, 0.82488571333845, 0.813415504789374, 0.188858894976501, 0.852108021949363, 0.179029573425824, 0.895698685680048, 0.98544972998846, 0.674287911628145, 0.908633496115883, 0.983700814811277, 0.83049737049197, 0.666869635003698, 0.881957806884948, 0.990104560337178, 0.539632048733969, 0.867423225594017, 0.999999682931835, 0.807558100405114, 0.58914475794227, 0.285952225104836, 0.556361022912784, 0.968715100118265, 0.139543114644236, 0.916803108771767, 0.149438132473599, 0.0299955002024957, 0.98710010101385, 0.257080551892155, 0.877200504274682, 0.999167945271476, 0.973484541695319, 0.928368967249167, 0.370920469412983, 0.109778300837175, 0.991458348191686, 0.979908061398614, 0.361615431964962, 0.78950373968995, 0.971148377921045, 0.227977523535188, 0.696135238627357, 0.997494986604054, 0.0399893341866342, 0.531186197920883, 0.613116851973434, 0.993868363411645, 0.636537182221968, 0.389418342308651, 0.380188415123161, 0.92460601240802, 0.994924349777581, 0.681638760023334, 0.996737752043143, 0.886626914449487, 0.0, 0.266731436688831, 0.912763940260521, 0.960835064206073, 0.40776045305957, 0.628793024018469, 0.452886285379068, 0.770738878898969, 0.29552020666134, 0.975723357826659, 0.247403959254523, 0.98185353037236, 0.952090341590516, 0.737931371109963, 0.920750597736136, 0.836025978600521, 0.00999983333416666, 0.276355648564114, 0.795601620036366, 0.305058636443443, 0.935616001553386, 0.966184951612734, 0.644217687237691, 0.963558185417193, 0.43496553411123, 0.35227423327509, 0.939099356319068, 0.724287174370143, 0.333487092140814, 0.999525830605479, 0.958015860289225, 0.857298989188603, 0.98865176285172, 0.237702626427135, 0.089878549198011, 0.945783999449539, 0.496880137843737, 0.60518640573604, 0.416870802429211, 0.659384671971473, 0.764328937025505, 0.942488801931697, 0.218229623080869, 0.324043028394868, 0.597195441362392, 0.999941720229966]
Output: [0.06832979603470501, 0.07188294966234705, 0.07930737937792995, 0.08344639857307247, 0.08768917338843706, 0.09217001637809853, 0.09664708624519457, 0.1066102438485228, 0.11203935402831483, 0.11703623526954221, 0.12881801421030872, 0.13444213906652, 0.1407623515421066, 0.15428756022784523, 0.16116006491609136, 0.16838956450957226, 0.17676825577576902, 0.18392998635380522, 0.19183224193204035, 0.1999549493351726, 0.20861923848066524, 0.21695142292624747, 0.2254944068247947, 0.23478172765311958, 0.2442313490895357, 0.2532282167115121, 0.2632763414759123, 0.27335049494243685, 0.293882742745364, 0.30424995761269197, 0.31612728364364107, 0.3255388281112265, 0.33623091084170353, 0.34695039995596416, 0.3579466552018494, 0.36899825011968745, 0.3807525572689545, 0.3919025819016242, 0.4036479098181607, 0.4261568699299743, 0.44866621012264574, 0.46207150368331334, 0.4960808088768313, 0.505909243665838, 0.5170351506778913, 0.5289316947049642, 0.5389909546883547, 0.5488603859182747, 0.5589632087312462, 0.5710399189656139, 0.5795531770655713, 0.591942379181924, 0.6028148414616932, 0.6125861182514303, 0.6194090685484328, 0.6299613547759062, 0.6395649293153173, 0.6479670987088021, 0.6658905969135563, 0.6741727648815279, 0.6830740810169882, 0.6910942178471575, 0.6994808235434818, 0.706492064468782, 0.7142625039176442, 0.7217971553919399, 0.7308521476194982, 0.7363510204416325, 0.7444271765376113, 0.7503286193223766, 0.7564791842938811, 0.7628687139623126, 0.7707134369540611, 0.774826849662103, 0.7814466331358575, 0.7860308584541381, 0.7970443008472805, 0.801903302176798, 0.8071970968314109, 0.8122287257448264, 0.8166443376250871, 0.8211737917627031, 0.8257926722291031, 0.82943643288968, 0.8337522634485414, 0.838045107391519, 0.8417281662131654, 0.8454733756062824, 0.8493775670050474, 0.8540327375255403, 0.8566445362604745, 0.8594441132363753, 0.8631492429380152, 0.8662869594026056, 0.8686980329277518, 0.8722426129918306, 0.8745371407286424, 0.8772641382276034, 0.8801263026695824, 0.8851749804100442, 0.8877775330692689, 0.8922043135205497, 0.8945589331124203, 0.8964322351754112, 0.8988641532502879, 0.9007632862723889, 0.9026750395342794, 0.9056632804702267, 0.9066057709250953, 0.9084650963295804, 0.9101109721140488, 0.9119834982957177, 0.9133056485514344, 0.9150004443903457, 0.9181118683832155, 0.9193777751282898, 0.9208315418451539, 0.9222389694400147, 0.9232019562975865, 0.9247762059595427, 0.9258734398586657, 0.9271809808922263, 0.9295145462496125, 0.930663490450994, 0.9316014878332637, 0.9326754734778978, 0.9335796401251216, 0.9349143684175102, 0.9355736692453765, 0.9366262696674178, 0.9375914018310393, 0.9383844882961927, 0.9392834672916675, 0.9407132713800537, 0.9409011470129044, 0.9417617943955217, 0.9426290557289392, 0.9435869676933324, 0.9438960524766639, 0.9448147633327104, 0.9461514039895705, 0.9466125126046925]
Test RMSE =  0.10464967241608952





-- run_XOR() Sample Run #1: hidden layer of three neurodes, 10001 epochs, 100% training factor, and is randomly ordered --
Sample [1. 1.] expected [0.] produced [0.6150869582654567]
Sample [1. 0.] expected [1.] produced [0.5915330270929355]
Sample [0. 1.] expected [1.] produced [0.6034731482657324]
Sample [0. 0.] expected [0.] produced [0.5816075813377802]
Epoch 0 RMSE =  0.5100681711791044
Epoch 100 RMSE =  0.501615411125597
Epoch 200 RMSE =  0.5010871311701888
Epoch 300 RMSE =  0.5010292362372026
Epoch 400 RMSE =  0.5010033678398855
Epoch 500 RMSE =  0.5009773144325932
Epoch 600 RMSE =  0.5009519644751111
Epoch 700 RMSE =  0.5009270536752622
Epoch 800 RMSE =  0.500900162474366
Epoch 900 RMSE =  0.5008785096646876
Sample [1. 1.] expected [0.] produced [0.5040270256358305]
Sample [0. 1.] expected [1.] produced [0.5021029286577413]
Sample [1. 0.] expected [1.] produced [0.5018112143758515]
Sample [0. 0.] expected [0.] produced [0.5032679633128658]
Epoch 1000 RMSE =  0.5008531334735873
Epoch 1100 RMSE =  0.500827929477002
Epoch 1200 RMSE =  0.5008025789544629
Epoch 1300 RMSE =  0.5007729562319896
Epoch 1400 RMSE =  0.5007480998709527
Epoch 1500 RMSE =  0.5007149097651649
Epoch 1600 RMSE =  0.50068765111162
Epoch 1700 RMSE =  0.5006554865519687
Epoch 1800 RMSE =  0.5006206755929785
Epoch 1900 RMSE =  0.5005830285136808
Sample [1. 1.] expected [0.] produced [0.5058767086580543]
Sample [1. 0.] expected [1.] produced [0.5002280290126613]
Sample [0. 0.] expected [0.] produced [0.49945600267507734]
Sample [0. 1.] expected [1.] produced [0.5029802817952904]
Epoch 2000 RMSE =  0.5005417480689626
Epoch 2100 RMSE =  0.5004954628594266
Epoch 2200 RMSE =  0.5004464162388665
Epoch 2300 RMSE =  0.5003972611892117
Epoch 2400 RMSE =  0.5003361569776058
Epoch 2500 RMSE =  0.500276504182371
Epoch 2600 RMSE =  0.5002060342009512
Epoch 2700 RMSE =  0.5001248447501828
Epoch 2800 RMSE =  0.500040224651645
Epoch 2900 RMSE =  0.4999397516434082
Sample [1. 0.] expected [1.] produced [0.5042694295961156]
Sample [1. 1.] expected [0.] produced [0.5120912422818669]
Sample [0. 1.] expected [1.] produced [0.5059866943836557]
Sample [0. 0.] expected [0.] produced [0.4972958448183195]
Epoch 3000 RMSE =  0.4998346082091305
Epoch 3100 RMSE =  0.4997118850451931
Epoch 3200 RMSE =  0.49957404219200213
Epoch 3300 RMSE =  0.4994150532246022
Epoch 3400 RMSE =  0.49924272603627473
Epoch 3500 RMSE =  0.4990445277143264
Epoch 3600 RMSE =  0.49881541703142473
Epoch 3700 RMSE =  0.49855835851390545
Epoch 3800 RMSE =  0.4982666352199603
Epoch 3900 RMSE =  0.49793771964536127
Sample [1. 0.] expected [1.] produced [0.5107086895363626]
Sample [0. 1.] expected [1.] produced [0.5117203551872662]
Sample [1. 1.] expected [0.] produced [0.5228833613398123]
Sample [0. 0.] expected [0.] produced [0.48889374791734364]
Epoch 4000 RMSE =  0.49755580198414956
Epoch 4100 RMSE =  0.4971279921118139
Epoch 4200 RMSE =  0.4966335413168149
Epoch 4300 RMSE =  0.4960792819331531
Epoch 4400 RMSE =  0.49544425112354135
Epoch 4500 RMSE =  0.49472529626405437
Epoch 4600 RMSE =  0.49391263141236863
Epoch 4700 RMSE =  0.49299642819733897
Epoch 4800 RMSE =  0.4919693885878862
Epoch 4900 RMSE =  0.49081988557667366
Sample [1. 0.] expected [1.] produced [0.5293746401479494]
Sample [0. 0.] expected [0.] produced [0.47000222280434784]
Sample [0. 1.] expected [1.] produced [0.5201492024839874]
Sample [1. 1.] expected [0.] produced [0.5347806834200532]
Epoch 5000 RMSE =  0.48955017261075306
Epoch 5100 RMSE =  0.4881405716854737
Epoch 5200 RMSE =  0.4866007771251381
Epoch 5300 RMSE =  0.4849179395643563
Epoch 5400 RMSE =  0.4830924961730295
Epoch 5500 RMSE =  0.48112598070064294
Epoch 5600 RMSE =  0.47901696929845283
Epoch 5700 RMSE =  0.47678389260185
Epoch 5800 RMSE =  0.47441600757984154
Epoch 5900 RMSE =  0.47192613412949563
Sample [1. 1.] expected [0.] produced [0.5325405277090128]
Sample [0. 0.] expected [0.] produced [0.43196022034320014]
Sample [0. 1.] expected [1.] produced [0.5344502352038032]
Sample [1. 0.] expected [1.] produced [0.559421648632933]
Epoch 6000 RMSE =  0.4693173001303747
Epoch 6100 RMSE =  0.4666031883079458
Epoch 6200 RMSE =  0.4637909937082222
Epoch 6300 RMSE =  0.4608853446628231
Epoch 6400 RMSE =  0.45788925487176646
Epoch 6500 RMSE =  0.4548030125769313
Epoch 6600 RMSE =  0.45165207306294997
Epoch 6700 RMSE =  0.448412426965448
Epoch 6800 RMSE =  0.44510642209136686
Epoch 6900 RMSE =  0.44171661387951994
Sample [0. 0.] expected [0.] produced [0.39340490695637337]
Sample [0. 1.] expected [1.] produced [0.5665770939546287]
Sample [1. 0.] expected [1.] produced [0.5940142449135251]
Sample [1. 1.] expected [0.] produced [0.51068888498731]
Epoch 7000 RMSE =  0.4382494742965714
Epoch 7100 RMSE =  0.4346896188927669
Epoch 7200 RMSE =  0.431038857646194
Epoch 7300 RMSE =  0.42727636935619817
Epoch 7400 RMSE =  0.4233865301073966
Epoch 7500 RMSE =  0.41935164212777715
Epoch 7600 RMSE =  0.4151579670751062
Epoch 7700 RMSE =  0.4107674533604755
Epoch 7800 RMSE =  0.4061657284446392
Epoch 7900 RMSE =  0.4013130178704777
Sample [0. 1.] expected [1.] produced [0.6114473225522723]
Sample [0. 0.] expected [0.] produced [0.35349184442450793]
Sample [1. 1.] expected [0.] produced [0.45700578826332544]
Sample [1. 0.] expected [1.] produced [0.6217608842596548]
Epoch 8000 RMSE =  0.3961845486666094
Epoch 8100 RMSE =  0.39075067801932434
Epoch 8200 RMSE =  0.384976040440661
Epoch 8300 RMSE =  0.37886374149637814
Epoch 8400 RMSE =  0.37238538959284934
Epoch 8500 RMSE =  0.36555669471889835
Epoch 8600 RMSE =  0.35838899634909493
Epoch 8700 RMSE =  0.35090249425430187
Epoch 8800 RMSE =  0.3431665936694665
Epoch 8900 RMSE =  0.33522379532488134
Sample [1. 1.] expected [0.] produced [0.35702899938866706]
Sample [0. 1.] expected [1.] produced [0.6767638672767305]
Sample [0. 0.] expected [0.] produced [0.3058357009738925]
Sample [1. 0.] expected [1.] produced [0.6797015053630121]
Epoch 9000 RMSE =  0.3271383138518327
Epoch 9100 RMSE =  0.3189864277668445
Epoch 9200 RMSE =  0.3108387230755795
Epoch 9300 RMSE =  0.30276177586617414
Epoch 9400 RMSE =  0.2948131281462414
Epoch 9500 RMSE =  0.28704552299747826
Epoch 9600 RMSE =  0.27949332264814825
Epoch 9700 RMSE =  0.2721914787068405
Epoch 9800 RMSE =  0.26515649885877823
Epoch 9900 RMSE =  0.2584089033349249
Sample [0. 0.] expected [0.] produced [0.2611670081772722]
Sample [1. 0.] expected [1.] produced [0.7528914690665364]
Sample [0. 1.] expected [1.] produced [0.7532796752462226]
Sample [1. 1.] expected [0.] produced [0.2525245333188245]
Epoch 10000 RMSE =  0.2519476090355202
Final Epoch RMSE =  0.2519476090355202





-- run_XOR() Sample Run #2: hidden layer of three neurodes, 50001 epochs, 100% training factor, and is randomly ordered --
Sample [1. 0.] expected [1.] produced [0.7839403821320643]
Sample [1. 1.] expected [0.] produced [0.8227561572288243]
Sample [0. 0.] expected [0.] produced [0.75489209404993]
Sample [0. 1.] expected [1.] produced [0.797480364271616]
Epoch 0 RMSE =  0.5775997163360249
Epoch 100 RMSE =  0.5142212337483002
Epoch 200 RMSE =  0.5021566462013546
Epoch 300 RMSE =  0.5011343532141063
Epoch 400 RMSE =  0.501036732648534
Epoch 500 RMSE =  0.5010145076700274
Epoch 600 RMSE =  0.5009942111564101
Epoch 700 RMSE =  0.5009802798190848
Epoch 800 RMSE =  0.5009629789410812
Epoch 900 RMSE =  0.5009516742405496
Sample [0. 1.] expected [1.] produced [0.5096021952090878]
Sample [1. 0.] expected [1.] produced [0.49313519211652607]
Sample [1. 1.] expected [0.] produced [0.5009862059159055]
Sample [0. 0.] expected [0.] produced [0.5053308373817221]
Epoch 1000 RMSE =  0.5009362170330228
Epoch 1100 RMSE =  0.5009263766954763
Epoch 1200 RMSE =  0.5009153381487499
Epoch 1300 RMSE =  0.5009054282593582
Epoch 1400 RMSE =  0.5008957854173766
Epoch 1500 RMSE =  0.5008863999159865
Epoch 1600 RMSE =  0.5008780058813718
Epoch 1700 RMSE =  0.5008678071629838
Epoch 1800 RMSE =  0.5008604953059655
Epoch 1900 RMSE =  0.5008559577816161
Sample [1. 0.] expected [1.] produced [0.4926068632695761]
Sample [0. 1.] expected [1.] produced [0.5101380184260571]
Sample [0. 0.] expected [0.] produced [0.5058116057432194]
Sample [1. 1.] expected [0.] produced [0.5001323772645992]
Epoch 2000 RMSE =  0.5008468656898938
Epoch 2100 RMSE =  0.5008428898112437
Epoch 2200 RMSE =  0.5008374133905886
Epoch 2300 RMSE =  0.5008318473051137
Epoch 2400 RMSE =  0.5008265783622596
Epoch 2500 RMSE =  0.5008215667695333
Epoch 2600 RMSE =  0.5008164579373732
Epoch 2700 RMSE =  0.5008097252499204
Epoch 2800 RMSE =  0.5008075208231407
Epoch 2900 RMSE =  0.5008007305543611
Sample [0. 1.] expected [1.] produced [0.5076778232440909]
Sample [1. 1.] expected [0.] produced [0.4998861666728546]
Sample [1. 0.] expected [1.] produced [0.49246473431254734]
Sample [0. 0.] expected [0.] produced [0.5033313114825467]
Epoch 3000 RMSE =  0.5007998004172254
Epoch 3100 RMSE =  0.500793414838072
Epoch 3200 RMSE =  0.5007918880357806
Epoch 3300 RMSE =  0.50078864924299
Epoch 3400 RMSE =  0.5007851461344851
Epoch 3500 RMSE =  0.5007819259004999
Epoch 3600 RMSE =  0.5007757260781599
Epoch 3700 RMSE =  0.5007726272425496
Epoch 3800 RMSE =  0.5007725905807219
Epoch 3900 RMSE =  0.500769655482711
Sample [0. 0.] expected [0.] produced [0.5016120444406652]
Sample [1. 1.] expected [0.] produced [0.49680279202644145]
Sample [0. 1.] expected [1.] produced [0.5043382074369467]
Sample [1. 0.] expected [1.] produced [0.49112941086015466]
Epoch 4000 RMSE =  0.5007638032025588
Epoch 4100 RMSE =  0.5007640532015751
Epoch 4200 RMSE =  0.5007587999716923
Epoch 4300 RMSE =  0.5007582838220324
Epoch 4400 RMSE =  0.5007557082781607
Epoch 4500 RMSE =  0.5007535525639978
Epoch 4600 RMSE =  0.5007480484262
Epoch 4700 RMSE =  0.5007484977384101
Epoch 4800 RMSE =  0.5007436411879659
Epoch 4900 RMSE =  0.5007437313231893
Sample [1. 1.] expected [0.] produced [0.4983993207276853]
Sample [1. 0.] expected [1.] produced [0.4907535511895818]
Sample [0. 1.] expected [1.] produced [0.5071788546415029]
Sample [0. 0.] expected [0.] produced [0.5023542755861696]
Epoch 5000 RMSE =  0.5007410328802718
Epoch 5100 RMSE =  0.500736207894564
Epoch 5200 RMSE =  0.5007338152675126
Epoch 5300 RMSE =  0.5007319507817095
Epoch 5400 RMSE =  0.5007322108827851
Epoch 5500 RMSE =  0.5007277406553742
Epoch 5600 RMSE =  0.5007247617834193
Epoch 5700 RMSE =  0.5007251832625567
Epoch 5800 RMSE =  0.5007208573789125
Epoch 5900 RMSE =  0.500718306430532
Sample [1. 0.] expected [1.] produced [0.4923188695783583]
Sample [0. 1.] expected [1.] produced [0.5092508797536384]
Sample [1. 1.] expected [0.] produced [0.502141138283988]
Sample [0. 0.] expected [0.] produced [0.5021429123793127]
Epoch 6000 RMSE =  0.5007165007313747
Epoch 6100 RMSE =  0.500713890942457
Epoch 6200 RMSE =  0.5007121708569303
Epoch 6300 RMSE =  0.5007124939531103
Epoch 6400 RMSE =  0.5007075011085155
Epoch 6500 RMSE =  0.5007081805236276
Epoch 6600 RMSE =  0.5007059833029625
Epoch 6700 RMSE =  0.5007013094851298
Epoch 6800 RMSE =  0.5007013065178072
Epoch 6900 RMSE =  0.5006990488570436
Sample [0. 0.] expected [0.] produced [0.5005940276301603]
Sample [1. 0.] expected [1.] produced [0.49031840874162874]
Sample [0. 1.] expected [1.] produced [0.5082699106594575]
Sample [1. 1.] expected [0.] produced [0.5006220053247106]
Epoch 7000 RMSE =  0.5006971584562977
Epoch 7100 RMSE =  0.5006924988632938
Epoch 7200 RMSE =  0.5006926744826781
Epoch 7300 RMSE =  0.5006904723531098
Epoch 7400 RMSE =  0.5006878763712181
Epoch 7500 RMSE =  0.5006856435919305
Epoch 7600 RMSE =  0.5006832473482572
Epoch 7700 RMSE =  0.50068093053408
Epoch 7800 RMSE =  0.5006789032311809
Epoch 7900 RMSE =  0.50067625878271
Sample [1. 1.] expected [0.] produced [0.4987843312487786]
Sample [0. 0.] expected [0.] produced [0.4987808947216282]
Sample [0. 1.] expected [1.] produced [0.5056682649319347]
Sample [1. 0.] expected [1.] produced [0.48935647317903935]
Epoch 8000 RMSE =  0.5006717651860728
Epoch 8100 RMSE =  0.5006714724264971
Epoch 8200 RMSE =  0.5006689672359753
Epoch 8300 RMSE =  0.5006662603500316
Epoch 8400 RMSE =  0.5006639335119341
Epoch 8500 RMSE =  0.5006609782898932
Epoch 8600 RMSE =  0.5006560437858761
Epoch 8700 RMSE =  0.5006553859679278
Epoch 8800 RMSE =  0.5006528719622403
Epoch 8900 RMSE =  0.5006472498032101
Sample [0. 1.] expected [1.] produced [0.5093150399262039]
Sample [0. 0.] expected [0.] produced [0.5011951544415612]
Sample [1. 1.] expected [0.] produced [0.499043691390815]
Sample [1. 0.] expected [1.] produced [0.4885549728881382]
Epoch 9000 RMSE =  0.5006468152681957
Epoch 9100 RMSE =  0.5006416218610752
Epoch 9200 RMSE =  0.5006409541005284
Epoch 9300 RMSE =  0.5006375243559901
Epoch 9400 RMSE =  0.5006344899857307
Epoch 9500 RMSE =  0.5006311927684148
Epoch 9600 RMSE =  0.5006256084006581
Epoch 9700 RMSE =  0.5006240411475049
Epoch 9800 RMSE =  0.5006206429200327
Epoch 9900 RMSE =  0.500616662867467
Sample [0. 1.] expected [1.] produced [0.5102290898898592]
Sample [0. 0.] expected [0.] produced [0.5008337116724392]
Sample [1. 1.] expected [0.] produced [0.4993178297959331]
Sample [1. 0.] expected [1.] produced [0.4877258874259508]
Epoch 10000 RMSE =  0.5006128775569396
Epoch 10100 RMSE =  0.5006062863995574
Epoch 10200 RMSE =  0.5006047806043272
Epoch 10300 RMSE =  0.5006004502870642
Epoch 10400 RMSE =  0.5005960436304233
Epoch 10500 RMSE =  0.5005918968448447
Epoch 10600 RMSE =  0.5005867666500654
Epoch 10700 RMSE =  0.5005817369802636
Epoch 10800 RMSE =  0.5005746520656812
Epoch 10900 RMSE =  0.5005686883533452
Sample [0. 0.] expected [0.] produced [0.4990598365484337]
Sample [1. 1.] expected [0.] produced [0.4980236018442995]
Sample [0. 1.] expected [1.] produced [0.5083497221782816]
Sample [1. 0.] expected [1.] produced [0.48673056423957867]
Epoch 11000 RMSE =  0.5005631173532746
Epoch 11100 RMSE =  0.5005599026555302
Epoch 11200 RMSE =  0.5005518122392041
Epoch 11300 RMSE =  0.500547518023216
Epoch 11400 RMSE =  0.5005408370130864
Epoch 11500 RMSE =  0.5005319552635852
Epoch 11600 RMSE =  0.500524679561955
Epoch 11700 RMSE =  0.5005190188136809
Epoch 11800 RMSE =  0.5005089692697275
Epoch 11900 RMSE =  0.5005006154909956
Sample [1. 1.] expected [0.] produced [0.5003847690729127]
Sample [0. 1.] expected [1.] produced [0.5111709269078004]
Sample [1. 0.] expected [1.] produced [0.48728328817236316]
Sample [0. 0.] expected [0.] produced [0.49975809981324515]
Epoch 12000 RMSE =  0.5004935975456182
Epoch 12100 RMSE =  0.5004819746711436
Epoch 12200 RMSE =  0.5004716656275733
Epoch 12300 RMSE =  0.5004637294917456
Epoch 12400 RMSE =  0.5004528772983902
Epoch 12500 RMSE =  0.500438836500145
Epoch 12600 RMSE =  0.5004260345621196
Epoch 12700 RMSE =  0.5004129854319644
Epoch 12800 RMSE =  0.5004011211858619
Epoch 12900 RMSE =  0.5003855252236025
Sample [1. 1.] expected [0.] produced [0.5010093319402967]
Sample [1. 0.] expected [1.] produced [0.4842418376987237]
Sample [0. 0.] expected [0.] produced [0.49723046735398213]
Sample [0. 1.] expected [1.] produced [0.512945130338865]
Epoch 13000 RMSE =  0.5003692177775669
Epoch 13100 RMSE =  0.5003520911799754
Epoch 13200 RMSE =  0.5003329622007148
Epoch 13300 RMSE =  0.5003129959057283
Epoch 13400 RMSE =  0.5002893072282449
Epoch 13500 RMSE =  0.5002659528429257
Epoch 13600 RMSE =  0.5002408269303592
Epoch 13700 RMSE =  0.5002130326337946
Epoch 13800 RMSE =  0.5001866178111308
Epoch 13900 RMSE =  0.5001541663501334
Sample [0. 1.] expected [1.] produced [0.5170054332947862]
Sample [1. 1.] expected [0.] produced [0.5044922695942113]
Sample [0. 0.] expected [0.] produced [0.49526036692817854]
Sample [1. 0.] expected [1.] produced [0.4828923771640293]
Epoch 14000 RMSE =  0.5001198171918351
Epoch 14100 RMSE =  0.5000814150899436
Epoch 14200 RMSE =  0.5000372395003693
Epoch 14300 RMSE =  0.4999917364571047
Epoch 14400 RMSE =  0.49994084282320295
Epoch 14500 RMSE =  0.4998887334862107
Epoch 14600 RMSE =  0.49982483000166356
Epoch 14700 RMSE =  0.4997600590196172
Epoch 14800 RMSE =  0.4996844412325457
Epoch 14900 RMSE =  0.4996005393012415
Sample [1. 0.] expected [1.] produced [0.4832436993541709]
Sample [1. 1.] expected [0.] produced [0.5081727682973272]
Sample [0. 0.] expected [0.] produced [0.49215321073172996]
Sample [0. 1.] expected [1.] produced [0.5198573060192173]
Epoch 15000 RMSE =  0.49950686334444033
Epoch 15100 RMSE =  0.49940155623100346
Epoch 15200 RMSE =  0.499280999891371
Epoch 15300 RMSE =  0.49915122789770916
Epoch 15400 RMSE =  0.4990008006303797
Epoch 15500 RMSE =  0.498831396899323
Epoch 15600 RMSE =  0.49863734130008924
Epoch 15700 RMSE =  0.4984173076995562
Epoch 15800 RMSE =  0.4981656428496177
Epoch 15900 RMSE =  0.4978781887363228
Sample [0. 1.] expected [1.] produced [0.5305529614186932]
Sample [1. 0.] expected [1.] produced [0.48383844427244627]
Sample [0. 0.] expected [0.] produced [0.4880260856746923]
Sample [1. 1.] expected [0.] produced [0.5150079608370547]
Epoch 16000 RMSE =  0.4975454586438185
Epoch 16100 RMSE =  0.49716696138233735
Epoch 16200 RMSE =  0.49673516310629856
Epoch 16300 RMSE =  0.49623083116706024
Epoch 16400 RMSE =  0.49565767501482216
Epoch 16500 RMSE =  0.4949976258095561
Epoch 16600 RMSE =  0.4942342492242422
Epoch 16700 RMSE =  0.49336496870357927
Epoch 16800 RMSE =  0.4923757485456878
Epoch 16900 RMSE =  0.49124308894148966
Sample [0. 1.] expected [1.] produced [0.554033029056306]
Sample [1. 0.] expected [1.] produced [0.4868263667574254]
Sample [1. 1.] expected [0.] produced [0.5265170059509079]
Sample [0. 0.] expected [0.] produced [0.46989890378817045]
Epoch 17000 RMSE =  0.4899639921374097
Epoch 17100 RMSE =  0.48852732389703146
Epoch 17200 RMSE =  0.4869105878232743
Epoch 17300 RMSE =  0.4851172197299485
Epoch 17400 RMSE =  0.4831374129796535
Epoch 17500 RMSE =  0.4809600485405324
Epoch 17600 RMSE =  0.4785826752562318
Epoch 17700 RMSE =  0.47600018660436416
Epoch 17800 RMSE =  0.4732244820664079
Epoch 17900 RMSE =  0.4702362857899548
Sample [1. 0.] expected [1.] produced [0.4983438186665342]
Sample [0. 0.] expected [0.] produced [0.43771043723186237]
Sample [1. 1.] expected [0.] produced [0.5166651864898404]
Sample [0. 1.] expected [1.] produced [0.5970670694532242]
Epoch 18000 RMSE =  0.4670511782934975
Epoch 18100 RMSE =  0.4636542301326043
Epoch 18200 RMSE =  0.46004494877309776
Epoch 18300 RMSE =  0.456215006678852
Epoch 18400 RMSE =  0.4521557659171511
Epoch 18500 RMSE =  0.44786151409091
Epoch 18600 RMSE =  0.44331123495785146
Epoch 18700 RMSE =  0.43849144749982444
Epoch 18800 RMSE =  0.433383973051738
Epoch 18900 RMSE =  0.42796791364629466
Sample [0. 1.] expected [1.] produced [0.6392374975082052]
Sample [1. 0.] expected [1.] produced [0.5443614769411876]
Sample [1. 1.] expected [0.] produced [0.47158567710583477]
Sample [0. 0.] expected [0.] produced [0.39112959797426083]
Epoch 19000 RMSE =  0.42223555634332643
Epoch 19100 RMSE =  0.41617649258028133
Epoch 19200 RMSE =  0.40976973347108125
Epoch 19300 RMSE =  0.4030181592414226
Epoch 19400 RMSE =  0.3959419262661664
Epoch 19500 RMSE =  0.38853539327851677
Epoch 19600 RMSE =  0.3808487743641256
Epoch 19700 RMSE =  0.3729029184729653
Epoch 19800 RMSE =  0.3647434555412751
Epoch 19900 RMSE =  0.3564367644488665
Sample [0. 1.] expected [1.] produced [0.6786476215730736]
Sample [1. 1.] expected [0.] produced [0.37006646462161547]
Sample [1. 0.] expected [1.] produced [0.6381874872395812]
Sample [0. 0.] expected [0.] produced [0.3366915324581485]
Epoch 20000 RMSE =  0.34802515059159467
Epoch 20100 RMSE =  0.33957188076154765
Epoch 20200 RMSE =  0.331138044006599
Epoch 20300 RMSE =  0.32277711881342774
Epoch 20400 RMSE =  0.31453890246546334
Epoch 20500 RMSE =  0.30646790690797143
Epoch 20600 RMSE =  0.2985969199619412
Epoch 20700 RMSE =  0.29095849142793184
Epoch 20800 RMSE =  0.28357547183235005
Epoch 20900 RMSE =  0.27645242061731823
Sample [0. 1.] expected [1.] produced [0.740769601587355]
Sample [1. 0.] expected [1.] produced [0.7312180747689063]
Sample [1. 1.] expected [0.] produced [0.26493641513454846]
Sample [0. 0.] expected [0.] produced [0.2848151109571548]
Epoch 21000 RMSE =  0.26960854691848796
Epoch 21100 RMSE =  0.2630440335672789
Epoch 21200 RMSE =  0.25675491979178905
Epoch 21300 RMSE =  0.2507403195578413
Epoch 21400 RMSE =  0.2449922615593501
Epoch 21500 RMSE =  0.23950379447543652
Epoch 21600 RMSE =  0.2342662609424523
Epoch 21700 RMSE =  0.2292653703403109
Epoch 21800 RMSE =  0.22449511050797766
Epoch 21900 RMSE =  0.2199401252970775
Sample [1. 0.] expected [1.] produced [0.7888731784699072]
Sample [0. 1.] expected [1.] produced [0.7925390879932745]
Sample [1. 1.] expected [0.] produced [0.19403783275977873]
Sample [0. 0.] expected [0.] produced [0.24627799945301332]
Epoch 22000 RMSE =  0.21559110506741241
Epoch 22100 RMSE =  0.2114387311113432
Epoch 22200 RMSE =  0.20746933380619895
Epoch 22300 RMSE =  0.20367590074800654
Epoch 22400 RMSE =  0.2000465787882866
Epoch 22500 RMSE =  0.19657245919348454
Epoch 22600 RMSE =  0.19324504148124788
Epoch 22700 RMSE =  0.19005592017955797
Epoch 22800 RMSE =  0.18699680967686558
Epoch 22900 RMSE =  0.18406195807638248
Sample [0. 1.] expected [1.] produced [0.8262806461333912]
Sample [1. 1.] expected [0.] produced [0.15050171102043475]
Sample [0. 0.] expected [0.] produced [0.21847099428129915]
Sample [1. 0.] expected [1.] produced [0.8243959609657573]
Epoch 23000 RMSE =  0.18124260867364342
Epoch 23100 RMSE =  0.17853301008077954
Epoch 23200 RMSE =  0.1759270465214544
Epoch 23300 RMSE =  0.17341871159276756
Epoch 23400 RMSE =  0.17100375308515367
Epoch 23500 RMSE =  0.16867619178194615
Epoch 23600 RMSE =  0.1664315209690606
Epoch 23700 RMSE =  0.1642653880253981
Epoch 23800 RMSE =  0.162173910092067
Epoch 23900 RMSE =  0.16015313377691667
Sample [1. 0.] expected [1.] produced [0.8484740626927811]
Sample [1. 1.] expected [0.] produced [0.12297600760664558]
Sample [0. 0.] expected [0.] produced [0.19812172904126843]
Sample [0. 1.] expected [1.] produced [0.8490931228033025]
Epoch 24000 RMSE =  0.15819948895876837
Epoch 24100 RMSE =  0.1563096377418942
Epoch 24200 RMSE =  0.15448050058954546
Epoch 24300 RMSE =  0.15270903306244482
Epoch 24400 RMSE =  0.1509925866846724
Epoch 24500 RMSE =  0.14932826574585967
Epoch 24600 RMSE =  0.14771423509363782
Epoch 24700 RMSE =  0.14614762079775281
Epoch 24800 RMSE =  0.14462644872924307
Epoch 24900 RMSE =  0.14314857053562513
Sample [1. 0.] expected [1.] produced [0.8652157263311389]
Sample [1. 1.] expected [0.] produced [0.10409023531798983]
Sample [0. 0.] expected [0.] produced [0.1823882541842378]
Sample [0. 1.] expected [1.] produced [0.8656027569177465]
Epoch 25000 RMSE =  0.14171244805016375
Epoch 25100 RMSE =  0.1403157312070804
Epoch 25200 RMSE =  0.13895733392814155
Epoch 25300 RMSE =  0.13763518354356155
Epoch 25400 RMSE =  0.13634787840302004
Epoch 25500 RMSE =  0.13509397576256105
Epoch 25600 RMSE =  0.13387202721310906
Epoch 25700 RMSE =  0.13268101572250807
Epoch 25800 RMSE =  0.1315195350347087
Epoch 25900 RMSE =  0.13038654303370145
Sample [0. 0.] expected [0.] produced [0.16978151311540776]
Sample [1. 1.] expected [0.] produced [0.09016613163214104]
Sample [1. 0.] expected [1.] produced [0.87748784300653]
Sample [0. 1.] expected [1.] produced [0.8779794010896456]
Epoch 26000 RMSE =  0.1292806527084546
Epoch 26100 RMSE =  0.12820116766074374
Epoch 26200 RMSE =  0.1271470158836958
Epoch 26300 RMSE =  0.12611719062031496
Epoch 26400 RMSE =  0.1251107108392676
Epoch 26500 RMSE =  0.1241269811695493
Epoch 26600 RMSE =  0.12316501268811086
Epoch 26700 RMSE =  0.12222399339665402
Epoch 26800 RMSE =  0.1213033960381614
Epoch 26900 RMSE =  0.12040233598121562
Sample [1. 1.] expected [0.] produced [0.08000011606996188]
Sample [0. 1.] expected [1.] produced [0.8876707337377757]
Sample [1. 0.] expected [1.] produced [0.8874162249845401]
Sample [0. 0.] expected [0.] produced [0.15952225121731342]
Epoch 27000 RMSE =  0.11952022596655806
Epoch 27100 RMSE =  0.11865641559902235
Epoch 27200 RMSE =  0.11781022303193285
Epoch 27300 RMSE =  0.11698125440170963
Epoch 27400 RMSE =  0.11616880627519147
Epoch 27500 RMSE =  0.11537238356823101
Epoch 27600 RMSE =  0.11459147614744866
Epoch 27700 RMSE =  0.11382560785912199
Epoch 27800 RMSE =  0.1130742798045934
Epoch 27900 RMSE =  0.11233717702951791
Sample [1. 1.] expected [0.] produced [0.07195215277442248]
Sample [0. 0.] expected [0.] produced [0.15068500628310563]
Sample [1. 0.] expected [1.] produced [0.8950908264204316]
Sample [0. 1.] expected [1.] produced [0.8953986468731582]
Epoch 28000 RMSE =  0.11161368776333808
Epoch 28100 RMSE =  0.11090354147853808
Epoch 28200 RMSE =  0.11020625447000107
Epoch 28300 RMSE =  0.10952149832321406
Epoch 28400 RMSE =  0.10884888620164229
Epoch 28500 RMSE =  0.10818809216117711
Epoch 28600 RMSE =  0.1075387375595544
Epoch 28700 RMSE =  0.1069005793924348
Epoch 28800 RMSE =  0.10627326588325182
Epoch 28900 RMSE =  0.10565644631632404
Sample [0. 1.] expected [1.] produced [0.9018767137531433]
Sample [1. 1.] expected [0.] produced [0.0655321844729433]
Sample [1. 0.] expected [1.] produced [0.9016694756888368]
Sample [0. 0.] expected [0.] produced [0.1433541658058718]
Epoch 29000 RMSE =  0.10504993499627224
Epoch 29100 RMSE =  0.10445334793098587
Epoch 29200 RMSE =  0.10386654786360927
Epoch 29300 RMSE =  0.10328917688715583
Epoch 29400 RMSE =  0.10272099952685483
Epoch 29500 RMSE =  0.1021618189172391
Epoch 29600 RMSE =  0.10161135480040008
Epoch 29700 RMSE =  0.1010694567428007
Epoch 29800 RMSE =  0.10053584237312577
Epoch 29900 RMSE =  0.10001032180631832
Sample [0. 0.] expected [0.] produced [0.13682335232683263]
Sample [1. 0.] expected [1.] produced [0.9070182707890654]
Sample [0. 1.] expected [1.] produced [0.9072344575097181]
Sample [1. 1.] expected [0.] produced [0.06019570677820066]
Epoch 30000 RMSE =  0.09949271417678227
Epoch 30100 RMSE =  0.09898280338784078
Epoch 30200 RMSE =  0.09848040889824035
Epoch 30300 RMSE =  0.09798537847705924
Epoch 30400 RMSE =  0.09749749362343682
Epoch 30500 RMSE =  0.0970166246692807
Epoch 30600 RMSE =  0.09654255923383419
Epoch 30700 RMSE =  0.09607520418665873
Epoch 30800 RMSE =  0.0956143561943347
Epoch 30900 RMSE =  0.09515989366559377
Sample [1. 1.] expected [0.] produced [0.055730064885276424]
Sample [1. 0.] expected [1.] produced [0.9117054241782021]
Sample [0. 0.] expected [0.] produced [0.13117150664685368]
Sample [0. 1.] expected [1.] produced [0.9118327742697028]
Epoch 31000 RMSE =  0.09471166256149649
Epoch 31100 RMSE =  0.09426949545440917
Epoch 31200 RMSE =  0.0938333238896032
Epoch 31300 RMSE =  0.09340296566709241
Epoch 31400 RMSE =  0.09297830314576608
Epoch 31500 RMSE =  0.09255921416182812
Epoch 31600 RMSE =  0.09214558260264526
Epoch 31700 RMSE =  0.09173729682117018
Epoch 31800 RMSE =  0.09133423062976487
Epoch 31900 RMSE =  0.09093628760195316
Sample [0. 0.] expected [0.] produced [0.1261220877386111]
Sample [0. 1.] expected [1.] produced [0.9158435193617115]
Sample [1. 0.] expected [1.] produced [0.9157311292211704]
Sample [1. 1.] expected [0.] produced [0.051981337169623325]
Epoch 32000 RMSE =  0.09054335459498075
Epoch 32100 RMSE =  0.09015533289475397
Epoch 32200 RMSE =  0.0897721155993031
Epoch 32300 RMSE =  0.08939360413879187
Epoch 32400 RMSE =  0.08901973028122556
Epoch 32500 RMSE =  0.08865037521653998
Epoch 32600 RMSE =  0.088285441359456
Epoch 32700 RMSE =  0.08792487469281375
Epoch 32800 RMSE =  0.08756856184627367
Epoch 32900 RMSE =  0.08721643096303523
Sample [0. 1.] expected [1.] produced [0.9194217024135981]
Sample [1. 1.] expected [0.] produced [0.04875242338276474]
Sample [0. 0.] expected [0.] produced [0.12163411273934588]
Sample [1. 0.] expected [1.] produced [0.9192537332976141]
Epoch 33000 RMSE =  0.08686840305294531
Epoch 33100 RMSE =  0.08652440093431461
Epoch 33200 RMSE =  0.08618433894167249
Epoch 33300 RMSE =  0.08584815013900156
Epoch 33400 RMSE =  0.08551576366914297
Epoch 33500 RMSE =  0.08518710001899482
Epoch 33600 RMSE =  0.0848621081038583
Epoch 33700 RMSE =  0.084540703009305
Epoch 33800 RMSE =  0.08422283410868771
Epoch 33900 RMSE =  0.08390842974286546
Sample [1. 1.] expected [0.] produced [0.04590635441033548]
Sample [1. 0.] expected [1.] produced [0.9224183416772859]
Sample [0. 1.] expected [1.] produced [0.9225535337717354]
Sample [0. 0.] expected [0.] produced [0.11760041360960895]
Epoch 34000 RMSE =  0.08359742743611567
Epoch 34100 RMSE =  0.08328976632279354
Epoch 34200 RMSE =  0.08298538392114714
Epoch 34300 RMSE =  0.08268424956717702
Epoch 34400 RMSE =  0.08238627494244048
Epoch 34500 RMSE =  0.08209141682255187
Epoch 34600 RMSE =  0.08179960790473433
Epoch 34700 RMSE =  0.08151083310226191
Epoch 34800 RMSE =  0.08122500615498747
Epoch 34900 RMSE =  0.08094209006780927
Sample [1. 0.] expected [1.] produced [0.925241034611679]
Sample [1. 1.] expected [0.] produced [0.04345725059403571]
Sample [0. 0.] expected [0.] produced [0.11388416179416969]
Sample [0. 1.] expected [1.] produced [0.9253111890160055]
Epoch 35000 RMSE =  0.08066203618941267
Epoch 35100 RMSE =  0.08038478973065409
Epoch 35200 RMSE =  0.08011032543886919
Epoch 35300 RMSE =  0.07983857376602208
Epoch 35400 RMSE =  0.07956950624952377
Epoch 35500 RMSE =  0.07930307081154613
Epoch 35600 RMSE =  0.07903922492088508
Epoch 35700 RMSE =  0.07877792956378017
Epoch 35800 RMSE =  0.07851914633752932
Epoch 35900 RMSE =  0.07826283070747257
Sample [0. 1.] expected [1.] produced [0.9278669687564491]
Sample [0. 0.] expected [0.] produced [0.11052840855642218]
Sample [1. 0.] expected [1.] produced [0.9277511824574097]
Sample [1. 1.] expected [0.] produced [0.041255212323990105]
Epoch 36000 RMSE =  0.078008953772305
Epoch 36100 RMSE =  0.07775746587860236
Epoch 36200 RMSE =  0.07750832992232337
Epoch 36300 RMSE =  0.07726151819068186
Epoch 36400 RMSE =  0.07701700407007683
Epoch 36500 RMSE =  0.07677473023428308
Epoch 36600 RMSE =  0.07653467232980243
Epoch 36700 RMSE =  0.07629679099095917
Epoch 36800 RMSE =  0.07606107436194576
Epoch 36900 RMSE =  0.07582746880071148
Sample [0. 1.] expected [1.] produced [0.9301579610367129]
Sample [1. 0.] expected [1.] produced [0.9300847226993116]
Sample [0. 0.] expected [0.] produced [0.10746217080441084]
Sample [1. 1.] expected [0.] produced [0.03930411838072619]
Epoch 37000 RMSE =  0.07559594612521224
Epoch 37100 RMSE =  0.07536649176057093
Epoch 37200 RMSE =  0.07513905636437336
Epoch 37300 RMSE =  0.07491361918294111
Epoch 37400 RMSE =  0.07469014868961862
Epoch 37500 RMSE =  0.07446862250442106
Epoch 37600 RMSE =  0.07424900612329635
Epoch 37700 RMSE =  0.07403126829862441
Epoch 37800 RMSE =  0.07381540280782585
Epoch 37900 RMSE =  0.07360136208485701
Sample [0. 0.] expected [0.] produced [0.10457035129816515]
Sample [1. 0.] expected [1.] produced [0.9321321565956905]
Sample [1. 1.] expected [0.] produced [0.03753475726287377]
Sample [0. 1.] expected [1.] produced [0.932220970651439]
Epoch 38000 RMSE =  0.07338912957855116
Epoch 38100 RMSE =  0.07317867424038112
Epoch 38200 RMSE =  0.07296998814991726
Epoch 38300 RMSE =  0.07276302496980948
Epoch 38400 RMSE =  0.0725577780536667
Epoch 38500 RMSE =  0.07235421058244754
Epoch 38600 RMSE =  0.07215231192140509
Epoch 38700 RMSE =  0.07195205093479246
Epoch 38800 RMSE =  0.07175341337037386
Epoch 38900 RMSE =  0.07155636896483646
Sample [1. 1.] expected [0.] produced [0.03596132079218143]
Sample [1. 0.] expected [1.] produced [0.9340768510820352]
Sample [0. 0.] expected [0.] produced [0.10194504778383989]
Sample [0. 1.] expected [1.] produced [0.9341392037062268]
Epoch 39000 RMSE =  0.0713609056355047
Epoch 39100 RMSE =  0.07116698957712458
Epoch 39200 RMSE =  0.07097461632545016
Epoch 39300 RMSE =  0.07078376042309809
Epoch 39400 RMSE =  0.07059439199803325
Epoch 39500 RMSE =  0.07040650866294217
Epoch 39600 RMSE =  0.0702200778640997
Epoch 39700 RMSE =  0.07003508822817053
Epoch 39800 RMSE =  0.06985151655647054
Epoch 39900 RMSE =  0.06966935057289883
Sample [0. 0.] expected [0.] produced [0.09948539252461563]
Sample [1. 0.] expected [1.] produced [0.9358287588278287]
Sample [0. 1.] expected [1.] produced [0.9359144318368237]
Sample [1. 1.] expected [0.] produced [0.03453103360314475]
Epoch 40000 RMSE =  0.06948856713131388
Epoch 40100 RMSE =  0.06930915222372361
Epoch 40200 RMSE =  0.06913108741449438
Epoch 40300 RMSE =  0.06895435215609001
Epoch 40400 RMSE =  0.06877894288466058
Epoch 40500 RMSE =  0.06860483126914968
Epoch 40600 RMSE =  0.06843200201004884
Epoch 40700 RMSE =  0.06826044856271637
Epoch 40800 RMSE =  0.06809014564524458
Epoch 40900 RMSE =  0.06792108370393417
Sample [1. 0.] expected [1.] produced [0.9374945777679115]
Sample [1. 1.] expected [0.] produced [0.03322785955089978]
Sample [0. 0.] expected [0.] produced [0.09721198605802388]
Sample [0. 1.] expected [1.] produced [0.9375434589952966]
Epoch 41000 RMSE =  0.06775324384552305
Epoch 41100 RMSE =  0.06758661564859597
Epoch 41200 RMSE =  0.06742118127032
Epoch 41300 RMSE =  0.06725692931492455
Epoch 41400 RMSE =  0.06709384414237654
Epoch 41500 RMSE =  0.06693191069660497
Epoch 41600 RMSE =  0.06677111906986113
Epoch 41700 RMSE =  0.0666114589156878
Epoch 41800 RMSE =  0.06645291087444703
Epoch 41900 RMSE =  0.06629546271054632
Sample [1. 0.] expected [1.] produced [0.939016110250107]
Sample [1. 1.] expected [0.] produced [0.03202652753723793]
Sample [0. 0.] expected [0.] produced [0.09507609569192789]
Sample [0. 1.] expected [1.] produced [0.9390629286468096]
Epoch 42000 RMSE =  0.06613910324506413
Epoch 42100 RMSE =  0.0659838174673554
Epoch 42200 RMSE =  0.06582960120414628
Epoch 42300 RMSE =  0.06567642992286588
Epoch 42400 RMSE =  0.06552430319525102
Epoch 42500 RMSE =  0.06537320771635512
Epoch 42600 RMSE =  0.06522312551877703
Epoch 42700 RMSE =  0.06507405033307125
Epoch 42800 RMSE =  0.0649259700314001
Epoch 42900 RMSE =  0.06477887154408977
Sample [1. 1.] expected [0.] produced [0.03091109867619372]
Sample [0. 0.] expected [0.] produced [0.09305892058234835]
Sample [0. 1.] expected [1.] produced [0.9404692610729858]
Sample [1. 0.] expected [1.] produced [0.9404164293732933]
Epoch 43000 RMSE =  0.06463274999399544
Epoch 43100 RMSE =  0.06448759005494022
Epoch 43200 RMSE =  0.06434338213438075
Epoch 43300 RMSE =  0.06420011392664493
Epoch 43400 RMSE =  0.06405777872655036
Epoch 43500 RMSE =  0.06391636385924712
Epoch 43600 RMSE =  0.06377586057971589
Epoch 43700 RMSE =  0.06363626152195279
Epoch 43800 RMSE =  0.06349755316828719
Epoch 43900 RMSE =  0.0633597286437629
Sample [0. 1.] expected [1.] produced [0.9418200978051712]
Sample [1. 0.] expected [1.] produced [0.9417698593398438]
Sample [1. 1.] expected [0.] produced [0.0299117671999339]
Sample [0. 0.] expected [0.] produced [0.09120368916311666]
Epoch 44000 RMSE =  0.06322277484160053
Epoch 44100 RMSE =  0.06308669107821882
Epoch 44200 RMSE =  0.06295145968552314
Epoch 44300 RMSE =  0.06281707604669955
Epoch 44400 RMSE =  0.06268353135162406
Epoch 44500 RMSE =  0.06255081593941433
Epoch 44600 RMSE =  0.06241891898013483
Epoch 44700 RMSE =  0.06228783721434687
Epoch 44800 RMSE =  0.06215756035919531
Epoch 44900 RMSE =  0.062028077040233665
Sample [0. 1.] expected [1.] produced [0.9430600622132437]
Sample [1. 1.] expected [0.] produced [0.028957103257896258]
Sample [1. 0.] expected [1.] produced [0.9430071681800851]
Sample [0. 0.] expected [0.] produced [0.08942752035947786]
Epoch 45000 RMSE =  0.06189938331830551
Epoch 45100 RMSE =  0.06177146913975899
Epoch 45200 RMSE =  0.061644329759257835
Epoch 45300 RMSE =  0.06151795447573898
Epoch 45400 RMSE =  0.061392334962300295
Epoch 45500 RMSE =  0.061267464557928954
Epoch 45600 RMSE =  0.06114333835254232
Epoch 45700 RMSE =  0.061019944798343446
Epoch 45800 RMSE =  0.06089728197021455
Epoch 45900 RMSE =  0.06077533977853405
Sample [0. 0.] expected [0.] produced [0.08773060227370418]
Sample [1. 1.] expected [0.] produced [0.028057490557806153]
Sample [0. 1.] expected [1.] produced [0.9442030625254295]
Sample [1. 0.] expected [1.] produced [0.9441564311497251]
Epoch 46000 RMSE =  0.060654109022358835
Epoch 46100 RMSE =  0.06053358856301145
Epoch 46200 RMSE =  0.060413764370438806
Epoch 46300 RMSE =  0.06029463564916506
Epoch 46400 RMSE =  0.06017619306510518
Epoch 46500 RMSE =  0.06005843329214241
Epoch 46600 RMSE =  0.05994134563673478
Epoch 46700 RMSE =  0.059824924440682506
Epoch 46800 RMSE =  0.05970916415508875
Epoch 46900 RMSE =  0.05959406192300878
Sample [1. 0.] expected [1.] produced [0.9452719617091473]
Sample [0. 0.] expected [0.] produced [0.08615498998018434]
Sample [0. 1.] expected [1.] produced [0.9453143973424472]
Sample [1. 1.] expected [0.] produced [0.027256904020849185]
Epoch 47000 RMSE =  0.05947960664898567
Epoch 47100 RMSE =  0.059365793811385695
Epoch 47200 RMSE =  0.059252617728596156
Epoch 47300 RMSE =  0.05914007231544414
Epoch 47400 RMSE =  0.05902815425756758
Epoch 47500 RMSE =  0.0589168537428908
Epoch 47600 RMSE =  0.058806168976805386
Epoch 47700 RMSE =  0.05869609081634499
Epoch 47800 RMSE =  0.0585866147807497
Epoch 47900 RMSE =  0.05847773739933763
Sample [0. 0.] expected [0.] produced [0.08463879932042467]
Sample [1. 0.] expected [1.] produced [0.9462946147392853]
Sample [1. 1.] expected [0.] produced [0.026486038546120916]
Sample [0. 1.] expected [1.] produced [0.9463486589916182]
Epoch 48000 RMSE =  0.05836945131288536
Epoch 48100 RMSE =  0.05826175113411457
Epoch 48200 RMSE =  0.05815463469637206
Epoch 48300 RMSE =  0.058048092295621835
Epoch 48400 RMSE =  0.05794212203541071
Epoch 48500 RMSE =  0.05783671622027993
Epoch 48600 RMSE =  0.057731872958438554
Epoch 48700 RMSE =  0.057627585863227374
Epoch 48800 RMSE =  0.057523847727484576
Epoch 48900 RMSE =  0.05742065878022768
Sample [1. 0.] expected [1.] produced [0.9472947769028914]
Sample [0. 1.] expected [1.] produced [0.9473501628626738]
Sample [0. 0.] expected [0.] produced [0.08322898776003332]
Sample [1. 1.] expected [0.] produced [0.025778029674746822]
Epoch 49000 RMSE =  0.057318010061792234
Epoch 49100 RMSE =  0.05721589957879714
Epoch 49200 RMSE =  0.05711432065762963
Epoch 49300 RMSE =  0.057013269555238985
Epoch 49400 RMSE =  0.056912741907067396
Epoch 49500 RMSE =  0.056812732762955336
Epoch 49600 RMSE =  0.05671323815060944
Epoch 49700 RMSE =  0.056614253051391124
Epoch 49800 RMSE =  0.05651577406031262
Epoch 49900 RMSE =  0.05641779564289278
Sample [0. 0.] expected [0.] produced [0.08184771913981283]
Sample [1. 1.] expected [0.] produced [0.025093618626439697]
Sample [1. 0.] expected [1.] produced [0.948208599149217]
Sample [0. 1.] expected [1.] produced [0.9482619716347279]
Epoch 50000 RMSE =  0.05632031517439128
Final Epoch RMSE =  0.05632031517439128

"""

