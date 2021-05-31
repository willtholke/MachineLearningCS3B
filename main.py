""" This project has been updated since its last instance such that...

Name: William Tholke
Course: CS3B w/ Professor Eric Reed
Date: 06/01/21
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


class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None
        self.back = None


class DoublyLinkedList:
    def __init__(self):
        self._head = None
        self._curr = None
        self._tail = None

    class EmptyListError(Exception):
        pass

    def add_to_head(self, data):
        """ Add data to the first node in the list. """
        new_node = Node(data)
        new_node.next = self._head
        self._head = new_node
        self.reset_to_head()

    def remove_from_head(self):
        """ Return data value if there is a node at the head or
        return None otherwise.
        """
        if self._head is None:
            return None
        ret_val = self._head.data
        self._head = self._head.next
        self.reset_to_head()
        return ret_val

    def reset_to_head(self):
        """ Reset current node to head. """
        self._curr = self._head
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        else:
            return self._curr.data

    def reset_to_tail(self):
        """ Reset current node to tail. """
        self._curr = self._tail
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        else:
            return self._curr.data

    def move_forward(self):
        """ Return the data from new node if it exists. """
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        self._curr = self._curr.next
        if self._curr is None:
            self.reset_to_head()
            raise IndexError
        else:
            return self._curr.data

    def move_back(self):
        """ Return the data from new node if it exists. """
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        self._curr = self._curr.next
        if self._curr is None:
            self.reset_to_tail()
            raise IndexError
        self._curr = self._curr.next
        return self._curr.data

    def add_after_curr(self, data):
        """ Alter the middle of a list by adding after a node. """
        if self._curr is None:
            self.add_to_head(data)
            raise DoublyLinkedList.EmptyListError
        new_node = Node(data)
        new_node.next = self._curr.next
        self.reset_to_tail()
        self._curr.next = new_node

    def remove_after_cur(self):
        """ Alter the middle of a list by removing after a node. """
        if self._curr is None or self._curr.next is None:
            return DoublyLinkedList.EmptyListError
        if self._curr == self._tail:
            raise IndexError
        ret_val = self._curr.next.data
        self._curr.next = self._curr.next.next
        self.reset_to_tail()
        return ret_val


    def get_current_data(self):
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        else:
            return self._curr.data

    def find(self, value):
        """ Return data at node if value parameter exists. """
        curr_pos = self._head
        while curr_pos is not None:
            if curr_pos.data == value:
                return curr_pos.data
            curr_pos = curr_pos.next
        return None

    def delete(self, value):
        """ Delete data at node if the current node's data is the
        same as the value parameter.
        """
        self.reset_to_head()
        if self._curr is None:
            return None
        if self._curr.data == value:
            return self.remove_from_head()
        while self._curr.next is not None:
            if self._curr.next.data == value:
                ret_val = self.remove_after_cur()
                self.reset_to_head()
                return ret_val
            self._curr = self._curr.next
        self.reset_to_head()
        return None

    def __iter__(self):
        self._curr_iter = self._head
        return self

    def __next__(self):
        if self._curr_iter is None:
            raise StopIteration
        ret_val = self._curr_iter.data
        self._curr_iter = self._curr_iter.next
        return ret_val


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


def layer_list_test():
    # create a LayerList with two inputs and four outputs
    my_list = LayerList(2, 4)
    # get a list of the input and output nodes, and make sure we have the right number
    inputs = my_list.input_nodes
    outputs = my_list.output_nodes
    assert len(inputs) == 2
    assert len(outputs) == 4
    # check that each has the right number of connections
    for node in inputs:
        assert len(node._neighbors[MultiLinkNode.Side.DOWNSTREAM]) == 4
    for node in outputs:
        assert len(node._neighbors[MultiLinkNode.Side.UPSTREAM]) == 2
    # check that the connections go to the right place
    for node in inputs:
        out_set = set(node._neighbors[MultiLinkNode.Side.DOWNSTREAM])
        check_set = set(outputs)
        assert out_set == check_set
    for node in outputs:
        in_set = set(node._neighbors[MultiLinkNode.Side.UPSTREAM])
        check_set = set(inputs)
        assert in_set == check_set
    # add a couple layers and check that they arrived in the right order, and that iterate and rev_iterate work
    my_list.reset_to_head()
    my_list.add_layer(3)
    my_list.add_layer(6)
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 6
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 3
    # save this layer to make sure it gets properly removed later
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.OUTPUT
    assert len(my_list.get_current_data()) == 4
    my_list.move_back()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 3
    # check that information flows through all layers
    save_vals = []
    for node in outputs:
        save_vals.append(node.value)
    for node in inputs:
        node.set_input(1)
    for i, node in enumerate(outputs):
        assert save_vals[i] != node.value
    # check that information flows back as well
    save_vals = []
    for node in inputs[1]._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
        save_vals.append(node.delta)
    for node in outputs:
        node.set_expected(1)
    for i, node in enumerate(inputs[1]._neighbors[MultiLinkNode.Side.DOWNSTREAM]):
        assert save_vals[i] != node.delta
    # try to remove an output layer
    try:
        my_list.remove_layer()
        assert False
    except IndexError:
        pass
    except:
        assert False
    # move and remove a hidden layer
    save_list = my_list.get_current_data()
    my_list.move_back()
    my_list.remove_layer()
    # check the order of layers again
    my_list.reset_to_head()
    assert my_list.get_current_data()[0].node_type == LayerType.INPUT
    assert len(my_list.get_current_data()) == 2
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 6
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.OUTPUT
    assert len(my_list.get_current_data()) == 4
    my_list.move_back()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 6
    my_list.move_back()
    assert my_list.get_current_data()[0].node_type == LayerType.INPUT
    assert len(my_list.get_current_data()) == 2
    # save a value from the removed layer to make sure it doesn't get changed
    saved_val = save_list[0].value
    # check that information still flows through all layers
    save_vals = []
    for node in outputs:
        save_vals.append(node.value)
    for node in inputs:
        node.set_input(1)
    for i, node in enumerate(outputs):
        assert save_vals[i] != node.value
    # check that information still flows back as well
    save_vals = []
    for node in inputs[1]._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
        save_vals.append(node.delta)
    for node in outputs:
        node.set_expected(1)
    for i, node in enumerate(inputs[1]._neighbors[MultiLinkNode.Side.DOWNSTREAM]):
        assert save_vals[i] != node.delta
    assert saved_val == save_list[0].value


if __name__ == "__main__":
    layer_list_test()

"""
-- Sample Run #1 --

"""