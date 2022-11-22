import numpy as np

_trace = {}

def sin(node):
    eval_node = Node(func_input=node, func='sin')
    result = Node(val=)
    try:
        return _trace[result]
    except KeyError:
        print('Node does not exist!')
        print(node.val, node.derivative)
        _trace[result] = result
    return result

class Node:
    def __init__(self, val=None, derivative=None, func_input=None, func=None):
        self.val = val
        self.derivative = derivative
        self.func_input = func_input
        self.func = func
        try:
            _trace[self]
        except KeyError:
            print('Node does not exist!')
            # print(self.val, self.derivative)
            _trace[self] = self

    def __add__(self, other):
        if not isinstance(other, Node):
            other = Node(other, 0)
        return Node(self.val + other.val, self.derivative + other.derivative)

    def __hash__(self):
        hash_key = '|'.join(str(s) for s in [self.val, self.derivative, self.func] if s is not None)
        hash_key = hash(hash_key)
        if self.func_input is not None:
            hash_key += hash(self.func_input)
        return hash_key

    def __eq__(self, other):
        return all([
            self.val == other.val,
            self.derivative == other.derivative,
            self.func_input = func_input,
            self.func == other.func
        ])

x = Node(5, 1)
f = sin(x) + (sin(x) + 2)