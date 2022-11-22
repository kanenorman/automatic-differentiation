


#importing Node class:

import pytest

from node import Node

import numpy as np

class Tests_Node:

    """ Test class for Milestone 2 """

    """ The testing suite that follows attempts to test every single method implemented as part"""
    """ of the Node() class implemented in Milestone2 """

    # Defining other types that are able to be converted into nodes:
    _COMPATIBLE_TYPES = (int, float, np.array)
    
    def test_init(self):

        """
        Testing the instantiation of a node which is the foundation of a computational graph.
        :param symbol: Symbolic representation of a Node instance that acts as a unique identifier.
        :param value: Analytical value of the node.
        :param derivative: Derivative with respect to the value attribute. Default=1.
        We test here an example instantiation:
        > x = Node('x', 10, 1)
        """

        node = Node('x', 10, 1)
        assert node._symbol == 'x'

        assert node._value == 10
        assert node._derivative == 1


    
    def test_value(self):   
        
        """Testing the `value` @property method"""

        node = Node('x', 100, 1)
        assert node.value == 100

    
    def test_symbol(self):

        """Testing the `symbol` method"""

        node = Node('x', 10, 1)
        assert node.symbol == 'x'

    
    def test_derivative(self):

        """Testing the `derivative` method"""

        node = Node('x', 10, 1)
        assert node.derivative == 1

    

    def test_check_foreign_type_compatibility(other_type):
        """
        Testing the method that sees if a datatype can be represented as a node.
        :param other_type: Type to check for conversion compatibility.
        :return: boolean. True if other_type is a supported type. False otherwise.

        We need to test all cases that are included in the _COMPATIBLE_TYPES options. 

        """
        
        

        assert Node._check_foreign_type_compatibility(1) == isinstance(1, Node._COMPATIBLE_TYPES)
        assert Node._check_foreign_type_compatibility(1.0) == isinstance(1.00, Node._COMPATIBLE_TYPES)
        assert Node._check_foreign_type_compatibility(np.ndarray([1,2,3])) == isinstance(np.ndarray([1,2,3]), Node._COMPATIBLE_TYPES)

    def test_check_node_exists(key):

        Node._NODE_REGISTRY = {'v0':Node('x', 10, 1)}

        assert Node._check_node_exists('v0') == True


    def test_check_get_existing_node(key):
        """
        Testing static method that returns existing Node instance to avoid recomputing nodes.
        :param key: Symbolic representation of a Node instance that acts as a unique identifier.
        :return: Node instance that matches the specified key.
        """
        Node._NODE_REGISTRY = {'v0':Node('x', 10, 1)}
        
        assert Node._get_existing_node('v0') == Node._NODE_REGISTRY['v0']

    
    def test__insert_node_to_registry(node):
        """
        Testing method that adds Node instance to the registry. Allows computational graph to keep track of what nodes have
        already been computed .
        :param node: Instance of class Node.
        :return: None.
        """

        node = Node('v0', 10, 1)

        Node._NODE_REGISTRY = {'v0':node}

        assert Node._NODE_REGISTRY[node._symbol] == node
        

    def test_add(self):

        """Function that tests correct output for __add__ method."""
        
        # Testing case where other value and derivative are integers
        node1 = Node('v0', 100, 1)
        node2 = Node('v1', 101, 1)

        new_node = node1 + node2

        assert new_node.symbol == "(v0+v1)"
        assert new_node.value == 201
        assert new_node.derivative == 2
        
        # Testing case where "other" value is float:
        node1 = Node('v0', 100, 1)
        node2 = Node('v1', 101.00, 1)

        new_node = node1 + node2

        assert new_node.symbol == "(v0+v1)"
        assert new_node.value == 201.00
        assert new_node.derivative == 2

        
        # Testing case where "other" value is a numpy array
        node1 = Node('v2', 100, 1)
        node2 = Node('v3', np.array([1,2,3]), 1)
        
        new_node = node1 + node2

        assert new_node.symbol == "(v2+v3)"
        print(new_node.value)
        assert all(new_node.value == np.array([101,102,103]))    
        assert new_node.derivative == 2
       
    def test_radd(self):
        
        """Testing __radd__ method for cases where:
        - other is of type `int`
        - other is of type `float`
        - other is of type `numpy.ndarray`"""

        # Int
        node1 = Node('v0', 100, 1)
        node2 = Node('v1', 101, 1)

        new_node = node1 + node2
        new_node_reverse = node2 + node1

        assert new_node.symbol == new_node_reverse.symbol
        assert new_node.value == new_node_reverse.value
        assert new_node.derivative == new_node_reverse.derivative

        # Float
        node1 = Node('v0', 100, 1)
        node2 = Node('v1', 101.00, 1)

        new_node = node1 + node2
        new_node_reverse = node2 + node1

        assert new_node.symbol == new_node_reverse.symbol
        assert new_node.value == new_node_reverse.value
        assert new_node.derivative == new_node_reverse.derivative
        

        # np.ndarray
        node1 = Node('v0', 100, 1)
        node2 = Node('v1', np.array([1,2,3]), 1)

        new_node = node1 + node2
        new_node_reverse = node2 + node1

        assert new_node.symbol == new_node_reverse.symbol
        assert new_node.value == new_node_reverse.value
        assert new_node.derivative == new_node_reverse.derivative

    def test_sub(self):

        """Function that tests correct output for __sub__ method."""
        
         # Testing case where other value and derivative are integers
        node1 = Node('v0', 100, 1)
        node2 = Node('v1', 101, 1)

        new_node = node1 - node2

        assert new_node.symbol == "(v0-v1)"
        assert new_node.value == -1
        assert new_node.derivative == 0
        
        # Testing case where "other" value is float:
        node1 = Node('v0', 100, 1)
        node2 = Node('v1', 101.00, 1)

        new_node = node1 - node2

        assert new_node.symbol == "(v0-v1)"
        assert new_node.value == -1.00
        assert new_node.derivative == 0

        
        # Testing case where "other" value is a numpy array
        node1 = Node('v2', 100, 1)
        node2 = Node('v3', np.array([1,2,3]), 1)
        
        new_node = node1.__sub__(node2)

        assert new_node.symbol == "(v2-v3)"
        assert all(new_node.value == np.array([99,98,97]))     
        assert new_node.derivative == 0



    def test_rsub(self):
        
        """Testing __rsub__ method for cases where:
        - other is of type `int`
        - other is of type `float`
        - other is of type `numpy.ndarray`"""

        # Testing case where other value and derivative are integers
        node1 = Node('v0', 100, 1)
        node2 = Node('v1', 101, 1)

        new_node = node2 - node1

        assert new_node.symbol == "(v1-v0)"
        assert new_node.value == 1
        assert new_node.derivative == 0
        
        # Testing case where "other" value is float:
        node1 = Node('v0', 100, 1)
        node2 = Node('v1', 101.00, 1)

        new_node = node2 - node1

        assert new_node.symbol == "(v1-v0)"
        assert new_node.value == 1.00
        assert new_node.derivative == 0

        
        # Testing case where "other" value is a numpy array
        node1 = Node('v0', 100, 1)
        node2 = Node('v1', np.array([1,2,3]), 1)
        
        new_node = node1.__sub__(node2)

        assert new_node.symbol == "(v0-v1)"
        #assert all(new_node.value == np.array([99,98,97]))     
        #assert new_node.derivative == 0 
    


    def test_mul(self):

        """Function that tests correct output for __mul__ method."""
        
        # Testing case where other value and derivative are integers
        node1 = Node('v0', 100, 1)
        node2 = Node('v1', 101, 1)

        new_node = node1 * node2

        assert new_node.symbol == "(v0*v1)"
        assert new_node.value == 10100
        assert new_node.derivative == 201
        
        # Testing case where "other" value is float:
        node1 = Node('v0', 100, 1)
        node2 = Node('v1', 101.00, 1)

        new_node = node1 * node2

        assert new_node.symbol == "(v0*v1)"
        assert new_node.value == 10100.00
        assert new_node.derivative == 201.00

        
        
    def test_mul_array(self):
        # Testing case where "other" value is a numpy array
        node1 = Node('v2', 100, 1)
        node2 = Node('v3', np.array([1,2,3]), 1)
        
        new_node = node1.__mul__(node2)

        assert new_node.symbol == "(v2*v3)"
        assert all(new_node.value == np.array([100,200,300])) == True   
        assert all(new_node.derivative == np.array([101,102,103]))

    



    def test_rmul(self):
        
        """Testing __rmul__ method for cases where:
        - other is of type `int`
        - other is of type `float`
        - other is of type `numpy.ndarray`"""

        # Int
        node1 = Node('v0', 100, 1)
        node2 = Node('v1', 101, 1)

        new_node = node1 * node2
        new_node_reverse = node2 * node1

        assert new_node.symbol == new_node_reverse.symbol
        assert new_node.value == new_node_reverse.value
        assert new_node.derivative == new_node_reverse.derivative

        # Float
        node1 = Node('v0', 100, 1)
        node2 = Node('v1', 101.00, 1)

        new_node = node1 * node2
        new_node_reverse = node2 * node1

        assert new_node.symbol == new_node_reverse.symbol
        assert new_node.value == new_node_reverse.value
        assert new_node.derivative == new_node_reverse.derivative
        
    def test_rmul_array(self):

        # np.ndarray
        node1 = Node('v0', 100, 1)
        node2 = Node('v1', np.array([1,2,3]), 1)

        new_node = node1 * node2
        new_node_reverse = node2 * node1

        assert new_node.symbol == new_node_reverse.symbol
        assert new_node.value == new_node_reverse.value
        assert new_node.derivative == new_node_reverse.derivative

    



    def test_truediv(self):

        """Function that tests correct output for __truediv__ method."""
        
        # Testing case where other value and derivative are integers
        node1 = Node('v0', 100, 1)
        node2 = Node('v1', 101, 1)

        new_node = node1 / node2

        assert new_node.symbol == "(v0/v1)"
        assert new_node.value == 100/101

        # Using chain rule:
        assert new_node.derivative == 1
        
        # Testing case where "other" value is float:
        node1 = Node('v0', 100, 1)
        node2 = Node('v1', 101.00, 1)

        new_node = node1/ node2

        assert new_node.symbol == "(v0/v1)"
        assert new_node.value == 100/101.00
        assert new_node.derivative == 1.00

    def test_truediv_array(self):

        # Testing case where "other" value is a numpy array
        node1 = Node('v2', 100, 1)
        node2 = Node('v3', np.array([1,2,3]), 1)
        
        new_node = node1.__truediv__(node2)

        assert new_node.symbol == "(v2/v3)"
        assert all(new_node.value == np.array([100, 50, 100/3]))    
        assert all(new_node.derivative == np.array([-99, -98, -97]))

    def test_rtruediv(self):

        """Function that tests correct output for __rtruediv__ method."""
        
        # Testing case where other value and derivative are integers
        node1 = Node('v0', 100, 1)
        node2 = Node('v1', 101, 1)

        new_node = node2 / node1

        assert new_node.symbol == "(v1/v0)"
        assert new_node.value == 101/100

        # Using chain rule:
        assert new_node.derivative == -1
        
        # Testing case where "other" value is float:
        node1 = Node('v0', 100, 1)
        node2 = Node('v1', 101.00, 1)

        new_node = node2/ node1

        assert new_node.symbol == "(v1/v0)"
        assert new_node.value == 101/100.00
        assert new_node.derivative == -1.00

    def test_rtruediv_array(self):
        #Testing case where "other" value is a numpy array
        node1 = Node('v2', 100, 1)
        node2 = Node('v3', np.array([1,2,3]), 1)
        
        new_node = node2.__truediv__(node1)

        assert new_node.symbol == "(v3/v2)"
        assert all(new_node.value == np.array([0.01, 0.02, 0.03]))    
        assert all(new_node.derivative == np.array([99,98,97]))

    
    def test_neg(self):

        # Int:
        node1 = Node('v2', 100, 1)
        
        new_node = Node.__neg__(node1)
        assert new_node.symbol == "-v2"
        assert new_node.derivative == -1
        assert new_node.value == -100

        # Float:
        node1 = Node('v3', 100.00, 1)       
        new_node = Node.__neg__(node1)
        assert new_node.symbol == "-v3"
        assert new_node.derivative == -1
        assert new_node.value == -100.00

    def test_neg_array(self):
        node1 = Node('v4', np.array([1,2,3]), 1) 
        new_node = Node.__neg__(node1)

        assert new_node.symbol == "-v4"
        assert all(new_node.value == -np.array([1,2,3]))
        assert new_node.derivative == -1

    def test_pow(self):
        # Int:
        node1 = Node('v2', 100, 1)
        node2 = Node('v3',100, 1)
        
        new_node = node1.__pow__(node2)
        assert new_node.symbol == "(v2**v3)"
        assert new_node.derivative == 100*100**(99)
        assert new_node.value == 10000

        # Float:
        node1 = Node('v2', 100, 1)
        node2 = Node('v3',100.00, 1)
        
        new_node = node1.__pow__(node2)
        assert new_node.symbol == "(v2**v3)"
        assert new_node.derivative == 100*100.00**(99)
        assert new_node.value == 10000.0

    def test_pow_array(self):
        node1 = Node('v2', 100, 1)
        node2 = Node('v3',np.array([1,1,1]), 1)

        new_node = node1.__pow__(node2)

        assert new_node.symbol == "(v2**v3)"
        assert new_node.derivative == np.array([1,1,1])*100**(100-np.array([1,1,1]))
        assert new_node.value == 100*(np.array([1,1,1]))

    def test_str(self):
        # Int
        node1 = Node('v1', 100, 1)
        assert Node.__str__(node1) == 'v1'
        
        # Float
        node2 = Node('v3',100.00, 1)
        assert Node.__str__(node2) == 'v3'

        # Array:
        node3 = Node('v4', 1001, np.array([34, 35, 39]))
        assert Node.__str__(node3) == 'v4'

    def test_repr(self):
        # Int case
        node1 = Node('v1', 1000, 1)
        assert Node.__repr__(node1) == f"Node({node1._symbol},{node1._value},{node1._derivative})"

        #Float case:
        node2 = Node('v2', 101.03, 2)
        assert Node.__repr__(node2) == f"Node({node2._symbol},{node2._value},{node2._derivative})"

        #Array case:
        node3 = Node('v3', np.array([76,77,78]), 3)
        assert Node.__repr__(node3) == f"Node({node3._symbol},{node3._value},{node3._derivative})"
        





        
             


