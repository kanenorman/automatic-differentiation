import numpy as np


class Node:

    # other types that are capable of being converted to Node
    _COMPATIBLE_TYPES = (int, float, np.ndarray)

    # store nodes that have been computed previously
    _NODE_REGISTRY = {}

    def __init__(self, symbol, value, derivative=1):
        """
        Represents a node which is the foundation of a computational graph.
        
        Parameters
        ----------
        symbol: str
                Symbolic representation of a Node instance that acts as a unique identifier.
        value: int, float, or np.ndarray
                Analytical value of the node.
        derivative: int, float, or np.ndarray consisting of either type int/float , optional, default = 1
                Derivative with respect to the value attribute

        Examples
        --------
        > x = Node('x', 10, 1)
        Node('x', 10, 1)
        > y = Node('y', 20)
        Node('y', 20, 1)
        """
        self._symbol = symbol
        self._value = value
        self._derivative = derivative

    @property
    def value(self):
        return self._value

    @property
    def symbol(self):
        return self._symbol

    @property
    def derivative(self):
        return self._derivative

    @staticmethod
    def _check_foreign_type_compatibility(other_type):
        """
        Checks to see if a datatype can be represented as a node.
        
        Parameters
        ----------
        other_type: Object
            Type of object that will be attempt being converted to a Node
            
        Returns
        -------
        bool
            True if other_type is a supported type. False otherwise.
            
        Examples
        --------
        > Node._check_foreign_type_compatibility(100)
        True
        > Node._check_foreign_type_compatibility("100")
        False
        """
        return isinstance(other_type, Node._COMPATIBLE_TYPES)

    @staticmethod
    def _convert_to_node(to_convert):
        """
        Attempts to convert an object into an instance of class Node.

        Parameters
        ----------
        to_convert: Object
            Object that will convert to type Node.
        
        Returns
        -------
        Node:
            instance of class Node created from to_convert.
            
        Raises
        ------
            TypeError if to_convert is an unsupported data type.
        """
        if isinstance(to_convert, Node):
            return to_convert

        if Node._check_foreign_type_compatibility(to_convert):
            return Node(symbol=str(to_convert), value=to_convert)

        raise TypeError(
            f"Unsupported type {type(to_convert)} for instance of class Node"
        )

    @staticmethod
    def _check_node_exists(key):
        """
        Checks if an instance of class Node has already been created.

        Parameters
        ----------
        key: str
            Symbolic representation of a Node instance that acts as a unique identifier.
            
        Returns
        -------
        bool:
            True if key argument is found. False otherwise.
        """
        return key in Node._NODE_REGISTRY

    @staticmethod
    def _get_existing_node(key):
        """
        Returns existing Node instance to avoid recomputing nodes.

        Parameters
        ----------
        key: str
            Symbolic representation of a Node instance that acts as a unique identifier.
            
        Returns
        -------
        Node:
            instance that matches the specified key.
        """
        return Node._NODE_REGISTRY[key]

    @staticmethod
    def _insert_node_to_registry(node):
        """
        Adds Node instance to the registry. Allows computational graph to keep track of what nodes have
        already been computed .

        Parameters
        ----------
        node: Node
            Instance of class Node.
        Returns
        -------
        None
        """
        Node._NODE_REGISTRY[node._symbol] = node

    def __add__(self, other):

        symbolic_representation = "({}+{})".format(*sorted([self._symbol, str(other)]))

        if self._check_node_exists(symbolic_representation):
            return self._get_existing_node(symbolic_representation)

        other = self._convert_to_node(other)
        primal_trace = self._value + other._value
        tangent_trace = self._derivative + other._derivative

        new_node = Node(
            symbol=symbolic_representation,
            value=primal_trace,
            derivative=tangent_trace,
        )

        Node._insert_node_to_registry(new_node)
        return new_node

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):

        symbolic_representation = "({}-{})".format(self._symbol, str(other))

        if self._check_node_exists(symbolic_representation):
            return self._get_existing_node(symbolic_representation)

        other = self._convert_to_node(other)
        primal_trace = self._value - other._value
        tangent_trace = self._derivative - other._derivative

        new_node = Node(
            symbol=symbolic_representation,
            value=primal_trace,
            derivative=tangent_trace,
        )

        Node._insert_node_to_registry(new_node)
        return new_node

    def __rsub__(self, other):

        symbolic_representation = "({}-{})".format(str(other), self._symbol)

        if self._check_node_exists(symbolic_representation):
            return self._get_existing_node(symbolic_representation)

        other = self._convert_to_node(other)
        primal_trace = other._value - self._value
        tangent_trace = other._derivative - self._derivative

        new_node = Node(
            symbol=symbolic_representation,
            value=primal_trace,
            derivative=tangent_trace,
        )

        Node._insert_node_to_registry(new_node)
        return new_node

    def __mul__(self, other):

        symbolic_representation = "({}*{})".format(*sorted([self._symbol, str(other)]))

        if self._check_node_exists(symbolic_representation):
            return self._get_existing_node(symbolic_representation)

        other = self._convert_to_node(other)
        primal_trace = self._value * other._value
        tangent_trace = (
            self._value * other._derivative + other._value * self._derivative
        )

        new_node = Node(
            symbol=symbolic_representation,
            value=primal_trace,
            derivative=tangent_trace,
        )

        Node._insert_node_to_registry(new_node)
        return new_node

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        symbolic_representation = "({}/{})".format(self._symbol, str(other))

        if self._check_node_exists(symbolic_representation):
            return self._get_existing_node(symbolic_representation)

        other = self._convert_to_node(other)
        primal_trace = self._value / other._value
        tangent_trace = (
            self._derivative * other._value - self._value * other._derivative
        ) / other._derivative**2

        new_node = Node(
            symbol=symbolic_representation,
            value=primal_trace,
            derivative=tangent_trace,
        )

        Node._insert_node_to_registry(new_node)
        return new_node

    def __rtruediv__(self, other):
        symbolic_representation = "({}/{})".format(str(other), self._symbol)

        if self._check_node_exists(symbolic_representation):
            return self._get_existing_node(symbolic_representation)

        other = self._convert_to_node(other)
        primal_trace = self._value / other._value
        tangent_trace = (
            other._derivative * self._value - other._value * self._derivative
        ) / self._derivative**2

        new_node = Node(
            symbol=symbolic_representation,
            value=primal_trace,
            derivative=tangent_trace,
        )

        Node._insert_node_to_registry(new_node)
        return new_node

    def __neg__(self):
        symbolic_representation = "-{}".format(self._symbol)

        if self._check_node_exists(symbolic_representation):
            return self._get_existing_node(symbolic_representation)

        primal_trace = -1 * self._value
        tangent_trace = -1 * self._derivative
        new_node = Node(symbolic_representation, primal_trace, tangent_trace)
        self._insert_node_to_registry(new_node)
        return new_node

    def __pow__(self, exponent):
        symbolic_representation = "({}**{})".format(str(exponent), self._symbol)

        if self._check_node_exists(symbolic_representation):
            return self._get_existing_node(symbolic_representation)

        exponent = self._convert_to_node(exponent)
        primal_trace = self._value**exponent._value
        tangent_trace = exponent._value * self._value ** (exponent._value - 1)

        new_node = Node(
            symbol=symbolic_representation,
            value=primal_trace,
            derivative=tangent_trace,
        )
        self._insert_node_to_registry(new_node)
        return new_node

    def __str__(self):
        return self._symbol

    def __repr__(self):
        return f"Node({self._symbol},{self._value},{self._derivative})"

    def __eq__(self, other):
        symbolic_representation_equal = self._symbol == other._symbol
        value_equal = self._value = other._value
        derivative_equal = self._derivative = other._derivative

        return all([symbolic_representation_equal, value_equal, derivative_equal])


if __name__ == "__main__":
    x = Node("x", 3, 1)
    z = -x
    print(z._NODE_REGISTRY)
