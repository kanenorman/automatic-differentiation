from __future__ import annotations
from typing import Union
import warnings

import numpy as np
from numpy.typing import NDArray


class Node:

    # other types that are capable of being converted to Node
    _COMPATIBLE_VALUE_TYPES = (int, float)
    _COMPATIBLE_DERIVATIVE_TYPES = (int, float, np.ndarray)

    # store nodes that have been computed previously
    _NODE_REGISTRY = {}

    def __new__(
        cls,
        symbol: str,
        value: Union[float, int],
        derivative: Union[int, float],
        overwrite_existing: bool = False,
        supress_warning: bool = False,
        **kwargs,
    ) -> Node:
        """
        Represents a node which is the foundation of a computational graph.

        Parameters
        ----------
        symbol : str
                Symbolic representation of a Node instance that acts as a unique identifier.
        value : int, float
                Analytical value of the node.
        derivative : int, float,
                Derivative with respect to the value attribute

        Optional Parameters
        -------------------
        overwrite_existing : bool, default=True
                If node with matching symbol already exists, override the existing node stored in the registry

        supress_warning : bool, default=False
                supresses warnings for existing nodes that are recomputed

        **kwargs
        ---------
        seed_vector : List
                A seed vector for computing partial derivatives of multi-variable functions.
                The seed vector allows us to cherry-pick a certain derivative of interest (choose direction).
                For F:Rm --> Rn, our seed vector should be of length m with a 1 in the direction of interest and 0 elsewhere.

        Examples
        --------
        >>> x = Node('x',10,1)
        >>> Node('x',10,1)
        >>> x + x
        >>> Node('x+x',20,2)

        """
        # check if node already exist before recreating
        if not overwrite_existing:
            if cls._check_node_exists(symbol):
                if not supress_warning:
                    warnings.warn(
                        f"Node with symbol {symbol} already exist. Using existing node!",
                        RuntimeWarning,
                    )
                return cls._get_existing_node(symbol)

        # ensure that the values and derivatives specified are of the correct datatype
        # if they are not these methods will raise an exception
        cls._check_foreign_value_type_compatibility(value)
        cls._check_foreign_derivative_type_compatibility(derivative)

        # creating an instance of the class
        instance = super().__new__(cls)
        instance._symbol = symbol
        instance._value = value

        # if kwargs are specified we are dealing with an n-dimensional function
        if "seed_vector" in kwargs:
            seed_vector = np.array(kwargs["seed_vector"])
            instance._derivative = derivative * seed_vector
        else:
            instance._derivative = derivative

        cls._insert_node_to_registry(instance)

        return instance

    @property
    def symbol(self) -> str:
        """
        Returns symbolic representation of the computational node

        """
        return self._symbol

    @property
    def value(self) -> float | int:
        """
        Returns analytical value of the computational node

        """
        return self._value

    @property
    def derivative(self) -> int | float:
        """
        Returns derivative value of the computational node

        """
        return self._derivative

    @staticmethod
    def _check_foreign_value_type_compatibility(other_type: Union[int, float]) -> None:
        """
        Checks to see if a datatype can be represented as a node.

        Parameters
        ----------
        other_type : Any
            Python object that will be attempt being converted to a Node

        Raises
        -------
        TypeError
            Raises TypeError if type is unsupported

        Examples
        --------
        >>> Node._check_foreign_derivative_type_compatibility(100)
        >>> Node._check_foreign_derivative_type_compatibility("100")
        >>> TypeError Unsupported type 'str' for value attribute in class Node

        """
        if not isinstance(other_type, Node._COMPATIBLE_VALUE_TYPES):
            raise TypeError(
                f"Unsupported type '{type(other_type)}' for value attribute in class Node"
            )

    @staticmethod
    def _check_foreign_derivative_type_compatibility(
        other_type: Union[int, float, NDArray]
    ) -> None:
        """
        Checks to see if a datatype can be represented as a node.

        Parameters
        ----------
        other_type : Any
            Python object that will be attempt being converted to a Node

        Raises
        -------
        TypeError
            Raises TypeError if type is unsupported

        Examples
        --------
        >>> Node._check_foreign_derivative_type_compatibility(100)
        >>> Node._check_foreign_derivative_type_compatibility("100")
        >>> TypeError: Unsupported type 'str' for value attribute in class Node

        """
        if not isinstance(other_type, Node._COMPATIBLE_DERIVATIVE_TYPES):
            raise TypeError(
                f"Unsupported type '{type(other_type)}' for value attribute in class Node"
            )

    @classmethod
    def convert_to_node(cls, to_convert) -> Node:
        """
        Attempts to convert an numeric value into an instance of class Node.

        Parameters
        ----------
        to_convert : int, float
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

        Node._check_foreign_value_type_compatibility(to_convert)
        return cls(
            symbol=str(to_convert), value=to_convert, derivative=0, supress_warning=True
        )

    @staticmethod
    def _check_node_exists(key: str) -> bool:
        """
        Checks if an instance of class Node has already been created.

        Parameters
        ----------
        key : str
            Symbolic representation of a Node instance that acts as a unique identifier.

        Returns
        -------
        bool :
            True if key argument is found. False otherwise.

        """
        return key in Node._NODE_REGISTRY

    @staticmethod
    def _get_existing_node(key: str) -> Node:
        """
        Returns existing Node instance to avoid recomputing nodes.

        Parameters
        ----------
        key : str
            Symbolic representation of a Node instance that acts as a unique identifier.

        Returns
        -------
        Node :
            instance that matches the specified key.

        """
        return Node._NODE_REGISTRY[key]

    @staticmethod
    def _insert_node_to_registry(node: Node) -> None:
        """
        Adds Node instance to the registry, and allows computational graph to keep track of what nodes have
        already been computed .

        Parameters
        ----------
        node : Node
            Instance of class Node.

        Returns
        -------
        None

        """
        Node._NODE_REGISTRY[node._symbol] = node

    @staticmethod
    def clear_node_registry() -> None:
        """
        Removes all key value pairs from the node registry.
        Will Erase ALL previous computations made by the graph.
        """
        Node._NODE_REGISTRY.clear()

    def __add__(self, other: Union[int, float, Node]) -> Node:

        symbolic_representation = "({}+{})".format(*sorted([self._symbol, str(other)]))

        if self._check_node_exists(symbolic_representation):
            return self._get_existing_node(symbolic_representation)

        other = self.convert_to_node(other)
        primal_trace = self._value + other._value
        tangent_trace = self._derivative + other._derivative

        return Node(
            symbolic_representation, primal_trace, tangent_trace, supress_warning=True
        )

    def __radd__(self, other: Union[int, float]) -> Node:
        return self.__add__(other)

    def __sub__(self, other: Union[int, float, Node]) -> Node:

        symbolic_representation = "({}-{})".format(self._symbol, str(other))

        if self._check_node_exists(symbolic_representation):
            return self._get_existing_node(symbolic_representation)

        other = self.convert_to_node(other)
        primal_trace = self._value - other._value
        tangent_trace = self._derivative - other._derivative

        return Node(
            symbolic_representation, primal_trace, tangent_trace, supress_warning=True
        )

    def __rsub__(self, other: Union[int, float]) -> Node:

        symbolic_representation = "({}-{})".format(str(other), self._symbol)

        if self._check_node_exists(symbolic_representation):
            return self._get_existing_node(symbolic_representation)

        other = self.convert_to_node(other)
        primal_trace = other._value - self._value
        tangent_trace = other._derivative - self._derivative

        return Node(
            symbolic_representation, primal_trace, tangent_trace, supress_warning=True
        )

    def __mul__(self, other: Union[int, float, Node]) -> Node:

        symbolic_representation = "({}*{})".format(*sorted([self._symbol, str(other)]))

        if self._check_node_exists(symbolic_representation):
            return self._get_existing_node(symbolic_representation)

        other = self.convert_to_node(other)
        primal_trace = self._value * other._value
        tangent_trace = (
            self._value * other._derivative + other._value * self._derivative
        )

        return Node(
            symbolic_representation, primal_trace, tangent_trace, supress_warning=True
        )

    def __rmul__(self, other: Union[int, float]) -> Node:
        return self.__mul__(other)

    def __truediv__(self, other: Union[int, float, Node]) -> Node:
        symbolic_representation = "({}/{})".format(self._symbol, str(other))

        if self._check_node_exists(symbolic_representation):
            return self._get_existing_node(symbolic_representation)

        other = self.convert_to_node(other)
        primal_trace = self._value / other._value
        tangent_trace = (
            self._derivative * other._value - self._value * other._derivative
        ) / other._value**2

        return Node(
            symbolic_representation, primal_trace, tangent_trace, supress_warning=True
        )

    def __rtruediv__(self, other: Union[int, float]) -> Node:
        symbolic_representation = "({}/{})".format(str(other), self._symbol)

        if self._check_node_exists(symbolic_representation):
            return self._get_existing_node(symbolic_representation)

        other = self.convert_to_node(other)
        primal_trace = other._value / self._value
        tangent_trace = (
            self._value * other._derivative - other._value * self._derivative
        ) / self._value**2

        return Node(
            symbolic_representation, primal_trace, tangent_trace, supress_warning=True
        )

    def __neg__(self) -> Node:
        symbolic_representation = "-{}".format(self._symbol)

        if self._check_node_exists(symbolic_representation):
            return self._get_existing_node(symbolic_representation)

        primal_trace = -1 * self._value
        tangent_trace = -1 * self._derivative

        return Node(
            symbolic_representation, primal_trace, tangent_trace, supress_warning=True
        )

    def __pow__(self, exponent: Union[int, float, Node]) -> Node:
        symbolic_representation = "({}**{})".format(self._symbol, str(exponent))

        if self._check_node_exists(symbolic_representation):
            return self._get_existing_node(symbolic_representation)

        exponent = self.convert_to_node(exponent)
        primal_trace = self._value**exponent._value
        tangent_trace = self._value**exponent._value * (exponent._derivative * np.log(self._value) + (self._derivative * exponent._value) / self._value)

        return Node(
            symbolic_representation, primal_trace, tangent_trace, supress_warning=True
        )

    def __str__(self) -> str:
        return self._symbol

    def __repr__(self) -> str:
        return f"Node({self._symbol},{self._value},{self._derivative})"

    def __eq__(self, other: Node) -> bool:
        symbolic_representation_equal = self._symbol == other._symbol
        value_equal = self._value = other._value
        derivative_equal = self._derivative = other._derivative

        return all([symbolic_representation_equal, value_equal, derivative_equal])
