from autodiff_team29.node import Node
from autodiff_team29.elementaries import sin
import autodiff_team29.elementaries as E
from autodiff_team29 import VectorFunction
import numpy as np

# Vector Case
x1 = Node("x1", np.pi, 1, seed_vector=[1, 0])
x2 = Node("x2", np.pi / 2, 1, seed_vector=[0, 1])
print('Node aspects:')
print(f"Symbols: x1: {x1.symbol},  x2: {x2.symbol}")
print(f'Values: x1: {x1.value}, x2: {x2.value}')
print(f"Derivatives: x1: {x1.derivative}, x2: {x2.derivative}")

# Defining a function `f`
f1, f2 = x1 * x2 + E.sin(x1), x1 + x2 + E.sin(x1 * x2)

# Instantiating the VectorFunction
f = VectorFunction([f1, f2])

# Inspect f --> new `Node` with consistent representation
print("Symbol:", f.symbol)
print("Value", f.value)
print("Jacobian", f.jacobian)






