from autodiff_team29.node import Node
from autodiff_team29.elementaries import sin
import autodiff_team29.elementaries as E
from autodiff_team29 import VectorFunction
import numpy as np

# 'Scalar to Vector' case
x = Node("x", value=2, derivative=1)  


print('Node aspects:')
print(f"Symbol: {x.symbol}")
print(f'Value: {x.value}')
print(f"Derivative: {x.derivative}")

# Defining multiple functions: sin(x), cos(x) and exp(x)
f1, f2, f3 = E.sin(x), E.cos(x), E.exp(x)

# Instantiating the VectorFunction
f = VectorFunction([f1, f2, f3])

# Inspect f --> new `Node` with consistent representation
print("New Node representation:")
print("Symbol:", f.symbol)
print("Value", f.value)
print("Jacobian", f.jacobian)
