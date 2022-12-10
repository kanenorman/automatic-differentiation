from autodiff_team29.node import Node
from autodiff_team29.elementaries import sin
import autodiff_team29.elementaries as E
from autodiff_team29 import VectorFunction
import numpy as np

# Scalar case
x  = Node(symbol = 'x', value = np.pi, derivative =1)

print('Node aspects:')
print(f"Symbol: {x.symbol}")
print(f"Value: {x.value}")
print("Derivative: {x.derivative}")



# Defining a function `f`
f = sin(x) + x


# Inspect f --> New 'Node' with consistent symbolic representation
# and correct value at derivative `x=pi`

print('Representation:')
print(repr(f))





