"""
Collection of the core mathematical operators used throughout the code base.
"""

import math

# ## Task 0.1
from typing import Callable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiply two numbers"""
    return float(x * y)


def id(x: float) -> float:
    """Return the input unchanged"""
    return float(x)


def add(x: float, y: float) -> float:
    """Add two numbers"""
    return float(x + y)


def neg(x: float) -> float:
    """Negates a number"""
    return float(-x)


def lt(x: float, y: float) -> float:
    """Checks if one number is less than another"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if two numbers are equal"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers"""
    return float(x) if x > y else float(y)


def is_close(x: float, y: float) -> float:
    """Checks if two numbers are close in value"""
    return 1.0 if abs(x - y) < 1e-7 else 0.0


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function"""
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function"""
    return float(x) if x > 0 else 0.0


def log(x: float) -> float:
    """Applies the ReLU activation function"""
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function"""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal"""
    return 1 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log for x and times a second arg"""
    return (1 / x) * y


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal for x and times a second arg"""
    return -(1 / x**2) * y


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU for x and times a second arg"""
    return (1.0 if x > 0 else 0.0) * y


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(f: Callable[[float], float], x: list[float]) -> list[float]:
    """Higher-order function that applies a given function to each element of an iterable"""
    new_x = []
    for item in x:
        new_x.append(f(item))
    return new_x


def zipWith(
    f: Callable[[float, float], float], x: list[float], y: list[float]
) -> list[float]:
    """Higher-order function that combines elements from two iterables using a given function"""
    comb = []
    for x_item, y_item in zip(x, y):
        comb.append(f(x_item, y_item))
    return comb


def reduce(f: Callable[[float, float], float], x: list[float]) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function"""
    ans = x[0]
    for item in x[1:]:
        ans = f(ans, item)
    return ans


def negList(x: list[float]) -> list[float]:
    """Negate all elements in a list"""
    return map(neg, x)


def addLists(x: list[float], y: list[float]) -> list[float]:
    """Add corresponding elements from two lists"""
    return zipWith(add, x, y)


def sum(x: list[float]) -> float:
    """Sum all elements in a list"""
    if not x:
        return 0
    return reduce(add, x)


def prod(x: list[float]) -> float:
    """Calculate the product of all elements in a list"""
    if not x:
        return 0
    return reduce(mul, x)
