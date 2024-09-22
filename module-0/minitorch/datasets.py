import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """
    if x_1 coordinate < 0.5, label is 1. Otherwise 0. The boundary is perpendicular at 0.5.

    Args:
        N: The number of points sampled.

    Returns:
        The dataset with information of the sampled points, labels and the number of points.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """
    if x_1 coordinate + x_2 coordinate < 0.5, label is 1. Otherwise 0. The boundary is a line passing through (0, 0.5) and (0.5, 0).

    Args:
        N: The number of points sampled.

    Returns:
        The dataset with information of the sampled points, labels and the number of points.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """
    if x_1 coordinate < 0.2 or x_1 coordinate > 0.8, label is 1. Otherwise 0. The boundary is two verticle lines passing through 0.2 and 0.8 respectively.

    Args:
        N: The number of points sampled.

    Returns:
        The dataset with information of the sampled points, labels and the number of points.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """
    The xor boundary where x_1 < 0.5 and x_2 > 0.5 is label 1 and x_1 > 0.5 and x_2 < 0.5 is label 1. Otherwise label 0.

    Args:
        N: The number of points sampled.

    Returns:
        The dataset with information of the sampled points, labels and the number of points.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """
    The circle boundary where the square Euclidean distance of point (x_1, x_2) to the origin >= 1 is label 1. Otherwise label 0.

    Args:
        N: The number of points sampled.

    Returns:
        The dataset with information of the sampled points, labels and the number of points.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """
    The spiral boundary.

    Args:
        N: The number of points sampled.

    Returns:
        The dataset with information of the sampled points, labels and the number of points.
    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
