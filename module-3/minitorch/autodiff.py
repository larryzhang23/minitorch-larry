from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
from collections import deque

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    delta_vals1 = list(vals)
    delta_vals1[arg] = vals[arg] + epsilon
    delta_vals2 = list(vals)
    delta_vals2[arg] = vals[arg] - epsilon
    return (f(*delta_vals1) - f(*delta_vals2)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # count the in-degree for each node except the right-most variable
    nodes = deque([variable])
    in_degrees = {}
    while nodes:
        node = nodes.popleft()
        if node.is_constant():
            continue
        parents = node.parents
        for parent in parents:
            if parent.unique_id not in in_degrees:
                nodes.append(parent)
            in_degrees[parent.unique_id] = in_degrees.get(parent.unique_id, 0) + 1
   
    queue = deque([variable])
    sorts = []
    while queue:
        node = queue.popleft()
        if node.is_constant():
            continue
        sorts.append(node)
        parents = node.parents
        for parent in parents:
            in_degrees[parent.unique_id] -= 1
            # if all ancestors are reached, the node can be enqueued
            if in_degrees[parent.unique_id] == 0:
                queue.append(parent)
    return sorts


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    nodes = topological_sort(variable)
    deriv_map = {variable.unique_id: deriv}
    for node in nodes:
        node_deriv = deriv_map[node.unique_id]
        if node.is_leaf():
            node.accumulate_derivative(node_deriv)
        else:
            back = node.chain_rule(node_deriv)
            for var, var_deriv in back:
                deriv_map[var.unique_id] = deriv_map.get(var.unique_id, 0.0) + var_deriv
    
    
@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
