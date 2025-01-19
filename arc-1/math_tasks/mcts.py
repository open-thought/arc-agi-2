from abc import ABC, abstractmethod
from typing import Iterator, Self
import math


class MctsParamsBase(ABC):
    def __init__(self):
        self.exploration_weight = 1.0

        # progressive widening parameters
        self.alpha = 0.5
        self.k = 1

    @abstractmethod
    def create_child(self, parent: "NodeBase") -> "NodeBase":
        pass


class NodeBase:
    def __init__(self, params: MctsParamsBase, parent: Self = None):
        self.params = params
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.terminal = False

    @property
    def depth(self) -> int:
        depth = 0
        for _ in self.ancestors(include_self=False):
            depth += 1
        return depth

    def ancestors(self, include_self: bool = True) -> Iterator[Self]:
        node = self if include_self else self.parent
        while node is not None:
            yield node
            node = node.parent

    @property
    def uct(self) -> float:
        if self.visits == 0: 
            return self.value
        exploration_weight = self.params.exploration_weight
        return (self.value / self.visits) + exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def update(self, reward: float) -> None:
        self.value += reward
        self.visits += 1
        if self.parent is not None:
            self.parent.update(reward)

    def select(self) -> Self:
        if self.terminal or self.visits == 0:
            return self
        node = self.get_or_create_child()
        return node.select()

    @property
    def max_children(self) -> int:
        alpha = self.params.alpha
        k = self.params.k
        # progressive widening, max_children = ⌊kN^α⌋
        return math.floor(k * math.pow(self.visits, alpha))

    def get_or_create_child(self) -> Self:
        max_children = self.max_children
        if len(self.children) < max_children:
            # expand new child
            child = self.params.create_child(self)
            self.children.append(child)
            return child
        else:
            # select existing child using UCT
            return self.select_existing_child()

    def select_existing_child(self) -> Self:
        return max(self.children, key=lambda child: child.uct)
