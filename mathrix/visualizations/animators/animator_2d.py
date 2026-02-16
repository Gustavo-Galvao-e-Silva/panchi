from abc import ABC, abstractmethod

from mathrix.primitives.vector import Vector
from mathrix.primitives.matrix import Matrix


class BaseAnimator2D(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def plot_vectors(self) -> None:
        pass
