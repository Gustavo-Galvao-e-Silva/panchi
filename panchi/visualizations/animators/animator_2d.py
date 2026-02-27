from abc import ABC, abstractmethod

from panchi.primitives.vector import Vector
from panchi.primitives.matrix import Matrix


class BaseAnimator2D(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def plot_vectors(self) -> None:
        pass
