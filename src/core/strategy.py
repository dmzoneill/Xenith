"""Strategy pattern base class"""

from abc import ABC, abstractmethod
from typing import Any


class Strategy(ABC):
    """Base class for all strategy implementations"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this strategy"""

    @abstractmethod
    def initialize(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the strategy with configuration"""

    def cleanup(self) -> None:
        """Cleanup resources (override if needed)"""
