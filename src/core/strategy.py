"""Strategy pattern base class"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Strategy(ABC):
    """Base class for all strategy implementations"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this strategy"""
        pass

    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the strategy with configuration"""
        pass

    def cleanup(self) -> None:
        """Cleanup resources (override if needed)"""
        pass
