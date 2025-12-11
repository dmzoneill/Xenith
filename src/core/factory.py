"""Factory pattern base class"""

from typing import Dict, Type, TypeVar, Optional, Any
from .strategy import Strategy

T = TypeVar('T', bound=Strategy)


class Factory:
    """Base factory class for creating strategy instances"""
    
    _strategies: Dict[str, Type[Strategy]] = {}
    
    @classmethod
    def register(cls, name: str, strategy_class: Type[T]) -> None:
        """Register a strategy class with the factory"""
        cls._strategies[name] = strategy_class
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a strategy class"""
        if name in cls._strategies:
            del cls._strategies[name]
    
    @classmethod
    def create(cls, name: str, config: Optional[Dict[str, Any]] = None) -> Strategy:
        """Create an instance of a registered strategy"""
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}. Available: {list(cls._strategies.keys())}")
        
        strategy_class = cls._strategies[name]
        instance = strategy_class()
        instance.initialize(config or {})
        return instance
    
    @classmethod
    def list_available(cls) -> list[str]:
        """List all available strategy names"""
        return list(cls._strategies.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a strategy is registered"""
        return name in cls._strategies



