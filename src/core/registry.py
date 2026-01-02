"""Component registry for plugin registration"""

from typing import Dict, List, Any, Optional
from .factory import Factory


class ComponentRegistry:
    """Central registry for all components (agents, LLMs, actions, etc.)"""

    def __init__(self):
        self._factories: Dict[str, Factory] = {}
        self._components: Dict[str, Dict[str, Any]] = {}

    def register_factory(self, component_type: str, factory: Factory) -> None:
        """Register a factory for a component type"""
        self._factories[component_type] = factory

    def get_factory(self, component_type: str) -> Optional[Factory]:
        """Get a factory for a component type"""
        return self._factories.get(component_type)

    def register_component(
        self, component_type: str, name: str, component: Any
    ) -> None:
        """Register a component instance"""
        if component_type not in self._components:
            self._components[component_type] = {}
        self._components[component_type][name] = component

    def get_component(self, component_type: str, name: str) -> Optional[Any]:
        """Get a registered component"""
        return self._components.get(component_type, {}).get(name)

    def list_components(self, component_type: str) -> List[str]:
        """List all registered components of a type"""
        return list(self._components.get(component_type, {}).keys())

    def unregister_component(self, component_type: str, name: str) -> None:
        """Unregister a component"""
        if component_type in self._components:
            self._components[component_type].pop(name, None)
