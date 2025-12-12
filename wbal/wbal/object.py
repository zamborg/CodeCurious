"""
Base class for all WBAL objects.

All WBAL components (LM, Agent, Env) inherit from WBALObject,
which enforces a consistent interface for observability and sandbox setup.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from sandbox.interface import SandboxInterface


class WBALObject(BaseModel, ABC):
    """
    Base class for all WBAL components.

    Provides:
    - `.observe()` -> str: Return a human-readable representation of current state
    - `.setup(sandbox)`: Initialize with a sandbox for execution context
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def observe(self) -> str:
        """
        Return a string representation of the object's current observable state.

        This is used for debugging, logging, and agent introspection.
        """
        ...

    async def setup(self, sandbox: SandboxInterface) -> None:
        """
        Initialize this object with a sandbox execution context.

        Override this method to perform sandbox-dependent initialization.
        The sandbox interface will be defined separately.

        Args:
            sandbox: The sandbox object providing read_file, write_file, run_command
        """
        print(f"WARNING: please overload this setup method as by default this *only* setups the children")
        await self._setup_children(sandbox) # by default we just setup all the children, this should be overloaded

    async def _setup_children(self, sandbox: SandboxInterface) -> None:
        """
        calls `setup` on all children objects
        """ 
        for child_name, child_obj in self.__iter__():
            if isinstance(child_obj, WBALObject):
                print(f"Setting up {child_name}")
                await child_obj.setup(sandbox)
            else:
                print(f"Skipping setup of {child_name} as it is not a WBALObject")
