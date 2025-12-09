"""
Base class for all WBAL objects.

All WBAL components (LM, Agent, Env) inherit from WBALObject,
which enforces a consistent interface for observability and sandbox setup.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict


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

    def setup(self, sandbox: Any) -> None:
        """
        Initialize this object with a sandbox execution context.

        Override this method to perform sandbox-dependent initialization.
        The sandbox interface will be defined separately.

        Args:
            sandbox: The sandbox object providing read_file, write_file, run_command
        """
        pass
