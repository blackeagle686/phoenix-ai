from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class ProtocolInterface(ABC):
    """
    Interface for hardware communication protocols.
    Standardizes how we send/receive raw bytes or messages.
    """
    
    @abstractmethod
    async def connect(self) -> bool:
        """Open the communication channel."""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Close the communication channel."""
        pass

    @abstractmethod
    async def send(self, data: Any) -> bool:
        """Send data through the protocol."""
        pass

    @abstractmethod
    async def receive(self) -> Any:
        """Receive data from the protocol."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check connection status."""
        pass
