import asyncio
import logging
from typing import Any, Optional
from .base import ProtocolInterface

logger = logging.getLogger(__name__)

try:
    import serial
except ImportError:
    serial = None

class SerialProtocol(ProtocolInterface):
    """
    Implementation of Serial/UART communication.
    Optimized for non-blocking async reads.
    """
    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 0.1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: Optional[Any] = None
        self._connected = False

    async def connect(self) -> bool:
        if not serial:
            logger.error("pyserial is not installed. Please run 'pip install pyserial'")
            return False
        
        try:
            # Run blocking open in a thread to keep event loop free
            loop = asyncio.get_event_loop()
            self._ser = await loop.run_in_executor(
                None, 
                lambda: serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            )
            self._connected = self._ser.is_open
            return self._connected
        except Exception as e:
            logger.error(f"Failed to connect to serial port {self.port}: {e}")
            return False

    async def disconnect(self) -> bool:
        if self._ser and self._ser.is_open:
            self._ser.close()
            self._connected = False
        return True

    async def send(self, data: Any) -> bool:
        if not self.is_connected():
            return False
        
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._ser.write, data)
            return True
        except Exception as e:
            logger.error(f"Serial send error: {e}")
            return False

    async def receive(self) -> Any:
        if not self.is_connected():
            return None
        
        try:
            loop = asyncio.get_event_loop()
            # Read until newline or timeout
            data = await loop.run_in_executor(None, self._ser.readline)
            return data.decode('utf-8').strip() if data else None
        except Exception as e:
            logger.error(f"Serial receive error: {e}")
            return None

    def is_connected(self) -> bool:
        return self._connected and self._ser and self._ser.is_open
