import asyncio
import logging
import json
from typing import Any, Callable, Dict, Optional
from .base import ProtocolInterface

logger = logging.getLogger(__name__)

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

class MQTTProtocol(ProtocolInterface):
    """
    Implementation of MQTT protocol for IoT communication.
    Uses paho-mqtt with async wrappers.
    """
    def __init__(self, broker: str, port: int = 1883, client_id: Optional[str] = None):
        self.broker = broker
        self.port = port
        self.client_id = client_id
        self._client: Optional[Any] = None
        self._connected = False
        self._loop = None
        self._on_message_callback: Optional[Callable] = None

    async def connect(self) -> bool:
        if not mqtt:
            logger.error("paho-mqtt is not installed. Please run 'pip install paho-mqtt'")
            return False

        try:
            self._client = mqtt.Client(self.client_id)
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message = self._on_message

            # Connect in a separate thread/executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self._client.connect(self.broker, self.port))
            
            # Start the MQTT network loop in the background
            self._client.loop_start()
            
            # Wait for connection (simple polling for demo, should use Event)
            for _ in range(20):
                if self._connected:
                    return True
                await asyncio.sleep(0.1)
            
            return False
        except Exception as e:
            logger.error(f"MQTT connection error: {e}")
            return False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            logger.info("MQTT Connected successfully")
        else:
            logger.error(f"MQTT Connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False

    def _on_message(self, client, userdata, msg):
        if self._on_message_callback:
            # Handle message in the event loop
            asyncio.run_coroutine_threadsafe(
                self._on_message_callback(msg.topic, msg.payload), 
                asyncio.get_event_loop()
            )

    async def disconnect(self) -> bool:
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
        self._connected = False
        return True

    async def send(self, topic: str, payload: Any) -> bool:
        if not self.is_connected():
            return False
        
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload)
        
        try:
            self._client.publish(topic, payload)
            return True
        except Exception as e:
            logger.error(f"MQTT publish error: {e}")
            return False

    async def receive(self, topic: str, callback: Callable) -> bool:
        """Subscribe to a topic and set a callback."""
        if not self.is_connected():
            return False
        
        self._on_message_callback = callback
        self._client.subscribe(topic)
        return True

    def is_connected(self) -> bool:
        return self._connected

