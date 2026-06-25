import asyncio
import json
import logging
from typing import Any, Callable, Dict, List
from .event_bus import EventBus
from ..core.models import DeviceEvent

try:
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
except ImportError:
    AIOKafkaProducer = None
    AIOKafkaConsumer = None

logger = logging.getLogger(__name__)

class KafkaEventBus(EventBus):
    """
    Distributed, high-throughput Event Bus using Apache Kafka.
    Perfect for scaling Sensorium across multiple servers, hospitals, or military bases
    where millions of sensors stream data simultaneously.
    """
    def __init__(self, bootstrap_servers: str = "localhost:9092", group_id: str = "sensorium_group"):
        super().__init__()
        if AIOKafkaProducer is None:
            raise ImportError("Please install aiokafka to use KafkaEventBus: pip install aiokafka")
            
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.producer = None
        self.consumer = None
        self._consumer_task = None
        self._subscribed_topics = set()

    async def start(self):
        """Starts Kafka producer and consumer connections."""
        self.producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap_servers)
        await self.producer.start()
        
        # Start consumer if there are already subscribed topics
        if self._subscribed_topics:
            await self._start_consumer()
            
        logger.info(f"KafkaEventBus connected to {self.bootstrap_servers}")

    async def _start_consumer(self):
        if self.consumer:
            await self.consumer.stop()
            
        self.consumer = AIOKafkaConsumer(
            *self._subscribed_topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id
        )
        await self.consumer.start()
        self._consumer_task = asyncio.create_task(self._consume_loop())

    async def _consume_loop(self):
        try:
            async for msg in self.consumer:
                topic = msg.topic
                data = json.loads(msg.value.decode('utf-8'))
                
                # Reconstruct the standard DeviceEvent from Kafka JSON
                event = DeviceEvent(
                    event_name=data.get("event_name"),
                    source_id=data.get("source_id"),
                    data=data.get("data"),
                    timestamp=data.get("timestamp")
                )
                
                # Trigger local memory callbacks that agents are listening to
                super().emit(topic, event)
        except asyncio.CancelledError:
            pass

    def subscribe(self, event_name: str, callback: Callable[[DeviceEvent], Any]) -> None:
        """Subscribe local callback and join the Kafka topic."""
        super().subscribe(event_name, callback)
        
        # If it's a new topic, we must restart the Kafka consumer to include it
        if event_name not in self._subscribed_topics:
            self._subscribed_topics.add(event_name)
            if self.producer:  # Means bus is already started
                asyncio.create_task(self._start_consumer())

    def emit(self, event_name: str, event: DeviceEvent) -> None:
        """Publish event to Kafka cluster asynchronously."""
        if not self.producer:
            # Fallback to local memory bus if Kafka isn't started yet
            super().emit(event_name, event)
            return
            
        payload = {
            "event_name": event.event_name,
            "source_id": event.source_id,
            "data": event.data,
            "timestamp": event.timestamp
        }
        
        # Fire-and-forget Kafka send to ensure zero-latency blocking
        asyncio.create_task(
            self.producer.send_and_wait(event_name, json.dumps(payload).encode('utf-8'))
        )
        
    async def emit_wait(self, event_name: str, event: DeviceEvent) -> None:
        if not self.producer:
            await super().emit_wait(event_name, event)
            return
            
        payload = {
            "event_name": event.event_name,
            "source_id": event.source_id,
            "data": event.data,
            "timestamp": event.timestamp
        }
        await self.producer.send_and_wait(event_name, json.dumps(payload).encode('utf-8'))

    async def stop(self):
        """Clean shutdown of Kafka connections."""
        if self._consumer_task:
            self._consumer_task.cancel()
        if self.consumer:
            await self.consumer.stop()
        if self.producer:
            await self.producer.stop()
        logger.info("KafkaEventBus shutdown complete.")
