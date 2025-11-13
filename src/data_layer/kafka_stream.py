"""
Kafka streaming for real-time market data and trading signals.

Architecture:
- Producers: Publish market data, news, and trading signals to Kafka topics
- Consumers: Subscribe to topics and process events in real-time
- Event-driven: Decouples data ingestion from signal generation and trading

Benefits:
- Scalable: Can handle high-frequency data
- Resilient: Messages are persisted and can be replayed
- Real-time: Sub-second latency for trading signals
"""

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import json
from typing import Dict, Any, Callable, Optional
from datetime import datetime
from src.utils.config import settings
from src.utils.logger import app_logger
import time


class KafkaStreamProducer:
    """
    Kafka producer for publishing market events and trading signals.

    Use this to publish:
    - Market quotes
    - News articles
    - SEC filings
    - Trading signals
    """

    def __init__(self):
        """Initialize Kafka producer with retry logic."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=settings.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',  # Wait for all replicas to acknowledge
                retries=3,
                max_in_flight_requests_per_connection=1,  # Ensure ordering
                compression_type='gzip',  # Reduce bandwidth
            )
            app_logger.info("Kafka producer initialized successfully")
        except KafkaError as e:
            app_logger.error(f"Failed to initialize Kafka producer: {e}")
            self.producer = None

    def publish_news(self, news_data: Dict[str, Any]) -> bool:
        """
        Publish news event to Kafka.

        Args:
            news_data: News article data

        Returns:
            True if published successfully
        """
        return self._publish(
            topic=settings.kafka_topic_news,
            key=news_data.get("symbol"),
            value=news_data
        )

    def publish_filing(self, filing_data: Dict[str, Any]) -> bool:
        """
        Publish SEC filing event to Kafka.

        Args:
            filing_data: SEC filing data

        Returns:
            True if published successfully
        """
        return self._publish(
            topic=settings.kafka_topic_filings,
            key=filing_data.get("symbol"),
            value=filing_data
        )

    def publish_signal(self, signal_data: Dict[str, Any]) -> bool:
        """
        Publish trading signal to Kafka.

        Args:
            signal_data: Trading signal data

        Returns:
            True if published successfully
        """
        return self._publish(
            topic=settings.kafka_topic_signals,
            key=signal_data.get("symbol"),
            value=signal_data
        )

    def _publish(
        self,
        topic: str,
        key: str,
        value: Dict[str, Any]
    ) -> bool:
        """
        Internal method to publish message to Kafka.

        Args:
            topic: Kafka topic name
            key: Message key (typically symbol)
            value: Message payload

        Returns:
            True if published successfully
        """
        if not self.producer:
            app_logger.error("Kafka producer not initialized")
            return False

        try:
            # Add timestamp to message
            value["_kafka_timestamp"] = datetime.utcnow().isoformat()

            # Publish async with callback
            future = self.producer.send(
                topic,
                key=key.encode('utf-8') if key else None,
                value=value
            )

            # Wait for acknowledgment (with timeout)
            record_metadata = future.get(timeout=10)

            app_logger.debug(
                f"Published to {topic} | "
                f"Partition: {record_metadata.partition} | "
                f"Offset: {record_metadata.offset}"
            )
            return True

        except KafkaError as e:
            app_logger.error(f"Failed to publish to {topic}: {e}")
            return False
        except Exception as e:
            app_logger.error(f"Unexpected error publishing to {topic}: {e}")
            return False

    def close(self):
        """Close Kafka producer and flush pending messages."""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            app_logger.info("Kafka producer closed")


class KafkaStreamConsumer:
    """
    Kafka consumer for processing market events and trading signals.

    Use this to subscribe to topics and process events in real-time.
    """

    def __init__(
        self,
        topics: list[str],
        group_id: str = "trading-platform",
        auto_offset_reset: str = "latest"
    ):
        """
        Initialize Kafka consumer.

        Args:
            topics: List of topics to subscribe to
            group_id: Consumer group ID for load balancing
            auto_offset_reset: Where to start reading ('earliest' or 'latest')
        """
        try:
            self.consumer = KafkaConsumer(
                *topics,
                bootstrap_servers=settings.kafka_bootstrap_servers,
                group_id=group_id,
                auto_offset_reset=auto_offset_reset,
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                consumer_timeout_ms=1000,  # Poll timeout
            )
            self.topics = topics
            app_logger.info(f"Kafka consumer initialized for topics: {topics}")
        except KafkaError as e:
            app_logger.error(f"Failed to initialize Kafka consumer: {e}")
            self.consumer = None

    def consume_messages(
        self,
        callback: Callable[[str, Dict[str, Any]], None],
        max_messages: Optional[int] = None
    ):
        """
        Consume messages from Kafka and process with callback.

        Args:
            callback: Function to call for each message (topic, message_data)
            max_messages: Maximum messages to process (None = infinite)

        Example:
            def process_signal(topic: str, data: dict):
                print(f"Received signal: {data}")

            consumer = KafkaStreamConsumer(['trading-signals'])
            consumer.consume_messages(process_signal)
        """
        if not self.consumer:
            app_logger.error("Kafka consumer not initialized")
            return

        messages_processed = 0
        app_logger.info(f"Starting to consume messages from {self.topics}")

        try:
            for message in self.consumer:
                try:
                    # Process message
                    topic = message.topic
                    data = message.value

                    app_logger.debug(
                        f"Consumed from {topic} | "
                        f"Partition: {message.partition} | "
                        f"Offset: {message.offset}"
                    )

                    # Call user-provided callback
                    callback(topic, data)

                    messages_processed += 1

                    # Stop if max_messages reached
                    if max_messages and messages_processed >= max_messages:
                        app_logger.info(
                            f"Reached max messages ({max_messages}), stopping"
                        )
                        break

                except Exception as e:
                    app_logger.error(f"Error processing message: {e}")
                    # Continue processing next message

        except KeyboardInterrupt:
            app_logger.info("Consumer interrupted by user")
        finally:
            self.close()

    def close(self):
        """Close Kafka consumer."""
        if self.consumer:
            self.consumer.close()
            app_logger.info("Kafka consumer closed")


# Example usage and helper functions
def create_news_stream():
    """Create a producer for streaming news events."""
    return KafkaStreamProducer()


def create_signal_consumer(callback: Callable):
    """
    Create a consumer for trading signals.

    Args:
        callback: Function to handle trading signals
    """
    consumer = KafkaStreamConsumer(
        topics=[settings.kafka_topic_signals],
        group_id="signal-processor"
    )
    return consumer
