"""
Message Queue Connector Module
==============================

Supports Kafka and RabbitMQ integration.
"""

from kafka import KafkaProducer, KafkaConsumer
import pika
from . import logger


# Kafka
def kafka_producer(bootstrap_servers="localhost:9092"):
    logger.info(f"Creating Kafka producer at {bootstrap_servers}")
    return KafkaProducer(bootstrap_servers=bootstrap_servers)


def kafka_consumer(topic, bootstrap_servers="localhost:9092", group_id="sentenial"):
    logger.info(f"Creating Kafka consumer for topic={topic} at {bootstrap_servers}")
    return KafkaConsumer(topic, bootstrap_servers=bootstrap_servers, group_id=group_id)


# RabbitMQ
def rabbitmq_channel(host="localhost"):
    logger.info(f"Connecting to RabbitMQ at {host}")
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
    return connection.channel()
