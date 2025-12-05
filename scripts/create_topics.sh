#!/bin/bash

# ============================================================
# Create Kafka Topics
# ============================================================

echo "Creating Kafka topics..."

# Wait for Kafka to be ready
sleep 5

# Create topics
docker exec kafka kafka-topics --create \
    --bootstrap-server localhost:9092 \
    --replication-factor 1 \
    --partitions 3 \
    --topic reddit-raw-posts \
    --if-not-exists

docker exec kafka kafka-topics --create \
    --bootstrap-server localhost:9092 \
    --replication-factor 1 \
    --partitions 3 \
    --topic sentiment-results \
    --if-not-exists

docker exec kafka kafka-topics --create \
    --bootstrap-server localhost:9092 \
    --replication-factor 1 \
    --partitions 1 \
    --topic sentiment-alerts \
    --if-not-exists

echo "Kafka topics created successfully!"

# List topics
echo ""
echo "Available topics:"
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092
