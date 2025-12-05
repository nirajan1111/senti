#!/bin/bash

# ============================================================
# Create HDFS Directories
# ============================================================

echo "Creating HDFS directories..."

# Wait for HDFS to be ready
sleep 5

# Create directories
docker exec namenode hdfs dfs -mkdir -p /data/raw
docker exec namenode hdfs dfs -mkdir -p /data/processed
docker exec namenode hdfs dfs -mkdir -p /data/batch
docker exec namenode hdfs dfs -mkdir -p /checkpoints

# Set permissions
docker exec namenode hdfs dfs -chmod -R 777 /data
docker exec namenode hdfs dfs -chmod -R 777 /checkpoints

echo "HDFS directories created successfully!"

# List directories
echo ""
echo "HDFS structure:"
docker exec namenode hdfs dfs -ls -R /
