#!/bin/bash

# ============================================================
# Health Check Script
# ============================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "============================================================"
echo "    System Health Check"
echo "============================================================"
echo ""

# Function to check service
check_service() {
    local name=$1
    local url=$2
    
    if curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null | grep -q "200\|204\|301\|302"; then
        echo -e "${GREEN}✓${NC} $name is healthy"
        return 0
    else
        echo -e "${RED}✗${NC} $name is not responding"
        return 1
    fi
}

# Check Docker containers
echo "Docker Containers:"
docker compose ps 2>/dev/null || echo "Docker Compose not available"
echo ""

# Check services
echo "Service Health:"
check_service "Serving API" "http://localhost:8000/health" || true
check_service "Kafka UI" "http://localhost:8080" || true
check_service "Spark Master" "http://localhost:8081" || true
check_service "HDFS NameNode" "http://localhost:9870" || true
check_service "Grafana" "http://localhost:3000" || true
check_service "Prometheus" "http://localhost:9090" || true

echo ""
echo "============================================================"
echo ""

# Check Kafka topics
echo "Kafka Topics:"
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092 2>/dev/null || echo "Unable to connect to Kafka"
echo ""

# Check MongoDB collections
echo "MongoDB Collections:"
docker exec mongodb mongosh --quiet --eval "db.getSiblingDB('sentiment_db').getCollectionNames()" 2>/dev/null || echo "Unable to connect to MongoDB"
echo ""

# Check Redis
echo "Redis Status:"
docker exec redis redis-cli ping 2>/dev/null || echo "Unable to connect to Redis"
echo ""

echo "============================================================"
echo "    Health Check Complete"
echo "============================================================"
