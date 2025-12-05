#!/bin/bash

# ============================================================
# Real-Time Sentiment Analysis System - Startup Script
# ============================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}    Real-Time Sentiment Analysis System${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi

# Check for .env file
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo -e "${YELLOW}Warning: .env file not found, copying from template...${NC}"
    if [ -f "$PROJECT_DIR/.env.example" ]; then
        cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    else
        echo -e "${RED}Error: No .env or .env.example file found${NC}"
        exit 1
    fi
fi

# Create necessary directories
echo -e "${BLUE}Creating directories...${NC}"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/models"
mkdir -p "$PROJECT_DIR/data"
mkdir -p "$PROJECT_DIR/checkpoints"

# Function to wait for service
wait_for_service() {
    local service=$1
    local url=$2
    local max_attempts=$3
    local attempt=1

    echo -e "${YELLOW}Waiting for $service to be ready...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "200\|204\|301\|302"; then
            echo -e "${GREEN}$service is ready!${NC}"
            return 0
        fi
        echo "  Attempt $attempt/$max_attempts..."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}$service failed to start within timeout${NC}"
    return 1
}

# Parse command line arguments
COMMAND=${1:-"start"}

case $COMMAND in
    "start")
        echo -e "${GREEN}Starting all services...${NC}"
        cd "$PROJECT_DIR"
        
        # Start infrastructure services first
        echo -e "${BLUE}Starting infrastructure services...${NC}"
        docker compose up -d zookeeper kafka kafka-ui namenode datanode redis mongodb prometheus grafana
        
        # Wait for core services
        sleep 10
        wait_for_service "Kafka UI" "http://localhost:8080" 30
        wait_for_service "HDFS" "http://localhost:9870" 30
        wait_for_service "MongoDB" "http://localhost:27017" 30
        
        # Start Spark
        echo -e "${BLUE}Starting Spark cluster...${NC}"
        docker compose up -d spark-master spark-worker
        sleep 10
        wait_for_service "Spark Master" "http://localhost:8081" 30
        
        # Start application services
        echo -e "${BLUE}Starting application services...${NC}"
        docker compose up -d reddit-scraper spark-streaming spark-batch serving-api
        
        sleep 10
        wait_for_service "Serving API" "http://localhost:8000/health" 60
        
        echo ""
        echo -e "${GREEN}============================================================${NC}"
        echo -e "${GREEN}    All services are running!${NC}"
        echo -e "${GREEN}============================================================${NC}"
        echo ""
        echo -e "Available endpoints:"
        echo -e "  ${BLUE}API:${NC}          http://localhost:8000"
        echo -e "  ${BLUE}API Docs:${NC}     http://localhost:8000/docs"
        echo -e "  ${BLUE}Kafka UI:${NC}     http://localhost:8080"
        echo -e "  ${BLUE}Spark UI:${NC}     http://localhost:8081"
        echo -e "  ${BLUE}HDFS UI:${NC}      http://localhost:9870"
        echo -e "  ${BLUE}Grafana:${NC}      http://localhost:3000 (admin/sentiment_grafana_2024)"
        echo -e "  ${BLUE}Prometheus:${NC}   http://localhost:9090"
        echo ""
        ;;
    
    "stop")
        echo -e "${YELLOW}Stopping all services...${NC}"
        cd "$PROJECT_DIR"
        docker compose down
        echo -e "${GREEN}All services stopped${NC}"
        ;;
    
    "restart")
        echo -e "${YELLOW}Restarting all services...${NC}"
        cd "$PROJECT_DIR"
        docker compose down
        $0 start
        ;;
    
    "logs")
        SERVICE=${2:-""}
        cd "$PROJECT_DIR"
        if [ -n "$SERVICE" ]; then
            docker compose logs -f "$SERVICE"
        else
            docker compose logs -f
        fi
        ;;
    
    "status")
        cd "$PROJECT_DIR"
        docker compose ps
        ;;
    
    "clean")
        echo -e "${RED}WARNING: This will delete all data!${NC}"
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cd "$PROJECT_DIR"
            docker compose down -v
            rm -rf "$PROJECT_DIR/logs/*"
            rm -rf "$PROJECT_DIR/checkpoints/*"
            echo -e "${GREEN}Cleanup complete${NC}"
        fi
        ;;
    
    "build")
        echo -e "${BLUE}Building Docker images...${NC}"
        cd "$PROJECT_DIR"
        docker compose build --no-cache
        echo -e "${GREEN}Build complete${NC}"
        ;;
    
    *)
        echo "Usage: $0 {start|stop|restart|logs|status|clean|build}"
        echo ""
        echo "Commands:"
        echo "  start   - Start all services"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        echo "  logs    - View logs (optionally specify service name)"
        echo "  status  - Show service status"
        echo "  clean   - Remove all data and volumes"
        echo "  build   - Rebuild Docker images"
        exit 1
        ;;
esac
