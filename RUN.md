# Real-Time Sentiment Analysis System

A big data project implementing **Lambda Architecture** for real-time sentiment analysis of Reddit posts using Kafka, HDFS, Spark, and MongoDB.

## 🏗️ Architecture Overview

```
┌─────────────┐    ┌─────────┐    ┌──────────────────┐    ┌─────────┐    ┌─────────┐
│   Reddit    │───▶│  Kafka  │───▶│  Spark Streaming │───▶│ MongoDB │───▶│ FastAPI │
│   Scraper   │    │         │    │  (VADER NLP)     │    │         │    │   API   │
└─────────────┘    └─────────┘    └──────────────────┘    └─────────┘    └─────────┘
                        │                                                     │
                        ▼                                                     ▼
                   ┌─────────┐                                          ┌─────────┐
                   │  HDFS   │                                          │  React  │
                   │ (Batch) │                                          │Dashboard│
                   └─────────┘                                          └─────────┘
```

### Components:

- **Speed Layer**: Kafka → Spark Streaming → MongoDB (real-time)
- **Batch Layer**: HDFS → Spark Batch (historical analysis)
- **Serving Layer**: FastAPI + Redis (query interface)
- **Presentation**: React Dashboard

---

## 📋 Prerequisites

### Required Software:

- **Docker Desktop** (with Docker Compose v2)
- **Node.js** v18+ (for frontend development)
- **Git**

### Reddit API Credentials:

1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Select "script" as the app type
4. Note down: `client_id`, `client_secret`
5. Your Reddit username and password

---

## 🚀 Quick Start

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd Real-time-senti
```

### Step 2: Configure Environment

Create a `.env` file in the project root:

```bash
cat > .env << 'EOF'
# Reddit API Credentials
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USERNAME=your_reddit_username
REDDIT_PASSWORD=your_reddit_password
REDDIT_USER_AGENT=SentimentAnalyzer/1.0

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=kafka:9092

# MongoDB Configuration
MONGODB_URI=mongodb://admin:sentiment_admin_2024@mongodb:27017
MONGODB_DATABASE=sentiment_db

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379

# Spark Configuration
SPARK_MASTER_URL=local[*]

# Scraper Configuration
SCRAPE_INTERVAL_SECONDS=60

# Topics to analyze (comma-separated subreddits)
TOPICS_TECHNOLOGY=technology,tech,gadgets,programming
TOPICS_FINANCE=cryptocurrency,wallstreetbets,stocks,finance
EOF
```

**⚠️ Important**: Replace the Reddit credentials with your actual values!

### Step 3: Start All Services

```bash
# Start all containers (this may take 5-10 minutes on first run)
docker compose up -d

# Check all containers are running
docker compose ps
```

Expected output - all services should be "healthy" or "running":

```
NAME                STATUS
zookeeper           healthy
kafka               healthy
namenode            healthy
datanode            running
spark-master        healthy
spark-worker        running
mongodb             healthy
redis               healthy
reddit-scraper      running
spark-streaming     running
spark-batch         running
api                 running
frontend            running
```

### Step 4: Wait for Services to Initialize

```bash
# Wait ~30 seconds for all services to be ready
sleep 30

# Check if data is flowing
docker logs reddit-scraper --tail 20
docker logs spark-streaming --tail 20
```

### Step 5: Access the Dashboard

Open your browser and go to:

- **Dashboard**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health

---

## 🔍 Verify the System

### Check Data Pipeline:

```bash
# 1. Verify Reddit scraper is collecting posts
docker logs reddit-scraper --tail 10

# 2. Check Kafka has messages
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic reddit-raw-posts \
  --from-beginning \
  --max-messages 3

# 3. Verify Spark is processing
docker logs spark-streaming --tail 20

# 4. Check MongoDB has sentiment results
docker exec mongodb mongosh -u admin -p "sentiment_admin_2024" \
  --authenticationDatabase admin --quiet \
  --eval "db.getSiblingDB('sentiment_db').sentiment_results.countDocuments()"
```

### Test API Endpoints:

```bash
# Get sentiment for technology topic
curl http://localhost:8000/api/v1/sentiment/technology | jq

# Get sentiment for finance topic
curl http://localhost:8000/api/v1/sentiment/finance | jq

# Get all available topics
curl http://localhost:8000/api/v1/topics | jq
```

---

## 🛠️ Common Commands

### Container Management:

```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down

# Restart a specific service
docker compose restart spark-streaming

# View logs for a service
docker logs -f spark-streaming

# Rebuild a service after code changes
docker compose build spark-streaming --no-cache
docker compose up -d spark-streaming
```

### Database Operations:

```bash
# Connect to MongoDB
docker exec -it mongodb mongosh -u admin -p "sentiment_admin_2024" --authenticationDatabase admin

# Inside MongoDB shell:
use sentiment_db
db.sentiment_results.find().limit(5)
db.sentiment_results.countDocuments()

# Clear all sentiment results (for testing)
docker exec mongodb mongosh -u admin -p "sentiment_admin_2024" \
  --authenticationDatabase admin --quiet \
  --eval "db.getSiblingDB('sentiment_db').sentiment_results.deleteMany({})"
```

### Kafka Operations:

```bash
# List topics
docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list

# Consume messages from a topic
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic reddit-raw-posts \
  --from-beginning
```

---

## 📊 API Reference

| Endpoint                             | Method | Description                 |
| ------------------------------------ | ------ | --------------------------- |
| `/health`                            | GET    | Health check                |
| `/api/v1/sentiment/{topic}`          | GET    | Get combined sentiment view |
| `/api/v1/sentiment/{topic}/realtime` | GET    | Get real-time sentiment     |
| `/api/v1/sentiment/{topic}/history`  | GET    | Get sentiment history       |
| `/api/v1/sentiment/{topic}/trends`   | GET    | Get hourly trends           |
| `/api/v1/sentiment/{topic}/keywords` | GET    | Get top keywords            |
| `/api/v1/topics`                     | GET    | List all topics             |
| `/api/v1/alerts`                     | GET    | Get sentiment alerts        |

---

## 🔧 Troubleshooting

### Problem: Containers not starting

```bash
# Check Docker resources (need at least 8GB RAM allocated to Docker)
docker system info | grep -i memory

# Remove all containers and volumes, start fresh
docker compose down -v
docker compose up -d
```

### Problem: No sentiment data appearing

```bash
# 1. Check Reddit scraper logs for API errors
docker logs reddit-scraper --tail 50

# 2. Verify .env file has correct Reddit credentials
cat .env | grep REDDIT

# 3. Check Kafka is receiving messages
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic reddit-raw-posts \
  --max-messages 1

# 4. Check Spark streaming is processing
docker logs spark-streaming --tail 50
```

### Problem: All sentiment scores are 0.0

```bash
# Rebuild spark-streaming with fresh dependencies
docker compose build spark-streaming --no-cache
docker compose up -d spark-streaming

# Clear old data and wait for new processing
docker exec mongodb mongosh -u admin -p "sentiment_admin_2024" \
  --authenticationDatabase admin --quiet \
  --eval "db.getSiblingDB('sentiment_db').sentiment_results.deleteMany({})"
```

### Problem: Frontend not loading

```bash
# Check frontend container logs
docker logs frontend --tail 20

# Alternatively, run frontend locally
cd frontend
npm install
npm run dev
```

### Problem: ARM64 / Apple Silicon issues

The project is configured for ARM64 (Apple Silicon). If running on x86/amd64:

```bash
# Edit docker-compose.yml and change these images:
# - namenode: use bde2020/hadoop-namenode:2.0.0-hadoop3.2.1-java8
# - datanode: use bde2020/hadoop-datanode:2.0.0-hadoop3.2.1-java8
```

---

## 📁 Project Structure

```
Real-time-senti/
├── config/                 # Configuration files
│   └── settings.py         # Pydantic settings
├── docker/                 # Dockerfiles for each service
│   ├── api/
│   ├── frontend/
│   ├── hdfs/
│   ├── reddit-scraper/
│   ├── spark-batch/
│   └── spark-streaming/
├── frontend/               # React dashboard
│   └── src/
├── requirements/           # Python dependencies per service
├── src/
│   ├── api/               # FastAPI serving layer
│   ├── batch/             # Spark batch processing
│   ├── ingestion/         # Reddit scraper + Kafka producer
│   ├── streaming/         # Spark streaming + VADER sentiment
│   └── utils/             # Shared utilities
├── docker-compose.yml      # Main orchestration file
├── .env                    # Environment variables (create this!)
└── RUN.md                  # This file
```

---

## 🧪 Development Mode

### Run Frontend Locally (Hot Reload):

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### Run API Locally:

```bash
# Install dependencies
pip install -r requirements/api.txt

# Set environment variables
export MONGODB_URI="mongodb://admin:sentiment_admin_2024@localhost:27017"
export REDIS_HOST="localhost"

# Run API
cd src/api
uvicorn main:app --reload --port 8000
```

---

## 📈 Monitoring

### View Real-time Metrics:

- **Spark UI**: http://localhost:8080 (when spark-master is running)
- **API Metrics**: http://localhost:8000/metrics (Prometheus format)

### Check System Resources:

```bash
# Container resource usage
docker stats

# Disk usage
docker system df
```

---

## 🎯 Topics Configuration

Default topics monitored:

- **technology**: r/technology, r/tech, r/gadgets, r/programming
- **finance**: r/cryptocurrency, r/wallstreetbets, r/stocks, r/finance

To add new topics, modify `config/settings.py` or create topics via API:

```bash
curl -X POST http://localhost:8000/api/v1/topics \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gaming",
    "subreddits": ["gaming", "games", "pcgaming"],
    "keywords": ["game", "gaming", "esports"],
    "active": true
  }'
```

---

## 📝 License

MIT License

---

## 🤝 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review container logs: `docker logs <container-name>`
3. Ensure Docker has sufficient resources (8GB+ RAM recommended)
