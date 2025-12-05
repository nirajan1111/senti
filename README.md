# Real-Time Sentiment Analysis System

A production-ready, real-time sentiment analysis pipeline using **Lambda Architecture** for processing Reddit data at scale.

![Architecture](docs/architecture.png)

## 🏗️ Architecture Overview

This system implements the **Lambda Architecture** pattern with three layers:

### 1. **Batch Layer** (Spark Batch)
- Processes historical data stored in HDFS
- Computes comprehensive aggregations hourly
- Generates batch views for accurate analytics
- Handles data reprocessing and backfilling

### 2. **Speed Layer** (Spark Streaming)
- Processes real-time data from Kafka
- Provides low-latency sentiment analysis
- Stores results in Redis for fast access
- Enables real-time alerting

### 3. **Serving Layer** (FastAPI)
- Combines batch and real-time views
- Exposes REST API for querying
- Handles data fusion and presentation
- Provides alerting and monitoring endpoints

## 🚀 Features

- **Real-time Reddit scraping** with PRAW
- **BERT-based sentiment analysis** (multilingual support)
- **Keyword extraction** and entity recognition
- **Sentiment alerts** with configurable thresholds
- **Topic-based partitioning** in HDFS
- **Trend analysis** and anomaly detection
- **Subreddit-level breakdown**
- **Topic comparison** functionality
- **Prometheus metrics** and Grafana dashboards
- **Fully containerized** with Docker Compose

## 📋 Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM recommended
- Reddit API credentials

## 🔧 Quick Start

### 1. Clone the repository

```bash
git clone <repository-url>
cd Real-time-senti
```

### 2. Configure environment

Create a `.env` file (or copy from `.env.example`):

```bash
# Reddit API Credentials
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=YourApp/1.0 by u/YourUsername

# Other configurations are pre-set with defaults
```

### 3. Start the system

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Start all services
./scripts/start.sh start
```

### 4. Verify the setup

```bash
# Check service health
./scripts/health_check.sh
```

## 🌐 Service Endpoints

| Service | URL | Description |
|---------|-----|-------------|
| **API** | http://localhost:8000 | Main REST API |
| **API Docs** | http://localhost:8000/docs | Swagger documentation |
| **Kafka UI** | http://localhost:8080 | Kafka management |
| **Spark UI** | http://localhost:8081 | Spark cluster dashboard |
| **HDFS UI** | http://localhost:9870 | Hadoop filesystem |
| **Grafana** | http://localhost:3000 | Monitoring dashboards |
| **Prometheus** | http://localhost:9090 | Metrics collection |

## 📡 API Usage

### Get Combined Sentiment View

```bash
curl http://localhost:8000/api/v1/sentiment/technology
```

### Get Real-time Sentiment

```bash
curl http://localhost:8000/api/v1/sentiment/technology/realtime
```

### Get Sentiment Trends

```bash
curl http://localhost:8000/api/v1/sentiment/technology/trends?hours=24
```

### Get Top Keywords

```bash
curl http://localhost:8000/api/v1/sentiment/technology/keywords?limit=20
```

### Compare Topics

```bash
curl "http://localhost:8000/api/v1/compare?topics=technology,finance"
```

### Create New Topic

```bash
curl -X POST http://localhost:8000/api/v1/topics \
  -H "Content-Type: application/json" \
  -d '{
    "name": "crypto",
    "subreddits": ["cryptocurrency", "bitcoin", "ethereum"],
    "keywords": ["btc", "eth", "blockchain"],
    "active": true
  }'
```

### Get Alerts

```bash
curl http://localhost:8000/api/v1/alerts?unacknowledged_only=true
```

## 📁 Project Structure

```
Real-time-senti/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration management
├── docker/
│   ├── api/                  # API Dockerfile
│   ├── grafana/              # Grafana provisioning
│   ├── mongodb/              # MongoDB initialization
│   ├── prometheus/           # Prometheus config
│   ├── scraper/              # Scraper Dockerfile
│   ├── spark/                # Spark master Dockerfile
│   ├── spark-batch/          # Batch processor Dockerfile
│   └── spark-streaming/      # Streaming processor Dockerfile
├── requirements/
│   ├── api.txt               # API dependencies
│   ├── scraper.txt           # Scraper dependencies
│   ├── spark-batch.txt       # Batch dependencies
│   └── spark-streaming.txt   # Streaming dependencies
├── scripts/
│   ├── create_topics.sh      # Kafka topic creation
│   ├── health_check.sh       # System health check
│   ├── setup_hdfs.sh         # HDFS directory setup
│   └── start.sh              # Main startup script
├── src/
│   ├── api/
│   │   └── main.py           # FastAPI serving layer
│   ├── batch/
│   │   └── spark_batch.py    # Spark batch processing
│   ├── models/
│   │   └── data_models.py    # Pydantic models
│   ├── processing/
│   │   └── sentiment_analyzer.py  # BERT sentiment analysis
│   ├── scraper/
│   │   └── reddit_scraper.py # Reddit data scraper
│   ├── streaming/
│   │   └── spark_streaming.py # Spark streaming
│   └── utils/
│       └── logging_config.py # Logging configuration
├── .env                      # Environment variables
├── .gitignore
├── docker-compose.yml        # Docker orchestration
└── README.md
```

## 🔄 Data Flow

```
┌─────────────┐    ┌─────────┐    ┌──────────────────┐
│   Reddit    │───▶│  Kafka  │───▶│ Spark Streaming  │
│   Scraper   │    │         │    │  (Speed Layer)   │
└─────────────┘    └─────────┘    └────────┬─────────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │    Redis     │
                                    │  (Real-time) │
                                    └──────┬───────┘
                                           │
                   ┌───────────────────────┴───────────────────────┐
                   │                                               │
                   ▼                                               ▼
            ┌──────────────┐                                ┌──────────────┐
            │     HDFS     │                                │   MongoDB    │
            │  (Raw Data)  │                                │  (Results)   │
            └──────┬───────┘                                └──────┬───────┘
                   │                                               │
                   ▼                                               │
            ┌──────────────┐                                       │
            │ Spark Batch  │                                       │
            │(Batch Layer) │───────────────────────────────────────┘
            └──────────────┘
                   │
                   ▼
            ┌──────────────┐
            │   FastAPI    │◀─── Serving Layer (combines views)
            │  (REST API)  │
            └──────────────┘
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REDDIT_CLIENT_ID` | Reddit API client ID | Required |
| `REDDIT_CLIENT_SECRET` | Reddit API secret | Required |
| `REDDIT_USER_AGENT` | Reddit API user agent | Required |
| `SCRAPE_INTERVAL_SECONDS` | Scraping interval | 60 |
| `BATCH_INTERVAL_HOURS` | Batch processing interval | 1 |
| `SENTIMENT_ALERT_THRESHOLD` | Alert threshold | -0.5 |
| `LOG_LEVEL` | Logging level | INFO |

### Adding New Topics

Topics can be managed via the API:

```bash
# Add a new topic
curl -X POST http://localhost:8000/api/v1/topics \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ai",
    "subreddits": ["artificial", "MachineLearning", "deeplearning"],
    "keywords": ["GPT", "LLM", "neural network"],
    "active": true
  }'

# Deactivate a topic
curl -X PUT http://localhost:8000/api/v1/topics/ai \
  -H "Content-Type: application/json" \
  -d '{"active": false}'
```

## 📊 Monitoring

### Grafana Dashboards

Access Grafana at http://localhost:3000 (admin/sentiment_grafana_2024)

Pre-configured dashboards:
- System Overview
- Sentiment Trends
- Kafka Metrics
- API Performance

### Prometheus Metrics

Access at http://localhost:9090

Available metrics:
- `api_requests_total` - Total API requests
- `api_request_latency_seconds` - Request latency
- Kafka consumer lag
- Spark job metrics

## 🛠️ Development

### Running Locally (without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements/api.txt

# Run individual services
python -m src.scraper.reddit_scraper
python -m src.streaming.spark_streaming
python -m src.batch.spark_batch
python -m src.api.main
```

### Running Tests

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📝 Commands Reference

```bash
# Start all services
./scripts/start.sh start

# Stop all services
./scripts/start.sh stop

# Restart services
./scripts/start.sh restart

# View logs
./scripts/start.sh logs                    # All services
./scripts/start.sh logs reddit-scraper     # Specific service

# Check status
./scripts/start.sh status

# Clean up (removes all data)
./scripts/start.sh clean

# Rebuild images
./scripts/start.sh build
```

## 🔍 Troubleshooting

### Common Issues

1. **Kafka connection errors**
   - Wait for Kafka to fully start (check Kafka UI)
   - Run `./scripts/create_topics.sh`

2. **HDFS permission errors**
   - Run `./scripts/setup_hdfs.sh`

3. **Out of memory errors**
   - Increase Docker memory allocation
   - Reduce `spark.executor.memory` in config

4. **Reddit rate limiting**
   - Increase `SCRAPE_INTERVAL_SECONDS`
   - Reduce `posts_per_subreddit`

### Viewing Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f reddit-scraper
docker compose logs -f spark-streaming
docker compose logs -f serving-api
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues and questions, please open a GitHub issue.
