# Architecture Documentation

## Lambda Architecture

This system implements the Lambda Architecture pattern for processing both real-time and batch data.

```
                                    ┌─────────────────────────────────────────────────────────────┐
                                    │                        DATA SOURCES                         │
                                    │  ┌─────────────────────────────────────────────────────┐   │
                                    │  │                    Reddit API                        │   │
                                    │  │         (Posts, Comments, Upvotes, etc.)            │   │
                                    │  └─────────────────────────────────────────────────────┘   │
                                    └───────────────────────────┬─────────────────────────────────┘
                                                                │
                                                                ▼
                                    ┌─────────────────────────────────────────────────────────────┐
                                    │                      DATA INGESTION                         │
                                    │  ┌─────────────────────────────────────────────────────┐   │
                                    │  │              Reddit Scraper (PRAW)                   │   │
                                    │  │        Extracts posts/comments → Kafka              │   │
                                    │  └─────────────────────────────────────────────────────┘   │
                                    └───────────────────────────┬─────────────────────────────────┘
                                                                │
                                                                ▼
                                    ┌─────────────────────────────────────────────────────────────┐
                                    │                      MESSAGE QUEUE                          │
                                    │  ┌─────────────────────────────────────────────────────┐   │
                                    │  │                 Apache Kafka                         │   │
                                    │  │    Topic: reddit-raw-posts (partitioned by topic)   │   │
                                    │  └─────────────────────────────────────────────────────┘   │
                                    └───────────────────────────┬─────────────────────────────────┘
                                                                │
                                    ┌───────────────────────────┴───────────────────────────┐
                                    │                                                       │
                                    ▼                                                       ▼
    ┌───────────────────────────────────────────────────────────┐   ┌───────────────────────────────────────────────────────────┐
    │                      BATCH LAYER                          │   │                      SPEED LAYER                          │
    │  ┌─────────────────────────────────────────────────────┐  │   │  ┌─────────────────────────────────────────────────────┐  │
    │  │                      HDFS                            │  │   │  │              Spark Structured Streaming             │  │
    │  │           Raw data storage (Parquet)                 │  │   │  │         Real-time sentiment analysis               │  │
    │  │         Partitioned by topic and date               │  │   │  │              VADER.                  │  │
    │  └─────────────────────────┬───────────────────────────┘  │   │  └─────────────────────────┬───────────────────────────┘  │
    │                            │                              │   │                            │                              │
    │                            ▼                              │   │                            ▼                              │
    │  ┌─────────────────────────────────────────────────────┐  │   │  ┌─────────────────────────────────────────────────────┐  │
    │  │                  Spark Batch Jobs                    │  │   │  │                     Redis                           │  │
    │  │      Hourly aggregations, trend analysis,           │  │   │  │    Real-time metrics cache (TTL: 5 min)            │  │
    │  │        anomaly detection, keyword extraction        │  │   │  │         Time-series sentiment data                 │  │
    │  └─────────────────────────┬───────────────────────────┘  │   │  └─────────────────────────┬───────────────────────────┘  │
    │                            │                              │   │                            │                              │
    │                            ▼                              │   │                            │                              │
    │  ┌─────────────────────────────────────────────────────┐  │   │                            │                              │
    │  │                    Batch Views                       │  │   │                            │                              │
    │  │    - Topic aggregations (avg, stddev, counts)       │  │   │                            │                              │
    │  │    - Hourly trends                                   │  │   │                            │                              │
    │  │    - Top keywords per topic                          │  │   │                            │                              │
    │  │    - Subreddit breakdown                             │  │   │                            │                              │
    │  └─────────────────────────────────────────────────────┘  │   │                            │                              │
    └───────────────────────────┬───────────────────────────────┘   └────────────────────────────┬──────────────────────────────┘
                                │                                                                │
                                └────────────────────────┬───────────────────────────────────────┘
                                                         │
                                                         ▼
                                    ┌─────────────────────────────────────────────────────────────┐
                                    │                     SERVING LAYER                           │
                                    │  ┌─────────────────────────────────────────────────────┐   │
                                    │  │                     MongoDB                          │   │
                                    │  │         Merged batch + real-time views              │   │
                                    │  │    Collections: sentiment_results, batch_aggs,      │   │
                                    │  │                 alerts, topics, hourly_trends       │   │
                                    │  └─────────────────────────────────────────────────────┘   │
                                    │                            │                               │
                                    │                            ▼                               │
                                    │  ┌─────────────────────────────────────────────────────┐   │
                                    │  │                   FastAPI Server                     │   │
                                    │  │      REST API endpoints for querying data           │   │
                                    │  │   Combines speed layer (Redis) + batch views       │   │
                                    │  └─────────────────────────────────────────────────────┘   │
                                    └─────────────────────────────────────────────────────────────┘
                                                                │
                                                                ▼
                                    ┌─────────────────────────────────────────────────────────────┐
                                    │                        CONSUMERS                            │
                                    │  ┌─────────────────────────────────────────────────────┐   │
                                    │  │  Frontend Dashboard  │  Monitoring  │  Alerting    │   │
                                    │  └─────────────────────────────────────────────────────┘   │
                                    └─────────────────────────────────────────────────────────────┘
```

## Component Details

### Reddit Scraper

- Uses PRAW library for Reddit API access
- Scrapes hot/new/rising posts from configured subreddits
- Extracts post metadata (title, content, score, author)
- Optionally includes comments
- Publishes to Kafka with topic-based keys

### Kafka

- Message broker for decoupling producers and consumers
- Topic: `reddit-raw-posts` (partitioned by topic)
- Enables multiple consumers (streaming + batch)
- Provides durability and replay capability

### Spark Streaming (Speed Layer)

- Consumes from Kafka in micro-batches
- Applies BERT sentiment analysis in real-time
- Extracts keywords and entities
- Writes to Redis for low-latency queries
- Writes to MongoDB for persistence
- Generates alerts for sentiment threshold breaches

### HDFS (Batch Layer Storage)

- Stores raw processed data as Parquet files
- Partitioned by topic for efficient queries
- Enables historical reprocessing

### Spark Batch (Batch Layer)

- Runs hourly via cron
- Computes comprehensive aggregations
- Detects anomalies using z-score analysis
- Generates keyword rankings
- Creates subreddit-level breakdowns

### Redis (Speed Layer Cache)

- Caches real-time sentiment metrics
- Stores time-series data for trends
- Low-latency access for serving layer
- TTL-based expiration

### MongoDB (Serving Layer)

- Stores merged batch and real-time views
- Indexed for fast queries
- Collections for different data types
- Stores alerts and topic configurations

### FastAPI (Serving Layer API)

- Combines speed layer (Redis) + batch views (MongoDB)
- RESTful endpoints for all data access
- Prometheus metrics for monitoring
- Swagger documentation

## Data Models

### Reddit Post

```python
{
    "post_id": "abc123",
    "topic": "technology",
    "subreddit": "programming",
    "title": "New AI breakthrough",
    "content": "...",
    "author": "user123",
    "score": 1234,
    "num_comments": 56,
    "created_utc": 1701456789.0,
    "is_comment": false
}
```

### Sentiment Result

```python
{
    "post_id": "abc123",
    "topic": "technology",
    "sentiment_score": 0.75,
    "sentiment_label": "positive",
    "confidence": 0.92,
    "keywords": ["AI", "machine", "learning"],
    "entities": [{"text": "OpenAI", "type": "ORG"}],
    "processed_at": "2024-12-01T12:00:00Z"
}
```

### Batch Aggregation

```python
{
    "topic": "technology",
    "period_start": "2024-12-01T00:00:00Z",
    "period_end": "2024-12-01T01:00:00Z",
    "avg_sentiment": 0.23,
    "std_sentiment": 0.45,
    "total_posts": 1234,
    "positive_count": 567,
    "negative_count": 234,
    "neutral_count": 433,
    "top_keywords": [{"word": "AI", "count": 89}]
}
```

## Scaling Considerations

1. **Kafka Partitions**: Increase for higher throughput
2. **Spark Executors**: Add more workers for parallelism
3. **HDFS Replication**: Increase for durability
4. **Redis Cluster**: Shard for larger datasets
5. **MongoDB Replica Set**: Add for high availability
