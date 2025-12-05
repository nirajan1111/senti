"""
Application Configuration Module
Centralized configuration management using Pydantic Settings
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
from functools import lru_cache


class RedditConfig(BaseSettings):
    """Reddit API Configuration"""
    client_id: str = Field(default="", env="REDDIT_CLIENT_ID")
    client_secret: str = Field(default="", env="REDDIT_CLIENT_SECRET")
    user_agent: str = Field(default="", env="REDDIT_USER_AGENT")
    
    class Config:
        env_prefix = "REDDIT_"


class KafkaConfig(BaseSettings):
    """Kafka Configuration"""
    bootstrap_servers: str = Field(default="kafka:9092", env="KAFKA_BOOTSTRAP_SERVERS")
    raw_posts_topic: str = Field(default="reddit-raw-posts")
    sentiment_results_topic: str = Field(default="sentiment-results")
    alerts_topic: str = Field(default="sentiment-alerts")
    consumer_group_id: str = Field(default="sentiment-processor-group")
    auto_offset_reset: str = Field(default="earliest")
    
    class Config:
        env_prefix = "KAFKA_"


class HDFSConfig(BaseSettings):
    """HDFS Configuration"""
    namenode: str = Field(default="hdfs://namenode:9000", env="HDFS_NAMENODE")
    raw_data_path: str = Field(default="/data/raw")
    processed_data_path: str = Field(default="/data/processed")
    batch_output_path: str = Field(default="/data/batch")
    
    class Config:
        env_prefix = "HDFS_"


class RedisConfig(BaseSettings):
    """Redis Configuration"""
    host: str = Field(default="redis", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0)
    cache_ttl: int = Field(default=300)  # 5 minutes
    
    class Config:
        env_prefix = "REDIS_"


class MongoDBConfig(BaseSettings):
    """MongoDB Configuration"""
    uri: str = Field(
        default="mongodb://admin:sentiment_admin_2024@mongodb:27017",
        env="MONGODB_URI"
    )
    database: str = Field(default="sentiment_db")
    
    class Config:
        env_prefix = "MONGODB_"


class SparkConfig(BaseSettings):
    """Spark Configuration"""
    master_url: str = Field(default="spark://spark-master:7077", env="SPARK_MASTER_URL")
    app_name: str = Field(default="SentimentAnalysis")
    streaming_batch_interval: int = Field(default=10)  # seconds
    checkpoint_dir: str = Field(default="/app/checkpoints")
    
    class Config:
        env_prefix = "SPARK_"


class SentimentConfig(BaseSettings):
    """Sentiment Analysis Configuration"""
    model_name: str = Field(default="nlptown/bert-base-multilingual-uncased-sentiment")
    max_length: int = Field(default=512)
    batch_size: int = Field(default=32)
    alert_threshold: float = Field(default=-0.5, env="SENTIMENT_ALERT_THRESHOLD")
    
    class Config:
        env_prefix = "SENTIMENT_"


class ScraperConfig(BaseSettings):
    """Scraper Configuration"""
    interval_seconds: int = Field(default=60, env="SCRAPE_INTERVAL_SECONDS")
    posts_per_subreddit: int = Field(default=25)
    include_comments: bool = Field(default=True)
    max_comments_per_post: int = Field(default=10)
    
    class Config:
        env_prefix = "SCRAPER_"


class BatchConfig(BaseSettings):
    """Batch Processing Configuration"""
    interval_hours: int = Field(default=1, env="BATCH_INTERVAL_HOURS")
    lookback_hours: int = Field(default=24)
    aggregation_window: str = Field(default="1 hour")
    
    class Config:
        env_prefix = "BATCH_"


class AppConfig(BaseSettings):
    """Main Application Configuration"""
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Sub-configurations
    reddit: RedditConfig = Field(default_factory=RedditConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    hdfs: HDFSConfig = Field(default_factory=HDFSConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    mongodb: MongoDBConfig = Field(default_factory=MongoDBConfig)
    spark: SparkConfig = Field(default_factory=SparkConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    scraper: ScraperConfig = Field(default_factory=ScraperConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_config() -> AppConfig:
    """Get cached application configuration"""
    return AppConfig()
