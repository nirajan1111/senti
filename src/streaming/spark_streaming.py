"""
Spark Structured Streaming Module
Real-time sentiment analysis using Spark Structured Streaming (Speed Layer)
"""

import os
import sys
import json
import signal
from datetime import datetime
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    from_json, col, udf, current_timestamp, lit, 
    window, avg, count, sum as spark_sum, expr,
    to_json, struct
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    FloatType, TimestampType, ArrayType, BooleanType, DoubleType
)

sys.path.insert(0, '/app')

from config.settings import get_config
from src.utils.logging_config import setup_logging, get_logger


class SparkStreamingProcessor:
    """
    Production-grade Spark Structured Streaming processor for sentiment analysis
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("SparkStreaming")
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize Spark
        self.spark = self._create_spark_session()
        
        # Initialize sentiment analyzer (lazy loading)
        self._analyzer = None
        
        # Define schemas
        self._define_schemas()
        
        self.logger.info("SparkStreamingProcessor initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, stopping streaming...")
        self.running = False
    
    def _create_spark_session(self) -> SparkSession:
        """Create and configure Spark session"""
        try:
            spark = SparkSession.builder \
                .appName("SentimentAnalysis-Streaming") \
                .master("local[*]") \
                .config("spark.jars.packages", 
                        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
                .config("spark.sql.streaming.checkpointLocation", 
                        self.config.spark.checkpoint_dir) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.streaming.backpressure.enabled", "true") \
                .config("spark.sql.shuffle.partitions", "2") \
                .config("spark.driver.memory", "1g") \
                .config("spark.executor.memory", "1g") \
                .config("spark.driver.host", "localhost") \
                .getOrCreate()
            
            spark.sparkContext.setLogLevel("WARN")
            
            self.logger.info("Spark session created successfully")
            return spark
            
        except Exception as e:
            self.logger.error(f"Failed to create Spark session: {e}")
            raise
    
    def _define_schemas(self):
        """Define schemas for incoming data"""
        
        # Schema for Reddit posts from Kafka
        self.reddit_post_schema = StructType([
            StructField("post_id", StringType(), False),
            StructField("topic", StringType(), False),
            StructField("subreddit", StringType(), True),
            StructField("title", StringType(), True),
            StructField("content", StringType(), True),
            StructField("author", StringType(), True),
            StructField("score", IntegerType(), True),
            StructField("num_comments", IntegerType(), True),
            StructField("url", StringType(), True),
            StructField("created_utc", DoubleType(), True),
            StructField("scraped_at", StringType(), True),
            StructField("is_comment", BooleanType(), True),
            StructField("parent_id", StringType(), True)
        ])
        
        # Schema for sentiment results
        self.sentiment_result_schema = StructType([
            StructField("sentiment_score", FloatType(), True),
            StructField("sentiment_label", StringType(), True),
            StructField("confidence", FloatType(), True),
            StructField("cleaned_text", StringType(), True),
            StructField("keywords", ArrayType(StringType()), True),
            StructField("entities", ArrayType(StringType()), True),
            StructField("processing_time_ms", FloatType(), True)
        ])
    
    # Class-level model cache for transformer
    _sentiment_pipeline = None
    _model_loaded = False
    
    @classmethod
    def get_sentiment_pipeline(cls):
        """Lazy load the tiny transformer sentiment model (singleton pattern)"""
        # Initialize VADER sentiment analyzer (lightweight and fast)
        if not cls._model_loaded:
            try:
                import nltk
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                
                # Download VADER lexicon if needed
                try:
                    nltk.data.find('sentiment/vader_lexicon.zip')
                except LookupError:
                    nltk.download('vader_lexicon', quiet=True)
                
                cls._sentiment_pipeline = SentimentIntensityAnalyzer()
                cls._model_loaded = True
                print("✓ Loaded VADER sentiment analyzer successfully")
            except Exception as e:
                print(f"✗ Failed to load VADER: {e}")
                cls._sentiment_pipeline = None
                cls._model_loaded = True
        return cls._sentiment_pipeline
    
    def _analyze_sentiment_udf(self):
        """Create UDF for sentiment analysis using VADER + TextBlob ensemble"""
        
        def analyze(text: str) -> str:
            if not text or not text.strip():
                return json.dumps({
                    "sentiment_score": 0.0,
                    "sentiment_label": "neutral",
                    "confidence": 0.5,
                    "cleaned_text": "",
                    "keywords": [],
                    "entities": [],
                    "processing_time_ms": 0.0
                })
            
            try:
                import re
                import time
                from collections import Counter
                start_time = time.time()
                
                # Clean text for display
                cleaned = re.sub(r'http\S+|www\S+', '', text)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                
                if not cleaned:
                    return json.dumps({
                        "sentiment_score": 0.0,
                        "sentiment_label": "neutral",
                        "confidence": 0.5,
                        "cleaned_text": "",
                        "keywords": [],
                        "entities": [],
                        "processing_time_ms": 0.0
                    })
                
                # Primary: Use VADER (specifically designed for social media)
                try:
                    from nltk.sentiment.vader import SentimentIntensityAnalyzer
                    import nltk
                    
                    # Ensure VADER lexicon is available
                    try:
                        nltk.data.find('sentiment/vader_lexicon.zip')
                    except LookupError:
                        nltk.download('vader_lexicon', quiet=True)
                    
                    vader = SentimentIntensityAnalyzer()
                    vader_scores = vader.polarity_scores(cleaned)
                    
                    # VADER compound score is already -1 to 1
                    vader_compound = vader_scores['compound']
                    vader_pos = vader_scores['pos']
                    vader_neg = vader_scores['neg']
                    
                    score = vader_compound
                    confidence = max(vader_pos, vader_neg, vader_scores['neu'])
                    
                except Exception as vader_error:
                    print(f"VADER failed: {vader_error}")
                    # Fallback to TextBlob
                    try:
                        from textblob import TextBlob
                        blob = TextBlob(cleaned)
                        score = blob.sentiment.polarity  # -1 to 1
                        confidence = abs(blob.sentiment.subjectivity)
                    except:
                        score = 0.0
                        confidence = 0.5
                
                # Determine label based on score
                if score >= 0.05:
                    label = "positive"
                elif score <= -0.05:
                    label = "negative"
                else:
                    label = "neutral"
                
                # Extract keywords (frequency-based)
                stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                            'as', 'into', 'through', 'during', 'before', 'after', 'above',
                            'below', 'between', 'under', 'again', 'further', 'then', 'once',
                            'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                            'neither', 'not', 'only', 'own', 'same', 'than', 'too', 'very',
                            'just', 'also', 'now', 'here', 'there', 'when', 'where', 'why',
                            'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
                            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                            'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                            'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                            'like', 'just', 'really', 'much', 'even', 'still', 'well'}
                
                words = re.sub(r'[^\w\s]', ' ', cleaned.lower()).split()
                word_counts = Counter(w for w in words if len(w) > 3 and w not in stopwords)
                keywords = [w for w, _ in word_counts.most_common(5)]
                
                processing_time = (time.time() - start_time) * 1000
                
                return json.dumps({
                    "sentiment_score": round(score, 4),
                    "sentiment_label": label,
                    "confidence": round(confidence, 4),
                    "cleaned_text": cleaned[:500],
                    "keywords": keywords,
                    "entities": [],
                    "processing_time_ms": round(processing_time, 2)
                })
                
            except Exception as e:
                return json.dumps({
                    "sentiment_score": 0.0,
                    "sentiment_label": "neutral",
                    "confidence": 0.0,
                    "cleaned_text": text[:500] if text else "",
                    "keywords": [],
                    "entities": [],
                    "processing_time_ms": 0.0,
                    "error": str(e)
                })
        
        return udf(analyze, StringType())
    
    def read_kafka_stream(self) -> DataFrame:
        """Read streaming data from Kafka"""
        
        self.logger.info(f"Connecting to Kafka at {self.config.kafka.bootstrap_servers}")
        
        return self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.config.kafka.bootstrap_servers) \
            .option("subscribe", self.config.kafka.raw_posts_topic) \
            .option("startingOffsets", "earliest") \
            .option("failOnDataLoss", "false") \
            .option("maxOffsetsPerTrigger", 100) \
            .load()
    
    def process_stream(self, df: DataFrame) -> DataFrame:
        """Process the streaming data"""
        
        # Parse JSON from Kafka value
        parsed_df = df.select(
            col("key").cast("string").alias("kafka_key"),
            from_json(col("value").cast("string"), self.reddit_post_schema).alias("data"),
            col("timestamp").alias("kafka_timestamp")
        ).select("kafka_key", "data.*", "kafka_timestamp")
        
        # Combine title and content for analysis
        combined_text_df = parsed_df.withColumn(
            "full_text",
            expr("concat_ws(' ', coalesce(title, ''), coalesce(content, ''))")
        )
        
        # Apply sentiment analysis
        analyze_udf = self._analyze_sentiment_udf()
        
        sentiment_df = combined_text_df.withColumn(
            "sentiment_result",
            analyze_udf(col("full_text"))
        )
        
        # Parse sentiment result JSON
        result_schema = StructType([
            StructField("sentiment_score", FloatType()),
            StructField("sentiment_label", StringType()),
            StructField("confidence", FloatType()),
            StructField("cleaned_text", StringType()),
            StructField("keywords", ArrayType(StringType())),
            StructField("entities", ArrayType(StringType())),
            StructField("processing_time_ms", FloatType())
        ])
        
        final_df = sentiment_df.withColumn(
            "sentiment_parsed",
            from_json(col("sentiment_result"), result_schema)
        ).select(
            col("post_id"),
            col("topic"),
            col("subreddit"),
            col("title"),
            col("content"),
            col("author"),
            col("score"),
            col("num_comments"),
            col("created_utc"),
            col("is_comment"),
            col("sentiment_parsed.sentiment_score").alias("sentiment_score"),
            col("sentiment_parsed.sentiment_label").alias("sentiment_label"),
            col("sentiment_parsed.confidence").alias("confidence"),
            col("sentiment_parsed.cleaned_text").alias("cleaned_text"),
            col("sentiment_parsed.keywords").alias("keywords"),
            col("sentiment_parsed.processing_time_ms").alias("processing_time_ms"),
            current_timestamp().alias("processed_at")
        )
        
        return final_df
    
    def write_to_console(self, df: DataFrame, query_name: str = "console_output"):
        """Write stream to console for debugging"""
        return df.writeStream \
            .outputMode("append") \
            .format("console") \
            .option("truncate", False) \
            .option("numRows", 5) \
            .queryName(query_name) \
            .start()
    
    def write_to_mongodb(self, df: DataFrame, query_name: str = "mongodb_sink"):
        """Write stream to MongoDB"""
        
        def write_batch(batch_df: DataFrame, batch_id: int):
            """Write each batch to MongoDB"""
            if batch_df.isEmpty():
                return
            
            try:
                from pymongo import MongoClient
                from datetime import datetime
                
                client = MongoClient(self.config.mongodb.uri)
                db = client[self.config.mongodb.database]
                
                # Convert to list of dicts
                records = batch_df.toPandas().to_dict('records')
                
                if records:
                    # Fix records to match MongoDB schema
                    for record in records:
                        # Add required timestamp field
                        record['timestamp'] = record.get('processed_at', datetime.utcnow())
                        # Ensure sentiment_score is float
                        if record.get('sentiment_score') is not None:
                            record['sentiment_score'] = float(record['sentiment_score'])
                        if record.get('confidence') is not None:
                            record['confidence'] = float(record['confidence'])
                    
                    # Insert sentiment results (use insert with ordered=False to continue on errors)
                    try:
                        db.sentiment_results.insert_many(records, ordered=False)
                    except Exception as insert_error:
                        print(f"Some inserts failed (may be duplicates): {insert_error}")
                    
                    # Check for alerts
                    for record in records:
                        if record.get('sentiment_score', 0) < self.config.sentiment.alert_threshold:
                            alert = {
                                "topic": record.get('topic'),
                                "alert_type": "low_sentiment",
                                "message": f"Low sentiment detected: {record.get('sentiment_score')}",
                                "severity": "warning",
                                "sentiment_value": record.get('sentiment_score'),
                                "threshold": self.config.sentiment.alert_threshold,
                                "post_id": record.get('post_id'),
                                "timestamp": datetime.utcnow(),
                                "acknowledged": False
                            }
                            try:
                                db.alerts.insert_one(alert)
                            except Exception:
                                pass
                
                client.close()
                
            except Exception as e:
                print(f"Error writing to MongoDB: {e}")
        
        return df.writeStream \
            .outputMode("append") \
            .foreachBatch(write_batch) \
            .option("checkpointLocation", f"{self.config.spark.checkpoint_dir}/mongodb") \
            .queryName(query_name) \
            .start()
    
    def write_to_redis(self, df: DataFrame, query_name: str = "redis_sink"):
        """Write real-time aggregations to Redis for speed layer"""
        
        def write_batch(batch_df: DataFrame, batch_id: int):
            """Write aggregated stats to Redis"""
            if batch_df.isEmpty():
                return
            
            try:
                import redis
                import json
                
                r = redis.Redis(
                    host=self.config.redis.host,
                    port=self.config.redis.port,
                    db=self.config.redis.db
                )
                
                # Aggregate by topic
                topic_stats = batch_df.groupBy("topic").agg(
                    avg("sentiment_score").alias("avg_sentiment"),
                    count("*").alias("count")
                ).toPandas()
                
                for _, row in topic_stats.iterrows():
                    topic = row['topic']
                    stats = {
                        "avg_sentiment": float(row['avg_sentiment']) if row['avg_sentiment'] else 0,
                        "count": int(row['count']),
                        "updated_at": datetime.utcnow().isoformat()
                    }
                    
                    # Store current stats
                    r.setex(
                        f"sentiment:realtime:{topic}",
                        self.config.redis.cache_ttl,
                        json.dumps(stats)
                    )
                    
                    # Add to time series
                    r.lpush(f"sentiment:history:{topic}", json.dumps(stats))
                    r.ltrim(f"sentiment:history:{topic}", 0, 1000)  # Keep last 1000 entries
                
            except Exception as e:
                print(f"Error writing to Redis: {e}")
        
        return df.writeStream \
            .outputMode("append") \
            .foreachBatch(write_batch) \
            .option("checkpointLocation", f"{self.config.spark.checkpoint_dir}/redis") \
            .queryName(query_name) \
            .start()
    
    def write_to_hdfs(self, df: DataFrame, query_name: str = "hdfs_sink"):
        """Write raw data to HDFS for batch processing"""
        
        return df.writeStream \
            .outputMode("append") \
            .format("parquet") \
            .option("path", f"{self.config.hdfs.namenode}{self.config.hdfs.raw_data_path}") \
            .option("checkpointLocation", f"{self.config.spark.checkpoint_dir}/hdfs") \
            .partitionBy("topic") \
            .queryName(query_name) \
            .trigger(processingTime="30 seconds") \
            .start()
    
    def run(self):
        """Main execution loop"""
        self.logger.info("Starting Spark Streaming processor...")
        
        try:
            # Read from Kafka
            kafka_df = self.read_kafka_stream()
            
            # Process stream
            processed_df = self.process_stream(kafka_df)
            
            # Write to multiple sinks
            queries = []
            
            # Console output for debugging
            try:
                queries.append(self.write_to_console(processed_df, "console_debug"))
            except Exception as e:
                self.logger.warning(f"Failed to start console sink: {e}")
            
            # MongoDB for serving layer
            try:
                queries.append(self.write_to_mongodb(processed_df, "mongodb_sink"))
            except Exception as e:
                self.logger.warning(f"Failed to start MongoDB sink: {e}")
            
            # Redis for real-time stats
            try:
                queries.append(self.write_to_redis(processed_df, "redis_sink"))
            except Exception as e:
                self.logger.warning(f"Failed to start Redis sink: {e}")
            
            # Skip HDFS for now - can be unstable
            # try:
            #     queries.append(self.write_to_hdfs(processed_df, "hdfs_sink"))
            # except Exception as e:
            #     self.logger.warning(f"Failed to start HDFS sink: {e}")
            
            self.logger.info(f"Started {len(queries)} streaming queries")
            
            # Wait for any query to terminate (use spark.streams.awaitAnyTermination)
            if queries:
                self.spark.streams.awaitAnyTermination()
                
        except Exception as e:
            self.logger.error(f"Streaming error: {e}", exc_info=True)
            raise
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down Spark Streaming...")
        
        try:
            for query in self.spark.streams.active:
                query.stop()
            
            self.spark.stop()
            self.logger.info("Spark session stopped")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


def main():
    """Main entry point"""
    setup_logging(log_level="INFO", json_format=True, service_name="spark-streaming")
    
    logger = get_logger("main")
    logger.info("Starting Spark Streaming Service...")
    
    try:
        processor = SparkStreamingProcessor()
        processor.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
