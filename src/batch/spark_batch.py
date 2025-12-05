"""
Spark Batch Processing Module
Hourly batch processing for historical analysis (Batch Layer)
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, avg, count, sum as spark_sum, stddev,
    collect_list, explode, lit, window, 
    from_unixtime, to_timestamp, hour, date_format,
    when, expr, udf, array_distinct, flatten,
    desc, row_number
)
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    FloatType, TimestampType, ArrayType, MapType, DoubleType
)

sys.path.insert(0, '/app')

from config.settings import get_config
from src.utils.logging_config import setup_logging, get_logger


class SparkBatchProcessor:
    """
    Production-grade Spark batch processor for historical sentiment analysis
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("SparkBatch")
        
        # Initialize Spark
        self.spark = self._create_spark_session()
        
        self.logger.info("SparkBatchProcessor initialized")
    
    def _create_spark_session(self) -> SparkSession:
        """Create and configure Spark session"""
        try:
            spark = SparkSession.builder \
                .appName("SentimentAnalysis-Batch") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.shuffle.partitions", "10") \
                .config("spark.driver.memory", "2g") \
                .config("spark.executor.memory", "2g") \
                .getOrCreate()
            
            spark.sparkContext.setLogLevel("WARN")
            
            self.logger.info("Spark session created successfully")
            return spark
            
        except Exception as e:
            self.logger.error(f"Failed to create Spark session: {e}")
            raise
    
    def load_data_from_hdfs(
        self, 
        hours_lookback: int = 24
    ) -> Optional[DataFrame]:
        """
        Load raw sentiment data from HDFS
        
        Args:
            hours_lookback: Number of hours to look back
            
        Returns:
            DataFrame with historical data
        """
        try:
            hdfs_path = f"{self.config.hdfs.namenode}{self.config.hdfs.raw_data_path}"
            
            self.logger.info(f"Loading data from HDFS: {hdfs_path}")
            
            df = self.spark.read.parquet(hdfs_path)
            
            # Filter to lookback period
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_lookback)
            
            filtered_df = df.filter(
                col("processed_at") >= lit(cutoff_time)
            )
            
            record_count = filtered_df.count()
            self.logger.info(f"Loaded {record_count} records from HDFS")
            
            return filtered_df
            
        except Exception as e:
            self.logger.warning(f"Failed to load from HDFS: {e}")
            return None
    
    def load_data_from_mongodb(
        self, 
        hours_lookback: int = 24
    ) -> Optional[DataFrame]:
        """
        Load data from MongoDB as fallback or supplement
        
        Args:
            hours_lookback: Number of hours to look back
            
        Returns:
            DataFrame with historical data
        """
        try:
            from pymongo import MongoClient
            import pandas as pd
            
            client = MongoClient(self.config.mongodb.uri)
            db = client[self.config.mongodb.database]
            
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_lookback)
            
            # Query sentiment results
            cursor = db.sentiment_results.find({
                "processed_at": {"$gte": cutoff_time}
            })
            
            records = list(cursor)
            client.close()
            
            if not records:
                self.logger.warning("No records found in MongoDB")
                return None
            
            # Convert to Spark DataFrame
            pandas_df = pd.DataFrame(records)
            
            # Drop MongoDB _id field
            if '_id' in pandas_df.columns:
                pandas_df = pandas_df.drop(columns=['_id'])
            
            spark_df = self.spark.createDataFrame(pandas_df)
            
            self.logger.info(f"Loaded {len(records)} records from MongoDB")
            return spark_df
            
        except Exception as e:
            self.logger.error(f"Failed to load from MongoDB: {e}")
            return None
    
    def compute_topic_aggregations(self, df: DataFrame) -> DataFrame:
        """
        Compute aggregations per topic for the batch period
        
        Args:
            df: Input DataFrame with sentiment results
            
        Returns:
            DataFrame with topic-level aggregations
        """
        self.logger.info("Computing topic aggregations...")
        
        # Basic aggregations
        agg_df = df.groupBy("topic").agg(
            avg("sentiment_score").alias("avg_sentiment"),
            stddev("sentiment_score").alias("std_sentiment"),
            count("*").alias("total_posts"),
            spark_sum(when(col("sentiment_label") == "very_positive", 1).otherwise(0)).alias("very_positive_count"),
            spark_sum(when(col("sentiment_label") == "positive", 1).otherwise(0)).alias("positive_count"),
            spark_sum(when(col("sentiment_label") == "neutral", 1).otherwise(0)).alias("neutral_count"),
            spark_sum(when(col("sentiment_label") == "negative", 1).otherwise(0)).alias("negative_count"),
            spark_sum(when(col("sentiment_label") == "very_negative", 1).otherwise(0)).alias("very_negative_count"),
        )
        
        # Calculate sentiment distribution percentages
        agg_df = agg_df.withColumn(
            "positive_pct",
            (col("very_positive_count") + col("positive_count")) / col("total_posts") * 100
        ).withColumn(
            "negative_pct",
            (col("very_negative_count") + col("negative_count")) / col("total_posts") * 100
        ).withColumn(
            "neutral_pct",
            col("neutral_count") / col("total_posts") * 100
        )
        
        return agg_df
    
    def compute_hourly_trends(self, df: DataFrame) -> DataFrame:
        """
        Compute hourly sentiment trends
        
        Args:
            df: Input DataFrame with sentiment results
            
        Returns:
            DataFrame with hourly trends
        """
        self.logger.info("Computing hourly trends...")
        
        # Add hour column
        hourly_df = df.withColumn(
            "hour",
            date_format(col("processed_at"), "yyyy-MM-dd HH:00:00")
        )
        
        # Aggregate by topic and hour
        trends_df = hourly_df.groupBy("topic", "hour").agg(
            avg("sentiment_score").alias("avg_sentiment"),
            count("*").alias("post_count"),
            avg("score").alias("avg_reddit_score")
        ).orderBy("topic", "hour")
        
        return trends_df
    
    def compute_subreddit_breakdown(self, df: DataFrame) -> DataFrame:
        """
        Compute sentiment breakdown by subreddit
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with subreddit-level stats
        """
        self.logger.info("Computing subreddit breakdown...")
        
        breakdown_df = df.groupBy("topic", "subreddit").agg(
            avg("sentiment_score").alias("avg_sentiment"),
            count("*").alias("post_count"),
            avg("score").alias("avg_reddit_score")
        ).orderBy("topic", desc("post_count"))
        
        return breakdown_df
    
    def extract_top_keywords(self, df: DataFrame, top_n: int = 20) -> DataFrame:
        """
        Extract top keywords per topic
        
        Args:
            df: Input DataFrame with keywords column
            top_n: Number of top keywords to extract
            
        Returns:
            DataFrame with top keywords per topic
        """
        self.logger.info("Extracting top keywords...")
        
        # Explode keywords array
        keywords_df = df.filter(col("keywords").isNotNull()) \
            .select("topic", explode("keywords").alias("keyword"))
        
        # Count keyword frequencies
        keyword_counts = keywords_df.groupBy("topic", "keyword").agg(
            count("*").alias("count")
        )
        
        # Get top N per topic
        window_spec = Window.partitionBy("topic").orderBy(desc("count"))
        
        top_keywords = keyword_counts.withColumn(
            "rank", row_number().over(window_spec)
        ).filter(col("rank") <= top_n)
        
        return top_keywords
    
    def detect_anomalies(self, df: DataFrame) -> DataFrame:
        """
        Detect sentiment anomalies (sudden drops or spikes)
        
        Args:
            df: Input DataFrame with hourly trends
            
        Returns:
            DataFrame with anomaly flags
        """
        self.logger.info("Detecting anomalies...")
        
        # Calculate rolling average and standard deviation
        window_spec = Window.partitionBy("topic") \
            .orderBy("hour") \
            .rowsBetween(-6, 0)  # 6-hour rolling window
        
        anomaly_df = df.withColumn(
            "rolling_avg", avg("avg_sentiment").over(window_spec)
        ).withColumn(
            "rolling_std", stddev("avg_sentiment").over(window_spec)
        ).withColumn(
            "z_score",
            (col("avg_sentiment") - col("rolling_avg")) / 
            when(col("rolling_std") > 0, col("rolling_std")).otherwise(1)
        ).withColumn(
            "is_anomaly",
            when(abs(col("z_score")) > 2, True).otherwise(False)
        )
        
        return anomaly_df
    
    def save_to_mongodb(
        self, 
        aggregations: DataFrame,
        trends: DataFrame,
        keywords: DataFrame,
        subreddit_breakdown: DataFrame
    ):
        """
        Save batch results to MongoDB
        
        Args:
            aggregations: Topic-level aggregations
            trends: Hourly trends
            keywords: Top keywords
            subreddit_breakdown: Subreddit breakdown
        """
        self.logger.info("Saving batch results to MongoDB...")
        
        try:
            from pymongo import MongoClient
            
            client = MongoClient(self.config.mongodb.uri)
            db = client[self.config.mongodb.database]
            
            batch_time = datetime.utcnow()
            period_start = batch_time - timedelta(hours=self.config.batch.lookback_hours)
            
            # Save aggregations
            agg_records = aggregations.toPandas().to_dict('records')
            for record in agg_records:
                record['period_start'] = period_start
                record['period_end'] = batch_time
                record['created_at'] = batch_time
                
                # Update or insert
                db.batch_aggregations.update_one(
                    {
                        "topic": record['topic'],
                        "period_start": period_start
                    },
                    {"$set": record},
                    upsert=True
                )
            
            # Save trends
            trends_collection = db['hourly_trends']
            trends_records = trends.toPandas().to_dict('records')
            for record in trends_records:
                record['batch_time'] = batch_time
                
            if trends_records:
                trends_collection.insert_many(trends_records)
            
            # Save top keywords
            keywords_collection = db['batch_keywords']
            keywords_records = keywords.toPandas().to_dict('records')
            for record in keywords_records:
                record['batch_time'] = batch_time
                
            if keywords_records:
                keywords_collection.insert_many(keywords_records)
            
            # Save subreddit breakdown
            subreddit_collection = db['subreddit_stats']
            subreddit_records = subreddit_breakdown.toPandas().to_dict('records')
            for record in subreddit_records:
                record['batch_time'] = batch_time
                
            if subreddit_records:
                subreddit_collection.insert_many(subreddit_records)
            
            client.close()
            
            self.logger.info("Batch results saved to MongoDB successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save to MongoDB: {e}")
            raise
    
    def generate_alerts(self, anomalies: DataFrame, aggregations: DataFrame):
        """
        Generate alerts based on batch analysis
        
        Args:
            anomalies: DataFrame with detected anomalies
            aggregations: Topic-level aggregations
        """
        self.logger.info("Generating alerts...")
        
        try:
            from pymongo import MongoClient
            
            client = MongoClient(self.config.mongodb.uri)
            db = client[self.config.mongodb.database]
            
            # Alert for anomalies
            anomaly_records = anomalies.filter(col("is_anomaly") == True).toPandas()
            
            for _, row in anomaly_records.iterrows():
                alert = {
                    "topic": row['topic'],
                    "alert_type": "sentiment_anomaly",
                    "message": f"Sentiment anomaly detected. Z-score: {row['z_score']:.2f}",
                    "severity": "warning" if abs(row['z_score']) < 3 else "critical",
                    "sentiment_value": float(row['avg_sentiment']),
                    "threshold": 2.0,  # Z-score threshold
                    "timestamp": datetime.utcnow(),
                    "acknowledged": False
                }
                db.alerts.insert_one(alert)
            
            # Alert for low average sentiment
            agg_records = aggregations.toPandas()
            
            for _, row in agg_records.iterrows():
                if row['avg_sentiment'] < self.config.sentiment.alert_threshold:
                    alert = {
                        "topic": row['topic'],
                        "alert_type": "low_avg_sentiment",
                        "message": f"Average sentiment below threshold: {row['avg_sentiment']:.3f}",
                        "severity": "warning",
                        "sentiment_value": float(row['avg_sentiment']),
                        "threshold": self.config.sentiment.alert_threshold,
                        "post_count": int(row['total_posts']),
                        "timestamp": datetime.utcnow(),
                        "acknowledged": False
                    }
                    db.alerts.insert_one(alert)
            
            client.close()
            
            self.logger.info("Alerts generated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to generate alerts: {e}")
    
    def run(self):
        """Main batch processing execution"""
        self.logger.info(f"Starting batch processing job at {datetime.utcnow()}")
        
        try:
            # Load data (try HDFS first, fallback to MongoDB)
            df = self.load_data_from_hdfs(self.config.batch.lookback_hours)
            
            if df is None or df.count() == 0:
                self.logger.info("Trying MongoDB as fallback...")
                df = self.load_data_from_mongodb(self.config.batch.lookback_hours)
            
            if df is None or df.count() == 0:
                self.logger.warning("No data available for batch processing")
                return
            
            # Compute aggregations
            aggregations = self.compute_topic_aggregations(df)
            
            # Compute hourly trends
            trends = self.compute_hourly_trends(df)
            
            # Extract top keywords
            keywords = self.extract_top_keywords(df)
            
            # Compute subreddit breakdown
            subreddit_breakdown = self.compute_subreddit_breakdown(df)
            
            # Detect anomalies
            anomalies = self.detect_anomalies(trends)
            
            # Save results
            self.save_to_mongodb(
                aggregations,
                trends,
                keywords,
                subreddit_breakdown
            )
            
            # Generate alerts
            self.generate_alerts(anomalies, aggregations)
            
            self.logger.info(f"Batch processing completed at {datetime.utcnow()}")
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}", exc_info=True)
            raise
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down Spark Batch Processor...")
        try:
            self.spark.stop()
            self.logger.info("Spark session stopped")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


def main():
    """Main entry point"""
    setup_logging(log_level="INFO", json_format=True, service_name="spark-batch")
    
    logger = get_logger("main")
    logger.info("Starting Spark Batch Processing Job...")
    
    try:
        processor = SparkBatchProcessor()
        processor.run()
    except Exception as e:
        logger.error(f"Batch job failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
