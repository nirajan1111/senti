"""
Serving Layer API
FastAPI application for querying sentiment analysis results
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import redis
from pymongo import MongoClient
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

sys.path.insert(0, "/app")

from config.settings import get_config
from src.utils.logging_config import setup_logging, get_logger
from src.models.data_models import (
    Topic,
    TopicRequest,
    SentimentResult,
    BatchAggregation,
    Alert,
    SentimentTrend,
)


# Prometheus metrics
REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "Request latency in seconds", ["method", "endpoint"]
)


class ServingAPI:
    """
    Serving layer API for Lambda Architecture
    Combines real-time (speed layer) and batch views
    """

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("ServingAPI")

        # Initialize connections
        self.redis_client = self._init_redis()
        self.mongo_client = self._init_mongodb()
        self.db = self.mongo_client[self.config.mongodb.database]

        self.logger.info("ServingAPI initialized")

    def _init_redis(self) -> redis.Redis:
        """Initialize Redis connection"""
        return redis.Redis(
            host=self.config.redis.host,
            port=self.config.redis.port,
            db=self.config.redis.db,
            decode_responses=True,
        )

    def _init_mongodb(self) -> MongoClient:
        """Initialize MongoDB connection"""
        return MongoClient(self.config.mongodb.uri)

    def get_realtime_sentiment(self, topic: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time sentiment from speed layer (Redis)

        Args:
            topic: Topic name

        Returns:
            Real-time sentiment data
        """
        try:
            import json

            data = self.redis_client.get(f"sentiment:realtime:{topic}")
            if data:
                return json.loads(data)
            return None

        except Exception as e:
            self.logger.error(f"Redis error: {e}")
            return None

    def get_batch_aggregation(
        self, topic: str, hours_back: int = 24
    ) -> Optional[Dict[str, Any]]:
        """
        Get batch aggregation from batch layer (MongoDB)

        Args:
            topic: Topic name
            hours_back: Hours to look back

        Returns:
            Batch aggregation data
        """
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours_back)

            result = self.db.batch_aggregations.find_one(
                {"topic": topic, "period_start": {"$gte": cutoff}},
                sort=[("period_start", -1)],
            )

            if result:
                result["_id"] = str(result["_id"])
                return result
            return None

        except Exception as e:
            self.logger.error(f"MongoDB error: {e}")
            return None

    def get_sentiment_stats(
        self, topic: str, hours_back: int = 24
    ) -> Optional[Dict[str, Any]]:
        """
        Get sentiment statistics directly from sentiment_results collection

        Args:
            topic: Topic name
            hours_back: Hours to look back

        Returns:
            Aggregated sentiment statistics
        """
        try:
            from datetime import datetime, timedelta

            cutoff = datetime.utcnow() - timedelta(hours=hours_back)

            # Aggregate sentiment results
            pipeline = [
                {"$match": {"topic": topic}},
                {
                    "$group": {
                        "_id": "$topic",
                        "average_score": {"$avg": "$sentiment_score"},
                        "post_count": {"$sum": 1},
                        "positive_count": {
                            "$sum": {
                                "$cond": [{"$gt": ["$sentiment_score", 0.2]}, 1, 0]
                            }
                        },
                        "negative_count": {
                            "$sum": {
                                "$cond": [{"$lt": ["$sentiment_score", -0.2]}, 1, 0]
                            }
                        },
                        "neutral_count": {
                            "$sum": {
                                "$cond": [
                                    {
                                        "$and": [
                                            {"$gte": ["$sentiment_score", -0.2]},
                                            {"$lte": ["$sentiment_score", 0.2]},
                                        ]
                                    },
                                    1,
                                    0,
                                ]
                            }
                        },
                    }
                },
            ]

            results = list(self.db.sentiment_results.aggregate(pipeline))

            if results:
                result = results[0]
                total = result.get("post_count", 1)
                return {
                    "average_score": result.get("average_score", 0),
                    "post_count": total,
                    "distribution": {
                        "positive": result.get("positive_count", 0) / total,
                        "negative": result.get("negative_count", 0) / total,
                        "neutral": result.get("neutral_count", 0) / total,
                    },
                }
            return None

        except Exception as e:
            self.logger.error(f"MongoDB aggregation error: {e}")
            return None

    def get_recent_results(self, topic: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent sentiment results
        """
        try:
            results = list(
                self.db.sentiment_results.find(
                    {"topic": topic}, sort=[("timestamp", -1)]
                ).limit(limit)
            )

            for r in results:
                r["_id"] = str(r["_id"])

            return results
        except Exception as e:
            self.logger.error(f"MongoDB error: {e}")
            return []

    def get_combined_view(self, topic: str) -> Dict[str, Any]:
        """
        Get combined view from speed + batch layers

        Args:
            topic: Topic name

        Returns:
            Combined sentiment view
        """
        # Get real-time data from Redis
        realtime = self.get_realtime_sentiment(topic)

        # Get batch data
        batch = self.get_batch_aggregation(topic)

        # Get direct stats from sentiment_results
        stats = self.get_sentiment_stats(topic)

        # Get recent posts
        recent_posts = self.get_recent_results(topic, limit=10)

        return {
            "topic": topic,
            "realtime": realtime,
            "batch": batch,
            "stats": stats,
            "average_score": stats.get("average_score", 0) if stats else 0,
            "post_count": stats.get("post_count", 0) if stats else 0,
            "distribution": stats.get("distribution") if stats else None,
            "recent_posts": recent_posts,
            "combined_at": datetime.utcnow().isoformat(),
        }

    def get_sentiment_history(
        self, topic: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get sentiment history from Redis

        Args:
            topic: Topic name
            limit: Number of records

        Returns:
            List of historical sentiment data
        """
        try:
            import json

            history = self.redis_client.lrange(
                f"sentiment:history:{topic}", 0, limit - 1
            )

            return [json.loads(item) for item in history]

        except Exception as e:
            self.logger.error(f"Redis error: {e}")
            return []

    def get_hourly_trends(self, topic: str, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get hourly trends from batch layer

        Args:
            topic: Topic name
            hours: Number of hours

        Returns:
            List of hourly trend data
        """
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)

            results = list(
                self.db.hourly_trends.find(
                    {"topic": topic, "batch_time": {"$gte": cutoff}},
                    sort=[("hour", -1)],
                ).limit(hours)
            )

            for r in results:
                r["_id"] = str(r["_id"])

            return results

        except Exception as e:
            self.logger.error(f"MongoDB error: {e}")
            return []

    def get_top_keywords(self, topic: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get top keywords for a topic

        Args:
            topic: Topic name
            limit: Number of keywords

        Returns:
            List of keyword data
        """
        try:
            results = list(
                self.db.batch_keywords.find(
                    {"topic": topic}, sort=[("batch_time", -1), ("count", -1)]
                ).limit(limit)
            )

            for r in results:
                r["_id"] = str(r["_id"])

            return results

        except Exception as e:
            self.logger.error(f"MongoDB error: {e}")
            return []

    def get_subreddit_stats(self, topic: str) -> List[Dict[str, Any]]:
        """
        Get subreddit breakdown for a topic

        Args:
            topic: Topic name

        Returns:
            List of subreddit stats
        """
        try:
            results = list(
                self.db.subreddit_stats.find(
                    {"topic": topic}, sort=[("batch_time", -1), ("post_count", -1)]
                ).limit(50)
            )

            for r in results:
                r["_id"] = str(r["_id"])

            return results

        except Exception as e:
            self.logger.error(f"MongoDB error: {e}")
            return []

    def get_recent_posts(
        self, topic: str, sentiment_filter: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get recent posts for a topic

        Args:
            topic: Topic name
            sentiment_filter: Optional sentiment label filter
            limit: Number of posts

        Returns:
            List of recent posts
        """
        try:
            query = {"topic": topic}

            if sentiment_filter:
                query["sentiment_label"] = sentiment_filter

            results = list(
                self.db.sentiment_results.find(
                    query, sort=[("processed_at", -1)]
                ).limit(limit)
            )

            for r in results:
                r["_id"] = str(r["_id"])

            return results

        except Exception as e:
            self.logger.error(f"MongoDB error: {e}")
            return []

    def get_alerts(
        self,
        topic: Optional[str] = None,
        unacknowledged_only: bool = True,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get alerts

        Args:
            topic: Optional topic filter
            unacknowledged_only: Only return unacknowledged alerts
            limit: Number of alerts

        Returns:
            List of alerts
        """
        try:
            query = {}

            if topic:
                query["topic"] = topic

            if unacknowledged_only:
                query["acknowledged"] = False

            results = list(
                self.db.alerts.find(query, sort=[("timestamp", -1)]).limit(limit)
            )

            for r in results:
                r["_id"] = str(r["_id"])

            return results

        except Exception as e:
            self.logger.error(f"MongoDB error: {e}")
            return []

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert

        Args:
            alert_id: Alert ID

        Returns:
            Success status
        """
        try:
            from bson import ObjectId

            result = self.db.alerts.update_one(
                {"_id": ObjectId(alert_id)},
                {"$set": {"acknowledged": True, "acknowledged_at": datetime.utcnow()}},
            )

            return result.modified_count > 0

        except Exception as e:
            self.logger.error(f"MongoDB error: {e}")
            return False

    def get_topics(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all topics

        Args:
            active_only: Only return active topics

        Returns:
            List of topics
        """
        try:
            query = {}
            if active_only:
                query["active"] = True

            results = list(self.db.topics.find(query))

            for r in results:
                r["_id"] = str(r["_id"])

            return results

        except Exception as e:
            self.logger.error(f"MongoDB error: {e}")
            return []

    def create_topic(self, topic: TopicRequest) -> Dict[str, Any]:
        """
        Create a new topic

        Args:
            topic: Topic data

        Returns:
            Created topic
        """
        try:
            doc = {
                "name": topic.name,
                "subreddits": topic.subreddits,
                "keywords": topic.keywords,
                "active": topic.active,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }

            result = self.db.topics.insert_one(doc)
            doc["_id"] = str(result.inserted_id)

            return doc

        except Exception as e:
            self.logger.error(f"MongoDB error: {e}")
            raise

    def update_topic(self, topic_name: str, updates: Dict[str, Any]) -> bool:
        """
        Update a topic

        Args:
            topic_name: Topic name
            updates: Fields to update

        Returns:
            Success status
        """
        try:
            updates["updated_at"] = datetime.utcnow()

            result = self.db.topics.update_one({"name": topic_name}, {"$set": updates})

            return result.modified_count > 0

        except Exception as e:
            self.logger.error(f"MongoDB error: {e}")
            return False

    def compare_topics(
        self, topics: List[str], hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple topics

        Args:
            topics: List of topic names
            hours: Hours to look back

        Returns:
            Comparison data
        """
        comparison = []

        for topic in topics:
            combined = self.get_combined_view(topic)
            comparison.append(combined)

        return comparison

    def close(self):
        """Close connections"""
        try:
            self.redis_client.close()
            self.mongo_client.close()
            self.logger.info("Connections closed")
        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")


# Global API instance
api = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global api

    setup_logging(log_level="INFO", json_format=True, service_name="serving-api")

    logger = get_logger("main")
    logger.info("Starting Serving API...")

    api = ServingAPI()

    yield

    # Cleanup
    if api:
        api.close()
    logger.info("Serving API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Real-time sentiment analysis serving layer API",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Topic endpoints
@app.get("/api/v1/topics")
async def get_topics(active_only: bool = True):
    """Get all topics"""
    REQUEST_COUNT.labels(method="GET", endpoint="/topics", status="success").inc()
    return {"topics": api.get_topics(active_only)}


@app.post("/api/v1/topics")
async def create_topic(topic: TopicRequest):
    """Create a new topic"""
    try:
        result = api.create_topic(topic)
        REQUEST_COUNT.labels(method="POST", endpoint="/topics", status="success").inc()
        return {"topic": result}
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/topics", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/v1/topics/{topic_name}")
async def update_topic(topic_name: str, updates: Dict[str, Any]):
    """Update a topic"""
    success = api.update_topic(topic_name, updates)
    if success:
        REQUEST_COUNT.labels(method="PUT", endpoint="/topics", status="success").inc()
        return {"message": "Topic updated"}
    REQUEST_COUNT.labels(method="PUT", endpoint="/topics", status="error").inc()
    raise HTTPException(status_code=404, detail="Topic not found")


# Sentiment endpoints
@app.get("/api/v1/sentiment/{topic}")
async def get_sentiment(topic: str):
    """Get combined sentiment view for a topic"""
    REQUEST_COUNT.labels(method="GET", endpoint="/sentiment", status="success").inc()
    return api.get_combined_view(topic)


@app.get("/api/v1/sentiment/{topic}/realtime")
async def get_realtime_sentiment(topic: str):
    """Get real-time sentiment from speed layer"""
    result = api.get_realtime_sentiment(topic)
    if result:
        return result
    raise HTTPException(status_code=404, detail="No real-time data available")


@app.get("/api/v1/sentiment/{topic}/batch")
async def get_batch_sentiment(topic: str, hours: int = Query(default=24, ge=1, le=168)):
    """Get batch aggregation from batch layer"""
    result = api.get_batch_aggregation(topic, hours)
    if result:
        return result
    raise HTTPException(status_code=404, detail="No batch data available")


@app.get("/api/v1/sentiment/{topic}/history")
async def get_sentiment_history(
    topic: str, limit: int = Query(default=100, ge=1, le=1000)
):
    """Get sentiment history"""
    return {"history": api.get_sentiment_history(topic, limit)}


@app.get("/api/v1/sentiment/{topic}/trends")
async def get_sentiment_trends(
    topic: str, hours: int = Query(default=24, ge=1, le=168)
):
    """Get hourly sentiment trends"""
    return {"trends": api.get_hourly_trends(topic, hours)}


@app.get("/api/v1/sentiment/{topic}/keywords")
async def get_keywords(topic: str, limit: int = Query(default=20, ge=1, le=100)):
    """Get top keywords for a topic"""
    return {"keywords": api.get_top_keywords(topic, limit)}


@app.get("/api/v1/sentiment/{topic}/subreddits")
async def get_subreddit_stats(topic: str):
    """Get subreddit breakdown for a topic"""
    return {"subreddits": api.get_subreddit_stats(topic)}


@app.get("/api/v1/sentiment/{topic}/posts")
async def get_recent_posts(
    topic: str,
    sentiment: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    """Get recent posts for a topic"""
    return {"posts": api.get_recent_posts(topic, sentiment, limit)}


# Comparison endpoint
@app.get("/api/v1/compare")
async def compare_topics(
    topics: str = Query(..., description="Comma-separated topic names"),
    hours: int = Query(default=24, ge=1, le=168),
):
    """Compare multiple topics"""
    topic_list = [t.strip() for t in topics.split(",")]
    return {"comparison": api.compare_topics(topic_list, hours)}


# Alert endpoints
@app.get("/api/v1/alerts")
async def get_alerts(
    topic: Optional[str] = None,
    unacknowledged_only: bool = True,
    limit: int = Query(default=100, ge=1, le=500),
):
    """Get alerts"""
    return {"alerts": api.get_alerts(topic, unacknowledged_only, limit)}


@app.post("/api/v1/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert"""
    success = api.acknowledge_alert(alert_id)
    if success:
        return {"message": "Alert acknowledged"}
    raise HTTPException(status_code=404, detail="Alert not found")


# Stats endpoint
@app.get("/api/v1/stats")
async def get_overall_stats():
    """Get overall system statistics"""
    try:
        topics = api.get_topics(active_only=True)

        stats = {
            "total_topics": len(topics),
            "active_topics": sum(1 for t in topics if t.get("active", False)),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Add per-topic stats
        topic_stats = []
        for topic in topics:
            topic_name = topic.get("name")
            combined = api.get_combined_view(topic_name)
            topic_stats.append(
                {
                    "topic": topic_name,
                    "realtime_available": combined.get("realtime") is not None,
                    "batch_available": combined.get("batch") is not None,
                }
            )

        stats["topics"] = topic_stats

        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
