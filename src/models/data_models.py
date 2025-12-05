"""
Data Models Module
Pydantic models for data validation and serialization
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class SentimentLabel(str, Enum):
    """Sentiment classification labels"""

    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class RedditPost(BaseModel):
    """Model for Reddit post data"""

    post_id: str = Field(..., description="Unique Reddit post ID")
    topic: str = Field(..., description="Topic/category for the post")
    subreddit: str = Field(..., description="Subreddit name")
    title: str = Field(..., description="Post title")
    content: str = Field(default="", description="Post body/selftext")
    author: str = Field(..., description="Post author username")
    score: int = Field(default=0, description="Post score (upvotes - downvotes)")
    num_comments: int = Field(default=0, description="Number of comments")
    url: str = Field(default="", description="Post URL")
    created_utc: float = Field(..., description="Unix timestamp of creation")
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    is_comment: bool = Field(default=False, description="Whether this is a comment")
    parent_id: Optional[str] = Field(
        default=None, description="Parent post ID for comments"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class RedditComment(BaseModel):
    """Model for Reddit comment data"""

    comment_id: str = Field(..., description="Unique Reddit comment ID")
    post_id: str = Field(..., description="Parent post ID")
    topic: str = Field(..., description="Topic/category")
    subreddit: str = Field(..., description="Subreddit name")
    content: str = Field(..., description="Comment body")
    author: str = Field(..., description="Comment author username")
    score: int = Field(default=0, description="Comment score")
    created_utc: float = Field(..., description="Unix timestamp of creation")
    scraped_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class Entity(BaseModel):
    """Model for extracted entities"""

    text: str = Field(..., description="Entity text")
    entity_type: str = Field(..., description="Entity type (PERSON, ORG, etc.)")
    confidence: float = Field(default=1.0, description="Extraction confidence")


class SentimentResult(BaseModel):
    """Model for sentiment analysis results"""

    post_id: str = Field(..., description="Source post/comment ID")
    topic: str = Field(..., description="Topic/category")
    subreddit: str = Field(..., description="Subreddit name")
    original_text: str = Field(..., description="Original text analyzed")
    cleaned_text: str = Field(default="", description="Preprocessed text")
    sentiment_score: float = Field(..., description="Sentiment score (-1 to 1)")
    sentiment_label: SentimentLabel = Field(..., description="Sentiment classification")
    confidence: float = Field(..., description="Model confidence score")
    entities: List[Entity] = Field(
        default_factory=list, description="Extracted entities"
    )
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    timestamp: datetime = Field(..., description="Original post timestamp")
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: float = Field(
        default=0, description="Processing time in milliseconds"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class BatchAggregation(BaseModel):
    """Model for batch aggregation results"""

    topic: str = Field(..., description="Topic/category")
    period_start: datetime = Field(..., description="Aggregation period start")
    period_end: datetime = Field(..., description="Aggregation period end")
    avg_sentiment: float = Field(..., description="Average sentiment score")
    std_sentiment: float = Field(default=0, description="Standard deviation")
    total_posts: int = Field(..., description="Total number of posts")
    positive_count: int = Field(default=0, description="Positive sentiment count")
    negative_count: int = Field(default=0, description="Negative sentiment count")
    neutral_count: int = Field(default=0, description="Neutral sentiment count")
    very_positive_count: int = Field(default=0)
    very_negative_count: int = Field(default=0)
    top_keywords: List[Dict[str, Any]] = Field(default_factory=list)
    top_entities: List[Dict[str, Any]] = Field(default_factory=list)
    sentiment_distribution: Dict[str, float] = Field(default_factory=dict)
    subreddit_breakdown: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class Alert(BaseModel):
    """Model for sentiment alerts"""

    alert_id: str = Field(..., description="Unique alert ID")
    topic: str = Field(..., description="Topic that triggered alert")
    alert_type: str = Field(..., description="Type of alert")
    message: str = Field(..., description="Alert message")
    severity: str = Field(default="warning", description="Alert severity")
    sentiment_value: float = Field(
        ..., description="Sentiment value that triggered alert"
    )
    threshold: float = Field(..., description="Threshold that was crossed")
    post_count: int = Field(default=1, description="Number of posts in window")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = Field(
        default=False, description="Whether alert was acknowledged"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class Topic(BaseModel):
    """Model for monitoring topics"""

    name: str = Field(..., description="Topic name")
    subreddits: List[str] = Field(..., description="Subreddits to monitor")
    keywords: List[str] = Field(default_factory=list, description="Keywords to filter")
    active: bool = Field(
        default=True, description="Whether topic is actively monitored"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class TopicRequest(BaseModel):
    """Model for topic creation/update requests"""

    name: str = Field(..., description="Topic name")
    subreddits: List[str] = Field(..., description="Subreddits to monitor")
    keywords: Optional[List[str]] = Field(
        default=None, description="Keywords to filter"
    )
    active: bool = Field(
        default=True, description="Whether topic is actively monitored"
    )


class SentimentTrend(BaseModel):
    """Model for sentiment trend data"""

    topic: str
    time_series: List[Dict[str, Any]]
    current_sentiment: float
    trend_direction: str  # "up", "down", "stable"
    change_percent: float


class WordCloudData(BaseModel):
    """Model for word cloud data"""

    topic: str
    words: List[Dict[str, Any]]  # {"word": str, "count": int, "sentiment": float}
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class TopicComparison(BaseModel):
    """Model for comparing multiple topics"""

    topics: List[str]
    period_start: datetime
    period_end: datetime
    comparison_data: List[Dict[str, Any]]
