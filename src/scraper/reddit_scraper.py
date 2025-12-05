"""
Reddit Scraper Module
Scrapes posts and comments from Reddit using PRAW and publishes to Kafka
"""

import praw
import json
import time
import signal
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Generator
from kafka import KafkaProducer
from kafka.errors import KafkaError
from tenacity import retry, stop_after_attempt, wait_exponential

# Add parent directory to path for imports
sys.path.insert(0, "/app")

from config.settings import get_config
from src.utils.logging_config import setup_logging, get_logger
from src.models.data_models import RedditPost, RedditComment


class RedditScraper:
    """
    Production-grade Reddit scraper with Kafka integration
    """

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("RedditScraper")
        self.running = True

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Initialize Reddit client
        self.reddit = self._init_reddit_client()

        # Initialize Kafka producer
        self.producer = self._init_kafka_producer()

        # Track scraped posts to avoid duplicates
        self.scraped_ids: set = set()
        self.max_tracked_ids = 100000  # Prevent memory bloat

        self.logger.info("RedditScraper initialized successfully")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _init_reddit_client(self) -> praw.Reddit:
        """Initialize Reddit API client with retry logic"""
        try:
            reddit = praw.Reddit(
                client_id=self.config.reddit.client_id,
                client_secret=self.config.reddit.client_secret,
                user_agent=self.config.reddit.user_agent,
                check_for_async=False,
            )
            # Test connection with a simple read-only request
            list(reddit.subreddit("all").hot(limit=1))
            self.logger.info("Reddit client initialized (read-only mode)")
            return reddit
        except Exception as e:
            self.logger.error(f"Failed to initialize Reddit client: {e}")
            raise

    @retry(
        stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=30)
    )
    def _init_kafka_producer(self) -> KafkaProducer:
        """Initialize Kafka producer with retry logic"""
        try:
            producer = KafkaProducer(
                bootstrap_servers=self.config.kafka.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
                max_in_flight_requests_per_connection=1,
                compression_type="gzip",
                linger_ms=100,
                batch_size=16384,
            )
            self.logger.info(
                f"Kafka producer connected to {self.config.kafka.bootstrap_servers}"
            )
            return producer
        except KafkaError as e:
            self.logger.error(f"Failed to initialize Kafka producer: {e}")
            raise

    def _get_active_topics(self) -> List[Dict[str, Any]]:
        """
        Get active topics from MongoDB
        Falls back to default topics if MongoDB is unavailable
        """
        from pymongo import MongoClient

        try:
            client = MongoClient(self.config.mongodb.uri, serverSelectionTimeoutMS=5000)
            db = client[self.config.mongodb.database]
            topics = list(db.topics.find({"active": True}))
            client.close()

            if topics:
                self.logger.info(f"Loaded {len(topics)} active topics from MongoDB")
                return topics
        except Exception as e:
            self.logger.warning(f"Failed to load topics from MongoDB: {e}")

        # Default topics if MongoDB is unavailable
        default_topics = [
            {
                "name": "technology",
                "subreddits": ["technology", "programming", "artificial"],
                "keywords": [],
            },
            {
                "name": "finance",
                "subreddits": ["finance", "stocks", "investing"],
                "keywords": [],
            },
        ]
        self.logger.info("Using default topics")
        return default_topics

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        # Remove excessive whitespace
        text = " ".join(text.split())
        # Limit length
        return text[:10000] if len(text) > 10000 else text

    def scrape_subreddit(
        self, subreddit_name: str, topic_name: str, limit: int = 25
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Scrape posts from a subreddit

        Args:
            subreddit_name: Name of the subreddit
            topic_name: Topic category for the posts
            limit: Maximum number of posts to fetch

        Yields:
            Dictionary representations of posts
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)

            # Scrape hot, new, and rising posts
            for post in subreddit.hot(limit=limit):
                if post.id in self.scraped_ids:
                    continue

                self.scraped_ids.add(post.id)

                reddit_post = RedditPost(
                    post_id=post.id,
                    topic=topic_name,
                    subreddit=subreddit_name,
                    title=self._clean_text(post.title),
                    content=self._clean_text(post.selftext),
                    author=str(post.author) if post.author else "[deleted]",
                    score=post.score,
                    num_comments=post.num_comments,
                    url=post.url,
                    created_utc=post.created_utc,
                    is_comment=False,
                )

                yield reddit_post.model_dump()

                # Scrape comments if enabled
                if self.config.scraper.include_comments:
                    yield from self._scrape_comments(post, topic_name, subreddit_name)

        except Exception as e:
            self.logger.error(
                f"Error scraping subreddit {subreddit_name}: {e}", exc_info=True
            )

    def _scrape_comments(
        self, post, topic_name: str, subreddit_name: str
    ) -> Generator[Dict[str, Any], None, None]:
        """Scrape comments from a post"""
        try:
            post.comments.replace_more(limit=0)  # Skip "more comments" links

            for i, comment in enumerate(post.comments.list()):
                if i >= self.config.scraper.max_comments_per_post:
                    break

                if comment.id in self.scraped_ids:
                    continue

                if not hasattr(comment, "body") or not comment.body:
                    continue

                self.scraped_ids.add(comment.id)

                reddit_comment = RedditComment(
                    comment_id=comment.id,
                    post_id=post.id,
                    topic=topic_name,
                    subreddit=subreddit_name,
                    content=self._clean_text(comment.body),
                    author=str(comment.author) if comment.author else "[deleted]",
                    score=comment.score,
                    created_utc=comment.created_utc,
                )

                # Convert to post-like format for unified processing
                yield {
                    "post_id": comment.id,
                    "topic": topic_name,
                    "subreddit": subreddit_name,
                    "title": "",
                    "content": reddit_comment.content,
                    "author": reddit_comment.author,
                    "score": reddit_comment.score,
                    "num_comments": 0,
                    "url": "",
                    "created_utc": reddit_comment.created_utc,
                    "scraped_at": datetime.utcnow().isoformat(),
                    "is_comment": True,
                    "parent_id": post.id,
                }

        except Exception as e:
            self.logger.warning(f"Error scraping comments for post {post.id}: {e}")

    def publish_to_kafka(self, data: Dict[str, Any]) -> bool:
        """
        Publish data to Kafka topic

        Args:
            data: Dictionary data to publish

        Returns:
            True if successful, False otherwise
        """
        try:
            topic = self.config.kafka.raw_posts_topic
            key = f"{data['topic']}:{data['post_id']}"

            future = self.producer.send(topic, key=key, value=data)
            # Wait for send to complete
            record_metadata = future.get(timeout=10)

            self.logger.debug(
                f"Published to {record_metadata.topic} partition {record_metadata.partition} offset {record_metadata.offset}"
            )
            return True

        except KafkaError as e:
            self.logger.error(f"Failed to publish to Kafka: {e}")
            return False

    def _cleanup_tracked_ids(self):
        """Clean up tracked IDs to prevent memory bloat"""
        if len(self.scraped_ids) > self.max_tracked_ids:
            # Keep the most recent half
            to_remove = len(self.scraped_ids) - (self.max_tracked_ids // 2)
            self.scraped_ids = set(list(self.scraped_ids)[to_remove:])
            self.logger.info(
                f"Cleaned up tracked IDs, now tracking {len(self.scraped_ids)}"
            )

    def run(self):
        """Main scraping loop"""
        self.logger.info("Starting Reddit scraping loop...")

        while self.running:
            try:
                topics = self._get_active_topics()
                posts_scraped = 0

                for topic in topics:
                    if not self.running:
                        break

                    topic_name = topic["name"]
                    subreddits = topic.get("subreddits", [])

                    for subreddit in subreddits:
                        if not self.running:
                            break

                        self.logger.info(
                            f"Scraping r/{subreddit} for topic '{topic_name}'"
                        )

                        for post_data in self.scrape_subreddit(
                            subreddit,
                            topic_name,
                            limit=self.config.scraper.posts_per_subreddit,
                        ):
                            if self.publish_to_kafka(post_data):
                                posts_scraped += 1

                        # Small delay between subreddits to respect rate limits
                        time.sleep(2)

                self.logger.info(
                    f"Scraping cycle complete. Published {posts_scraped} items"
                )

                # Cleanup tracked IDs periodically
                self._cleanup_tracked_ids()

                # Flush producer
                self.producer.flush()

                # Wait for next cycle
                if self.running:
                    self.logger.info(
                        f"Waiting {self.config.scraper.interval_seconds}s until next scrape cycle..."
                    )
                    time.sleep(self.config.scraper.interval_seconds)

            except Exception as e:
                self.logger.error(f"Error in scraping loop: {e}", exc_info=True)
                if self.running:
                    time.sleep(30)  # Wait before retry

        self.shutdown()

    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down RedditScraper...")
        try:
            self.producer.flush(timeout=10)
            self.producer.close(timeout=10)
            self.logger.info("Kafka producer closed")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

        self.logger.info("RedditScraper shutdown complete")


def main():
    """Main entry point"""
    # Setup logging
    setup_logging(log_level="INFO", json_format=True, service_name="reddit-scraper")

    logger = get_logger("main")
    logger.info("Starting Reddit Scraper Service...")

    try:
        scraper = RedditScraper()
        scraper.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
