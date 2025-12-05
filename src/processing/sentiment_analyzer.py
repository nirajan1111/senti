"""
Sentiment Analyzer Module
BERT-based sentiment analysis with text preprocessing and entity extraction
"""

import re
import time
from typing import List, Tuple, Optional, Dict, Any
from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

import sys

sys.path.insert(0, "/app")

from src.models.data_models import SentimentLabel, Entity
from src.utils.logging_config import get_logger


# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)


class SentimentAnalyzer:
    """
    Production-grade sentiment analyzer using BERT
    """

    def __init__(
        self,
        model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment",
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        self.logger = get_logger("SentimentAnalyzer")
        self.model_name = model_name
        self.max_length = max_length

        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        self._load_model()

        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        # Add custom stop words for Reddit
        self.stop_words.update(
            [
                "http",
                "https",
                "www",
                "com",
                "reddit",
                "post",
                "comment",
                "edit",
                "deleted",
                "removed",
                "amp",
            ]
        )

        self.logger.info("SentimentAnalyzer initialized successfully")

    def _load_model(self):
        """Load BERT model and tokenizer"""
        try:
            self.logger.info(f"Loading model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.to(self.device)
            self.model.eval()

            self.logger.info("Model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis

        Args:
            text: Raw text input

        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)

        # Remove Reddit-specific patterns
        text = re.sub(r"/r/\w+", "", text)  # Subreddit mentions
        text = re.sub(r"/u/\w+", "", text)  # User mentions
        text = re.sub(r"u/\w+", "", text)  # User mentions (alternate)

        # Remove HTML entities
        text = re.sub(r"&\w+;", "", text)

        # Remove special characters but keep punctuation for sentiment
        text = re.sub(r"[^\w\s!?.,\'-]", "", text)

        # Remove extra whitespace
        text = " ".join(text.split())

        return text.strip()

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract important keywords from text

        Args:
            text: Preprocessed text
            top_n: Number of top keywords to return

        Returns:
            List of keywords
        """
        if not text:
            return []

        try:
            # Tokenize
            tokens = word_tokenize(text.lower())

            # Remove stopwords and short tokens
            tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token.isalpha() and token not in self.stop_words and len(token) > 2
            ]

            # Count frequencies
            word_freq = Counter(tokens)

            # Return top keywords
            return [word for word, _ in word_freq.most_common(top_n)]

        except Exception as e:
            self.logger.warning(f"Keyword extraction failed: {e}")
            return []

    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract named entities from text using pattern matching
        (Lightweight alternative to full NER)

        Args:
            text: Text to analyze

        Returns:
            List of Entity objects
        """
        entities = []

        if not text:
            return entities

        # Pattern for potential company/organization names (capitalized words)
        org_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"

        # Pattern for money amounts
        money_pattern = (
            r"\$[\d,]+(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD)"
        )

        # Pattern for percentages
        percent_pattern = r"\d+(?:\.\d+)?%"

        try:
            # Find organizations/names
            org_matches = re.findall(org_pattern, text)
            for match in org_matches[:5]:  # Limit to 5
                if len(match) > 2 and match.lower() not in self.stop_words:
                    entities.append(
                        Entity(text=match, entity_type="ORG_OR_PERSON", confidence=0.7)
                    )

            # Find money mentions
            money_matches = re.findall(money_pattern, text, re.IGNORECASE)
            for match in money_matches[:3]:
                entities.append(Entity(text=match, entity_type="MONEY", confidence=0.9))

            # Find percentages
            percent_matches = re.findall(percent_pattern, text)
            for match in percent_matches[:3]:
                entities.append(
                    Entity(text=match, entity_type="PERCENT", confidence=0.9)
                )

        except Exception as e:
            self.logger.warning(f"Entity extraction failed: {e}")

        return entities

    def _score_to_label(self, score: float) -> SentimentLabel:
        """Convert sentiment score to label"""
        if score <= -0.6:
            return SentimentLabel.VERY_NEGATIVE
        elif score <= -0.2:
            return SentimentLabel.NEGATIVE
        elif score <= 0.2:
            return SentimentLabel.NEUTRAL
        elif score <= 0.6:
            return SentimentLabel.POSITIVE
        else:
            return SentimentLabel.VERY_POSITIVE

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Perform sentiment analysis on text

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment results
        """
        start_time = time.time()

        # Preprocess
        cleaned_text = self.preprocess_text(text)

        if not cleaned_text or len(cleaned_text) < 3:
            return {
                "sentiment_score": 0.0,
                "sentiment_label": SentimentLabel.NEUTRAL.value,
                "confidence": 0.0,
                "cleaned_text": cleaned_text,
                "keywords": [],
                "entities": [],
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

        try:
            # Tokenize for BERT
            inputs = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self.device)

            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)

            # The model outputs 5 classes (1-5 stars)
            # Convert to sentiment score (-1 to 1)
            probs = probabilities[0].cpu().numpy()

            # Weighted average: 1-star = -1, 5-star = 1
            weights = [-1.0, -0.5, 0.0, 0.5, 1.0]
            sentiment_score = sum(p * w for p, w in zip(probs, weights))

            # Confidence is the probability of the predicted class
            predicted_class = probs.argmax()
            confidence = float(probs[predicted_class])

            # Get label
            sentiment_label = self._score_to_label(sentiment_score)

            # Extract keywords and entities
            keywords = self.extract_keywords(cleaned_text)
            entities = self.extract_entities(text)  # Use original text for entities

            processing_time = (time.time() - start_time) * 1000

            return {
                "sentiment_score": round(float(sentiment_score), 4),
                "sentiment_label": sentiment_label.value,
                "confidence": round(confidence, 4),
                "cleaned_text": cleaned_text,
                "keywords": keywords,
                "entities": [e.model_dump() for e in entities],
                "processing_time_ms": round(processing_time, 2),
            }

        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}", exc_info=True)
            return {
                "sentiment_score": 0.0,
                "sentiment_label": SentimentLabel.NEUTRAL.value,
                "confidence": 0.0,
                "cleaned_text": cleaned_text,
                "keywords": [],
                "entities": [],
                "processing_time_ms": (time.time() - start_time) * 1000,
                "error": str(e),
            }

    def analyze_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts in batches for efficiency

        Args:
            texts: List of texts to analyze
            batch_size: Number of texts per batch

        Returns:
            List of sentiment results
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_results = [self.analyze(text) for text in batch]
            results.extend(batch_results)

        return results


# Singleton instance for efficiency
_analyzer_instance: Optional[SentimentAnalyzer] = None


def get_sentiment_analyzer(
    model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment",
) -> SentimentAnalyzer:
    """Get or create sentiment analyzer instance"""
    global _analyzer_instance

    if _analyzer_instance is None:
        _analyzer_instance = SentimentAnalyzer(model_name=model_name)

    return _analyzer_instance


if __name__ == "__main__":
    # Test the analyzer
    from src.utils.logging_config import setup_logging

    setup_logging(log_level="INFO", json_format=False)

    analyzer = SentimentAnalyzer()

    test_texts = [
        "This is absolutely amazing! I love it!",
        "This is terrible, worst experience ever.",
        "It's okay, nothing special.",
        "The stock market crashed today, investors are worried.",
        "Apple announces revolutionary new product, shares soar!",
    ]

    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\nText: {text}")
        print(f"Score: {result['sentiment_score']}, Label: {result['sentiment_label']}")
        print(f"Confidence: {result['confidence']}, Keywords: {result['keywords']}")
