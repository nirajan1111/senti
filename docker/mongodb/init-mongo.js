// MongoDB initialization script
db = db.getSiblingDB('sentiment_db');

// Create collections with validation
db.createCollection('raw_posts', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['post_id', 'topic', 'timestamp'],
            properties: {
                post_id: { bsonType: 'string' },
                topic: { bsonType: 'string' },
                subreddit: { bsonType: 'string' },
                title: { bsonType: 'string' },
                content: { bsonType: 'string' },
                author: { bsonType: 'string' },
                score: { bsonType: 'int' },
                num_comments: { bsonType: 'int' },
                created_utc: { bsonType: 'double' },
                timestamp: { bsonType: 'date' }
            }
        }
    }
});

db.createCollection('sentiment_results', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['post_id', 'topic', 'sentiment_score', 'timestamp'],
            properties: {
                post_id: { bsonType: 'string' },
                topic: { bsonType: 'string' },
                sentiment_score: { bsonType: 'double' },
                sentiment_label: { bsonType: 'string' },
                confidence: { bsonType: 'double' },
                entities: { bsonType: 'array' },
                keywords: { bsonType: 'array' },
                timestamp: { bsonType: 'date' },
                processed_at: { bsonType: 'date' }
            }
        }
    }
});

db.createCollection('batch_aggregations', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['topic', 'period_start', 'period_end'],
            properties: {
                topic: { bsonType: 'string' },
                period_start: { bsonType: 'date' },
                period_end: { bsonType: 'date' },
                avg_sentiment: { bsonType: 'double' },
                total_posts: { bsonType: 'int' },
                positive_count: { bsonType: 'int' },
                negative_count: { bsonType: 'int' },
                neutral_count: { bsonType: 'int' },
                top_keywords: { bsonType: 'array' },
                top_entities: { bsonType: 'array' },
                sentiment_distribution: { bsonType: 'object' }
            }
        }
    }
});

db.createCollection('alerts', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['topic', 'alert_type', 'timestamp'],
            properties: {
                topic: { bsonType: 'string' },
                alert_type: { bsonType: 'string' },
                message: { bsonType: 'string' },
                severity: { bsonType: 'string' },
                sentiment_value: { bsonType: 'double' },
                threshold: { bsonType: 'double' },
                timestamp: { bsonType: 'date' },
                acknowledged: { bsonType: 'bool' }
            }
        }
    }
});

db.createCollection('topics', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['name', 'subreddits', 'active'],
            properties: {
                name: { bsonType: 'string' },
                subreddits: { bsonType: 'array' },
                keywords: { bsonType: 'array' },
                active: { bsonType: 'bool' },
                created_at: { bsonType: 'date' },
                updated_at: { bsonType: 'date' }
            }
        }
    }
});

// Create indexes for performance
db.raw_posts.createIndex({ 'post_id': 1 }, { unique: true });
db.raw_posts.createIndex({ 'topic': 1, 'timestamp': -1 });
db.raw_posts.createIndex({ 'subreddit': 1, 'timestamp': -1 });
db.raw_posts.createIndex({ 'timestamp': -1 });

db.sentiment_results.createIndex({ 'post_id': 1 }, { unique: true });
db.sentiment_results.createIndex({ 'topic': 1, 'timestamp': -1 });
db.sentiment_results.createIndex({ 'sentiment_label': 1, 'topic': 1 });
db.sentiment_results.createIndex({ 'timestamp': -1 });

db.batch_aggregations.createIndex({ 'topic': 1, 'period_start': -1 });
db.batch_aggregations.createIndex({ 'period_start': -1 });

db.alerts.createIndex({ 'topic': 1, 'timestamp': -1 });
db.alerts.createIndex({ 'acknowledged': 1, 'timestamp': -1 });

db.topics.createIndex({ 'name': 1 }, { unique: true });
db.topics.createIndex({ 'active': 1 });

// Insert default topics
db.topics.insertMany([
    {
        name: 'technology',
        subreddits: ['technology', 'tech', 'gadgets', 'programming'],
        keywords: ['AI', 'machine learning', 'software', 'hardware'],
        active: true,
        created_at: new Date(),
        updated_at: new Date()
    },
    {
        name: 'finance',
        subreddits: ['finance', 'investing', 'stocks', 'cryptocurrency'],
        keywords: ['stock', 'crypto', 'bitcoin', 'investment'],
        active: true,
        created_at: new Date(),
        updated_at: new Date()
    },
    {
        name: 'politics',
        subreddits: ['politics', 'worldnews', 'news'],
        keywords: ['election', 'government', 'policy'],
        active: false,
        created_at: new Date(),
        updated_at: new Date()
    }
]);

print('MongoDB initialization completed successfully!');
