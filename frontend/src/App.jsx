import { useState, useEffect, useCallback } from 'react';
import { Header, StatsBar, LoadingSpinner, ErrorMessage } from './components/Layout';
import { SentimentCard, SentimentMeter } from './components/SentimentCard';
import { SentimentTrendChart, SentimentDistributionChart, MultiTopicChart } from './components/Charts';
import { TopicManager } from './components/TopicManager';
import { AlertList } from './components/AlertList';
import { useTopics, useAlerts, useTrends } from './hooks/useSentiment';
import { sentimentApi } from './api/sentiment';

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [topicData, setTopicData] = useState({});
  const [stats, setStats] = useState({ totalPosts: 0, activeTopics: 0, processingRate: 0 });

  const { topics, loading: topicsLoading, addTopic, removeTopic, refetch: refetchTopics } = useTopics();
  const { alerts, loading: alertsLoading, refetch: refetchAlerts } = useAlerts();
  const { trends, loading: trendsLoading, refetch: refetchTrends } = useTrends();

  // Check connection
  useEffect(() => {
    const checkConnection = async () => {
      try {
        await sentimentApi.getHealth();
        setIsConnected(true);
      } catch {
        setIsConnected(false);
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  // Fetch sentiment data for all topics
  const fetchAllTopicData = useCallback(async () => {
    const data = {};
    let totalPosts = 0;

    for (const topic of topics) {
      try {
        const topicName = topic.name || topic;
        const result = await sentimentApi.getSentiment(topicName);
        data[topicName] = result;
        totalPosts += result.post_count || 0;
      } catch (err) {
        console.error(`Failed to fetch data for ${topic.name || topic}:`, err);
      }
    }

    setTopicData(data);
    setStats((prev) => ({
      ...prev,
      totalPosts,
      activeTopics: topics.length,
    }));
    setLastUpdate(new Date().toISOString());
  }, [topics]);

  useEffect(() => {
    if (topics.length > 0) {
      fetchAllTopicData();
      const interval = setInterval(fetchAllTopicData, 30000);
      return () => clearInterval(interval);
    }
  }, [topics, fetchAllTopicData]);

  const handleRefresh = () => {
    refetchTopics();
    refetchAlerts();
    refetchTrends();
    fetchAllTopicData();
  };

  const handleTopicClick = (topic) => {
    setSelectedTopic(selectedTopic === topic ? null : topic);
  };

  // Prepare data for multi-topic chart
  const topicsWithTrends = topics
    .map((t) => {
      const name = t.name || t;
      const data = topicData[name];
      return {
        name,
        trend: data?.trend || [],
      };
    })
    .filter((t) => t.trend.length > 0);

  return (
    <div className="min-h-screen bg-gray-900">
      <Header 
        isConnected={isConnected} 
        onRefresh={handleRefresh} 
        lastUpdate={lastUpdate} 
      />

      <main className="container mx-auto px-4 py-6">
        <StatsBar stats={stats} />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Topics */}
          <div className="lg:col-span-2 space-y-6">
            {/* Topic Cards */}
            <div>
              <h2 className="text-xl font-semibold mb-4">Sentiment by Topic</h2>
              {topicsLoading ? (
                <LoadingSpinner />
              ) : topics.length === 0 ? (
                <div className="card text-center py-8">
                  <p className="text-gray-400 mb-4">
                    No topics configured. Add a topic to start monitoring.
                  </p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {topics.map((topic) => {
                    const topicName = topic.name || topic;
                    const data = topicData[topicName];
                    return (
                      <SentimentCard
                        key={topicName}
                        topic={topicName}
                        sentiment={data}
                        postCount={data?.post_count}
                        onClick={() => handleTopicClick(topicName)}
                      />
                    );
                  })}
                </div>
              )}
            </div>

            {/* Selected Topic Details */}
            {selectedTopic && topicData[selectedTopic] && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-semibold capitalize">
                    {selectedTopic} Details
                  </h2>
                  <button
                    onClick={() => setSelectedTopic(null)}
                    className="text-gray-400 hover:text-white"
                  >
                    Close
                  </button>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="card flex flex-col items-center justify-center">
                    <SentimentMeter 
                      score={topicData[selectedTopic].average_score || 0} 
                      size="lg"
                    />
                    <p className="mt-2 text-gray-400">Overall Sentiment</p>
                  </div>
                  <div className="md:col-span-2">
                    <SentimentDistributionChart 
                      distribution={topicData[selectedTopic].distribution}
                    />
                  </div>
                </div>

                {topicData[selectedTopic].trend && (
                  <SentimentTrendChart data={topicData[selectedTopic].trend} />
                )}

                {/* Recent Posts */}
                {topicData[selectedTopic].recent_posts && (
                  <div className="card">
                    <h3 className="text-lg font-semibold mb-4">Recent Posts</h3>
                    <div className="space-y-3 max-h-64 overflow-y-auto">
                      {topicData[selectedTopic].recent_posts.slice(0, 5).map((post, idx) => (
                        <div key={idx} className="p-3 bg-gray-700/30 rounded-lg">
                          <p className="text-sm">{post.title || post.text}</p>
                          <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
                            <span>r/{post.subreddit}</span>
                            <span className={post.score > 0 ? 'text-green-500' : post.score < 0 ? 'text-red-500' : ''}>
                              Score: {(post.score * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Multi-topic Comparison Chart */}
            {topicsWithTrends.length > 1 && (
              <MultiTopicChart topics={topicsWithTrends} />
            )}
          </div>

          {/* Right Column - Sidebar */}
          <div className="space-y-6">
            <TopicManager
              topics={topics}
              onAddTopic={addTopic}
              onRemoveTopic={removeTopic}
              loading={topicsLoading}
            />
            
            <AlertList alerts={alerts} loading={alertsLoading} />
          </div>
        </div>
      </main>

      <footer className="bg-gray-800 border-t border-gray-700 py-4 mt-8">
        <div className="container mx-auto px-4 text-center text-gray-500 text-sm">
          <p>Real-time Sentiment Analysis Dashboard</p>
          <p className="text-xs mt-1">Powered by Lambda Architecture • Kafka • Spark • HDFS</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
