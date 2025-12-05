import { useState, useEffect, useCallback } from 'react';
import { sentimentApi } from '../api/sentiment';

export const useSentiment = (topic, refreshInterval = 30000) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    if (!topic) return;
    
    try {
      setLoading(true);
      const result = await sentimentApi.getSentiment(topic);
      setData(result);
      setError(null);
    } catch (err) {
      setError(err.message || 'Failed to fetch sentiment data');
    } finally {
      setLoading(false);
    }
  }, [topic]);

  useEffect(() => {
    fetchData();
    
    if (refreshInterval > 0) {
      const interval = setInterval(fetchData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [fetchData, refreshInterval]);

  return { data, loading, error, refetch: fetchData };
};

export const useTopics = (refreshInterval = 60000) => {
  const [topics, setTopics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchTopics = useCallback(async () => {
    try {
      setLoading(true);
      const result = await sentimentApi.getTopics();
      setTopics(result.topics || []);
      setError(null);
    } catch (err) {
      setError(err.message || 'Failed to fetch topics');
    } finally {
      setLoading(false);
    }
  }, []);

  const addTopic = async (topic, subreddits = []) => {
    try {
      await sentimentApi.addTopic(topic, subreddits);
      await fetchTopics();
      return true;
    } catch (err) {
      setError(err.message || 'Failed to add topic');
      return false;
    }
  };

  const removeTopic = async (topic) => {
    try {
      await sentimentApi.removeTopic(topic);
      await fetchTopics();
      return true;
    } catch (err) {
      setError(err.message || 'Failed to remove topic');
      return false;
    }
  };

  useEffect(() => {
    fetchTopics();
    
    if (refreshInterval > 0) {
      const interval = setInterval(fetchTopics, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [fetchTopics, refreshInterval]);

  return { topics, loading, error, addTopic, removeTopic, refetch: fetchTopics };
};

export const useAlerts = (limit = 50, refreshInterval = 15000) => {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchAlerts = useCallback(async () => {
    try {
      setLoading(true);
      const result = await sentimentApi.getAlerts(limit);
      setAlerts(result.alerts || []);
      setError(null);
    } catch (err) {
      setError(err.message || 'Failed to fetch alerts');
    } finally {
      setLoading(false);
    }
  }, [limit]);

  useEffect(() => {
    fetchAlerts();
    
    if (refreshInterval > 0) {
      const interval = setInterval(fetchAlerts, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [fetchAlerts, refreshInterval]);

  return { alerts, loading, error, refetch: fetchAlerts };
};

export const useTrends = (hours = 24, refreshInterval = 60000) => {
  const [trends, setTrends] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchTrends = useCallback(async () => {
    try {
      setLoading(true);
      const result = await sentimentApi.getTrends(hours);
      setTrends(result);
      setError(null);
    } catch (err) {
      setError(err.message || 'Failed to fetch trends');
    } finally {
      setLoading(false);
    }
  }, [hours]);

  useEffect(() => {
    fetchTrends();
    
    if (refreshInterval > 0) {
      const interval = setInterval(fetchTrends, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [fetchTrends, refreshInterval]);

  return { trends, loading, error, refetch: fetchTrends };
};
