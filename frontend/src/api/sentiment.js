import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const sentimentApi = {
  // Health check
  getHealth: async () => {
    const response = await api.get('/health');
    return response.data;
  },

  // Get sentiment for a topic
  getSentiment: async (topic, hours = 24) => {
    const response = await api.get(`/api/v1/sentiment/${topic}`, {
      params: { hours },
    });
    return response.data;
  },

  // Get all topics
  getTopics: async () => {
    const response = await api.get('/api/v1/topics');
    return response.data;
  },

  // Add a new topic
  addTopic: async (topic, subreddits = []) => {
    const response = await api.post('/api/v1/topics', {
      topic,
      subreddits,
    });
    return response.data;
  },

  // Remove a topic
  removeTopic: async (topic) => {
    const response = await api.delete(`/api/v1/topics/${topic}`);
    return response.data;
  },

  // Get trends
  getTrends: async (hours = 24) => {
    const response = await api.get('/api/v1/trends', {
      params: { hours },
    });
    return response.data;
  },

  // Get alerts
  getAlerts: async (limit = 50) => {
    const response = await api.get('/api/v1/alerts', {
      params: { limit },
    });
    return response.data;
  },

  // Compare topics
  compareTopics: async (topics) => {
    const response = await api.post('/api/v1/topics/compare', { topics });
    return response.data;
  },

  // Get topic stats
  getTopicStats: async (topic) => {
    const response = await api.get(`/api/v1/stats/${topic}`);
    return response.data;
  },
};

export default api;
