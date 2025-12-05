import { useState } from 'react';
import { Plus, X, Search, Hash } from 'lucide-react';

export const TopicManager = ({ topics, onAddTopic, onRemoveTopic, loading }) => {
  const [newTopic, setNewTopic] = useState('');
  const [subreddits, setSubreddits] = useState('');
  const [isAdding, setIsAdding] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!newTopic.trim()) return;

    const subredditList = subreddits
      .split(',')
      .map((s) => s.trim())
      .filter((s) => s);

    const success = await onAddTopic(newTopic.trim().toLowerCase(), subredditList);
    if (success) {
      setNewTopic('');
      setSubreddits('');
      setIsAdding(false);
    }
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Topics</h3>
        <button
          onClick={() => setIsAdding(!isAdding)}
          className="btn-primary flex items-center gap-2"
        >
          {isAdding ? <X className="w-4 h-4" /> : <Plus className="w-4 h-4" />}
          {isAdding ? 'Cancel' : 'Add Topic'}
        </button>
      </div>

      {isAdding && (
        <form onSubmit={handleSubmit} className="mb-4 p-4 bg-gray-700/50 rounded-lg">
          <div className="space-y-3">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Topic Name</label>
              <input
                type="text"
                value={newTopic}
                onChange={(e) => setNewTopic(e.target.value)}
                placeholder="e.g., technology, sports, politics"
                className="input w-full"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Subreddits (comma separated, optional)
              </label>
              <input
                type="text"
                value={subreddits}
                onChange={(e) => setSubreddits(e.target.value)}
                placeholder="e.g., technology, gadgets, tech"
                className="input w-full"
              />
            </div>
            <button type="submit" className="btn-primary w-full" disabled={loading}>
              {loading ? 'Adding...' : 'Add Topic'}
            </button>
          </div>
        </form>
      )}

      <div className="space-y-2">
        {topics.length === 0 ? (
          <p className="text-gray-400 text-center py-4">
            No topics yet. Add one to start tracking!
          </p>
        ) : (
          topics.map((topic) => (
            <div
              key={topic.name || topic}
              className="flex items-center justify-between p-3 bg-gray-700/30 rounded-lg hover:bg-gray-700/50 transition-colors"
            >
              <div className="flex items-center gap-2">
                <Hash className="w-4 h-4 text-blue-500" />
                <span className="capitalize">{topic.name || topic}</span>
                {topic.subreddits && (
                  <span className="text-xs text-gray-500">
                    ({topic.subreddits.length} subreddits)
                  </span>
                )}
              </div>
              <button
                onClick={() => onRemoveTopic(topic.name || topic)}
                className="text-gray-400 hover:text-red-500 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export const TopicSearch = ({ onSearch }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit} className="relative">
      <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search topics..."
        className="input w-full pl-10"
      />
    </form>
  );
};

export default TopicManager;
