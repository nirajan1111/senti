import { Activity, Database, Server, Wifi, WifiOff, RefreshCw } from 'lucide-react';

export const Header = ({ isConnected, onRefresh, lastUpdate }) => {
  return (
    <header className="bg-gray-800 border-b border-gray-700 sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity className="w-8 h-8 text-blue-500" />
            <div>
              <h1 className="text-xl font-bold">Sentiment Analysis</h1>
              <p className="text-sm text-gray-400">Real-time Reddit Monitoring</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm">
              {isConnected ? (
                <>
                  <Wifi className="w-4 h-4 text-green-500" />
                  <span className="text-green-500">Connected</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-4 h-4 text-red-500" />
                  <span className="text-red-500">Disconnected</span>
                </>
              )}
            </div>
            
            {lastUpdate && (
              <span className="text-xs text-gray-500">
                Updated: {new Date(lastUpdate).toLocaleTimeString()}
              </span>
            )}

            <button
              onClick={onRefresh}
              className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 transition-colors"
              title="Refresh"
            >
              <RefreshCw className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export const StatsBar = ({ stats }) => {
  const statItems = [
    {
      icon: <Database className="w-5 h-5 text-blue-500" />,
      label: 'Total Posts',
      value: stats?.totalPosts || 0,
    },
    {
      icon: <Activity className="w-5 h-5 text-green-500" />,
      label: 'Active Topics',
      value: stats?.activeTopics || 0,
    },
    {
      icon: <Server className="w-5 h-5 text-purple-500" />,
      label: 'Processing Rate',
      value: `${stats?.processingRate || 0}/min`,
    },
  ];

  return (
    <div className="grid grid-cols-3 gap-4 mb-6">
      {statItems.map((item, index) => (
        <div key={index} className="card flex items-center gap-4">
          <div className="p-3 bg-gray-700 rounded-lg">{item.icon}</div>
          <div>
            <p className="text-2xl font-bold">{item.value}</p>
            <p className="text-sm text-gray-400">{item.label}</p>
          </div>
        </div>
      ))}
    </div>
  );
};

export const LoadingSpinner = ({ size = 'md' }) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
  };

  return (
    <div className="flex items-center justify-center p-8">
      <div
        className={`${sizeClasses[size]} border-4 border-gray-600 border-t-blue-500 rounded-full animate-spin`}
      />
    </div>
  );
};

export const ErrorMessage = ({ message, onRetry }) => {
  return (
    <div className="card border-red-500/30 bg-red-500/10 text-center">
      <p className="text-red-400 mb-4">{message}</p>
      {onRetry && (
        <button onClick={onRetry} className="btn-primary">
          Retry
        </button>
      )}
    </div>
  );
};

export default Header;
