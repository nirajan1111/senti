import { AlertTriangle, Bell, Clock, TrendingDown } from 'lucide-react';
import { format, formatDistanceToNow } from 'date-fns';

export const AlertList = ({ alerts, loading }) => {
  if (loading) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Bell className="w-5 h-5 text-yellow-500" />
          Alerts
        </h3>
        <div className="animate-pulse space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-16 bg-gray-700 rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Bell className="w-5 h-5 text-yellow-500" />
        Alerts
        {alerts.length > 0 && (
          <span className="bg-red-500 text-white text-xs px-2 py-0.5 rounded-full">
            {alerts.length}
          </span>
        )}
      </h3>

      {alerts.length === 0 ? (
        <p className="text-gray-400 text-center py-4">No alerts at this time</p>
      ) : (
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {alerts.map((alert, index) => (
            <AlertItem key={index} alert={alert} />
          ))}
        </div>
      )}
    </div>
  );
};

const AlertItem = ({ alert }) => {
  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical':
        return 'border-red-500 bg-red-500/10';
      case 'warning':
        return 'border-yellow-500 bg-yellow-500/10';
      default:
        return 'border-blue-500 bg-blue-500/10';
    }
  };

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'critical':
        return <AlertTriangle className="w-5 h-5 text-red-500" />;
      case 'warning':
        return <TrendingDown className="w-5 h-5 text-yellow-500" />;
      default:
        return <Bell className="w-5 h-5 text-blue-500" />;
    }
  };

  return (
    <div
      className={`p-3 rounded-lg border-l-4 ${getSeverityColor(alert.severity)}`}
    >
      <div className="flex items-start gap-3">
        {getSeverityIcon(alert.severity)}
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <span className="font-medium capitalize">{alert.topic}</span>
            <span className="text-xs text-gray-500 flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {alert.timestamp
                ? formatDistanceToNow(new Date(alert.timestamp), { addSuffix: true })
                : 'Just now'}
            </span>
          </div>
          <p className="text-sm text-gray-400 mt-1">{alert.message}</p>
          {alert.score && (
            <p className="text-xs text-gray-500 mt-1">
              Score: {(alert.score * 100).toFixed(1)}%
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default AlertList;
