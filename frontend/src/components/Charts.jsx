import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { format } from 'date-fns';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 shadow-lg">
        <p className="text-gray-400 text-sm">{label}</p>
        <p className="text-white font-semibold">
          Score: {(payload[0].value * 100).toFixed(1)}%
        </p>
        {payload[0].payload.count && (
          <p className="text-gray-400 text-sm">
            Posts: {payload[0].payload.count}
          </p>
        )}
      </div>
    );
  }
  return null;
};

export const SentimentTrendChart = ({ data, height = 300 }) => {
  const chartData = (data || []).map((item) => ({
    time: format(new Date(item.timestamp), 'HH:mm'),
    score: item.average_score || item.score || 0,
    count: item.count || 0,
  }));

  return (
    <div className="card">
      <h3 className="text-lg font-semibold mb-4">Sentiment Trend</h3>
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={chartData}>
          <defs>
            <linearGradient id="sentimentGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis 
            dataKey="time" 
            stroke="#9ca3af" 
            fontSize={12}
            tickLine={false}
          />
          <YAxis 
            stroke="#9ca3af" 
            fontSize={12}
            tickLine={false}
            domain={[-1, 1]}
            tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="score"
            stroke="#3b82f6"
            strokeWidth={2}
            fill="url(#sentimentGradient)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export const SentimentDistributionChart = ({ distribution, height = 200 }) => {
  const data = [
    { name: 'Positive', value: (distribution?.positive || 0) * 100, color: '#22c55e' },
    { name: 'Neutral', value: (distribution?.neutral || 0) * 100, color: '#6b7280' },
    { name: 'Negative', value: (distribution?.negative || 0) * 100, color: '#ef4444' },
  ];

  return (
    <div className="card">
      <h3 className="text-lg font-semibold mb-4">Sentiment Distribution</h3>
      <div className="space-y-4">
        {data.map((item) => (
          <div key={item.name}>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-400">{item.name}</span>
              <span style={{ color: item.color }}>{item.value.toFixed(1)}%</span>
            </div>
            <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{ width: `${item.value}%`, backgroundColor: item.color }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export const MultiTopicChart = ({ topics, height = 300 }) => {
  const colors = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6'];
  
  // Merge data from all topics
  const mergedData = {};
  topics.forEach((topic, index) => {
    (topic.trend || []).forEach((point) => {
      const time = format(new Date(point.timestamp), 'HH:mm');
      if (!mergedData[time]) {
        mergedData[time] = { time };
      }
      mergedData[time][topic.name] = point.average_score || 0;
    });
  });

  const chartData = Object.values(mergedData);

  return (
    <div className="card">
      <h3 className="text-lg font-semibold mb-4">Topic Comparison</h3>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="time" stroke="#9ca3af" fontSize={12} />
          <YAxis 
            stroke="#9ca3af" 
            fontSize={12}
            domain={[-1, 1]}
            tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#1f2937', 
              border: '1px solid #374151',
              borderRadius: '8px'
            }}
          />
          {topics.map((topic, index) => (
            <Line
              key={topic.name}
              type="monotone"
              dataKey={topic.name}
              stroke={colors[index % colors.length]}
              strokeWidth={2}
              dot={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-4 mt-4 justify-center">
        {topics.map((topic, index) => (
          <div key={topic.name} className="flex items-center gap-2">
            <div 
              className="w-3 h-3 rounded-full" 
              style={{ backgroundColor: colors[index % colors.length] }}
            />
            <span className="text-sm text-gray-400 capitalize">{topic.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SentimentTrendChart;
