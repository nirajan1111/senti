import { TrendingUp, TrendingDown, Minus, Activity } from 'lucide-react';

const getSentimentColor = (score) => {
  if (score > 0.2) return 'text-green-500';
  if (score < -0.2) return 'text-red-500';
  return 'text-gray-400';
};

const getSentimentBg = (score) => {
  if (score > 0.2) return 'bg-green-500/20 border-green-500/30';
  if (score < -0.2) return 'bg-red-500/20 border-red-500/30';
  return 'bg-gray-500/20 border-gray-500/30';
};

const getSentimentIcon = (score) => {
  if (score > 0.2) return <TrendingUp className="w-5 h-5 text-green-500" />;
  if (score < -0.2) return <TrendingDown className="w-5 h-5 text-red-500" />;
  return <Minus className="w-5 h-5 text-gray-400" />;
};

const getSentimentLabel = (score) => {
  if (score > 0.5) return 'Very Positive';
  if (score > 0.2) return 'Positive';
  if (score > -0.2) return 'Neutral';
  if (score > -0.5) return 'Negative';
  return 'Very Negative';
};

export const SentimentCard = ({ topic, sentiment, postCount, onClick }) => {
  const score = sentiment?.average_score || 0;
  
  return (
    <div 
      onClick={onClick}
      className={`card cursor-pointer hover:scale-[1.02] transition-transform border ${getSentimentBg(score)}`}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold capitalize">{topic}</h3>
        {getSentimentIcon(score)}
      </div>
      
      <div className="space-y-3">
        <div>
          <div className="flex items-baseline gap-2">
            <span className={`text-3xl font-bold ${getSentimentColor(score)}`}>
              {(score * 100).toFixed(1)}%
            </span>
            <span className="text-sm text-gray-400">{getSentimentLabel(score)}</span>
          </div>
        </div>
        
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <Activity className="w-4 h-4" />
          <span>{postCount || 0} posts analyzed</span>
        </div>
        
        {sentiment?.distribution && (
          <div className="flex gap-1 h-2 rounded-full overflow-hidden bg-gray-700">
            <div 
              className="bg-green-500 transition-all"
              style={{ width: `${(sentiment.distribution.positive || 0) * 100}%` }}
            />
            <div 
              className="bg-gray-500 transition-all"
              style={{ width: `${(sentiment.distribution.neutral || 0) * 100}%` }}
            />
            <div 
              className="bg-red-500 transition-all"
              style={{ width: `${(sentiment.distribution.negative || 0) * 100}%` }}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export const SentimentMeter = ({ score, size = 'md' }) => {
  const normalizedScore = ((score + 1) / 2) * 100; // Convert -1 to 1 range to 0-100
  
  const sizeClasses = {
    sm: 'w-24 h-24',
    md: 'w-32 h-32',
    lg: 'w-40 h-40',
  };
  
  return (
    <div className={`relative ${sizeClasses[size]}`}>
      <svg className="w-full h-full transform -rotate-90">
        <circle
          cx="50%"
          cy="50%"
          r="45%"
          fill="none"
          stroke="#374151"
          strokeWidth="8"
        />
        <circle
          cx="50%"
          cy="50%"
          r="45%"
          fill="none"
          stroke={score > 0.2 ? '#22c55e' : score < -0.2 ? '#ef4444' : '#6b7280'}
          strokeWidth="8"
          strokeDasharray={`${normalizedScore * 2.83} 283`}
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`text-2xl font-bold ${getSentimentColor(score)}`}>
          {(score * 100).toFixed(0)}%
        </span>
        <span className="text-xs text-gray-400">{getSentimentLabel(score)}</span>
      </div>
    </div>
  );
};

export default SentimentCard;
