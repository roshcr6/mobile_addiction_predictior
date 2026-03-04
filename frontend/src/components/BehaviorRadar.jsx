import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from 'recharts'

/**
 * Normalise each feature to a 0-100 scale for display.
 */
function normalise(features) {
  return [
    {
      subject: 'Screen Time',
      value: Math.min(((features.screen_time_hours || 0) / 16) * 100, 100),
      fullMark: 100,
    },
    {
      subject: 'Social Media',
      value: Math.min(((features.social_media_hours || 0) / 8) * 100, 100),
      fullMark: 100,
    },
    {
      subject: 'Gaming',
      value: Math.min(((features.gaming_hours || 0) / 5) * 100, 100),
      fullMark: 100,
    },
    {
      subject: 'Unlocks',
      value: Math.min(((features.unlock_count || 0) / 300) * 100, 100),
      fullMark: 100,
    },
    {
      subject: 'Night Usage',
      value: (features.night_usage || 0) * 100,
      fullMark: 100,
    },
  ]
}

export default function BehaviorRadar({ features }) {
  const data = normalise(features)

  return (
    <ResponsiveContainer width="100%" height={280}>
      <RadarChart data={data} cx="50%" cy="50%">
        <PolarGrid stroke="#2e3250" />
        <PolarAngleAxis
          dataKey="subject"
          tick={{ fill: '#8892b0', fontSize: 12 }}
        />
        <PolarRadiusAxis
          angle={90}
          domain={[0, 100]}
          tick={{ fill: '#8892b0', fontSize: 10 }}
          tickCount={4}
        />
        <Radar
          name="Your Usage"
          dataKey="value"
          stroke="#6c63ff"
          fill="#6c63ff"
          fillOpacity={0.35}
        />
        <Tooltip
          contentStyle={{
            background: '#1a1d26',
            border: '1px solid #2e3250',
            borderRadius: '8px',
            color: '#e8eaf6',
          }}
          formatter={(val) => [`${val.toFixed(0)}%`, 'Intensity']}
        />
      </RadarChart>
    </ResponsiveContainer>
  )
}
