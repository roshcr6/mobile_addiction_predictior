import BehaviorRadar from './BehaviorRadar'
import Advice from './Advice'
import styles from './Dashboard.module.css'

const LABEL_META = {
  Low:      { color: 'var(--low)',      emoji: '✅', bg: 'rgba(67,170,139,0.12)',  border: 'rgba(67,170,139,0.35)'  },
  Moderate: { color: 'var(--moderate)', emoji: '⚠️', bg: 'rgba(249,199,79,0.12)',  border: 'rgba(249,199,79,0.35)'  },
  High:     { color: 'var(--high)',     emoji: '🚨', bg: 'rgba(249,65,68,0.12)',   border: 'rgba(249,65,68,0.35)'   },
}

function ScoreGauge({ score }) {
  // score range 0 – ~12+; clamp to 10 for display
  const pct = Math.min(score / 10, 1) * 100
  return (
    <div className={styles.gaugeWrap}>
      <div className={styles.gauge}>
        <div className={styles.gaugeFill} style={{ width: `${pct}%` }} />
      </div>
      <span className={styles.gaugeLabel}>{score.toFixed(2)} / 10</span>
    </div>
  )
}

function FeatureGrid({ features }) {
  const items = [
    { label: 'Screen Time',   value: `${(features.screen_time_hours || 0).toFixed(1)} h`,   icon: '📱' },
    { label: 'Social Media',  value: `${(features.social_media_hours || 0).toFixed(1)} h`,  icon: '💬' },
    { label: 'Gaming',        value: `${(features.gaming_hours || 0).toFixed(1)} h`,        icon: '🎮' },
    { label: 'Unlocks',       value: `${features.unlock_count || 0}x`,                       icon: '🔓' },
    { label: 'Night Usage',   value: features.night_usage ? 'Yes' : 'No',                   icon: '🌙' },
  ]
  return (
    <div className={styles.featureGrid}>
      {items.map((i) => (
        <div key={i.label} className={styles.featureCard}>
          <span className={styles.featureIcon}>{i.icon}</span>
          <span className={styles.featureValue}>{i.value}</span>
          <span className={styles.featureLabel}>{i.label}</span>
        </div>
      ))}
    </div>
  )
}

function ProbBar({ label, value, color }) {
  return (
    <div className={styles.probRow}>
      <span className={styles.probLabel}>{label}</span>
      <div className={styles.probTrack}>
        <div className={styles.probFill} style={{ width: `${(value * 100).toFixed(1)}%`, background: color }} />
      </div>
      <span className={styles.probPct}>{(value * 100).toFixed(1)}%</span>
    </div>
  )
}

export default function Dashboard({ result, onReset }) {
  const meta = LABEL_META[result.label] || LABEL_META['Moderate']
  const { Low = 0, Moderate = 0, High = 0 } = result.probabilities || {}

  return (
    <div className={styles.wrapper}>
      {/* ── Risk badge ── */}
      <div
        className={styles.badge}
        style={{ background: meta.bg, borderColor: meta.border }}
      >
        <span className={styles.badgeEmoji}>{meta.emoji}</span>
        <div>
          <p className={styles.badgeLabel}>Addiction Risk Level</p>
          <h2 className={styles.badgeLevel} style={{ color: meta.color }}>
            {result.label}
          </h2>
        </div>
        <div className={styles.badgeScore}>
          <p className={styles.badgeScoreLabel}>Addiction Score</p>
          <ScoreGauge score={result.addiction_score} />
        </div>
      </div>

      {/* ── Main grid ── */}
      <div className={styles.grid}>
        {/* Left column */}
        <div className={styles.col}>
          <section className={styles.section}>
            <h3>Extracted Features</h3>
            <FeatureGrid features={result.features} />
          </section>

          <section className={styles.section}>
            <h3>Prediction Confidence</h3>
            <div className={styles.probList}>
              <ProbBar label="Low"      value={Low}      color="var(--low)"      />
              <ProbBar label="Moderate" value={Moderate} color="var(--moderate)" />
              <ProbBar label="High"     value={High}     color="var(--high)"     />
            </div>
          </section>
        </div>

        {/* Right column */}
        <div className={styles.col}>
          <section className={styles.section}>
            <h3>Behaviour Radar</h3>
            <BehaviorRadar features={result.features} />
          </section>
        </div>
      </div>

      {/* ── Advice ── */}
      <Advice advice={result.advice} source={result.advice_source} label={result.label} />

      <button className={styles.resetBtn} onClick={onReset}>
        ← Analyse Another Screenshot
      </button>
    </div>
  )
}
