import styles from './Advice.module.css'

const LABEL_SOURCE = {
  ollama: { badge: '🤖 AI-Generated', cls: 'ai' },
  'rule-based': { badge: '📋 Rule-Based', cls: 'rule' },
}

export default function Advice({ advice, source, label }) {
  const meta = LABEL_SOURCE[source] || LABEL_SOURCE['rule-based']

  // Split advice into bullet lines where possible
  const lines = advice
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)

  return (
    <section className={styles.card}>
      <div className={styles.header}>
        <h3 className={styles.title}>💡 Personalised Recommendations</h3>
        <span className={`${styles.sourceBadge} ${styles[meta.cls]}`}>
          {meta.badge}
        </span>
      </div>

      <div className={styles.body}>
        {lines.map((line, i) => {
          const isBullet = line.startsWith('•') || line.startsWith('-') || line.startsWith('*')
          return (
            <p key={i} className={isBullet ? styles.bullet : styles.para}>
              {line}
            </p>
          )
        })}
      </div>
    </section>
  )
}
