import { useState } from 'react'
import UploadSection from './components/UploadSection'
import Dashboard from './components/Dashboard'
import styles from './App.module.css'

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleReset = () => {
    setResult(null)
    setError(null)
  }

  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <div className={styles.logo}>📱</div>
        <div>
          <h1 className={styles.title}>Smartphone Addiction Predictor</h1>
          <p className={styles.subtitle}>
            Upload your screen-time screenshot for an AI-powered addiction risk assessment
          </p>
        </div>
      </header>

      <main className={styles.main}>
        {!result ? (
          <UploadSection
            onResult={setResult}
            setLoading={setLoading}
            setError={setError}
            loading={loading}
            error={error}
          />
        ) : (
          <Dashboard result={result} onReset={handleReset} />
        )}
      </main>

      <footer className={styles.footer}>
        <p>Powered by EasyOCR · scikit‑learn · FastAPI · React</p>
      </footer>
    </div>
  )
}
