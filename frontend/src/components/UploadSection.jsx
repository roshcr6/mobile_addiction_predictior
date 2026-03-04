import { useRef, useState } from 'react'
import axios from 'axios'
import styles from './UploadSection.module.css'

const API = import.meta.env.VITE_API_URL || '/api'

/** Green spinner SVG */
function Spinner() {
  return (
    <svg className={styles.spinner} viewBox="0 0 50 50">
      <circle cx="25" cy="25" r="20" fill="none" strokeWidth="5" />
    </svg>
  )
}

export default function UploadSection({ onResult, setLoading, setError, loading, error }) {
  const inputRef = useRef(null)
  const [preview, setPreview] = useState(null)
  const [file, setFile] = useState(null)
  const [dragOver, setDragOver] = useState(false)

  const handleFile = (f) => {
    if (!f) return
    if (!f.type.startsWith('image/')) {
      setError('Please upload a valid image file (JPEG, PNG, WebP).')
      return
    }
    setFile(f)
    setError(null)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }

  const handleSubmit = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const { data } = await axios.post(`${API}/predict`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      onResult(data)
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.message ||
        'Prediction failed. Is the backend running?'
      setError(Array.isArray(msg) ? msg.map((m) => m.msg).join(' | ') : msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.wrapper}>
      <div className={styles.card}>
        <h2 className={styles.heading}>Upload Screenshot</h2>
        <p className={styles.hint}>
          Take a screenshot of your phone's <strong>Screen Time</strong> or{' '}
          <strong>Digital Wellbeing</strong> page and upload it here.
        </p>

        {/* Drop zone */}
        <div
          className={`${styles.dropZone} ${dragOver ? styles.dragOver : ''} ${preview ? styles.hasPreview : ''}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          {preview ? (
            <img src={preview} alt="preview" className={styles.preview} />
          ) : (
            <>
              <span className={styles.dropIcon}>🖼️</span>
              <p>Drag & drop or <span className={styles.link}>browse</span></p>
              <p className={styles.dropHint}>JPEG · PNG · WebP · max 10 MB</p>
            </>
          )}
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => handleFile(e.target.files[0])}
          />
        </div>

        {error && <div className={styles.error}>{error}</div>}

        <div className={styles.actions}>
          {preview && (
            <button
              className={styles.resetBtn}
              onClick={() => { setPreview(null); setFile(null); setError(null) }}
            >
              Clear
            </button>
          )}
          <button
            className={styles.submitBtn}
            onClick={handleSubmit}
            disabled={!file || loading}
          >
            {loading ? <><Spinner /> Analysing…</> : 'Analyse Now'}
          </button>
        </div>
      </div>

      {/* Instructions */}
      <div className={styles.instructions}>
        <h3>How it works</h3>
        <ol>
          <li>📸 Take a screenshot of your Screen Time / Digital Wellbeing dashboard.</li>
          <li>📤 Upload it here.</li>
          <li>🤖 Our OCR engine extracts usage data automatically.</li>
          <li>🧠 The ML model predicts your addiction risk level.</li>
          <li>💡 Receive personalised detox recommendations.</li>
        </ol>
      </div>
    </div>
  )
}
