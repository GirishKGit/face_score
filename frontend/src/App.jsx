import { useCallback, useMemo, useState } from 'react';
import clsx from 'clsx';
import UploadDropzone from './components/UploadDropzone.jsx';
import ScoreResult from './components/ScoreResult.jsx';
import './App.css';

const API_ENDPOINT = import.meta.env.VITE_API_URL ?? '/api/score';
const MAX_FILE_SIZE_MB = 5;
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

const featureCards = [
  {
    title: 'Deep Learning Score',
    description:
      'Advanced neural networks benchmark your photo against thousands of labelled examples to estimate perceived attractiveness.'
  },
  {
    title: 'Feature Analysis',
    description:
      'Individual assessments for facial components including eyes and nose symmetry keep the score transparent and interpretable.'
  },
  {
    title: 'Symmetry Detection',
    description:
      'Dlib facial landmarks and geometric heuristics quantify left/right balance, a core component across open-source face score tools.'
  }
];

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState(null);

  const acceptedExtensions = useMemo(
    () => ACCEPTED_TYPES.map((type) => type.split('/')[1].toUpperCase()).join(', '),
    []
  );

  const resetState = useCallback(() => {
    setError('');
    setResult(null);
  }, []);

  const validateFile = useCallback((file) => {
    if (!file) {
      setError('Please select an image to analyze.');
      return false;
    }

    if (!ACCEPTED_TYPES.includes(file.type)) {
      setError('Unsupported file type. Please use JPEG, PNG, or WEBP images.');
      return false;
    }

    const sizeInMb = file.size / (1024 * 1024);
    if (sizeInMb > MAX_FILE_SIZE_MB) {
      setError(`Image is too large. Please upload a file under ${MAX_FILE_SIZE_MB}MB.`);
      return false;
    }

    return true;
  }, []);

  const handleFileChange = useCallback(
    (file) => {
      if (!validateFile(file)) {
        setSelectedFile(null);
        setPreviewUrl('');
        setResult(null);
        return;
      }

      resetState();
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    },
    [resetState, validateFile]
  );

  const fileToBase64 = useCallback(async (file) => {
    const base64 = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error('Unable to read the selected file.'));
      reader.readAsDataURL(file);
    });

    const [, data] = String(base64).split(',');
    return data ?? String(base64);
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!validateFile(selectedFile)) {
      return;
    }

    setIsSubmitting(true);
    setError('');

    try {
      const payload = { image_base64: await fileToBase64(selectedFile) };
      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const message = await response.json().catch(() => ({ error: 'Unexpected server error.' }));
        throw new Error(message.error ?? 'Server returned an unexpected error.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setResult(null);
      setError(err.message ?? 'Something went wrong while scoring the face.');
    } finally {
      setIsSubmitting(false);
    }
  }, [fileToBase64, selectedFile, validateFile]);

  return (
    <div className="app">
      <header className="hero" role="banner">
        <nav className="hero__nav" aria-label="Primary">
          <span className="logo" aria-label="Face Score home">
            <span className="logo__dot" aria-hidden="true" />
            Face Score
          </span>
          <div className="hero__actions">
            <a className="hero__link" href="#features">
              Features
            </a>
            <a className="hero__link" href="#upload">
              Upload
            </a>
          </div>
        </nav>
        <div className="hero__content">
          <p className="eyebrow">Discover Your Beauty Score</p>
          <h1 className="headline">
            Upload your portrait and receive an instant AI-powered assessment with transparent metrics and confidence.
          </h1>
          <p className="subheadline">
            Inspired by open-source face attractiveness evaluators, we combine neural predictions with classic symmetry analysis to deliver a balanced beauty score.
          </p>
          <div className="hero__cta">
            <a className="button button--primary" href="#upload">
              Analyze Your Photo
            </a>
            <a
              className="button button--ghost"
              href="https://github.com/aqeelanwar/Face-Attractiveness-Score"
              target="_blank"
              rel="noreferrer"
            >
              Reference Repo
            </a>
          </div>
        </div>
      </header>

      <main>
        <section id="features" className="section">
          <div className="section__intro">
            <h2>Advanced AI Analysis</h2>
            <p>
              Each score is accompanied by interpretable explanations so you can understand exactly how facial features influence the result.
            </p>
          </div>
          <div className="features__grid" role="list">
            {featureCards.map((card) => (
              <article key={card.title} className="feature-card" role="listitem">
                <h3>{card.title}</h3>
                <p>{card.description}</p>
              </article>
            ))}
          </div>
        </section>

        <section id="upload" className="section section--alt" aria-labelledby="upload-title">
          <div className="section__intro">
            <h2 id="upload-title">Upload Your Photo</h2>
            <p>Get instant AI-powered beauty analysis in seconds.</p>
          </div>

          <UploadDropzone
            acceptedExtensions={acceptedExtensions}
            previewUrl={previewUrl}
            file={selectedFile}
            onFileSelect={handleFileChange}
          />

          <div className="actions">
            <button
              type="button"
              className={clsx('button button--primary', isSubmitting && 'button--loading')}
              onClick={handleSubmit}
              disabled={isSubmitting}
            >
              {isSubmitting ? 'Scoring...' : 'Score My Face'}
            </button>
            <p className="helper-text">Supports {acceptedExtensions}. Max {MAX_FILE_SIZE_MB}MB.</p>
          </div>

          {error && (
            <div className="alert alert--error" role="alert">
              {error}
            </div>
          )}

          {result && <ScoreResult result={result} />}
        </section>
      </main>

      <footer className="footer" role="contentinfo">
        <p>© {new Date().getFullYear()} Face Score. Built for ethical experimentation and transparency.</p>
        <a className="footer__link" href="mailto:hello@facescore.ai">
          Contact for feedback
        </a>
      </footer>
    </div>
  );
}

export default App;
