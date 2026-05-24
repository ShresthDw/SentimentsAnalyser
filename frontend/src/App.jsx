import { useMemo, useState } from 'react';

const EMOTION_COLORS = {
  anger: 'text-red-600',
  fear: 'text-violet-600',
  joy: 'text-emerald-600',
  love: 'text-pink-600',
  sadness: 'text-blue-600',
  surprise: 'text-amber-600'
};

const SAMPLE_TEXTS = [
  'I can not believe how amazing this day feels.',
  'Everything seems heavy and quiet today.',
  'That surprise party was the best thing ever.'
];

const API_URL = import.meta.env.VITE_API_URL || '';

export default function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [prediction, setPrediction] = useState(null);

  const words = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text]);
  const confidence = prediction ? Math.max(0, Math.min(100, prediction.confidence * 100)) : 0;
  const probabilities = prediction?.probabilities
    ? Object.entries(prediction.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  async function analyzeEmotion() {
    if (!text.trim()) {
      setError('Enter some text before analyzing.');
      setPrediction(null);
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Prediction failed.');
      setPrediction(data);
    } catch (err) {
      setPrediction(null);
      setError(err.message || 'Could not connect to the API.');
    } finally {
      setLoading(false);
    }
  }

  function clearAll() {
    setText('');
    setError('');
    setPrediction(null);
  }

  function loadSample(sample) {
    setText(sample);
    setError('');
    setPrediction(null);
  }

  const predictedEmotion = prediction?.predicted_emotion || 'No result yet';
  const emotionColor = EMOTION_COLORS[predictedEmotion] || 'text-slate-900';

  return (
    <main className="min-h-screen bg-slate-50 px-4 py-8 text-slate-900 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-5xl">
        <header className="mb-8">
          <p className="mb-2 text-lg font-semibold uppercase tracking-wider text-emerald-600">Sentiment analyser</p>
          <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">Understand the feeling behind your words.</h1>
          <p className="mt-3 max-w-2xl text-slate-600">Enter a sentence and get a quick emotion prediction from the model.</p>
        </header>

        <div className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
            <div className="flex items-center justify-between gap-4">
              <div>
                <h2 className="text-lg font-semibold">Analyze text</h2>
                <p className="mt-1 text-sm text-slate-500">Write something or try an example below.</p>
              </div>
              <span className="text-sm text-slate-400">{words} words</span>
            </div>

            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="How are you feeling today?"
              className="mt-5 min-h-48 w-full resize-y rounded-xl border border-slate-200 bg-slate-50 p-4 text-base outline-none transition placeholder:text-slate-400 focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10"
            />

            <div className="mt-4 flex flex-col-reverse gap-3 sm:flex-row sm:items-center sm:justify-between">
              <button type="button" onClick={clearAll} disabled={loading} className="text-sm font-medium text-slate-500 hover:text-slate-900 disabled:opacity-50">
                Clear text
              </button>
              <button type="button" onClick={analyzeEmotion} disabled={loading} className="rounded-lg bg-emerald-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-60">
                {loading ? 'Analyzing…' : 'Analyze emotion'}
              </button>
            </div>

            {error && <p className="mt-4 rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</p>}


          </section>

          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
            <p className="text-sm font-medium text-slate-500">Prediction result</p>
            <div className="mt-3 flex items-end justify-between gap-3">
              <h2 className={`text-3xl font-bold capitalize ${emotionColor}`}>{predictedEmotion}</h2>
              <span className="text-sm text-slate-500">{prediction ? `${confidence.toFixed(1)}%` : '—'}</span>
            </div>

            <div className="mt-5 h-2 overflow-hidden rounded-full bg-slate-100">
              <div className="h-full rounded-full bg-emerald-500 transition-all duration-300" style={{ width: `${confidence}%` }} />
            </div>

            {probabilities.length > 0 ? (
              <div className="mt-7 space-y-4">
                {probabilities.map(([emotion, value]) => (
                  <div key={emotion}>
                    <div className="mb-1 flex justify-between text-sm">
                      <span className="capitalize text-slate-600">{emotion}</span>
                      <span className="text-slate-400">{(value * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-slate-100">
                      <div className="h-full rounded-full bg-slate-400" style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="mt-8 rounded-xl bg-slate-50 p-4 text-sm leading-6 text-slate-500">Your emotion breakdown will appear here after you analyze some text.</p>
            )}
          </section>
        </div>
 </div>
    </main>
  );
}
