import { useMemo, useState } from 'react';

const EMOTION_STYLES = {
  anger: 'text-rose-400',
  fear: 'text-indigo-400',
  joy: 'text-emerald-400',
  love: 'text-orange-300',
  sadness: 'text-blue-400',
  surprise: 'text-amber-300'
};

const SAMPLE_TEXTS = [
  'I can not believe how amazing this day feels.',
  'Everything seems heavy and quiet today.',
  'That surprise party was the best thing ever.'
];

export default function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [prediction, setPrediction] = useState(null);

  const characters = text.length;
  const words = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text]);
  const confidence = prediction ? Math.max(0, Math.min(100, prediction.confidence * 100)) : 0;
  const emotion = prediction?.predicted_emotion || 'Awaiting analysis';
  const emotionClass = prediction ? EMOTION_STYLES[prediction.predicted_emotion] || 'text-slate-200' : 'text-slate-400';
  const probabilities = prediction?.probabilities ? Object.entries(prediction.probabilities).sort((a, b) => b[1] - a[1]) : [];

  async function analyzeEmotion() {
    if (!text.trim()) {
      setError('Enter a sentence or a short paragraph first.');
      setPrediction(null);
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed.');
      }

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

  return (
    <main className="min-h-screen bg-slate-950 text-slate-50">
      <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
        <section className="relative overflow-hidden rounded-[2rem] border border-white/10 bg-slate-900/70 p-6 shadow-2xl shadow-slate-950/60 backdrop-blur-xl sm:p-8 lg:p-10">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(56,189,248,0.18),transparent_26%),radial-gradient(circle_at_bottom_left,rgba(16,185,129,0.16),transparent_30%)]" />
          <div className="relative grid gap-10 lg:grid-cols-[1.1fr_0.9fr] lg:items-center">
            <div className="space-y-6">
              <span className="inline-flex items-center gap-2 rounded-full border border-cyan-400/20 bg-white/5 px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] text-slate-300">
                <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_18px_rgba(52,211,153,0.6)]" />
                Emotion analysis workspace
              </span>
              <div className="space-y-4">
                <h1 className="max-w-3xl text-4xl font-black tracking-[-0.06em] text-white sm:text-5xl lg:text-7xl">
                  React UI with Tailwind for emotion detection.
                </h1>
                <p className="max-w-2xl text-base leading-8 text-slate-300 sm:text-lg">
                  Paste a tweet, review, or any short text and the model will classify the dominant emotion in real time.
                  The frontend is built with Vite and Tailwind and talks to the Express API.
                </p>
              </div>

              <div className="grid gap-3 sm:grid-cols-3">
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <span className="block text-2xl font-extrabold text-white">6</span>
                  <span className="mt-1 block text-sm text-slate-400">emotion classes</span>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <span className="block text-2xl font-extrabold text-white">{characters}</span>
                  <span className="mt-1 block text-sm text-slate-400">characters typed</span>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <span className="block text-2xl font-extrabold text-white">{words}</span>
                  <span className="mt-1 block text-sm text-slate-400">words detected</span>
                </div>
              </div>
            </div>

            <div className="rounded-[1.75rem] border border-white/10 bg-slate-950/70 p-5 shadow-xl shadow-cyan-950/20">
              <div className="flex items-center justify-between gap-3">
                <h2 className="text-sm font-semibold uppercase tracking-[0.24em] text-slate-400">Try samples</h2>
                <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">No history saved</span>
              </div>

              <div className="mt-5 space-y-3">
                {SAMPLE_TEXTS.map((sample) => (
                  <button
                    key={sample}
                    type="button"
                    onClick={() => loadSample(sample)}
                    className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-left text-sm leading-6 text-slate-200 transition hover:border-cyan-400/40 hover:bg-cyan-400/10"
                  >
                    {sample}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="rounded-[2rem] border border-white/10 bg-slate-900/70 p-5 shadow-2xl shadow-slate-950/60 backdrop-blur-xl sm:p-6">
            <div className="flex items-center justify-between gap-4">
              <div>
                <h2 className="text-lg font-bold text-white sm:text-xl">Analyze text</h2>
                <p className="mt-1 text-sm text-slate-400">Send text to the backend and forward the request to the Python model service.</p>
              </div>
              <div className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">
                {characters} characters
              </div>
            </div>

            <textarea
              className="mt-5 min-h-56 w-full resize-y rounded-[1.5rem] border border-white/10 bg-slate-950/80 p-5 text-base text-slate-100 outline-none transition placeholder:text-slate-500 focus:border-cyan-400/50 focus:ring-4 focus:ring-cyan-400/10"
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Write something emotional..."
            />

            <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-sm text-slate-400">The frontend never stores history or sends data anywhere except the prediction API.</p>
              <div className="flex gap-3">
                <button
                  className="rounded-full border border-white/10 bg-white/5 px-5 py-3 text-sm font-semibold text-slate-200 transition hover:border-white/20 hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
                  onClick={clearAll}
                  type="button"
                  disabled={loading}
                >
                  Clear
                </button>
                <button
                  className="rounded-full bg-gradient-to-r from-cyan-400 to-emerald-400 px-5 py-3 text-sm font-bold text-slate-950 shadow-lg shadow-emerald-500/20 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
                  onClick={analyzeEmotion}
                  disabled={loading}
                  type="button"
                >
                  {loading ? 'Analyzing...' : 'Analyze emotion'}
                </button>
              </div>
            </div>

            <div className="mt-4 rounded-[1.75rem] border border-white/10 bg-slate-950/70 p-5">
              <p className="text-sm font-medium uppercase tracking-[0.24em] text-slate-400">Status</p>
              <p className={`mt-3 text-sm leading-7 ${error ? 'text-rose-300' : prediction ? 'text-emerald-300' : 'text-slate-300'}`}>
                {error || (prediction ? 'Model response received successfully.' : 'Type anything and inspect the emotion breakdown.')}
              </p>
            </div>
          </div>

          <div className="rounded-[2rem] border border-white/10 bg-slate-900/70 p-5 shadow-2xl shadow-slate-950/60 backdrop-blur-xl sm:p-6">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">Prediction result</p>
                <h2 className="mt-1 text-2xl font-black tracking-[-0.04em] text-white">{emotion.toUpperCase()}</h2>
              </div>
              <div className={`rounded-full border border-white/10 bg-white/5 px-3 py-1 text-sm font-semibold ${emotionClass}`}>
                {prediction ? `${confidence.toFixed(2)}% confidence` : 'Waiting'}
              </div>
            </div>

            <div className="mt-6">
              <div className="mb-2 flex items-center justify-between text-sm text-slate-400">
                <span>Confidence</span>
                <span>{prediction ? `${confidence.toFixed(2)}%` : '0.00%'}</span>
              </div>
              <div className="h-3 w-full overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-emerald-400 transition-all duration-300"
                  style={{ width: `${confidence}%` }}
                />
              </div>
            </div>

            {probabilities.length > 0 && (
              <div className="mt-6 space-y-3">
                {probabilities.map(([emotionName, value]) => (
                  <div key={emotionName} className="space-y-2">
                    <div className="flex items-center justify-between text-sm text-slate-300">
                      <span className="capitalize">{emotionName}</span>
                      <span>{(value * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-2 w-full overflow-hidden rounded-full bg-white/10">
                      <div className="h-full rounded-full bg-gradient-to-r from-indigo-400 to-cyan-400" style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            )}

            <div className="mt-6 rounded-[1.5rem] border border-white/10 bg-white/5 p-4 text-sm leading-7 text-slate-300">
              The request is sent to <span className="font-semibold text-white">/api/predict</span>, which is proxied to the Express backend and then forwarded to the Python ML service.
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}