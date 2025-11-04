"use client";
import { useState, useRef, DragEvent } from "react";
import ThemeToggle from "@/components/ThemeToggle";

type PredictRes = {
  label: string;
  score: number;
  classNames: string[];
  modelVersion: string;
  processingMs: number;
  details?: {
    all_probs?: Record<string, number>;
    model?: string;
    temperature?: number;
  };
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [res, setRes] = useState<PredictRes | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // normalize any legacy labels coming from older backends
  const pretty = (s: string) => {
    if (s === "AiArtData" || s === "AI Art") return "Bot-Made";
    if (s === "RealArt" || s === "Real Art") return "Brush-Made";
    return s;
  };

  const onFile = (f: File | null) => {
    setFile(f);
    setRes(null);
    setError(null);
    setShowDetails(false);
  };

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) =>
    onFile(e.target.files?.[0] ?? null);

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f);
  };

  const classify = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const r = await fetch("/api/predict", { method: "POST", body: form });
      if (!r.ok) throw new Error(await r.text());
      const data: PredictRes = await r.json();

      // pretty-up labels for UI
      data.label = pretty(data.label);
      data.classNames = data.classNames?.map(pretty) ?? ["Bot-Made", "Brush-Made"];

      setRes(data);
    } catch (e: any) {
      setError(e.message ?? String(e));
    } finally {
      setLoading(false);
    }
  };

  // compute per-class probabilities for the dual bars
  const botProb =
    res?.details?.all_probs?.["Bot-Made"] ??
    (res ? (res.label === "Bot-Made" ? res.score : 1 - res.score) : 0);
  const brushProb =
    res?.details?.all_probs?.["Brush-Made"] ??
    (res ? (res.label === "Brush-Made" ? res.score : 1 - res.score) : 0);

  const confidence = Math.round((res?.score ?? 0) * 100);

  return (
    <main className="relative min-h-screen isolate bg-aurora text-zinc-900 smooth">
      {/* floating theme toggle */}
      <ThemeToggle />

      <div className="mx-auto max-w-3xl px-4 py-16">
        {/* Brand */}
        <header className="text-center">
          <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight">
            <span className="text-gradient animate-gradient">ArtSentinel</span>
          </h1>

          <p
            className="
            mx-auto mt-3 inline-flex items-center justify-center
            rounded-full border px-4 py-2
            text-sm sm:text-base font-medium text-zinc-800 dark:text-zinc-100
            bg-white/70 dark:bg-black/40 border-zinc-200/70 dark:border-white/10
            backdrop-blur text-glow"
          >
            <span className="font-semibold">Brush or Bot?</span>
            <span className="mx-2 opacity-50">•</span>
            Drop your art to reveal the truth.
          </p>
        </header>

        {/* Card */}
        <section className="mt-8 rounded-2xl border border-zinc-200/80 bg-white/70 p-5 shadow-lg backdrop-blur supports-[backdrop-filter]:bg-white/50 shadow-[0_10px_40px_-20px_rgba(0,0,0,0.3)] dark:bg-zinc-900/60 dark:border-zinc-800">
          {/* Upload zone */}
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={onDrop}
            className="group relative grid place-items-center rounded-xl border-2 border-dashed border-zinc-300 bg-zinc-50/60 p-6 text-center transition hover:border-zinc-400 dark:bg-zinc-800/30 dark:border-zinc-700"
          >
            {!file ? (
              <>
                <div className="mx-auto mb-3 inline-flex h-12 w-12 items-center justify-center rounded-full bg-zinc-900 text-white shadow">
                  ⬆️
                </div>
                <p className="text-sm text-zinc-700 dark:text-zinc-300">
                  Drag & drop an image here, or
                </p>
                <button
                  onClick={() => inputRef.current?.click()}
                  className="mt-3 inline-flex items-center justify-center rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white shadow hover:bg-black smooth"
                >
                  Choose file
                </button>
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/*"
                  onChange={onInputChange}
                  className="hidden"
                />
                <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
                  JPG / PNG / WEBP • Max ~10 MB
                </p>
              </>
            ) : (
              <div className="w-full">
                <img
                  src={URL.createObjectURL(file)}
                  alt="preview"
                  className="mx-auto max-h-[420px] w-full rounded-xl object-contain shadow"
                />
                <div className="mt-4 flex flex-wrap items-center justify-center gap-3">
                  <button
                    onClick={classify}
                    disabled={loading}
                    className="inline-flex items-center justify-center rounded-xl bg-gradient-to-r from-zinc-900 to-zinc-800 px-5 py-3 text-sm font-semibold text-white shadow transition disabled:opacity-60 hover:from-black hover:to-black active:scale-[0.98]"
                  >
                    {loading ? "Analyzing…" : "Classify"}
                  </button>
                  <button
                    onClick={() => onFile(null)}
                    className="inline-flex items-center justify-center rounded-xl border border-zinc-300 bg-white px-5 py-3 text-sm font-medium text-zinc-700 hover:bg-zinc-50 dark:bg-zinc-900 dark:text-zinc-200 dark:border-zinc-700 dark:hover:bg-zinc-800 smooth"
                  >
                    Choose another
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Result */}
          {error && (
            <div className="mt-5 rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-500/30 dark:bg-red-500/10">
              Error: {error}
            </div>
          )}

          {res && (
            <div className="mt-6 rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm dark:bg-zinc-900 dark:border-zinc-800">
              <div className="flex items-center justify-between gap-3">
                <div className="text-lg font-semibold">
                  Prediction:{" "}
                  <span
                    className={`ml-1 inline-flex items-center rounded-full px-2.5 py-1 text-sm font-semibold ${
                      res.label === "Bot-Made"
                        ? "bg-fuchsia-100 text-fuchsia-800 dark:bg-fuchsia-500/15 dark:text-fuchsia-300"
                        : "bg-emerald-100 text-emerald-800 dark:bg-emerald-500/15 dark:text-emerald-300"
                    }`}
                  >
                    {res.label}
                  </span>
                </div>
                <div className="text-xs text-zinc-500 dark:text-zinc-400">
                  {res.modelVersion} • {res.processingMs} ms
                </div>
              </div>

              {/* Dual Bars */}
              <div className="mt-4 space-y-3">
                <div>
                  <div className="mb-1 flex items-center justify-between text-sm">
                    <span className="font-medium">Bot-Made</span>
                    <span className="tabular-nums">{Math.round(botProb * 100)}%</span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
                    <div
                      className="h-full rounded-full bg-fuchsia-500 transition-all"
                      style={{ width: `${Math.min(100, Math.max(0, botProb * 100))}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="mb-1 flex items-center justify-between text-sm">
                    <span className="font-medium">Brush-Made</span>
                    <span className="tabular-nums">{Math.round(brushProb * 100)}%</span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
                    <div
                      className="h-full rounded-full bg-emerald-500 transition-all"
                      style={{ width: `${Math.min(100, Math.max(0, brushProb * 100))}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Simple confidence line (top-level verdict) */}
              <div className="mt-3 text-sm text-zinc-700 dark:text-zinc-300">
                Top confidence: <span className="font-medium">{confidence}%</span>
              </div>

              {/* Details panel */}
              <div className="mt-4">
                <button
                  onClick={() => setShowDetails((s) => !s)}
                  className="text-xs underline text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white"
                >
                  {showDetails ? "Hide details" : "Show details"}
                </button>
                {showDetails && (
                  <div className="mt-2 rounded-lg border border-zinc-200 p-3 text-xs text-zinc-700 dark:text-zinc-300 dark:border-zinc-800">
                    <div>Model: <span className="font-medium">{res.details?.model ?? "efficientnet_b3"}</span></div>
                    <div>Temperature: <span className="font-medium">{res.details?.temperature ?? 1.0}</span></div>
                    {res.details?.all_probs && (
                      <div className="mt-1">
                        Raw probs:{" "}
                        <code className="rounded bg-zinc-100 px-1 py-0.5 dark:bg-zinc-800">
                          {JSON.stringify(res.details.all_probs)}
                        </code>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </section>

        <footer className="mt-8 text-center text-xs text-zinc-500 dark:text-zinc-400">
          Built with ❤️ for art & truth.
        </footer>
      </div>
    </main>
  );
}
