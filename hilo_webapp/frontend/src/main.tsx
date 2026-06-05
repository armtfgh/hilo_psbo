import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Atom,
  Beaker,
  Bot,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  ClipboardList,
  Crosshair,
  Database,
  FlaskConical,
  Gauge,
  Layers,
  LineChart,
  Moon,
  Play,
  Plus,
  Rocket,
  RotateCcw,
  Send,
  Settings2,
  Shield,
  ShieldAlert,
  Sparkles,
  Sun,
  Target,
  Trash2,
  TrendingDown,
  TrendingUp,
  User,
  Waypoints,
  Zap
} from "lucide-react";
import "./styles.css";

type HistoryRow = {
  iter: number;
  y: number | null;
  best_so_far: number | null;
  method: string;
  x1?: number;
  x2?: number;
  x3?: number;
  x4?: number;
  awcd_score?: number | null;
  awcd_constraint?: number | null;
  awcd_mean_disagree?: number | null;
  weight_believer?: number | null;
  prior_active?: boolean;
};

type ParamKey = "x1" | "x2" | "x3" | "x4";

type AppState = {
  domain: {
    feature_names: string[];
    ranges: { name: string; min: number; max: number }[];
    metadata: Record<string, unknown>;
  };
  settings: Record<string, number | string>;
  readout: Record<string, unknown>;
  history: HistoryRow[];
  summary: {
    iteration: number;
    n_observations: number;
    best_yield: number | null;
    latest_yield: number | null;
    awcd_score: number | null;
    c_user: number;
    prior_active: boolean;
    status: "not_started" | "prior_trusted" | "prior_gated";
    last_error: string | null;
  };
};

type Surface = {
  x_label: string;
  y_label: string;
  x_index?: number;
  y_index?: number;
  x_values: number[];
  y_values: number[];
  values: number[][];
  forbidden?: number[][];
  min: number;
  max: number;
};

type SurfaceBundle = {
  surfaces: Surface[];
  min: number;
  max: number;
};

type AskTellParam = { name: string; min: number; max: number; unit?: string };
type AskTellData = {
  configured: boolean;
  objective_name?: string;
  goal?: "maximize" | "minimize";
  parameters?: AskTellParam[];
  batch_size?: number;
  n_init?: number;
  round?: number;
  n_observations?: number;
  best?: { y: number; x: Record<string, number> } | null;
  pending?: { id: number; x: Record<string, number> }[];
  history?: { n: number; x: Record<string, number>; y: number }[];
  series?: { n: number; y: number; best: number }[];
  readout?: Record<string, unknown>;
  c_user?: number;
  prior_active?: boolean;
  awcd?: number | null;
  has_prior?: boolean;
};

const DEFAULT_READOUT = JSON.stringify({ effects: {}, bumps: [], constraints: [] }, null, 2);
const PARAM_LABELS: Record<string, string> = {
  amine_mM: "amine",
  aldehyde_mM: "aldehyde",
  isocyanide_mM: "isocyanide",
  ptsa: "pTSA"
};

type ExpertMessage = {
  role: "expert" | "hilo";
  text: string;
  time: string;
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const payload = await res.json();
      detail = payload.detail || detail;
    } catch {
      // keep default
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

function fmt(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return value.toFixed(digits);
}

function statusText(status: AppState["summary"]["status"]): string {
  if (status === "prior_trusted") return "Prior trusted";
  if (status === "prior_gated") return "Prior gated";
  return "Idle";
}

function statusClass(status: AppState["summary"]["status"]): string {
  if (status === "prior_trusted") return "ok";
  if (status === "prior_gated") return "danger";
  return "idle";
}

/* ---- viridis colormap ----------------------------------------------------- */
const VIRIDIS: [number, number, number][] = [
  [68, 1, 84],
  [72, 40, 120],
  [62, 74, 137],
  [49, 104, 142],
  [38, 130, 142],
  [31, 158, 137],
  [53, 183, 121],
  [110, 206, 88],
  [181, 222, 43],
  [253, 231, 37]
];

function viridis(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = VIRIDIS[i];
  const b = VIRIDIS[Math.min(VIRIDIS.length - 1, i + 1)];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f)
  ];
}

/* ---- hooks ---------------------------------------------------------------- */
// Snap counters instantly when the user prefers reduced motion, or for static
// captures (?anim=off) — otherwise the count-up may not settle in a screenshot.
const NO_ANIM: boolean = (() => {
  try {
    if (new URLSearchParams(window.location.search).get("anim") === "off") return true;
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
})();

function useAnimatedNumber(target: number | null, duration = 650): number {
  const [val, setVal] = useState(target ?? 0);
  const fromRef = useRef(target ?? 0);
  const startRef = useRef(0);
  const rafRef = useRef(0);
  useEffect(() => {
    if (target === null || Number.isNaN(target)) {
      setVal(0);
      return;
    }
    if (NO_ANIM) {
      setVal(target);
      return;
    }
    fromRef.current = val;
    startRef.current = performance.now();
    const from = fromRef.current;
    const delta = target - from;
    const tick = (now: number) => {
      const p = Math.min(1, (now - startRef.current) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setVal(from + delta * eased);
      if (p < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target]);
  return val;
}

function useElementSize<T extends HTMLElement>(): [React.RefObject<T | null>, { w: number; h: number }] {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  useLayoutEffect(() => {
    if (!ref.current) return;
    const el = ref.current;
    const measure = () => {
      const r = el.getBoundingClientRect();
      setSize((prev) => (prev.w === r.width && prev.h === r.height ? prev : { w: r.width, h: r.height }));
    };
    measure(); // synchronous initial measure so the first paint never waits on the observer
    let raf = requestAnimationFrame(measure); // catch a not-yet-laid-out first frame
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, []);
  return [ref, size];
}

/* ---- readout fallback translator (unchanged behaviour) -------------------- */
function draftReadoutFromExpertText(text: string, state: AppState): Record<string, unknown> {
  const lower = text.toLowerCase();
  const current = JSON.parse(JSON.stringify(state.readout || {}));
  const effects: Record<string, Record<string, unknown>> = { ...(current.effects || {}) };
  const constraints: Array<Record<string, unknown>> = Array.isArray(current.constraints) ? [...current.constraints] : [];
  const ranges = Object.fromEntries(state.domain.ranges.map((r) => [r.name, r]));
  const aliases: Array<[string, string[]]> = [
    ["amine_mM", ["amine", "x1"]],
    ["aldehyde_mM", ["aldehyde", "x2"]],
    ["isocyanide_mM", ["isocyanide", "x3"]],
    ["ptsa", ["ptsa", "p-tsa", "ptsoh", "p-tsoh", "acid", "x4"]]
  ];
  const hasPositive = /(beneficial|good|prefer|productive|support|increase|higher|high|favorable|favored)/.test(lower);
  const hasNegative = /(bad|poor|failed|harmful|avoid|forbid|exclude|unproductive|penalty)/.test(lower);
  const avoidLow = /(avoid|forbid|exclude|bad|poor|failed|harmful).{0,42}(low|very low|lower|below|under|less than)|(?:low|very low|lower|below|under|less than).{0,42}(bad|poor|failed|harmful|avoid)/.test(lower);
  const avoidHigh = /(avoid|forbid|exclude|bad|poor|failed|harmful).{0,42}(high|very high|higher|above|over|greater than)|(?:high|very high|higher|above|over|greater than).{0,42}(bad|poor|failed|harmful|avoid)/.test(lower);
  const preferHigh = /(prefer|beneficial|good|productive|favorable|favored).{0,30}(high|higher|upper)|(?:high|higher|upper).{0,30}(beneficial|good|productive|favorable|favored)/.test(lower);
  const preferLow = /(prefer|beneficial|good|productive|favorable|favored).{0,30}(low|lower)|(?:low|lower).{0,30}(beneficial|good|productive|favorable|favored)/.test(lower);
  const numbers = (lower.match(/(?<![a-z0-9.])-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/g) || []).map(Number).filter(Number.isFinite);

  for (const [name, words] of aliases) {
    if (!words.some((word) => lower.includes(word))) continue;
    const r = ranges[name];
    if (!r) continue;
    const span = r.max - r.min;
    const lowBand = [r.min, r.min + span / 3];
    const highBand = [r.max - span / 3, r.max];
    const threshold = numbers.find((v) => v >= r.min && v <= r.max);
    if (avoidLow) {
      constraints.push({ var: name, range: [r.min, threshold ?? lowBand[1]], penalty: 7.5, reason: `Expert text discouraged ${name} below threshold` });
    } else if (avoidHigh) {
      constraints.push({ var: name, range: [threshold ?? highBand[0], r.max], penalty: 7.5, reason: `Expert text discouraged ${name} above threshold` });
    } else if (preferHigh || (hasPositive && lower.includes("high"))) {
      effects[name] = { effect: "increasing", scale: 0.55, confidence: 0.7, range_hint: highBand };
    } else if (preferLow || (hasPositive && lower.includes("low"))) {
      effects[name] = { effect: "decreasing", scale: 0.55, confidence: 0.7, range_hint: lowBand };
    } else if (hasNegative) {
      effects[name] = { effect: "flat", scale: 0.15, confidence: 0.35, range_hint: [r.min, r.max] };
    }
  }

  return {
    effects,
    bumps: Array.isArray(current.bumps) ? current.bumps : [],
    constraints
  };
}

/* ---- sparkline ------------------------------------------------------------ */
function Sparkline({ data, color, width = 78, height = 30 }: { data: number[]; color: string; width?: number; height?: number }) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const span = Math.max(1e-9, max - min);
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / span) * (height - 4) - 2;
    return [x, y];
  });
  const line = pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  const area = `0,${height} ${line} ${width},${height}`;
  const id = `spark-${color.replace(/[^a-z0-9]/gi, "")}`;
  return (
    <svg className="kpiSpark" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.35" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={area} fill={`url(#${id})`} />
      <polyline points={line} fill="none" stroke={color} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={pts[pts.length - 1][0]} cy={pts[pts.length - 1][1]} r="2" fill={color} />
    </svg>
  );
}

/* ---- line chart with hover ------------------------------------------------ */
function LineChartSvg({
  rows,
  yKey,
  color,
  yLabel,
  height = 250,
  threshold,
  thresholdLabel
}: {
  rows: HistoryRow[];
  yKey: keyof HistoryRow;
  color: string;
  yLabel: string;
  height?: number;
  threshold?: number;
  thresholdLabel?: string;
}) {
  const [hover, setHover] = useState<{ x: number; y: number; iter: number; val: number } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const data = rows.filter((r) => r.iter >= 0 && typeof r[yKey] === "number");
  const width = 760;
  const pad = { l: 56, r: 18, t: 20, b: 32 };
  const values = data.map((r) => Number(r[yKey]));
  const xVals = data.map((r) => r.iter + 1);
  let yMin = values.length ? Math.min(...values) : 0;
  let yMax = values.length ? Math.max(...values) : 1;
  if (threshold !== undefined) { yMin = Math.min(yMin, threshold); yMax = Math.max(yMax, threshold); }
  const pad5 = (yMax - yMin) * 0.08 || 0.05;
  yMin -= pad5; yMax += pad5;
  const xMin = xVals.length ? Math.min(...xVals) : 1;
  const xMax = xVals.length ? Math.max(...xVals) : 2;
  const ySpan = Math.max(1e-9, yMax - yMin);
  const xSpan = Math.max(1, xMax - xMin);
  const sx = (x: number) => pad.l + ((x - xMin) / xSpan) * (width - pad.l - pad.r);
  const sy = (y: number) => pad.t + (1 - (y - yMin) / ySpan) * (height - pad.t - pad.b);
  const points = data.map((r) => `${sx(r.iter + 1)},${sy(Number(r[yKey]))}`).join(" ");
  const area = data.length > 1
    ? `${sx(data[0].iter + 1)},${height - pad.b} ${points} ${sx(data[data.length - 1].iter + 1)},${height - pad.b}`
    : "";
  const gid = `grad-${yLabel.replace(/[^a-z0-9]/gi, "")}-${color.replace(/[^a-z0-9]/gi, "")}`;
  const gridY = [0, 0.25, 0.5, 0.75, 1].map((f) => yMin + f * ySpan);

  const onMove = (e: React.MouseEvent) => {
    if (!svgRef.current || !data.length) return;
    const rect = svgRef.current.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const vbX = fx * width;
    let nearest = data[0];
    let best = Infinity;
    for (const r of data) {
      const d = Math.abs(sx(r.iter + 1) - vbX);
      if (d < best) { best = d; nearest = r; }
    }
    const px = sx(nearest.iter + 1) / width * rect.width;
    const py = sy(Number(nearest[yKey])) / height * rect.height;
    setHover({ x: px, y: py, iter: nearest.iter + 1, val: Number(nearest[yKey]) });
  };

  return (
    <div className="chartWrap" onMouseLeave={() => setHover(null)}>
      <svg ref={svgRef} className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={yLabel} onMouseMove={onMove}>
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.32" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
          <filter id={`glow-${gid}`}><feGaussianBlur stdDeviation="2.4" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>
        {gridY.map((gv, i) => (
          <g key={i}>
            <line className="grid" x1={pad.l} y1={sy(gv)} x2={width - pad.r} y2={sy(gv)} />
            <text className="axisText" x={pad.l - 8} y={sy(gv) + 3} textAnchor="end">{fmt(gv, 2)}</text>
          </g>
        ))}
        <line className="axis" x1={pad.l} y1={height - pad.b} x2={width - pad.r} y2={height - pad.b} />
        <text className="axisText" x={width - pad.r} y={height - 8} textAnchor="end">iteration</text>
        <text className="axisText" x={pad.l} y={height - 8} textAnchor="start">{xMin}</text>
        {threshold !== undefined && (
          <g>
            <line className="threshold" x1={pad.l} y1={sy(threshold)} x2={width - pad.r} y2={sy(threshold)} />
            <text className="thresholdText" x={width - pad.r} y={sy(threshold) - 5} textAnchor="end">{thresholdLabel || `c_user ${fmt(threshold, 2)}`}</text>
          </g>
        )}
        {area && <polygon points={area} fill={`url(#${gid})`} />}
        {points && (
          <polyline points={points} fill="none" stroke={color} strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" filter={`url(#glow-${gid})`} />
        )}
        {data.map((r) => (
          <circle key={`${String(yKey)}-${r.iter}`} cx={sx(r.iter + 1)} cy={sy(Number(r[yKey]))} r="2.6" fill={color} />
        ))}
        {hover && (
          <line className="grid" x1={hover.x / (svgRef.current?.getBoundingClientRect().width || 1) * width}
            y1={pad.t} x2={hover.x / (svgRef.current?.getBoundingClientRect().width || 1) * width} y2={height - pad.b}
            stroke={color} strokeOpacity="0.35" />
        )}
        {data.length === 0 && <text className="axisText" x={width / 2} y={height / 2} textAnchor="middle">no BO iterations yet</text>}
      </svg>
      {hover && (
        <div className="chartTooltip" style={{ left: hover.x, top: hover.y }}>
          iter {hover.iter} · <b>{fmt(hover.val, 3)}</b>
        </div>
      )}
    </div>
  );
}

/* ---- scatter -------------------------------------------------------------- */
function ScatterSvg({ rows, xKey, xLabel, color = "#5b8cff", height = 160 }: {
  rows: HistoryRow[]; xKey: ParamKey; xLabel: string; color?: string; height?: number;
}) {
  const data = rows.filter((r) => typeof r[xKey] === "number" && typeof r.y === "number");
  const width = 360;
  const pad = { l: 46, r: 14, t: 14, b: 28 };
  const xs = data.map((r) => Number(r[xKey]));
  const ys = data.map((r) => Number(r.y));
  const xMin = xs.length ? Math.min(...xs) : 0;
  const xMax = xs.length ? Math.max(...xs) : 1;
  const yMin = ys.length ? Math.min(...ys) : 0;
  const yMax = ys.length ? Math.max(...ys) : 1;
  const xSpan = Math.max(1e-9, xMax - xMin);
  const ySpan = Math.max(1e-9, yMax - yMin);
  const sx = (x: number) => pad.l + ((x - xMin) / xSpan) * (width - pad.l - pad.r);
  const sy = (y: number) => pad.t + (1 - (y - yMin) / ySpan) * (height - pad.t - pad.b);
  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${xLabel} vs yield`}>
      {[0, 0.5, 1].map((f, i) => (
        <line key={i} className="grid" x1={pad.l} y1={sy(yMin + f * ySpan)} x2={width - pad.r} y2={sy(yMin + f * ySpan)} />
      ))}
      <line className="axis" x1={pad.l} y1={height - pad.b} x2={width - pad.r} y2={height - pad.b} />
      <text className="axisText" x={width - pad.r} y={height - 8} textAnchor="end">{xLabel}</text>
      <text className="axisText" x={pad.l - 6} y={pad.t + 4} textAnchor="end">{fmt(yMax, 2)}</text>
      <text className="axisText" x={pad.l - 6} y={height - pad.b + 4} textAnchor="end">{fmt(yMin, 2)}</text>
      {data.map((r, i) => (
        <circle key={`${xKey}-${r.iter}-${i}`} cx={sx(Number(r[xKey]))} cy={sy(Number(r.y))}
          r={r.iter < 0 ? 2.6 : 3.4} fill={r.iter < 0 ? "#5f677b" : color} opacity={r.iter < 0 ? 0.5 : 0.92} />
      ))}
    </svg>
  );
}

/* ---- canvas heatmap ------------------------------------------------------- */
type ObsPoint = { x: number; y: number; isBO: boolean; best: boolean };

function Heatmap({ surface, globalMin, globalMax, points }: {
  surface: Surface; globalMin: number; globalMax: number; points: ObsPoint[];
}) {
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{ x: number; y: number; xv: number; yv: number; v: number; forbidden: boolean } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || size.w < 2 || size.h < 2) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const W = Math.round(size.w * dpr);
    const H = Math.round(size.h * dpr);
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const nY = surface.values.length;
    const nX = surface.values[0]?.length || 0;
    const lo = globalMin;
    const rawSpan = globalMax - globalMin;
    const flat = rawSpan < 1e-6; // no preference signal anywhere -> neutral fill
    const span = Math.max(1e-9, rawSpan);

    // render low-res grid to an offscreen canvas, then scale up with smoothing
    const off = document.createElement("canvas");
    off.width = nX; off.height = nY;
    const offCtx = off.getContext("2d");
    if (!offCtx) return;
    const img = offCtx.createImageData(nX, nY);
    for (let yi = 0; yi < nY; yi++) {
      for (let xi = 0; xi < nX; xi++) {
        const v = surface.values[nY - 1 - yi][xi]; // flip so max-y is at top
        const [r, g, b] = flat ? [38, 44, 58] : viridis((v - lo) / span);
        const idx = (yi * nX + xi) * 4;
        img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b; img.data[idx + 3] = 255;
      }
    }
    offCtx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(off, 0, 0, nX, nY, 0, 0, W, H);

    // forbidden-region overlay (hatched red), kept separate from the colormap
    const fb = surface.forbidden;
    if (fb && fb.length) {
      const maskC = document.createElement("canvas");
      maskC.width = nX; maskC.height = nY;
      const mctx = maskC.getContext("2d");
      if (mctx) {
        const mimg = mctx.createImageData(nX, nY);
        let any = false;
        for (let yi = 0; yi < nY; yi++) {
          for (let xi = 0; xi < nX; xi++) {
            const f = fb[nY - 1 - yi]?.[xi];
            const idx = (yi * nX + xi) * 4;
            if (f) { any = true; mimg.data[idx + 3] = 255; }
          }
        }
        mctx.putImageData(mimg, 0, 0);
        if (any) {
          const t = Math.max(8, Math.round(11 * dpr));
          const tile = document.createElement("canvas");
          tile.width = t; tile.height = t;
          const tctx = tile.getContext("2d");
          if (tctx) {
            tctx.strokeStyle = "rgba(251,113,133,0.95)";
            tctx.lineWidth = Math.max(1, 1.4 * dpr);
            tctx.beginPath(); tctx.moveTo(0, t); tctx.lineTo(t, 0); tctx.stroke();
          }
          const ov = document.createElement("canvas");
          ov.width = W; ov.height = H;
          const octx = ov.getContext("2d");
          if (octx) {
            octx.fillStyle = "rgba(244,80,107,0.30)";
            octx.fillRect(0, 0, W, H);
            const pat = tctx ? octx.createPattern(tile, "repeat") : null;
            if (pat) { octx.fillStyle = pat; octx.fillRect(0, 0, W, H); }
            octx.globalCompositeOperation = "destination-in";
            octx.imageSmoothingEnabled = true;
            octx.imageSmoothingQuality = "high";
            octx.drawImage(maskC, 0, 0, nX, nY, 0, 0, W, H);
            ctx.drawImage(ov, 0, 0);
          }
        }
      }
    }

    // observed points overlay
    const xmin = surface.x_values[0];
    const xmax = surface.x_values[surface.x_values.length - 1];
    const ymin = surface.y_values[0];
    const ymax = surface.y_values[surface.y_values.length - 1];
    const px = (xv: number) => ((xv - xmin) / Math.max(1e-9, xmax - xmin)) * W;
    const py = (yv: number) => (1 - (yv - ymin) / Math.max(1e-9, ymax - ymin)) * H;
    for (const p of points) {
      const cx = px(p.x), cy = py(p.y);
      if (!Number.isFinite(cx) || !Number.isFinite(cy)) continue;
      ctx.beginPath();
      ctx.arc(cx, cy, (p.best ? 6.5 : p.isBO ? 4.2 : 3.2) * dpr, 0, Math.PI * 2);
      ctx.fillStyle = p.best ? "rgba(46,230,166,0.95)" : p.isBO ? "rgba(255,255,255,0.92)" : "rgba(255,255,255,0.45)";
      ctx.strokeStyle = "rgba(0,0,0,0.55)";
      ctx.lineWidth = 1.4 * dpr;
      ctx.fill();
      ctx.stroke();
      if (p.best) {
        ctx.beginPath();
        ctx.arc(cx, cy, 10 * dpr, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(46,230,166,0.6)";
        ctx.lineWidth = 1.6 * dpr;
        ctx.stroke();
      }
    }
  }, [surface, globalMin, globalMax, points, size.w, size.h]);

  const onMove = (e: React.MouseEvent) => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const rect = wrap.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    const nY = surface.values.length;
    const nX = surface.values[0]?.length || 1;
    const xi = Math.max(0, Math.min(nX - 1, Math.round(fx * (nX - 1))));
    const yiTop = Math.max(0, Math.min(nY - 1, Math.round(fy * (nY - 1))));
    const yi = nY - 1 - yiTop;
    setHover({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      xv: surface.x_values[xi],
      yv: surface.y_values[yi],
      v: surface.values[yi][xi],
      forbidden: !!surface.forbidden?.[yi]?.[xi]
    });
  };

  return (
    <div className="heatCanvasWrap" ref={wrapRef} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
      <canvas ref={canvasRef} />
      {hover && (
        <div className="heatHover" style={{ left: hover.x, top: hover.y }}>
          {fmt(hover.xv, 1)}, {fmt(hover.yv, 1)} → <b>{fmt(hover.v, 2)}</b>
          {hover.forbidden && <span style={{ color: "var(--rose)" }}> · forbidden</span>}
        </div>
      )}
    </div>
  );
}

function HeatmapGrid({ bundle, history }: { bundle: SurfaceBundle | null; history: HistoryRow[] }) {
  if (!bundle?.surfaces?.length) return <div className="empty">Prior surfaces not loaded.</div>;
  const bestVal = Math.max(...history.filter((r) => typeof r.y === "number").map((r) => Number(r.y)), -Infinity);
  return (
    <div className="heatGrid">
      {bundle.surfaces.map((s) => {
        const xi = (s.x_index ?? 0) + 1;
        const yi = (s.y_index ?? 1) + 1;
        const points: ObsPoint[] = history
          .filter((r) => typeof (r as any)[`x${xi}`] === "number" && typeof (r as any)[`x${yi}`] === "number")
          .map((r) => ({
            x: Number((r as any)[`x${xi}`]),
            y: Number((r as any)[`x${yi}`]),
            isBO: r.iter >= 0,
            best: typeof r.y === "number" && Math.abs(Number(r.y) - bestVal) < 1e-9
          }));
        return (
          <div className="heatCard" key={`${s.x_label}-${s.y_label}`}>
            <div className="heatTitle">
              <span>{PARAM_LABELS[s.x_label] || s.x_label} × {PARAM_LABELS[s.y_label] || s.y_label}</span>
              <small>[{fmt(s.min, 1)}, {fmt(s.max, 1)}]</small>
            </div>
            <Heatmap surface={s} globalMin={bundle.min} globalMax={bundle.max} points={points} />
            <div className="axisLabels">
              <span>{PARAM_LABELS[s.x_label] || s.x_label} →</span>
              <span>↑ {PARAM_LABELS[s.y_label] || s.y_label}</span>
            </div>
          </div>
        );
      })}
      <div className="colorbar heatLegend">
        <span>low preference</span>
        <div className="bar" />
        <span>high preference</span>
        <span className="forbiddenKey">forbidden</span>
        <span style={{ marginLeft: "auto", color: "var(--green)" }}>● best obs</span>
        <span style={{ color: "var(--ink)" }}>● BO point</span>
      </div>
    </div>
  );
}

/* ---- KPI ------------------------------------------------------------------ */
function Kpi({ label, icon, value, note, accent, spark, sparkColor, delta }: {
  label: string;
  icon: React.ReactNode;
  value: string;
  note?: string;
  accent: string;
  spark?: number[];
  sparkColor?: string;
  delta?: number;
}) {
  const dCls = delta === undefined ? "" : delta > 1e-6 ? "up" : delta < -1e-6 ? "down" : "flat";
  return (
    <div className="kpi" style={{ ["--accent" as any]: accent }}>
      <div className="kpiTop">
        <span className="label">{icon} {label}</span>
        {delta !== undefined && (
          <span className={`delta ${dCls}`}>
            {dCls === "up" ? <TrendingUp size={11} /> : dCls === "down" ? <TrendingDown size={11} /> : null}
            {delta > 0 ? "+" : ""}{fmt(delta, 2)}
          </span>
        )}
      </div>
      <strong>{value}</strong>
      {note && <small>{note}</small>}
      {spark && spark.length > 1 && <Sparkline data={spark} color={sparkColor || "#2ee6a6"} />}
    </div>
  );
}

/* ---- diagnostics ---------------------------------------------------------- */
function DiagCard({ icon, title, value, sub, meter, meterClass }: {
  icon: React.ReactNode; title: string; value: string; sub?: string; meter?: number; meterClass?: string;
}) {
  return (
    <div className="diagCard">
      <div className="dHead">{icon} {title}</div>
      <div className="dVal">{value}</div>
      {sub && <div className="dSub">{sub}</div>}
      {meter !== undefined && (
        <div className={`meter ${meterClass || ""}`}>
          <i style={{ width: `${Math.max(0, Math.min(100, meter * 100))}%` }} />
        </div>
      )}
    </div>
  );
}

/* ---- prior effect chips --------------------------------------------------- */
function ReadoutChips({ readout }: { readout: Record<string, unknown> }) {
  const effects = (readout.effects || {}) as Record<string, any>;
  const constraints = (readout.constraints || []) as any[];
  const bumps = (readout.bumps || []) as any[];
  const items: React.ReactNode[] = [];
  for (const [k, v] of Object.entries(effects)) {
    const eff = String(v?.effect || "flat");
    const cls = eff.includes("increas") ? "increasing" : eff.includes("decreas") ? "decreasing" : "flat";
    const arrow = eff.includes("increas") ? "↑" : eff.includes("decreas") ? "↓" : eff.includes("peak") ? "∩" : eff.includes("valley") ? "∪" : "→";
    items.push(
      <span className={`chip ${cls}`} key={`e-${k}`}>
        <span className="dir">{arrow}</span> {PARAM_LABELS[k] || k}
        {v?.confidence !== undefined && <small style={{ opacity: 0.7 }}>c={fmt(v.confidence, 2)}</small>}
      </span>
    );
  }
  for (const c of constraints) {
    items.push(
      <span className="chip constraint" key={`c-${JSON.stringify(c).slice(0, 24)}`}>
        <ShieldAlert size={12} /> avoid {PARAM_LABELS[c?.var] || c?.var} [{fmt(c?.range?.[0], 1)}, {fmt(c?.range?.[1], 1)}]
      </span>
    );
  }
  bumps.forEach((b, i) => items.push(<span className="chip bump" key={`b-${i}`}><Sparkles size={12} /> bump amp={fmt(b?.amp, 2)}</span>));
  if (!items.length) return <div className="empty" style={{ minHeight: 80 }}>No prior primitives yet — describe intuition in the chat, then Apply.</div>;
  return <div className="chips">{items}</div>;
}

/* ============================================================ APP ========== */
function App() {
  const [state, setState] = useState<AppState | null>(null);
  const [surfaceBundle, setSurfaceBundle] = useState<SurfaceBundle | null>(null);
  const [theme, setTheme] = useState<"dark" | "light">(
    () => (typeof document !== "undefined" && document.documentElement.dataset.theme === "light" ? "light" : "dark")
  );
  const initialTab = window.location.hash.replace("#", "") || "run";
  const [tab, setTab] = useState(["run", "prior", "safety", "data", "asktell"].includes(initialTab) ? initialTab : "run");
  const [readoutText, setReadoutText] = useState(DEFAULT_READOUT);
  const [busy, setBusy] = useState(false);
  const [translating, setTranslating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [settingsDraft, setSettingsDraft] = useState<Record<string, number | string>>({});
  const [expertText, setExpertText] = useState("");
  const [llmModel, setLlmModel] = useState("claude-opus-4-5-20251101");
  const [llmApiKey, setLlmApiKey] = useState("");
  const messageEndRef = useRef<HTMLDivElement>(null);
  const [expertMessages, setExpertMessages] = useState<ExpertMessage[]>(() => {
    const welcome: ExpertMessage = {
      role: "hilo",
      text: "Describe your reaction intuition in plain language. I keep the transcript here while you turn it into a structured prior with Claude.",
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    };
    // ?demo=chat seeds an example expert transcript (used for documentation/figures).
    try {
      if (new URLSearchParams(window.location.search).get("demo") === "chat") {
        return [
          welcome,
          {
            role: "expert",
            text: "In the Ugi 4-component reaction the isocyanide is usually rate-limiting — higher isocyanide loading tends to raise yield. Aldehyde works best in a mid range, ~180–240 mM; too much promotes side products. The condensation needs acid, so avoid very low pTSA (below ~0.05 eq).",
            time: "10:24"
          },
          { role: "hilo", text: "Captured the transcript. Translating it into a structured HILO prior with Claude Opus…", time: "10:24" },
          {
            role: "hilo",
            text: "Drafted a prior: isocyanide → increasing, aldehyde → peak at 180–240 mM, plus a soft constraint forbidding pTSA < 0.05. Open Prior Studio to review, then Apply before the next BO steps.",
            time: "10:25"
          }
        ];
      }
    } catch { /* ignore */ }
    return [welcome];
  });

  const load = async () => {
    const next = await api<AppState>("/api/state");
    setState(next);
    setSettingsDraft(next.settings);
    setReadoutText(JSON.stringify(next.readout, null, 2));
    setSurfaceBundle(await api<SurfaceBundle>("/api/prior-surfaces?n_grid=44"));
  };

  useEffect(() => { load().catch((e) => setError(String(e.message || e))); }, []);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("hilo-theme", theme); } catch { /* ignore */ }
  }, [theme]);
  const toggleTheme = () => setTheme((t) => (t === "dark" ? "light" : "dark"));
  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [expertMessages, translating]);

  const chooseTab = (next: string) => {
    setTab(next);
    window.history.replaceState(null, "", `#${next}`);
  };

  const boRows = useMemo(() => (state?.history || []).filter((r) => r.iter >= 0), [state]);
  const bestRows = useMemo(() => boRows.filter((r) => typeof r.best_so_far === "number"), [boRows]);

  const run = async (steps: number) => {
    setBusy(true); setError(null);
    try {
      const next = await api<AppState>("/api/run", { method: "POST", body: JSON.stringify({ steps, settings: settingsDraft }) });
      setState(next);
      setSurfaceBundle(await api<SurfaceBundle>("/api/prior-surfaces?n_grid=44"));
    } catch (e) { setError(String((e as Error).message || e)); } finally { setBusy(false); }
  };

  const reset = async () => {
    setBusy(true); setError(null);
    try {
      const next = await api<AppState>("/api/reset", { method: "POST", body: JSON.stringify({ settings: settingsDraft }) });
      setState(next);
      setSurfaceBundle(await api<SurfaceBundle>("/api/prior-surfaces?n_grid=44"));
    } catch (e) { setError(String((e as Error).message || e)); } finally { setBusy(false); }
  };

  const applyReadout = async () => {
    setBusy(true); setError(null);
    try {
      const parsed = JSON.parse(readoutText);
      const next = await api<AppState>("/api/readout", { method: "PUT", body: JSON.stringify({ readout: parsed }) });
      setState(next);
      setSurfaceBundle(await api<SurfaceBundle>("/api/prior-surfaces?n_grid=44"));
    } catch (e) { setError(String((e as Error).message || e)); } finally { setBusy(false); }
  };

  const addExpertKnowledge = () => {
    const text = expertText.trim();
    if (!text) return;
    const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    setExpertMessages((items) => [
      ...items,
      { role: "expert", text, time },
      { role: "hilo", text: "Captured. Hit “Claude prior” to translate the transcript into a structured HILO JSON prior.", time }
    ]);
    setExpertText("");
  };

  const clearChat = async () => {
    setBusy(true); setError(null);
    try {
      const next = await api<AppState>("/api/clear-readout", { method: "POST" });
      setState(next);
      setReadoutText(JSON.stringify(next.readout, null, 2));
      setSurfaceBundle(await api<SurfaceBundle>("/api/prior-surfaces?n_grid=44"));
    } catch (e) {
      setError(String((e as Error).message || e));
    } finally {
      setBusy(false);
    }
    setExpertText("");
    setExpertMessages([
      {
        role: "hilo",
        text: "Cleared. The transcript and the active prior are both reset to neutral — start fresh.",
        time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      }
    ]);
  };

  const translateExpertKnowledge = async () => {
    if (!state) return;
    setBusy(true); setTranslating(true); setError(null);
    const transcript = [...expertMessages.filter((m) => m.role === "expert").map((m) => m.text), expertText].join("\n");
    try {
      const translated = await api<{ readout: Record<string, unknown> }>("/api/translate-readout", {
        method: "POST",
        body: JSON.stringify({ transcript, model: llmModel, temperature: 0.0, api_key: llmApiKey.trim() || null })
      });
      setReadoutText(JSON.stringify(translated.readout, null, 2));
      setExpertMessages((items) => [
        ...items,
        { role: "hilo", text: `${llmModel.includes("opus") ? "Opus" : "Haiku"} drafted a structured prior. Review it in Prior Studio, then Apply before running the next BO steps.`, time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) }
      ]);
      chooseTab("prior");
    } catch (e) {
      const message = String((e as Error).message || e);
      setError(message);
      const fallback = draftReadoutFromExpertText(transcript, state);
      setReadoutText(JSON.stringify(fallback, null, 2));
      setExpertMessages((items) => [
        ...items,
        { role: "hilo", text: `Claude translation failed: ${message}. I placed a conservative local fallback draft in Prior Studio.`, time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) }
      ]);
      chooseTab("prior");
    } finally { setBusy(false); setTranslating(false); }
  };

  const updateSetting = (key: string, value: string) => {
    const old = settingsDraft[key];
    const parsed = typeof old === "number" ? Number(value) : value;
    setSettingsDraft((s) => ({ ...s, [key]: Number.isNaN(parsed) ? 0 : parsed }));
  };

  // animated KPI values
  const animBest = useAnimatedNumber(state?.summary.best_yield ?? null);
  const animLatest = useAnimatedNumber(state?.summary.latest_yield ?? null);
  const animAwcd = useAnimatedNumber(state?.summary.awcd_score ?? null);

  if (!state) {
    return (
      <>
        <div className="aurora"><i /></div>
        <main className="boot">
          <div className="spinner" />
          <p>Booting HILO console…</p>
          {error && <pre>{error}</pre>}
        </main>
      </>
    );
  }

  const summary = state.summary;
  const latest = state.history.at(-1);
  const cUser = summary.c_user;

  // derived diagnostics
  const bestSeries = bestRows.map((r) => Number(r.best_so_far));
  const awcdSeries = boRows.map((r) => Number(r.awcd_score ?? 0));
  const weightSeries = boRows.map((r) => Number(r.weight_believer ?? 0));
  const yieldSeries = boRows.map((r) => Number(r.y ?? 0));
  const bestDelta = bestSeries.length > 1 ? bestSeries[bestSeries.length - 1] - bestSeries[bestSeries.length - 2] : 0;
  const recentImprove = bestSeries.length > 5 ? bestSeries[bestSeries.length - 1] - bestSeries[bestSeries.length - 6] : (bestSeries.length ? bestSeries[bestSeries.length - 1] - bestSeries[0] : 0);
  const gateMargin = summary.awcd_score === null ? null : cUser - summary.awcd_score;
  const priorOnFrac = boRows.length ? boRows.filter((r) => r.prior_active).length / boRows.length : 0;
  // exploration spread: std of last-up-to-8 BO points across normalized params
  const explSpread = (() => {
    const recent = boRows.slice(-8);
    if (recent.length < 2) return 0;
    const ranges = state.domain.ranges;
    let acc = 0; let cnt = 0;
    ranges.forEach((rg, di) => {
      const key = `x${di + 1}`;
      const vals = recent.map((r) => Number((r as any)[key])).filter(Number.isFinite).map((v) => (v - rg.min) / Math.max(1e-9, rg.max - rg.min));
      if (vals.length < 2) return;
      const m = vals.reduce((a, b) => a + b, 0) / vals.length;
      const sd = Math.sqrt(vals.reduce((a, b) => a + (b - m) ** 2, 0) / vals.length);
      acc += sd; cnt++;
    });
    return cnt ? acc / cnt : 0;
  })();
  const latestConstraint = boRows.at(-1)?.awcd_constraint ?? null;
  const latestDisagree = boRows.at(-1)?.awcd_mean_disagree ?? null;

  const suggestions = [
    "high isocyanide is beneficial for yield",
    "avoid pTSA below 0.05",
    "very low amine performs poorly",
    "mid-range aldehyde looks best"
  ];

  const expertPanel = (
    <section className="panel chatPanel">
      <div className="panelHead">
        <h2><BrainCircuit size={16} /> Expert knowledge channel</h2>
        <div className="headActions">
          <span className="tag">{llmModel.includes("opus") ? "Claude Opus" : "Claude Haiku"}</span>
          <button className="ghost clearBtn" disabled={busy} onClick={clearChat} title="Forget transcript and reset the active prior to neutral">
            <Trash2 size={14} /> Clear
          </button>
        </div>
      </div>
      <div className="llmControls">
        <label>
          <span>Translator</span>
          <select value={llmModel} onChange={(e) => setLlmModel(e.target.value)}>
            <option value="claude-opus-4-5-20251101">Claude Opus 4.5</option>
            <option value="claude-haiku-4-5-20251001">Claude Haiku 4.5</option>
          </select>
        </label>
        <label>
          <span>Anthropic key</span>
          <input type="password" value={llmApiKey} onChange={(e) => setLlmApiKey(e.target.value)} placeholder="optional · uses env if set" />
        </label>
      </div>
      <div className="messageList">
        {expertMessages.map((msg, index) => (
          <div className={`message ${msg.role}`} key={`${msg.role}-${index}`}>
            <div className="messageMeta">
              {msg.role === "expert" ? <><User size={13} /> Human expert</> : <><Bot size={13} /> HILO</>} · {msg.time}
            </div>
            <div>{msg.text}</div>
          </div>
        ))}
        {translating && (
          <div className="message hilo">
            <div className="messageMeta"><Bot size={13} /> HILO</div>
            <div className="typing"><span /><span /><span /></div>
          </div>
        )}
        <div ref={messageEndRef} />
      </div>
      <div>
        <div className="suggestChips">
          {suggestions.map((s) => (
            <button key={s} className="ghost" onClick={() => setExpertText((t) => (t ? t + "; " + s : s))}>{s}</button>
          ))}
        </div>
        <div className="composer">
          <textarea
            value={expertText}
            onChange={(e) => setExpertText(e.target.value)}
            onKeyDown={(e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") addExpertKnowledge(); }}
            placeholder="e.g. high isocyanide is beneficial; very low pTSA seems poor; avoid strongly imbalanced stoichiometry…"
          />
          <div className="composerActions">
            <span className="grow">⌘/Ctrl + ⏎ to add</span>
            <button onClick={addExpertKnowledge}><Send size={15} /> Add</button>
            <button className="primary" disabled={busy} onClick={translateExpertKnowledge}><Sparkles size={15} /> Claude prior</button>
          </div>
        </div>
      </div>
    </section>
  );

  return (
    <>
      <div className="aurora"><i /></div>
      {busy && <div className="topProgress"><i /></div>}
      <main className="appShell">
        <aside className="sidebar">
          <div className="brand">
            <div className="brandMark"><Atom size={22} /></div>
            <div>
              <h1>HILO</h1>
              <p>Human-in-the-loop BO · UGI</p>
            </div>
          </div>

          <nav className="nav">
            <button className={tab === "run" ? "active" : ""} onClick={() => chooseTab("run")}><LineChart size={16} /> Run</button>
            <button className={tab === "prior" ? "active" : ""} onClick={() => chooseTab("prior")}><Layers size={16} /> Prior studio</button>
            <button className={tab === "safety" ? "active" : ""} onClick={() => chooseTab("safety")}><Shield size={16} /> Safety</button>
            <button className={tab === "data" ? "active" : ""} onClick={() => chooseTab("data")}><Database size={16} /> Data</button>
            <div className="navDivider"><span>General tool</span></div>
            <button className={tab === "asktell" ? "active" : ""} onClick={() => chooseTab("asktell")}><ClipboardList size={16} /> Ask–Tell</button>
          </nav>

          {tab === "asktell" ? (
            <section className="sideSection">
              <h2><ClipboardList size={14} /> Ask–Tell</h2>
              <p className="sideNote">A domain-agnostic optimizer: define your own parameters and objective, run the suggested batch, feed back the measured results, and get the next batch.</p>
            </section>
          ) : (
            <>
              <section className="sideSection">
                <h2><Settings2 size={14} /> Campaign</h2>
                {["n_init", "seed", "constraint_hardness", "constraint_pool_size", "c_user", "awcd_top_frac", "awcd_warmup", "awcd_window"].map((key) => (
                  <label key={key}>
                    <span>{key}</span>
                    <input value={String(settingsDraft[key] ?? "")} onChange={(e) => updateSetting(key, e.target.value)} />
                  </label>
                ))}
              </section>

              <section className="sideSection">
                <h2><Beaker size={14} /> Domain</h2>
                {state.domain.ranges.map((r) => (
                  <div className="range" key={r.name}>
                    <span>{PARAM_LABELS[r.name] || r.name}</span>
                    <small>{fmt(r.min, 2)} – {fmt(r.max, 2)}</small>
                  </div>
                ))}
              </section>
            </>
          )}
        </aside>

        <section className="workspace">
          <header className="topbar">
            <div>
              <h1>{tab === "asktell" ? "Ask–Tell optimizer" : "Ugi reaction optimization"}</h1>
              <p>{tab === "asktell" ? "Domain-agnostic, manual human-in-the-loop optimization" : "Natural-language prior shaping with AWCD safety gating"}</p>
            </div>
            <div className="topbarRight">
              <button className="ghost themeToggle" onClick={toggleTheme} title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`} aria-label="Toggle theme">
                {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
                {theme === "dark" ? "Light" : "Dark"}
              </button>
              {tab !== "asktell" && (
                <div className={`status ${statusClass(summary.status)}`}>
                  <span className="dot" />
                  {summary.status === "prior_gated" ? <AlertTriangle size={15} /> : <Activity size={15} />}
                  {statusText(summary.status)}
                </div>
              )}
            </div>
          </header>

          {error && <div className="errorBox"><AlertTriangle size={16} /> {error}</div>}

          {tab !== "asktell" && (
            <section className="kpiGrid">
              <Kpi label="Iteration" icon={<Waypoints size={13} />} accent="var(--grad-violet)"
                value={String(summary.iteration)} note={`${summary.n_observations} observations`} />
              <Kpi label="Best yield" icon={<Target size={13} />} accent="var(--grad-primary)"
                value={fmt(animBest)} note="oracle estimate" spark={bestSeries} sparkColor="#2ee6a6" delta={bestDelta} />
              <Kpi label="Latest yield" icon={<FlaskConical size={13} />} accent="linear-gradient(135deg,#5b8cff,#22d3ee)"
                value={fmt(animLatest)} note={latest ? `pTSA ${fmt(latest.x4, 3)}` : "waiting"} spark={yieldSeries} sparkColor="#5b8cff" />
              <Kpi label="AWCD score" icon={<Shield size={13} />} accent="var(--grad-warm)"
                value={fmt(animAwcd)} note={`gate at c_user ${fmt(cUser, 2)}`} spark={awcdSeries} sparkColor="#fbbf24" />
              <Kpi label="Prior weight" icon={<Zap size={13} />} accent={summary.prior_active ? "var(--grad-primary)" : "rgba(139,147,167,0.6)"}
                value={summary.prior_active ? "ON" : "OFF"} note={`${(priorOnFrac * 100).toFixed(0)}% of iters trusted`} spark={weightSeries} sparkColor="#a78bfa" />
            </section>
          )}

          {tab !== "asktell" && (
            <section className="toolbar">
              <button className="primary" disabled={busy} onClick={() => run(1)}><Play size={15} /> Step</button>
              <button disabled={busy} onClick={() => run(5)}>Run 5</button>
              <button disabled={busy} onClick={() => run(10)}>Run 10</button>
              <button disabled={busy} onClick={() => run(20)}>Run 20</button>
              <button disabled={busy} onClick={() => run(30)}>Run 30</button>
              <button disabled={busy} onClick={reset}><RotateCcw size={15} /> Reset</button>
              <span className="spacer" />
              <span className="hint"><Activity size={13} /> {busy ? "optimizing…" : "ready"}</span>
            </section>
          )}

          {tab === "asktell" && <AskTellView />}

          {tab === "run" && (
            <div className="contentGrid viewEnter">
              <section className="panel wide">
                <div className="panelHead">
                  <h2><TrendingUp size={16} /> Best-so-far trajectory</h2>
                  <span className="tag">{boRows.length} BO iterations</span>
                </div>
                <LineChartSvg rows={bestRows} yKey="best_so_far" color="#2ee6a6" yLabel="Best yield" height={290} />
              </section>

              {expertPanel}

              <section className="panel fullSpan">
                <div className="panelHead">
                  <h2><Gauge size={16} /> Optimization diagnostics</h2>
                  <span className="tag">live campaign telemetry</span>
                </div>
                <div className="diagGrid">
                  <DiagCard icon={<TrendingUp size={14} />} title="Improvement (last 5)" value={`${recentImprove >= 0 ? "+" : ""}${fmt(recentImprove, 3)}`}
                    sub="best-yield gain over recent steps" />
                  <DiagCard icon={<Shield size={14} />} title="Gate margin" value={gateMargin === null ? "--" : fmt(gateMargin, 3)}
                    sub={gateMargin === null ? "no AWCD yet" : gateMargin >= 0 ? "headroom below c_user" : "over threshold — gated"}
                    meter={gateMargin === null ? 0 : Math.max(0, Math.min(1, gateMargin / Math.max(1e-6, cUser)))}
                    meterClass={gateMargin !== null && gateMargin < 0 ? "danger" : gateMargin !== null && gateMargin < cUser * 0.25 ? "warn" : ""} />
                  <DiagCard icon={<Crosshair size={14} />} title="Exploration spread" value={fmt(explSpread, 3)}
                    sub="std of last 8 picks (normalized)" meter={Math.min(1, explSpread / 0.4)} />
                  <DiagCard icon={<Zap size={14} />} title="Prior trust" value={`${(priorOnFrac * 100).toFixed(0)}%`}
                    sub="share of iterations using the prior" meter={priorOnFrac} />
                </div>
              </section>

              <section className="panel fullSpan">
                <div className="panelHead">
                  <h2><Layers size={16} /> Pairwise prior surfaces</h2>
                  <span className="tag">observations overlaid · other dims at midpoint</span>
                </div>
                <HeatmapGrid bundle={surfaceBundle} history={state.history} />
              </section>

              <section className="panel fullSpan">
                <div className="panelHead">
                  <h2><Beaker size={16} /> Parameter–yield relationships</h2>
                  <span className="tag">gray = initial · blue = BO</span>
                </div>
                <div className="smallChartGrid">
                  <div><h3>x1 <small>amine</small></h3><ScatterSvg rows={state.history} xKey="x1" xLabel="amine_mM" /></div>
                  <div><h3>x2 <small>aldehyde</small></h3><ScatterSvg rows={state.history} xKey="x2" xLabel="aldehyde_mM" /></div>
                  <div><h3>x3 <small>isocyanide</small></h3><ScatterSvg rows={state.history} xKey="x3" xLabel="isocyanide_mM" /></div>
                  <div><h3>x4 <small>pTSA</small></h3><ScatterSvg rows={state.history} xKey="x4" xLabel="ptsa" color="#a78bfa" /></div>
                </div>
              </section>

              <section className="panel fullSpan">
                <div className="panelHead">
                  <h2><ShieldAlert size={16} /> Safety & prior status</h2>
                  <span className="tag">AWCD constraint {latestConstraint !== null ? fmt(latestConstraint, 2) : "--"} · disagree {latestDisagree !== null ? fmt(latestDisagree, 2) : "--"}</span>
                </div>
                <div className="smallChartGrid two">
                  <div><h3>AWCD score <small>vs c_user gate</small></h3>
                    <LineChartSvg rows={boRows} yKey="awcd_score" color="#fbbf24" yLabel="AWCD" height={200} threshold={cUser} /></div>
                  <div><h3>Prior weight <small>believer contribution</small></h3>
                    <LineChartSvg rows={boRows} yKey="weight_believer" color="#a78bfa" yLabel="Weight" height={200} /></div>
                </div>
              </section>
            </div>
          )}

          {tab === "prior" && (
            <div className="priorLayout viewEnter">
              <section className="panel editorPanel">
                <div className="panelHead">
                  <h2><FlaskConical size={16} /> JSON prior</h2>
                  <button className="primary" disabled={busy} onClick={applyReadout}><ChevronRight size={14} /> Apply</button>
                </div>
                <ReadoutChips readout={state.readout} />
                <textarea value={readoutText} onChange={(e) => setReadoutText(e.target.value)} spellCheck={false} />
              </section>
              <section className="panel">
                <div className="panelHead">
                  <h2><Layers size={16} /> Prior surfaces</h2>
                  <span className="tag">all parameter pairs</span>
                </div>
                <HeatmapGrid bundle={surfaceBundle} history={state.history} />
              </section>
            </div>
          )}

          {tab === "safety" && (
            <div className="contentGrid viewEnter">
              <section className="panel wide">
                <div className="panelHead">
                  <h2><Shield size={16} /> AWCD score</h2>
                  <span className="tag">guard telemetry · gate at {fmt(cUser, 2)}</span>
                </div>
                <LineChartSvg rows={boRows} yKey="awcd_score" color="#fbbf24" yLabel="AWCD" height={290} threshold={cUser} />
              </section>
              <section className="panel">
                <div className="panelHead">
                  <h2><Zap size={16} /> Prior weight</h2>
                  <span className="tag">believer contribution</span>
                </div>
                <LineChartSvg rows={boRows} yKey="weight_believer" color="#a78bfa" yLabel="Weight" height={290} />
              </section>
              <section className="panel fullSpan">
                <div className="panelHead">
                  <h2><Gauge size={16} /> Guard breakdown</h2>
                  <span className="tag">constraint pressure vs prior–skeptic disagreement</span>
                </div>
                <div className="smallChartGrid two">
                  <div><h3>AWCD constraint <small>forbidden-region pressure</small></h3>
                    <LineChartSvg rows={boRows} yKey="awcd_constraint" color="#fb7185" yLabel="constraint" height={200} threshold={cUser} /></div>
                  <div><h3>AWCD mean disagreement <small>prior vs skeptic GP</small></h3>
                    <LineChartSvg rows={boRows} yKey="awcd_mean_disagree" color="#22d3ee" yLabel="disagree" height={200} threshold={cUser} /></div>
                </div>
              </section>
            </div>
          )}

          {tab === "data" && (
            <section className="panel viewEnter">
              <div className="panelHead">
                <h2><Database size={16} /> Campaign history</h2>
                <span className="tag">{state.history.length} rows</span>
              </div>
              <FullTable rows={state.history} />
            </section>
          )}
        </section>
      </main>
    </>
  );
}

function FullTable({ rows }: { rows: HistoryRow[] }) {
  if (!rows.length) return <div className="empty">No data yet — Reset to draw initial points, then Run.</div>;
  return (
    <div className="tableScroll">
      <table className="miniTable full">
        <thead>
          <tr>
            {["iter", "method", "y", "best", "x1", "x2", "x3", "x4", "awcd", "weight", "prior"].map((h) => <th key={h}>{h}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.slice().reverse().map((r, i) => (
            <tr key={`${r.method}-${r.iter}-${i}`}>
              <td>{r.iter}</td>
              <td><span className={`badge ${r.method === "hilo" ? "hilo" : "init"}`}>{r.method}</span></td>
              <td>{fmt(r.y)}</td>
              <td>{fmt(r.best_so_far)}</td>
              <td>{fmt(r.x1, 2)}</td>
              <td>{fmt(r.x2, 2)}</td>
              <td>{fmt(r.x3, 2)}</td>
              <td>{fmt(r.x4, 3)}</td>
              <td>{fmt(r.awcd_score)}</td>
              <td>{fmt(r.weight_believer, 2)}</td>
              <td><span className={`badge ${r.prior_active ? "on" : "off"}`}>{r.prior_active ? "on" : "off"}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ============================================================ ASK-TELL ===== */
function AskTellView() {
  const now = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const [data, setData] = useState<AskTellData | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // setup form (prefilled with a practical generic example)
  const [objective, setObjective] = useState("Yield (%)");
  const [goal, setGoal] = useState<"maximize" | "minimize">("maximize");
  const [params, setParams] = useState<AskTellParam[]>([
    { name: "Temperature", min: 40, max: 120, unit: "°C" },
    { name: "Time", min: 1, max: 24, unit: "h" },
    { name: "Catalyst loading", min: 0.5, max: 10, unit: "mol%" },
    { name: "Concentration", min: 0.05, max: 1, unit: "M" }
  ]);
  const [batchSize, setBatchSize] = useState(3);
  const [nInit, setNInit] = useState(5);
  const [seed, setSeed] = useState(0);
  const [results, setResults] = useState<Record<number, string>>({});

  // human prior shaping
  const [readoutText, setReadoutText] = useState("{}");
  const [bundle, setBundle] = useState<SurfaceBundle | null>(null);
  const [expertText, setExpertText] = useState("");
  const [llmModel, setLlmModel] = useState("claude-opus-4-5-20251101");
  const [llmApiKey, setLlmApiKey] = useState("");
  const [translating, setTranslating] = useState(false);
  const [messages, setMessages] = useState<ExpertMessage[]>(() => {
    const welcome: ExpertMessage = { role: "hilo", text: "Tell me what you already know about this system in plain language — which settings tend to help or hurt, and any ranges to avoid. I'll turn it into a prior that steers the suggested batches.", time: now() };
    try {
      if (new URLSearchParams(window.location.search).get("demo") === "chat") {
        return [
          welcome,
          { role: "expert", text: "From past runs on this coupling: higher temperature clearly helps the yield, at least up to ~110 °C. Reaction time has a sweet spot around 12–18 h — longer than that and it starts to degrade. And too much catalyst hurts, so avoid loadings above ~8 mol%.", time: "10:24" },
          { role: "hilo", text: "Captured the transcript. Building a structured prior from it with Claude Opus…", time: "10:24" },
          { role: "hilo", text: "Drafted and applied a prior: Temperature → higher is better, Time → peak around 12–18 h, and a soft constraint avoiding catalyst above 8 mol%. It now steers the next suggested batch — review or edit it on the right.", time: "10:25" }
        ];
      }
    } catch { /* ignore */ }
    return [welcome];
  });
  const endRef = useRef<HTMLDivElement>(null);

  const refresh = async (next?: AskTellData, syncEditor = true) => {
    const d = next ?? (await api<AskTellData>("/api/asktell/state"));
    setData(d);
    if (d.configured) {
      if (syncEditor) setReadoutText(JSON.stringify(d.readout ?? {}, null, 2));
      try { setBundle(await api<SurfaceBundle>("/api/asktell/prior-surfaces?n_grid=40")); } catch { /* ignore */ }
    }
  };

  useEffect(() => { refresh().catch((e) => setErr(String(e.message || e))); }, []);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [messages, translating]);

  const updateParam = (i: number, key: keyof AskTellParam, val: string) =>
    setParams((ps) => ps.map((p, j) => (j === i ? { ...p, [key]: key === "name" || key === "unit" ? val : Number(val) } : p)));
  const addParam = () => setParams((ps) => [...ps, { name: "", min: 0, max: 1, unit: "" }]);
  const removeParam = (i: number) => setParams((ps) => ps.filter((_, j) => j !== i));

  const start = async () => {
    setBusy(true); setErr(null);
    try {
      const next = await api<AskTellData>("/api/asktell/init", {
        method: "POST",
        body: JSON.stringify({ objective_name: objective, goal, parameters: params, batch_size: batchSize, n_init: nInit, seed })
      });
      await refresh(next); setResults({});
    } catch (e) { setErr(String((e as Error).message || e)); } finally { setBusy(false); }
  };
  const submit = async () => {
    if (!data?.pending) return;
    setBusy(true); setErr(null);
    try {
      const payload = data.pending.map((p) => ({ x: p.x, y: results[p.id] ?? "" }));
      const next = await api<AskTellData>("/api/asktell/tell", { method: "POST", body: JSON.stringify({ results: payload }) });
      await refresh(next, false); setResults({});
    } catch (e) { setErr(String((e as Error).message || e)); } finally { setBusy(false); }
  };
  const reSuggest = async () => {
    setBusy(true); setErr(null);
    try { await refresh(await api<AskTellData>("/api/asktell/suggest", { method: "POST" }), false); }
    catch (e) { setErr(String((e as Error).message || e)); } finally { setBusy(false); }
  };
  const newCampaign = async () => {
    setBusy(true); setErr(null);
    try {
      setData(await api<AskTellData>("/api/asktell/reset", { method: "POST" }));
      setBundle(null); setResults({}); setReadoutText("{}");
      setMessages([{ role: "hilo", text: "New campaign. Define your problem, then tell me what you know.", time: now() }]);
    } catch (e) { setErr(String((e as Error).message || e)); } finally { setBusy(false); }
  };

  const addKnowledge = () => {
    const t = expertText.trim();
    if (!t) return;
    setMessages((m) => [...m, { role: "expert", text: t, time: now() },
      { role: "hilo", text: "Captured. Hit “Build prior” to translate the transcript into a prior that steers the suggestions.", time: now() }]);
    setExpertText("");
  };
  const buildPrior = async () => {
    setBusy(true); setTranslating(true); setErr(null);
    const transcript = [...messages.filter((m) => m.role === "expert").map((m) => m.text), expertText].join("\n").trim();
    if (!transcript) { setBusy(false); setTranslating(false); return; }
    try {
      const { readout } = await api<{ readout: Record<string, unknown> }>("/api/asktell/translate", {
        method: "POST",
        body: JSON.stringify({ transcript, model: llmModel, temperature: 0.0, api_key: llmApiKey.trim() || null })
      });
      const next = await api<AskTellData>("/api/asktell/readout", { method: "PUT", body: JSON.stringify({ readout }) });
      await refresh(next, true); setExpertText("");
      setMessages((m) => [...m, { role: "hilo", text: `${llmModel.includes("opus") ? "Opus" : "Haiku"} drafted and applied a prior. It now steers the next suggested batch — review or edit it on the right, and check the surfaces below.`, time: now() }]);
    } catch (e) {
      const msg = String((e as Error).message || e);
      setErr(msg);
      setMessages((m) => [...m, { role: "hilo", text: `Could not build the prior: ${msg}. You can still edit the JSON prior directly on the right.`, time: now() }]);
    } finally { setBusy(false); setTranslating(false); }
  };
  const applyReadout = async () => {
    setBusy(true); setErr(null);
    try {
      const parsed = JSON.parse(readoutText);
      await refresh(await api<AskTellData>("/api/asktell/readout", { method: "PUT", body: JSON.stringify({ readout: parsed }) }), true);
    } catch (e) { setErr(String((e as Error).message || e)); } finally { setBusy(false); }
  };
  const clearPrior = async () => {
    setBusy(true); setErr(null);
    try { await refresh(await api<AskTellData>("/api/asktell/clear-readout", { method: "POST" }), true); }
    catch (e) { setErr(String((e as Error).message || e)); } finally { setBusy(false); }
  };

  if (!data) return <div className="empty">Loading optimizer…</div>;

  // ---- setup screen -------------------------------------------------------
  if (!data.configured) {
    return (
      <div className="viewEnter">
        {err && <div className="errorBox"><AlertTriangle size={16} /> {err}</div>}
        <section className="panel atSetupPanel">
          <div className="panelHead">
            <h2><Rocket size={16} /> Define your optimization</h2>
            <span className="tag">domain-agnostic · bring your own experiment</span>
          </div>
          <p className="atLede">
            Describe what you want to optimize and the knobs you can turn. The optimizer proposes a batch of conditions to run in
            the lab; you measure them, type the results back in, and it proposes the next batch — no dataset or simulator required.
          </p>
          <div className="atSetupGrid">
            <label><span>Objective</span><input value={objective} onChange={(e) => setObjective(e.target.value)} placeholder="e.g. Yield (%)" /></label>
            <label><span>Goal</span>
              <select value={goal} onChange={(e) => setGoal(e.target.value as "maximize" | "minimize")}>
                <option value="maximize">Maximize</option>
                <option value="minimize">Minimize</option>
              </select>
            </label>
            <label><span>Batch size</span><input type="number" value={batchSize} min={1} max={12} onChange={(e) => setBatchSize(Number(e.target.value))} /></label>
            <label><span>Initial points</span><input type="number" value={nInit} min={1} max={48} onChange={(e) => setNInit(Number(e.target.value))} /></label>
            <label><span>Seed</span><input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} /></label>
          </div>

          <div className="atParamHead">
            <h3><SlidersIcon /> Parameters</h3>
            <button className="ghost" onClick={addParam}><Plus size={14} /> Add parameter</button>
          </div>
          <div className="atParamTable">
            <div className="atParamRow atParamHeadRow">
              <span>Name</span><span>Min</span><span>Max</span><span>Unit</span><span />
            </div>
            {params.map((p, i) => (
              <div className="atParamRow" key={i}>
                <input value={p.name} onChange={(e) => updateParam(i, "name", e.target.value)} placeholder="parameter" />
                <input type="number" value={p.min} onChange={(e) => updateParam(i, "min", e.target.value)} />
                <input type="number" value={p.max} onChange={(e) => updateParam(i, "max", e.target.value)} />
                <input value={p.unit ?? ""} onChange={(e) => updateParam(i, "unit", e.target.value)} placeholder="unit" />
                <button className="ghost iconOnly" onClick={() => removeParam(i)} title="Remove" disabled={params.length <= 1}><Trash2 size={14} /></button>
              </div>
            ))}
          </div>
          <div className="atSetupActions">
            <button className="primary" disabled={busy} onClick={start}><Rocket size={15} /> Start campaign</button>
          </div>
        </section>
      </div>
    );
  }

  // ---- active campaign ----------------------------------------------------
  const pnames = (data.parameters || []).map((p) => p.name);
  const unit = (n: string) => { const p = (data.parameters || []).find((q) => q.name === n); return p?.unit ? ` ${p.unit}` : ""; };
  const seriesRows: HistoryRow[] = (data.series || []).map((p) => ({ iter: p.n - 1, y: p.y, best_so_far: p.best, method: "obs" }));
  const goalLabel = data.goal === "minimize" ? "minimize" : "maximize";
  // map completed runs into HistoryRow shape (x1..xN) so the heatmaps can overlay them
  const heatHistory: HistoryRow[] = (data.history || []).map((h) => {
    const row: Record<string, unknown> = { iter: h.n - 1, y: h.y, best_so_far: null, method: "obs" };
    (data.parameters || []).forEach((p, i) => { row[`x${i + 1}`] = h.x[p.name]; });
    return row as HistoryRow;
  });
  const priorValue = !data.has_prior ? "OFF" : data.prior_active ? "ON" : "GATED";
  const priorNote = data.awcd != null
    ? `AWCD ${fmt(data.awcd, 2)} / c_user ${fmt(data.c_user ?? 0.95, 2)}`
    : data.has_prior ? "evaluated once GP is active" : "no prior set yet";
  const suggestions = pnames.length
    ? [`higher ${pnames[0].toLowerCase()} improves ${(data.objective_name || "the result").toLowerCase()}`,
       `${(pnames[1] || pnames[0]).toLowerCase()} works best in a mid range`,
       `avoid very low ${pnames[pnames.length - 1].toLowerCase()}`]
    : [];

  return (
    <div className="viewEnter">
      {err && <div className="errorBox"><AlertTriangle size={16} /> {err}</div>}

      <section className="kpiGrid atKpis">
        <Kpi label="Round" icon={<Waypoints size={13} />} accent="var(--grad-violet)" value={String(data.round)} note={`${data.batch_size} per batch`} />
        <Kpi label="Observations" icon={<Database size={13} />} accent="linear-gradient(135deg,#5b8cff,#22d3ee)" value={String(data.n_observations)} note={`${data.parameters?.length || 0} parameters`} />
        <Kpi label={`Best ${data.objective_name}`} icon={<Target size={13} />} accent="var(--grad-primary)"
          value={data.best ? fmt(data.best.y, 3) : "--"} note={`goal: ${goalLabel}`} spark={(data.series || []).map((p) => p.best)} sparkColor="#2ee6a6" />
        <Kpi label="Prior" icon={<Zap size={13} />} accent={data.has_prior && data.prior_active ? "var(--grad-primary)" : data.has_prior ? "var(--grad-warm)" : "rgba(139,147,167,0.6)"}
          value={priorValue} note={priorNote} />
      </section>

      <div className="atGrid">
        <section className="panel wide">
          <div className="panelHead">
            <h2><ClipboardList size={16} /> Suggested experiments to run</h2>
            <span className="tag">enter the measured {data.objective_name}, then submit</span>
          </div>
          {data.pending && data.pending.length ? (
            <div className="tableScroll atTableScroll">
              <table className="miniTable atTable">
                <thead>
                  <tr>
                    <th>#</th>
                    {pnames.map((n) => <th key={n}>{n}{unit(n) && <small> /{unit(n).trim()}</small>}</th>)}
                    <th className="atYcol">{data.objective_name}</th>
                  </tr>
                </thead>
                <tbody>
                  {data.pending.map((row, i) => (
                    <tr key={row.id}>
                      <td>{i + 1}</td>
                      {pnames.map((n) => <td key={n}>{fmt(row.x[n], decimalsFor(data.parameters, n))}</td>)}
                      <td className="atYcol">
                        <input className="atYinput" inputMode="decimal" placeholder="measure"
                          value={results[row.id] ?? ""} onChange={(e) => setResults((r) => ({ ...r, [row.id]: e.target.value }))} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <div className="empty">No pending suggestions. Submit results to generate the next batch.</div>}
          <div className="atActions">
            <button className="ghost" disabled={busy} onClick={reSuggest}><RotateCcw size={14} /> Re-suggest batch</button>
            <span className="spacer" />
            <button className="primary" disabled={busy} onClick={submit}><CheckCircle2 size={15} /> Submit results & get next batch <ArrowRight size={14} /></button>
          </div>
        </section>

        <section className="panel">
          <div className="panelHead">
            <h2><TrendingUp size={16} /> Best so far</h2>
            <span className="tag">{data.n_observations} runs</span>
          </div>
          {seriesRows.length ? (
            <LineChartSvg rows={seriesRows} yKey="best_so_far" color="#2ee6a6" yLabel={data.objective_name || "objective"} height={230} />
          ) : <div className="empty">Run the first batch to see progress.</div>}
          {data.best && (
            <div className="atBest">
              <div className="atBestVal"><Target size={15} /> {fmt(data.best.y, 3)} <small>best {data.objective_name}</small></div>
              <div className="atBestCond">
                {pnames.map((n) => <span key={n} className="chip">{n} {fmt(data.best!.x[n], decimalsFor(data.parameters, n))}{unit(n)}</span>)}
              </div>
            </div>
          )}
        </section>
      </div>

      <section className="panel fullSpan">
        <div className="panelHead">
          <h2><Database size={16} /> Completed experiments</h2>
          <span className="tag">{data.history?.length || 0} rows</span>
        </div>
        {data.history && data.history.length ? (
          <div className="tableScroll">
            <table className="miniTable">
              <thead>
                <tr><th>#</th>{pnames.map((n) => <th key={n}>{n}</th>)}<th>{data.objective_name}</th></tr>
              </thead>
              <tbody>
                {data.history.slice().reverse().map((r) => {
                  const isBest = data.best && Math.abs(r.y - data.best.y) < 1e-9;
                  return (
                    <tr key={r.n}>
                      <td>{r.n}</td>
                      {pnames.map((n) => <td key={n}>{fmt(r.x[n], decimalsFor(data.parameters, n))}</td>)}
                      <td>{isBest ? <span className="badge hilo">{fmt(r.y, 3)}</span> : fmt(r.y, 3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : <div className="empty">No completed experiments yet.</div>}
        <div className="atActions">
          <span className="spacer" />
          <button className="ghost" disabled={busy} onClick={newCampaign}><Trash2 size={14} /> New campaign</button>
        </div>
      </section>

      <section className="panel fullSpan">
        <div className="panelHead">
          <h2><BrainCircuit size={16} /> Shape the prior from your knowledge</h2>
          <span className="tag">optional · natural language → structured prior → steers the suggestions</span>
        </div>
        <div className="atPriorGrid">
          <div className="atChat">
            <div className="llmControls">
              <label><span>Translator</span>
                <select value={llmModel} onChange={(e) => setLlmModel(e.target.value)}>
                  <option value="claude-opus-4-5-20251101">Claude Opus 4.5</option>
                  <option value="claude-haiku-4-5-20251001">Claude Haiku 4.5</option>
                </select>
              </label>
              <label><span>Anthropic key</span>
                <input type="password" value={llmApiKey} onChange={(e) => setLlmApiKey(e.target.value)} placeholder="optional · uses env if set" />
              </label>
            </div>
            <div className="messageList">
              {messages.map((m, i) => (
                <div className={`message ${m.role}`} key={i}>
                  <div className="messageMeta">{m.role === "expert" ? <><User size={13} /> You</> : <><Bot size={13} /> HILO</>} · {m.time}</div>
                  <div>{m.text}</div>
                </div>
              ))}
              {translating && <div className="message hilo"><div className="messageMeta"><Bot size={13} /> HILO</div><div className="typing"><span /><span /><span /></div></div>}
              <div ref={endRef} />
            </div>
            <div className="suggestChips">
              {suggestions.map((s) => <button key={s} className="ghost" onClick={() => setExpertText((t) => (t ? t + "; " + s : s))}>{s}</button>)}
            </div>
            <div className="composer">
              <textarea value={expertText} onChange={(e) => setExpertText(e.target.value)}
                onKeyDown={(e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") addKnowledge(); }}
                placeholder={`e.g. ${pnames[0] ? `higher ${pnames[0].toLowerCase()} helps; ` : ""}avoid extreme values; there's a sweet spot in the middle…`} />
              <div className="composerActions">
                <span className="grow">⌘/Ctrl + ⏎ to add</span>
                <button onClick={addKnowledge}><Send size={15} /> Add</button>
                <button className="primary" disabled={busy} onClick={buildPrior}><Sparkles size={15} /> Build prior</button>
              </div>
            </div>
          </div>

          <div className="atPriorEdit editorPanel">
            <div className="miniHead">
              <span>Structured prior</span>
              <div className="miniHeadBtns">
                <button className="ghost clearBtn" disabled={busy} onClick={clearPrior}><Trash2 size={14} /> Clear</button>
                <button className="primary" disabled={busy} onClick={applyReadout}><ChevronRight size={14} /> Apply</button>
              </div>
            </div>
            <ReadoutChips readout={data.readout || {}} />
            <textarea className="atReadout" value={readoutText} onChange={(e) => setReadoutText(e.target.value)} spellCheck={false} />
          </div>
        </div>
      </section>

      <section className="panel fullSpan">
        <div className="panelHead">
          <h2><Layers size={16} /> Prior surfaces</h2>
          <span className="tag">how your knowledge shapes preference across the space · runs overlaid</span>
        </div>
        <HeatmapGrid bundle={bundle} history={heatHistory} />
      </section>
    </div>
  );
}

function SlidersIcon() {
  return <Settings2 size={15} />;
}

function decimalsFor(params: AskTellParam[] | undefined, name: string): number {
  const p = (params || []).find((q) => q.name === name);
  if (!p) return 3;
  const span = Math.abs(p.max - p.min);
  if (span < 1) return 4;
  if (span < 10) return 3;
  if (span < 100) return 2;
  return 1;
}

createRoot(document.getElementById("root")!).render(<App />);
