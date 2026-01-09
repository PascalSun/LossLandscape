
'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import MetricChart from './MetricChart';
import { useTheme } from '../theme';
import * as Tooltip from '@radix-ui/react-tooltip';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  ReferenceLine
} from 'recharts';

// --- KDE Helpers ---
function kernelDensityEstimator(kernel: (u: number) => number, X: number[]) {
  return function (x: number) {
    let sum = 0;
    for (let i = 0; i < X.length; i++) {
      sum += kernel(x - X[i]);
    }
    return sum / X.length;
  };
}

function gaussianKernel(bandwidth: number) {
  return function (u: number) {
    return (1 / (Math.sqrt(2 * Math.PI) * bandwidth)) * Math.exp(-0.5 * (u / bandwidth) ** 2);
  };
}

function computeBandwidth(data: number[]) {
  const n = data.length;
  if (n === 0) return 1;
  const mean = data.reduce((a, b) => a + b) / n;
  const std = Math.sqrt(data.map((x) => (x - mean) ** 2).reduce((a, b) => a + b) / n);
  // Scott's Rule
  return 1.06 * std * Math.pow(n, -0.2) || 0.1;
}

interface HessianData {
  epochs: number[];
  max_eigenvalue: (number | null)[];
  trace: (number | null)[];
  top_eigenvalues: (number[] | null)[];
}

interface HessianAnalysisProps {
  data: HessianData;
  metadata?: any;
}

export default function HessianAnalysis({ data, metadata }: HessianAnalysisProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const isSnapshot = (data.epochs?.length || 0) <= 1;
  const [selectedEpochIndex, setSelectedEpochIndex] = useState<number>(
    isSnapshot ? 0 : data.epochs.length - 1
  );
  // Separate slider value so we can debounce expensive recomputations
  const [sliderEpochIndex, setSliderEpochIndex] = useState<number>(
    isSnapshot ? 0 : data.epochs.length - 1
  );
  const sliderDebounceRef = useRef<number | null>(null);

  // If the dataset changes (different run / different epoch count), keep the slider in range.
  useEffect(() => {
    const n = data.epochs?.length || 0;
    if (n <= 0) {
      setSelectedEpochIndex(0);
      setSliderEpochIndex(0);
      return;
    }
    setSelectedEpochIndex((prev) => Math.max(0, Math.min(prev, n - 1)));
    setSliderEpochIndex((prev) => Math.max(0, Math.min(prev, n - 1)));
  }, [data.epochs?.length]);

  const ui = useMemo(() => {
    return {
      bg: 'var(--bg-main)',
      cardBg: 'var(--bg-card)',
      cardBorder: 'var(--border)',
      text: 'var(--text-primary)',
      muted: 'var(--text-muted)',
      header: 'var(--text-primary)',
      accent: 'var(--accent)',
      panelBg: 'var(--bg-glass)',
      panelBorder: 'var(--border)',
      panelShadow: 'var(--shadow-card)',
      divider: 'var(--border)',
      surfaceBorder: 'var(--border)',
      badgeBg: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(15,23,42,0.06)',
    };
  }, [isDark]);

  const currentEpoch = data.epochs[selectedEpochIndex] ?? 0;

  const topEigenvalues = useMemo(() => {
    if (!data.top_eigenvalues) return null;
    return data.top_eigenvalues.map((row) => {
      if (!Array.isArray(row)) return null;
      const vals = row
        .map((v) => Number(v))
        .filter((v) => Number.isFinite(v));
      return vals.length ? vals : null;
    });
  }, [data.top_eigenvalues]);

  // --- 1. Global Metrics Data (MetricChart handled internally) ---
  const lrData = useMemo(() => {
    const lrHist = (metadata?.training_statistics?.learning_rate_history as number[]) || [];
    const baseLr =
      Number(metadata?.training?.optimizer?.learning_rate) ||
      Number(metadata?.training?.optimizer?.lr) ||
      null;

    const getLr = (ep: number, i: number) => {
      if (lrHist.length > 0) return lrHist[Math.min(i, lrHist.length - 1)];
      return baseLr;
    };

    const stabilityBoundary = data.epochs.map((ep, i) => {
      const lr = getLr(ep, i);
      if (!lr) return null;
      return 2.0 / lr;
    });

    return { stabilityBoundary, hasData: stabilityBoundary.some((v) => v !== null) };
  }, [data.epochs, metadata]);

  // --- 2. Density Plot Data (Recharts Area) ---
  const densityChartData = useMemo(() => {
    if (!topEigenvalues) return null;

    // 1. Pick Background Epochs (evolution context)
    const n = data.epochs.length;
    // Select a few fixed epochs for context (Start, Middle, End)
    const contextIndices = new Set([
      0,
      Math.floor(n / 2),
      n - 1
    ]);
    // Always add current selected epoch
    const activeIndex = selectedEpochIndex;
    
    // Combine for range computation
    const allIndices = new Set([...contextIndices, activeIndex]);
    const indices = Array.from(allIndices).sort((a, b) => a - b);

    // 2. Extract Values
    const series = indices.map((idx) => {
      const vals = topEigenvalues[idx];
      if (!vals || vals.length === 0) return null;
      return {
        epoch: data.epochs[idx],
        values: vals.filter((v) => Number.isFinite(v)),
        id: `e${data.epochs[idx]}`,
        isCurrent: idx === activeIndex,
        isContext: contextIndices.has(idx) && idx !== activeIndex
      };
    }).filter((s) => s !== null) as { epoch: number; values: number[]; id: string; isCurrent: boolean; isContext: boolean }[];

    if (series.length === 0) return null;

    // 3. Compute Global Range (X Axis)
    const allVals = series.flatMap((s) => s.values).sort((a, b) => a - b);
    if (allVals.length === 0) return null;
    
    let minX = allVals[0];
    let maxX = allVals[allVals.length - 1];
    // Add padding
    const rangeX = maxX - minX || 1;
    minX -= rangeX * 0.1;
    maxX += rangeX * 0.1;

    // 4. Compute KDE on Grid
    const resolution = 100;
    const grid = Array.from({ length: resolution }, (_, i) => minX + (i / (resolution - 1)) * (maxX - minX));

    // Pre-compute KDE functions
    const kdes = series.map((s) => {
      const bw = computeBandwidth(s.values) * 1.5;
      const fn = kernelDensityEstimator(gaussianKernel(bw), s.values);
      return { ...s, fn };
    });

    // 5. Generate Recharts Data Array
    const chartData = grid.map((x) => {
      const item: any = { x };
      kdes.forEach((s) => {
        item[s.id] = s.fn(x);
      });
      return item;
    });

    return { chartData, series };
  }, [data.epochs, selectedEpochIndex, topEigenvalues]);

  const getSeriesColor = (s: { isCurrent: boolean, epoch: number }) => {
      if (s.isCurrent) return ui.accent; // Highlight current
      // Gradient or fixed for context
      return isDark ? '#475569' : '#cbd5e1'; // Muted for context
  };

  // --- 3. Scree Plot Data (Recharts Bar) ---
  const screeChartData = useMemo(() => {
    const vals = topEigenvalues?.[selectedEpochIndex];
    if (!vals || vals.length === 0) return [];

    // Sort: Largest to Smallest (Algebraic)
    // Most standard for "Spectrum" is usually Magnitude descending, 
    // but for Hessian, sign matters (Negative = saddle).
    // Let's stick to Algebraic (Descending) so Positive is Left, Negative is Right.
    // Wait, typical Scree is Magnitude. But Hessian researchers often care about "Bulk" vs "Outliers".
    // Let's sort Descending (Algebraic): 100, 50, ..., -1, -5.
    const sorted = [...vals].sort((a, b) => b - a);

    return sorted.map((v, i) => ({
      rank: i + 1,
      value: v,
      isPositive: v >= 0
    }));
  }, [selectedEpochIndex, topEigenvalues]);


  // --- Helper Components ---
  const HelpIcon = ({ text }: { text: React.ReactNode }) => {
    const [isVisible, setIsVisible] = useState(false);
    return (
        <div className="relative inline-flex items-center ml-2 z-50">
          <button 
            style={{
              width: 16, height: 16, borderRadius: '50%', 
              border: `1px solid ${ui.cardBorder}`, 
              color: ui.muted, fontSize: 10, 
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              cursor: 'help',
              background: 'transparent'
            }}
            onMouseEnter={() => setIsVisible(true)}
            onMouseLeave={() => setIsVisible(false)}
            onClick={() => setIsVisible(!isVisible)}
          >
            ?
          </button>
          {isVisible && (
            <div style={{
              position: 'absolute', bottom: '100%', left: '50%', transform: 'translateX(-50%)',
              marginBottom: 8, width: 280, padding: 12, fontSize: 12,
              background: isDark ? 'rgba(15, 23, 42, 0.95)' : 'rgba(255, 255, 255, 0.95)',
              color: ui.text, borderRadius: 8, boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
              border: `1px solid ${ui.cardBorder}`, pointerEvents: 'none', lineHeight: 1.5, zIndex: 100
            }}>
              {text}
            </div>
          )}
        </div>
    );
  };

  const containerStyle: React.CSSProperties = {
    height: '100%',
    // overflowY: 'auto', // Handled by parent
    padding: '20px',
    background: ui.bg,
    color: ui.text,
    display: 'flex',
    flexDirection: 'column',
    position: 'relative', // anchor floating monitor
    minHeight: 0,
  };

  const cardStyle: React.CSSProperties = {
    padding: '16px',
    borderRadius: 'var(--radius-lg)',
    border: `1px solid ${ui.cardBorder}`,
    background: ui.cardBg,
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    boxShadow: ui.panelShadow as React.CSSProperties['boxShadow'],
  };

  const sectionHeaderStyle: React.CSSProperties = {
    fontSize: '15px',
    fontWeight: 700,
    color: ui.header,
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    letterSpacing: '0.01em',
  };

  // Show the Spectral column when the run contains eigenvalue data at all.
  // (Even if some epochs are null, we still render the panel with a clear empty-state message.)
  const showSpectralColumn = Array.isArray(data.top_eigenvalues) && data.top_eigenvalues.length > 0;
  const hasCurvatureCharts = !isSnapshot; // left column only makes sense when we have a trajectory

  return (
    <Tooltip.Provider delayDuration={200}>
      <div style={containerStyle}>
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'flex-start',
            gap: 12,
            marginBottom: 12,
            flexShrink: 0,
          }}
        >
          <div>
            <h1
              style={{
                fontSize: '22px',
                fontWeight: 800,
                color: ui.header,
                display: 'flex',
                alignItems: 'center',
                gap: 10,
              }}
            >
              <span role="img" aria-label="hessian">
                📉
              </span>
              Hessian Analysis
            </h1>
            <p
              style={{
                fontSize: 14,
                color: ui.muted,
                marginTop: 4,
                maxWidth: 620,
                lineHeight: 1.5,
              }}
            >
              {isSnapshot
                ? 'Static analysis of the loss landscape curvature at the current checkpoint.'
                : 'Evolution of loss landscape curvature throughout training, analyzing the Hessian spectrum.'}
            </p>
          </div>
        </div>

        {/* Inline spectral monitor controller bar – full width under title */}
        {!isSnapshot && (
          <div
            style={{
              marginBottom: 16,
              padding: '10px 18px',
              borderRadius: 999,
              border: `1px solid ${ui.surfaceBorder}`,
              background: isDark ? 'rgba(15,23,42,0.9)' : 'rgba(255,255,255,0.9)',
              boxShadow: ui.panelShadow as React.CSSProperties['boxShadow'],
              width: '100%',
              maxWidth: '100%',
              display: 'flex',
              flexDirection: 'column',
              gap: 6,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
              <span style={{ fontSize: 16 }}>⚡</span>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <span style={{ fontSize: 13, fontWeight: 700 }}>Spectral Monitor</span>
                <span style={{ fontSize: 11, color: ui.muted }}>
                  Epoch {currentEpoch} · λmax / Trace · slide to explore spectrum history
                </span>
              </div>
              <div
                style={{
                  marginLeft: 'auto',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  fontSize: 11,
                  color: ui.muted,
                }}
              >
                <span style={{ textTransform: 'uppercase', letterSpacing: '0.06em' }}>λmax</span>
                <span style={{ fontFamily: 'monospace', fontSize: 11, color: ui.text }}>
                  {data.max_eigenvalue[selectedEpochIndex]?.toExponential(3) ?? '-'}
                </span>
                <span style={{ textTransform: 'uppercase', letterSpacing: '0.06em' }}>Trace</span>
                <span style={{ fontFamily: 'monospace', fontSize: 11, color: ui.text }}>
                  {data.trace[selectedEpochIndex]?.toExponential(3) ?? '-'}
                </span>
              </div>
            </div>
            <input
              type="range"
              min={0}
              max={data.epochs.length - 1}
              step={1}
              value={sliderEpochIndex}
              onChange={(e) => {
                const idx = Number(e.target.value);
                setSliderEpochIndex(idx);
                if (sliderDebounceRef.current != null) {
                  window.clearTimeout(sliderDebounceRef.current);
                }
                sliderDebounceRef.current = window.setTimeout(() => {
                  setSelectedEpochIndex(idx);
                }, 80);
              }}
              style={{
                width: '100%',
                accentColor: ui.accent,
                cursor: 'pointer',
              }}
            />
          </div>
        )}

        <div
          style={{
            display: 'grid',
            gap: 24,
            gridTemplateColumns:
              hasCurvatureCharts && showSpectralColumn
                ? 'minmax(0, 1.4fr) minmax(0, 1.2fr)'
                : 'minmax(0, 1fr)',
            alignItems: 'stretch',
            flex: 1,
            minHeight: 0,
            paddingRight: 0,
            overflow: 'auto', // scroll inside remaining height so columns stay aligned
          }}
          className="custom-scrollbar"
        >
          {/* Left column: curvature metrics (only when we have a trajectory) */}
          {hasCurvatureCharts && (
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: 24,
                minHeight: 0,
                height: '100%',
              }}
            >
              <div style={{ ...cardStyle, flex: 1, minHeight: 0 }}>
                <div style={sectionHeaderStyle}>
                  <span>Curvature Dynamics</span>
                  <HelpIcon
                    text={
                      <div>
                        <p className="mb-2">
                          <strong>λmax (Sharpness):</strong> Largest eigenvalue of the Hessian,
                          tracking the sharpest direction in the loss landscape.
                        </p>
                        <p className="mb-2">
                          <strong>Stability Boundary (2/η):</strong> If λmax exceeds this, training
                          may become unstable for plain SGD.
                        </p>
                        <p>
                          <strong>Trace:</strong> Sum of eigenvalues, representing total curvature
                          around the current point.
                        </p>
                      </div>
                    }
                  />
                </div>
                {/* Stack plots vertically and split height evenly */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16, flex: 1, minHeight: 0 }}>
                  <div style={{ ...cardStyle, flex: 1, minHeight: 0 }}>
                    <MetricChart
                      title="Sharpness (λmax)"
                      color="#ef4444"
                      epochs={data.epochs}
                      values={data.max_eigenvalue}
                      yScale="log10"
                      height="100%"
                      valueLabel="λmax"
                      highlightEpoch={currentEpoch}
                      extraSeries={
                        lrData.hasData
                          ? [
                              {
                                label: 'Stability (2/η)',
                                color: '#f97316',
                                values: lrData.stabilityBoundary,
                                dashed: true,
                              },
                            ]
                          : []
                      }
                    />
                  </div>
                  <div style={{ ...cardStyle, flex: 1, minHeight: 0 }}>
                    <MetricChart
                      title="Total Curvature (Trace)"
                      color="#3b82f6"
                      epochs={data.epochs}
                      values={data.trace}
                      yScale="log10"
                      height="100%"
                      valueLabel="trace"
                      highlightEpoch={currentEpoch}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Right column: spectral analysis (only when we have data) */}
          {showSpectralColumn && (
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: 24,
                minHeight: 0,
                height: '100%',
              }}
            >
              <div style={{ ...cardStyle, flex: 1, minHeight: 0 }}>
                <div style={sectionHeaderStyle}>
                  <span>Spectral Analysis</span>
                  <HelpIcon
                    text={
                      <div>
                        <p className="mb-2">
                          <strong>Spectral Density:</strong> KDE of Hessian eigenvalues; the current
                          epoch is highlighted.
                        </p>
                        <p>
                          <strong>Scree Plot:</strong> Top eigenvalues (algebraic order) to separate
                          bulk spectrum from outliers.
                        </p>
                      </div>
                    }
                  />
                </div>

                {/* Stack plots vertically and split height evenly */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16, flex: 1, minHeight: 0 }}>
                  <div style={{ ...cardStyle, flex: 1, minHeight: 0 }}>
                    <div
                      style={{
                        fontSize: 13,
                        fontWeight: 700,
                        color: ui.muted,
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        flexShrink: 0,
                      }}
                    >
                      <span>Spectral Density (KDE)</span>
                      {densityChartData && (
                        <div className="flex gap-3 text-xs">
                          <div className="flex items-center gap-1.5">
                            <span
                              className="w-2 h-2 rounded-full"
                              style={{ background: ui.accent }}
                            ></span>
                            <span style={{ fontWeight: 600 }}>Epoch {currentEpoch}</span>
                          </div>
                          <div className="flex items-center gap-1.5 opacity-60">
                            <span
                              className="w-2 h-2 rounded-full"
                              style={{ background: isDark ? '#94a3b8' : '#cbd5e1' }}
                            ></span>
                            <span>History</span>
                          </div>
                        </div>
                      )}
                    </div>
                    <div
                      style={{
                        width: '100%',
                        minWidth: 0,
                        position: 'relative',
                        flex: 1,
                        minHeight: 0,
                      }}
                    >
                      {densityChartData ? (
                        <ResponsiveContainer
                          width="100%"
                          height="100%"
                          debounce={50}
                          initialDimension={{ width: 1, height: 1 }}
                        >
                          <AreaChart
                            data={densityChartData.chartData}
                            margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              stroke={isDark ? '#333' : '#e5e7eb'}
                              vertical={false}
                            />
                            <XAxis
                              dataKey="x"
                              type="number"
                              tick={{ fontSize: 10, fill: ui.muted }}
                              domain={['auto', 'auto']}
                              tickFormatter={(v) => v.toFixed(2)}
                              axisLine={false}
                              tickLine={false}
                            />
                            <YAxis hide />
                            <RechartsTooltip
                              contentStyle={{
                                backgroundColor: isDark ? 'rgba(26, 31, 38, 0.95)' : '#fff',
                                borderColor: isDark ? '#333' : '#e5e7eb',
                                fontSize: '12px',
                                borderRadius: '6px',
                              }}
                              labelFormatter={(v) => `Eigenvalue: ${Number(v).toFixed(4)}`}
                              formatter={(val: any) => [Number(val).toFixed(4), `Density`]}
                            />
                            {densityChartData.series.map((s) => {
                              const color = getSeriesColor(s);
                              return (
                                <Area
                                  key={s.id}
                                  type="monotone"
                                  dataKey={s.id}
                                  stroke={color}
                                  fill={color}
                                  fillOpacity={s.isCurrent ? 0.2 : 0.05}
                                  strokeWidth={s.isCurrent ? 2.2 : 1.4}
                                  isAnimationActive={false}
                                />
                              );
                            })}
                          </AreaChart>
                        </ResponsiveContainer>
                      ) : (
                        <div
                          className="flex items-center justify-center h-full text-muted-foreground"
                          style={{ fontSize: 13, opacity: 0.7 }}
                        >
                          No density data available
                        </div>
                      )}
                    </div>
                  </div>

                  <div style={{ ...cardStyle, flex: 1, minHeight: 0 }}>
                    <div style={{ fontSize: 13, fontWeight: 700, color: ui.muted, flexShrink: 0 }}>
                      Top-k Spectrum (Ranked)
                    </div>
                    <div
                      style={{
                        width: '100%',
                        minWidth: 0,
                        position: 'relative',
                        flex: 1,
                        minHeight: 0,
                      }}
                    >
                      {screeChartData.length > 0 ? (
                        <ResponsiveContainer
                          width="100%"
                          height="100%"
                          debounce={50}
                          initialDimension={{ width: 1, height: 1 }}
                        >
                          <BarChart
                            data={screeChartData}
                            margin={{ top: 10, right: 10, left: 0, bottom: 16 }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              stroke={isDark ? '#333' : '#e5e7eb'}
                              vertical={false}
                            />
                            <XAxis
                              dataKey="rank"
                              tick={{ fontSize: 10, fill: ui.muted }}
                              label={{
                                value: 'Rank',
                                position: 'insideBottom',
                                offset: -8,
                                fontSize: 10,
                                fill: ui.muted,
                              }}
                              axisLine={false}
                              tickLine={false}
                            />
                            <YAxis
                              tick={{ fontSize: 10, fill: ui.muted }}
                              tickFormatter={(val) => val.toExponential(1)}
                              width={48}
                              axisLine={false}
                              tickLine={false}
                            />
                            <RechartsTooltip
                              cursor={{ fill: isDark ? '#ffffff05' : '#00000005' }}
                              contentStyle={{
                                backgroundColor: isDark ? 'rgba(26, 31, 38, 0.95)' : '#fff',
                                borderColor: isDark ? '#333' : '#e5e7eb',
                                fontSize: '12px',
                                borderRadius: '6px',
                              }}
                              formatter={(val: any) => [Number(val).toExponential(4), 'Eigenvalue']}
                            />
                            <ReferenceLine y={0} stroke={isDark ? '#4b5563' : '#cbd5e1'} />
                            <Bar dataKey="value" isAnimationActive={false} radius={[2, 2, 0, 0]}>
                              {screeChartData.map((entry, index) => (
                                <Cell
                                  key={`cell-${index}`}
                                  fill={entry.isPositive ? '#60a5fa' : '#ef4444'}
                                />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      ) : (
                        <div
                          className="flex items-center justify-center h-full text-muted-foreground"
                          style={{ fontSize: 13, opacity: 0.7 }}
                        >
                          No spectrum data available
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </Tooltip.Provider>
  );
}
