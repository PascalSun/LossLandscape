'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useI18n } from '../i18n';
import { useTheme } from '../theme';
import { HoverCard } from './HoverCard';
import { getViridisColorNormalized } from '../lib/colormap';

interface LossLandscape1DProps {
  X: number[];
  lossLine: number[];
  baselineLoss?: number;
  trajectory?: {
    traj_1: number[];
    epochs: number[];
  };
  trajectoryHighlight?: {
    traj_1: number[];
    epochs: number[];
  };
  /** Optional axis label for hover/tooltips (defaults to α). */
  xLabel?: string;
}

function fmtLoss(x: number) {
  return Number.isFinite(x) ? x.toExponential(3) : String(x);
}

export default function LossLandscape1D({
  X,
  lossLine,
  trajectory,
  trajectoryHighlight,
  baselineLoss,
  xLabel = 'α',
}: LossLandscape1DProps) {
  const { t } = useI18n();
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const ui = useMemo(() => {
    return isDark
      ? {
          panelBg: 'rgba(26, 31, 38, 0.85)',
          panelBorder: 'rgba(148, 163, 184, 0.15)',
          panelShadow: '0 12px 40px rgba(0, 0, 0, 0.4)',
          text: '#e2e8f0',
          surface: 'rgba(148, 163, 184, 0.06)',
          surfaceBorder: 'rgba(148, 163, 184, 0.12)',
          divider: 'rgba(148, 163, 184, 0.15)',
          tooltipBg: 'rgba(26, 31, 38, 0.95)',
          tooltipText: '#cbd5e1',
        }
      : {
          panelBg: 'rgba(252, 253, 255, 0.9)',
          panelBorder: 'rgba(59, 130, 246, 0.1)',
          panelShadow: '0 12px 40px rgba(59, 130, 246, 0.08)',
          text: '#334155',
          surface: 'rgba(59, 130, 246, 0.04)',
          surfaceBorder: 'rgba(59, 130, 246, 0.08)',
          divider: 'rgba(59, 130, 246, 0.1)',
          tooltipBg: 'rgba(252, 253, 255, 0.95)',
          tooltipText: '#64748b',
        };
  }, [isDark]);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [useLog, setUseLog] = useState(true);
  const [legendPos, setLegendPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const isDraggingLegend = useRef(false);
  const dragStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const posStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const [hoverInfo, setHoverInfo] = useState<{ x: number; loss: number; px: number; py: number } | null>(null);

  // Compute epoch range
  const { minEpoch, maxEpoch } = useMemo(() => {
    if (!trajectory || !trajectory.epochs || trajectory.epochs.length === 0) {
      return { minEpoch: 0, maxEpoch: 0 };
    }
    return {
      minEpoch: Math.min(...trajectory.epochs),
      maxEpoch: Math.max(...trajectory.epochs),
    };
  }, [trajectory]);

  const [viewEpoch, setViewEpoch] = useState<number>(maxEpoch);

  // Update viewEpoch if maxEpoch changes
  useEffect(() => {
    setViewEpoch(maxEpoch);
  }, [maxEpoch]);

  // Filter trajectory by viewEpoch
  const filteredTrajectory = useMemo(() => {
    if (!trajectory || !trajectory.epochs) return undefined;
    const indices: number[] = [];
    for (let i = 0; i < trajectory.epochs.length; i++) {
      if (trajectory.epochs[i] <= viewEpoch) {
        indices.push(i);
      }
    }
    if (indices.length === 0) return undefined;
    return {
      traj_1: indices.map((i) => trajectory.traj_1[i]),
      epochs: indices.map((i) => trajectory.epochs[i]),
    };
  }, [trajectory, viewEpoch]);

  // Compute statistics
  const stats = useMemo(() => {
    if (!X.length || !lossLine.length) {
      return { xMin: 0, xMax: 0, zMin: 0, zMax: 0 };
    }
    const xVals = X.filter((v) => Number.isFinite(v));
    const zVals = lossLine.filter((v) => Number.isFinite(v));
    return {
      xMin: Math.min(...xVals),
      xMax: Math.max(...xVals),
      zMin: Math.min(...zVals),
      zMax: Math.max(...zVals),
    };
  }, [X, lossLine]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const render = () => {
      const w = wrap.clientWidth;
      const h = wrap.clientHeight;
      if (w <= 10 || h <= 10) return;

      // Set canvas to device pixels for crisp lines
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      ctx.clearRect(0, 0, w, h);

      if (!X.length || !lossLine.length) return;

      // Padding
      const padding = { top: 40, right: 20, bottom: 40, left: 60 };
      const plotW = w - padding.left - padding.right;
      const plotH = h - padding.top - padding.bottom;

      // Transform functions
      const zEps = 1e-12;
      const zT = (z: number) => (useLog ? Math.log10(Math.max(z, 0) + zEps) : z);
      const zMinT = zT(stats.zMin);
      const zMaxT = zT(stats.zMax);
      const zRangeT = zMaxT - zMinT || 1;

      const toX = (x: number) => padding.left + ((x - stats.xMin) / (stats.xMax - stats.xMin || 1)) * plotW;
      const toY = (z: number) => padding.top + plotH - ((zT(z) - zMinT) / zRangeT) * plotH;

      // Draw grid
      ctx.strokeStyle = isDark ? 'rgba(148, 163, 184, 0.15)' : 'rgba(59, 130, 246, 0.1)';
      ctx.lineWidth = 1;

      // Horizontal grid lines (loss values)
      const numGridLines = 5;
      for (let i = 0; i <= numGridLines; i++) {
        const t = i / numGridLines;
        const z = useLog ? Math.pow(10, zMinT + t * zRangeT) - zEps : stats.zMin + t * (stats.zMax - stats.zMin);
        const y = toY(z);
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(w - padding.right, y);
        ctx.stroke();

        // Label
        ctx.fillStyle = isDark ? '#94a3b8' : '#64748b';
        ctx.font = '11px system-ui';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(fmtLoss(z), padding.left - 8, y);
      }

      // Vertical grid lines (x values)
      for (let i = 0; i <= numGridLines; i++) {
        const t = i / numGridLines;
        const x = stats.xMin + t * (stats.xMax - stats.xMin);
        const px = toX(x);
        ctx.beginPath();
        ctx.moveTo(px, padding.top);
        ctx.lineTo(px, h - padding.bottom);
        ctx.stroke();

        // Label
        ctx.fillStyle = isDark ? '#94a3b8' : '#64748b';
        ctx.font = '11px system-ui';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(x.toFixed(3), px, h - padding.bottom + 8);
      }

      // Draw loss line
      ctx.strokeStyle = '#3b82f6'; // blue-500
      ctx.lineWidth = 2;
      ctx.beginPath();
      let first = true;
      for (let i = 0; i < X.length; i++) {
        const x = X[i];
        const z = lossLine[i];
        if (!Number.isFinite(x) || !Number.isFinite(z)) continue;
        const px = toX(x);
        const py = toY(z);
        if (first) {
          ctx.moveTo(px, py);
          first = false;
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.stroke();

      // Draw baseline point
      if (baselineLoss !== undefined && Number.isFinite(baselineLoss)) {
        const baselineX = 0; // Baseline is at origin
        const px = toX(baselineX);
        const py = toY(baselineLoss);
        ctx.fillStyle = '#10b981'; // emerald-500
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = isDark ? '#0b1220' : '#f1f5f9';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Interpolate loss value for a given x coordinate (helper function)
      const interpolateLoss = (x: number, XArr: number[], lossLineArr: number[]): number | null => {
        if (XArr.length === 0 || lossLineArr.length === 0) return null;
        if (x < stats.xMin || x > stats.xMax) return null;

        // Find the two closest points
        let idx = 0;
        for (let i = 0; i < XArr.length - 1; i++) {
          if (XArr[i] <= x && x <= XArr[i + 1]) {
            idx = i;
            break;
          }
          if (i === XArr.length - 2) idx = i;
        }

        const x0 = XArr[idx];
        const x1 = XArr[Math.min(idx + 1, XArr.length - 1)];
        const y0 = lossLineArr[idx];
        const y1 = lossLineArr[Math.min(idx + 1, lossLineArr.length - 1)];

        if (!Number.isFinite(x0) || !Number.isFinite(x1) || !Number.isFinite(y0) || !Number.isFinite(y1)) {
          return null;
        }

        const t = (x1 - x0) !== 0 ? (x - x0) / (x1 - x0) : 0;
        return y0 + (y1 - y0) * t;
      };

      // Draw trajectory if available
      if (filteredTrajectory && filteredTrajectory.traj_1 && filteredTrajectory.traj_1.length > 0) {
        ctx.strokeStyle = '#ef4444'; // red-500
        ctx.lineWidth = 2;
        ctx.beginPath();
        first = true;
        for (let i = 0; i < filteredTrajectory.traj_1.length; i++) {
          const alpha = filteredTrajectory.traj_1[i];
          // Interpolate loss from lossLine
          const loss = interpolateLoss(alpha, X, lossLine);
          if (loss === null) continue;
          const px = toX(alpha);
          const py = toY(loss);
          if (first) {
            ctx.moveTo(px, py);
            first = false;
          } else {
            ctx.lineTo(px, py);
          }
        }
        ctx.stroke();

        // Draw trajectory points
        ctx.fillStyle = '#ef4444';
        for (let i = 0; i < filteredTrajectory.traj_1.length; i++) {
          const alpha = filteredTrajectory.traj_1[i];
          const loss = interpolateLoss(alpha, X, lossLine);
          if (loss === null) continue;
          const px = toX(alpha);
          const py = toY(loss);
          ctx.beginPath();
          ctx.arc(px, py, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // Draw trajectory highlight if available
      if (trajectoryHighlight && trajectoryHighlight.traj_1 && trajectoryHighlight.traj_1.length > 0) {
        const highlightEpoch = viewEpoch;
        for (let i = 0; i < trajectoryHighlight.traj_1.length; i++) {
          if (trajectoryHighlight.epochs[i] > highlightEpoch) continue;
          const alpha = trajectoryHighlight.traj_1[i];
          const loss = interpolateLoss(alpha, X, lossLine);
          if (loss === null) continue;
          const px = toX(alpha);
          const py = toY(loss);
          ctx.fillStyle = '#fbbf24'; // amber-400
          ctx.beginPath();
          ctx.arc(px, py, 5, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = isDark ? '#0b1220' : '#f1f5f9';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }

      // Draw axes labels
      ctx.fillStyle = isDark ? '#e2e8f0' : '#334155';
      ctx.font = '12px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(xLabel, w / 2, h - 8);

      ctx.save();
      ctx.translate(20, h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(useLog ? `${t.loss} (log₁₀)` : t.loss, 0, 0);
      ctx.restore();
    };

    render();
  }, [X, lossLine, stats, useLog, isDark, baselineLoss, filteredTrajectory, trajectoryHighlight, viewEpoch, xLabel]);

  // Interpolate loss value for a given x coordinate
  const interpolateLoss = (x: number, XArr: number[], lossLineArr: number[]): number | null => {
    if (XArr.length === 0 || lossLineArr.length === 0) return null;
    if (x < stats.xMin || x > stats.xMax) return null;

    // Find the two closest points
    let idx = 0;
    for (let i = 0; i < XArr.length - 1; i++) {
      if (XArr[i] <= x && x <= XArr[i + 1]) {
        idx = i;
        break;
      }
      if (i === XArr.length - 2) idx = i;
    }

    const x0 = XArr[idx];
    const x1 = XArr[Math.min(idx + 1, XArr.length - 1)];
    const y0 = lossLineArr[idx];
    const y1 = lossLineArr[Math.min(idx + 1, lossLineArr.length - 1)];

    if (!Number.isFinite(x0) || !Number.isFinite(x1) || !Number.isFinite(y0) || !Number.isFinite(y1)) {
      return null;
    }

    const t = (x1 - x0) !== 0 ? (x - x0) / (x1 - x0) : 0;
    return y0 + (y1 - y0) * t;
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const rect = wrap.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const padding = { top: 40, right: 20, bottom: 40, left: 60 };
    const plotW = wrap.clientWidth - padding.left - padding.right;
    const plotH = wrap.clientHeight - padding.top - padding.bottom;

    if (x < padding.left || x > wrap.clientWidth - padding.right || y < padding.top || y > wrap.clientHeight - padding.bottom) {
      setHoverInfo(null);
      return;
    }

    // Convert pixel to data coordinates
    const dataX = stats.xMin + ((x - padding.left) / plotW) * (stats.xMax - stats.xMin);
    const loss = interpolateLoss(dataX, X, lossLine);

    if (loss !== null) {
      setHoverInfo({
        x: dataX,
        loss,
        px: e.clientX,
        py: e.clientY,
      });
    } else {
      setHoverInfo(null);
    }
  };

  const handleMouseLeave = () => {
    setHoverInfo(null);
  };

  // Legend drag handlers
  const startLegendDrag = (e: React.MouseEvent) => {
    if (e.button !== 0) return; // Only left mouse button
    e.preventDefault();
    isDraggingLegend.current = true;
    dragStart.current = { x: e.clientX, y: e.clientY };
    posStart.current = { ...legendPos };
    document.addEventListener('mousemove', handleLegendDrag);
    document.addEventListener('mouseup', stopLegendDrag);
  };

  const handleLegendDrag = (e: MouseEvent) => {
    if (!isDraggingLegend.current) return;
    const dx = e.clientX - dragStart.current.x;
    const dy = e.clientY - dragStart.current.y;
    setLegendPos({ x: posStart.current.x + dx, y: posStart.current.y + dy });
  };

  const stopLegendDrag = () => {
    isDraggingLegend.current = false;
    document.removeEventListener('mousemove', handleLegendDrag);
    document.removeEventListener('mouseup', stopLegendDrag);
  };

  useEffect(() => {
    return () => {
      document.removeEventListener('mousemove', handleLegendDrag);
      document.removeEventListener('mouseup', stopLegendDrag);
    };
  }, []);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div
        ref={wrapRef}
        style={{
          width: '100%',
          height: '100%',
          position: 'relative',
          background: isDark ? '#0b1220' : '#f1f5f9',
        }}
      >
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{
            display: 'block',
            width: '100%',
            height: '100%',
            cursor: hoverInfo ? 'crosshair' : 'default',
          }}
        />
      </div>

      {/* Control Panel - Draggable (same as 3D view) */}
      <div
        style={{
          position: 'absolute',
          top: 16,
          right: 16,
          transform: `translate(${legendPos.x}px, ${legendPos.y}px)`,
          padding: '18px 20px',
          borderRadius: 18,
          border: `1px solid ${ui.panelBorder}`,
          background: ui.panelBg,
          backdropFilter: 'blur(12px)',
          color: ui.text,
          zIndex: 10,
          fontSize: 13,
          lineHeight: 1.6,
          width: 300,
          boxShadow: ui.panelShadow,
          cursor: isDraggingLegend.current ? 'grabbing' : 'grab',
        }}
        onMouseDown={startLegendDrag}
      >
        <div style={{ 
          fontWeight: 800, 
          marginBottom: 16, 
          fontSize: 15, 
          letterSpacing: '0.02em',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}>
          <span>🌐</span>
          <span>Loss Landscape (1D)</span>
        </div>
        
        <div style={{ 
          display: 'flex', 
          gap: 14, 
          marginBottom: 16,
          alignItems: 'flex-start',
        }}>
          {/* Colorbar */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
            <div
              title={t.lowerIsBetter}
              style={{
                width: 24,
                height: 160,
                borderRadius: 0,
                border: 'none',
                background: `linear-gradient(to top, ${(() => {
                  const steps = 20;
                  const colors: string[] = [];
                  for (let i = 0; i <= steps; i++) {
                    const t = i === 0 ? 0 : i === steps ? 1 : i / steps;
                    const c = getViridisColorNormalized(t);
                    const rgb = `rgb(${Math.round(c.r * 255)},${Math.round(c.g * 255)},${Math.round(c.b * 255)})`;
                    const stopPercent = i === 0 ? '0%' : i === steps ? '100%' : `${(i / steps * 100).toFixed(2)}%`;
                    colors.push(`${rgb} ${stopPercent}`);
                  }
                  return colors.join(', ');
                })()})`,
                boxShadow: isDark
                  ? '0 4px 12px rgba(0,0,0,0.3), inset 0 0 20px rgba(255,255,255,0.1)'
                  : '0 4px 12px rgba(15,23,42,0.10), inset 0 0 20px rgba(255,255,255,0.25)',
                cursor: 'help',
                position: 'relative',
              }}
            />
            <div style={{ 
              fontSize: 9, 
              opacity: isDark ? 0.6 : 0.75, 
              marginTop: 4,
              textAlign: 'center',
              lineHeight: 1.2,
            }}>
              Hover for hint
            </div>
          </div>
          
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', minHeight: 160 }}>
            <div style={{ 
              padding: '8px 10px',
              borderRadius: 8,
              background: isDark ? 'rgba(255,100,100,0.15)' : 'rgba(239,68,68,0.10)',
              border: isDark ? '1px solid rgba(255,100,100,0.3)' : '1px solid rgba(239,68,68,0.22)',
            }}>
              <div style={{ fontSize: 10, opacity: 0.9, fontWeight: 700, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.5px' }}>{t.highLoss}</div>
              <div style={{ fontFamily: 'monospace', fontSize: 13, fontWeight: 700, color: '#ff6b6b' }}>{fmtLoss(stats.zMax)}</div>
            </div>
            
            <div style={{ 
              textAlign: 'center',
              padding: '6px 0',
              fontSize: 10,
              opacity: isDark ? 0.7 : 0.8,
              fontStyle: 'italic',
            }}>
              {t.lowerIsBetterArrow}
            </div>
            
            <div style={{ 
              padding: '8px 10px',
              borderRadius: 8,
              background: isDark ? 'rgba(100,255,150,0.15)' : 'rgba(16,185,129,0.10)',
              border: isDark ? '1px solid rgba(100,255,150,0.3)' : '1px solid rgba(16,185,129,0.22)',
            }}>
              <div style={{ fontSize: 10, opacity: 0.9, fontWeight: 700, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.5px' }}>{t.lowLoss}</div>
              <div style={{ fontFamily: 'monospace', fontSize: 13, fontWeight: 700, color: '#51cf66' }}>{fmtLoss(stats.zMin)}</div>
            </div>
          </div>
        </div>

        <div style={{ 
          borderTop: `2px solid ${ui.divider}`, 
          paddingTop: 14, 
          marginTop: 14, 
          display: 'grid', 
          gap: 10 
        }}>
          {baselineLoss !== undefined && (
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center',
              fontSize: 12,
              padding: '8px 10px',
              borderRadius: 8,
              background: ui.surface,
            }}>
              <span style={{ opacity: 0.9, fontWeight: 600 }}>📍 {t.baselineLoss}</span>
              <span style={{ fontFamily: 'monospace', fontWeight: 700, color: '#fbbf24' }}>{fmtLoss(baselineLoss)}</span>
            </div>
          )}
          
          {filteredTrajectory && filteredTrajectory.traj_1 && filteredTrajectory.traj_1.length > 0 && (
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: 10, 
              fontSize: 12,
              padding: '8px 10px',
              borderRadius: 8,
              background: isDark ? 'rgba(255, 80, 80, 0.1)' : 'rgba(239, 68, 68, 0.08)',
              border: isDark ? '1px solid rgba(255, 80, 80, 0.3)' : '1px solid rgba(239, 68, 68, 0.18)',
            }}>
              <div style={{ 
                width: 24, 
                height: 4, 
                background: 'rgba(255, 80, 80, 1)', 
                borderRadius: 2,
                boxShadow: '0 0 8px rgba(255, 80, 80, 0.5)',
              }} />
              <span style={{ opacity: 0.95, fontWeight: 600 }}>{t.trainingTrajectory}</span>
            </div>
          )}
          
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center', 
            fontSize: 12,
            padding: '8px 10px',
            borderRadius: 8,
            background: ui.surface,
          }}>
            <span style={{ opacity: 0.9, fontWeight: 600 }}>{t.scale}</span>
            <button
              type="button"
              onClick={() => setUseLog((v) => !v)}
              style={{
                border: `2px solid ${
                  useLog
                    ? 'rgba(251, 191, 36, 0.55)'
                    : isDark
                      ? 'rgba(255,255,255,0.30)'
                      : 'rgba(15,23,42,0.18)'
                }`,
                background: useLog
                  ? 'rgba(251, 191, 36, 0.20)'
                  : isDark
                    ? 'rgba(255,255,255,0.08)'
                    : 'rgba(15,23,42,0.05)',
                color: ui.text,
                borderRadius: 8,
                padding: '6px 14px',
                cursor: 'pointer',
                fontWeight: 700,
                fontSize: 11,
                transition: 'all 0.2s ease',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.background = useLog
                  ? 'rgba(251, 191, 36, 0.28)'
                  : isDark
                    ? 'rgba(255,255,255,0.12)'
                    : 'rgba(15,23,42,0.07)')
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.background = useLog
                  ? 'rgba(251, 191, 36, 0.20)'
                  : isDark
                    ? 'rgba(255,255,255,0.08)'
                    : 'rgba(15,23,42,0.05)')
              }
            >
              {useLog ? `📊 ${t.log}` : `📈 ${t.linear}`}
            </button>
          </div>

          {filteredTrajectory && filteredTrajectory.epochs && filteredTrajectory.epochs.length > 0 && maxEpoch > 0 && (
            <div 
              style={{ 
                display: 'flex', 
                gap: 12, 
                alignItems: 'center', 
                fontSize: 12,
                padding: '8px 10px',
                borderRadius: 8,
                background: 'rgba(255,255,255,0.05)',
                marginTop: 8,
              }}
              onMouseDown={(e) => e.stopPropagation()}
            >
              <span style={{ opacity: 0.9, fontWeight: 600 }}>{t.epoch}:</span>
              <input
                type="range"
                min={minEpoch}
                max={maxEpoch}
                step={1}
                value={viewEpoch}
                onChange={(e) => setViewEpoch(parseInt(e.target.value, 10))}
                style={{ 
                  flex: 1,
                  accentColor: '#ff4444',
                }}
              />
              <span style={{ 
                minWidth: 50, 
                textAlign: 'right', 
                opacity: 0.9, 
                fontFamily: 'monospace', 
                fontSize: 12 
              }}>
                {viewEpoch}
              </span>
            </div>
          )}
        </div>
        
        <div style={{
          marginTop: 12,
          paddingTop: 12,
          borderTop: `1px solid ${ui.surfaceBorder}`,
          fontSize: 10,
          opacity: isDark ? 0.6 : 0.75,
          lineHeight: 1.4,
        }}>
          💡 <strong>Tip:</strong> Hover to see values, drag panel to move
        </div>
      </div>

      {/* Hover tooltip */}
      {hoverInfo && (
        <HoverCard
          x={hoverInfo.px}
          y={hoverInfo.py}
          title={`${xLabel} = ${hoverInfo.x.toFixed(4)}`}
          content={`Loss: ${fmtLoss(hoverInfo.loss)}`}
        />
      )}
    </div>
  );
}
