'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import type { MouseEvent as ReactMouseEvent } from 'react';
import { useTheme } from '../theme';

export type MetricChartPoint = number | null | undefined;

interface MetricChartProps {
  title: string;
  color: string;
  epochs: number[];
  values: MetricChartPoint[];
  height?: number;
  valueLabel?: string;
  fillArea?: boolean;
}

export default function MetricChart({
  title,
  color,
  epochs,
  values,
  height = 160,
  valueLabel = 'loss',
  fillArea = true,
}: MetricChartProps) {
  const { theme } = useTheme();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null);
  const hoverRef = useRef<{ i: number; x: number; y: number } | null>(null);
  const renderRef = useRef<(() => void) | null>(null);
  const rafRef = useRef<number | null>(null);

  const chartTheme = useMemo(() => ({
    background: theme === 'dark' ? '#1a1f26' : '#fcfdff',
    grid: theme === 'dark' ? 'rgba(148, 163, 184, 0.1)' : 'rgba(59, 130, 246, 0.06)',
    textMuted: theme === 'dark' ? '#94a3b8' : '#94a3b8',
    textPrimary: theme === 'dark' ? '#e2e8f0' : '#334155',
    tooltipBg: theme === 'dark' ? 'rgba(26, 31, 38, 0.95)' : 'rgba(252, 253, 255, 0.95)',
    tooltipText: theme === 'dark' ? '#cbd5e1' : '#64748b',
    tooltipBorder: theme === 'dark' ? 'rgba(148, 163, 184, 0.15)' : 'rgba(59, 130, 246, 0.12)',
    dotFill: theme === 'dark' ? '#0f1419' : '#fcfdff'
  }), [theme]);

  const data = useMemo(() => {
    if (!epochs?.length) return null;
    const pts: Array<{ epoch: number; value: number }> = [];
    for (let i = 0; i < epochs.length; i++) {
      const v = values && i < values.length ? values[i] : null;
      const n = typeof v === 'number' && Number.isFinite(v) ? v : null;
      if (n !== null) pts.push({ epoch: epochs[i], value: n });
    }
    if (!pts.length) return null;

    const minEpoch = Math.min(...pts.map((p) => p.epoch));
    const maxEpoch = Math.max(...pts.map((p) => p.epoch));
    const epochRange = maxEpoch - minEpoch || 1;

    const minV = Math.min(...pts.map((p) => p.value));
    const maxV = Math.max(...pts.map((p) => p.value));
    let range = maxV - minV;
    if (range === 0) range = 0.1; // avoid divide by zero

    // Add 10% padding to Y axis
    const paddedMin = minV - range * 0.1;
    const paddedMax = maxV + range * 0.1;
    const paddedRange = paddedMax - paddedMin;

    return {
      pts,
      minEpoch,
      maxEpoch,
      epochRange,
      minV: paddedMin,
      maxV: paddedMax,
      range: paddedRange,
      rawMin: minV,
      rawMax: maxV,
      last: pts[pts.length - 1]?.value ?? null,
    };
  }, [epochs, values]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !data) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const render = () => {
      const width = container.clientWidth;
      const h = height;
      if (width <= 10 || h <= 10) return;

      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // Clear
      ctx.clearRect(0, 0, width, h);

      const padding = { top: 24, right: 12, bottom: 20, left: 48 };
      const chartW = width - padding.left - padding.right;
      const chartH = h - padding.top - padding.bottom;

      const toX = (epoch: number) => padding.left + ((epoch - data.minEpoch) / data.epochRange) * chartW;
      const toY = (v: number) => padding.top + ((data.maxV - v) / data.range) * chartH;

      // --- Grid & Axes ---
      ctx.strokeStyle = chartTheme.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      
      // Horizontal grid lines
      const steps = 4;
      for (let i = 0; i <= steps; i++) {
        const y = padding.top + (chartH / steps) * i;
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
      }
      ctx.stroke();

      // --- Y Axis Labels ---
      ctx.fillStyle = chartTheme.textMuted; // text-muted
      ctx.font = '500 10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (let i = 0; i <= steps; i++) {
        const val = data.maxV - (data.range / steps) * i;
        const y = padding.top + (chartH / steps) * i;
        ctx.fillText(val.toExponential(1), padding.left - 8, y);
      }

      // --- Plot Line ---
      if (data.pts.length > 1) {
        ctx.beginPath();
        let first = true;
        for (const p of data.pts) {
          const x = toX(p.epoch);
          const y = toY(p.value);
          if (first) {
            ctx.moveTo(x, y);
            first = false;
          } else {
            ctx.lineTo(x, y);
          }
        }
        
        // Stroke
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Fill Area
        if (fillArea) {
          ctx.lineTo(toX(data.pts[data.pts.length - 1].epoch), h - padding.bottom);
          ctx.lineTo(toX(data.pts[0].epoch), h - padding.bottom);
          ctx.closePath();
          const gradient = ctx.createLinearGradient(0, padding.top, 0, h - padding.bottom);
          gradient.addColorStop(0, color + '33'); // 20% opacity
          gradient.addColorStop(1, color + '00'); // 0% opacity
          ctx.fillStyle = gradient;
          ctx.fill();
        }
      }

      // --- Title & Last Value ---
      ctx.fillStyle = chartTheme.textPrimary; // text-primary
      ctx.font = '600 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(title, padding.left, 4);

      // --- Hover Interaction ---
      const hv = hoverRef.current;
      if (hv && hv.i >= 0 && hv.i < data.pts.length) {
        const p = data.pts[hv.i];
        const x = toX(p.epoch);
        const y = toY(p.value);

        // Crosshair
        ctx.strokeStyle = color; // 'rgba(0,0,0,0.2)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, h - padding.bottom);
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
        ctx.setLineDash([]);

        // Highlight dot
        ctx.fillStyle = chartTheme.dotFill;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    };

    renderRef.current = render;
    render();
    const ro = new ResizeObserver(render);
    ro.observe(container);
    return () => {
      ro.disconnect();
      renderRef.current = null;
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [color, data, height, title, valueLabel, fillArea, chartTheme]);

  const scheduleRender = () => {
    if (!renderRef.current) return;
    if (rafRef.current !== null) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      renderRef.current?.();
    });
  };

  const onMove = (e: ReactMouseEvent<HTMLDivElement>) => {
    if (!data) return;
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const padding = { top: 24, right: 12, bottom: 20, left: 48 };
    const chartW = rect.width - padding.left - padding.right;
    if (chartW <= 1) return;

    // Find nearest epoch
    const t = Math.max(0, Math.min(1, (x - padding.left) / chartW));
    const epochGuess = data.minEpoch + t * data.epochRange;

    let bestI = 0;
    let bestD = Infinity;
    // Simple linear scan is fast enough for <10k points
    for (let i = 0; i < data.pts.length; i++) {
      const d = Math.abs(data.pts[i].epoch - epochGuess);
      if (d < bestD) {
        bestD = d;
        bestI = i;
      }
    }

    const next = { i: bestI, x, y };
    hoverRef.current = next;
    setHover(next);
    scheduleRender();
  };

  const onLeave = () => {
    hoverRef.current = null;
    setHover(null);
    scheduleRender();
  };

  if (!data) return null;

  const hoveredPoint = hover ? data.pts[hover.i] : null;

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height,
        borderRadius: '12px',
        border: '1px solid var(--border)',
        background: chartTheme.background,
        position: 'relative',
        cursor: 'crosshair',
        boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
      }}
      onMouseMove={onMove}
      onMouseLeave={onLeave}
    >
      <canvas ref={canvasRef} style={{ display: 'block', width: '100%', height: '100%' }} />
      
      {/* Tooltip */}
      {hoveredPoint && (
        <div
          style={{
            position: 'absolute',
            left: 12, // Fixed position at top-left of chart for stability
            top: 24,
            pointerEvents: 'none',
            fontSize: '11px',
            color: chartTheme.tooltipText,
            background: chartTheme.tooltipBg,
            padding: '2px 6px',
            borderRadius: '4px',
            border: `1px solid ${chartTheme.tooltipBorder}`,
            boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
            zIndex: 10,
          }}
        >
          <span style={{ fontWeight: 600 }}>Epoch {hoveredPoint.epoch}</span>
          <span style={{ margin: '0 4px', color: chartTheme.textMuted }}>|</span>
          <span style={{ fontWeight: 600, color: color }}>{hoveredPoint.value.toExponential(4)}</span>
        </div>
      )}
    </div>
  );
}