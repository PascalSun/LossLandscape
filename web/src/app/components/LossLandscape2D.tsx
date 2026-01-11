'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useI18n } from '../i18n';
import { getViridisColor } from '../lib/colormap';
import { useTheme } from '../theme';

interface LossLandscape2DProps {
  X: number[][];
  Y: number[][];
  lossGrid: number[][];
  baselineLoss?: number;
  trajectory?: {
    traj_1: number[];
    traj_2: number[];
    epochs: number[];
  };
  trajectoryHighlight?: {
    traj_1: number[];
    traj_2: number[];
    traj_3?: number[];
    epochs: number[];
  };
  sliceGamma?: number;
  sliceThreshold?: number;
  /** Optional axis labels for hover/tooltips (defaults to α / β). */
  xLabel?: string;
  planeLabel?: string;
}


function fmtLoss(x: number) {
  return Number.isFinite(x) ? x.toExponential(3) : String(x);
}

type Pt = { x: number; y: number };

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function interp(a: Pt, b: Pt, va: number, vb: number, iso: number): Pt {
  const denom = vb - va;
  const t = denom === 0 ? 0.5 : (iso - va) / denom;
  return { x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t) };
}

// Helper to draw a star
function drawStar(ctx: CanvasRenderingContext2D, cx: number, cy: number, spikes: number, outerRadius: number, innerRadius: number) {
  let rot = Math.PI / 2 * 3;
  let x = cx;
  let y = cy;
  let step = Math.PI / spikes;

  ctx.beginPath();
  ctx.moveTo(cx, cy - outerRadius);
  for (let i = 0; i < spikes; i++) {
    x = cx + Math.cos(rot) * outerRadius;
    y = cy + Math.sin(rot) * outerRadius;
    ctx.lineTo(x, y);
    rot += step;

    x = cx + Math.cos(rot) * innerRadius;
    y = cy + Math.sin(rot) * innerRadius;
    ctx.lineTo(x, y);
    rot += step;
  }
  ctx.lineTo(cx, cy - outerRadius);
  ctx.closePath();
}

export default function LossLandscape2D({
  X,
  Y,
  lossGrid,
  trajectory,
  trajectoryHighlight,
  baselineLoss,
  sliceGamma,
  sliceThreshold,
  xLabel,
  planeLabel,
}: LossLandscape2DProps) {
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
  const [isDraggingLegend, setIsDraggingLegend] = useState(false);
  const isDraggingLegendRef = useRef(false);
  const dragStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const posStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  
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

  // Filter trajectory based on viewEpoch
  const filteredTrajectory = useMemo(() => {
    if (!trajectory) return undefined;
    const len = trajectory.traj_1.length;
    const filtered = {
      traj_1: [] as number[],
      traj_2: [] as number[],
      epochs: [] as number[],
    };
    for (let i = 0; i < len; i++) {
      const ep = trajectory.epochs[i] ?? i;
      if (ep > viewEpoch) continue;
      filtered.traj_1.push(trajectory.traj_1[i]);
      filtered.traj_2.push(trajectory.traj_2[i]);
      filtered.epochs.push(ep);
    }
    return filtered;
  }, [trajectory, viewEpoch]);
  
  // Debug: log trajectory data
  useEffect(() => {
    console.log('[LossLandscape2D] Trajectory data:', {
      hasTrajectory: !!trajectory,
      hasTraj1: !!trajectory?.traj_1,
      traj1Length: trajectory?.traj_1?.length,
      traj2Length: trajectory?.traj_2?.length,
      trajectory,
    });
  }, [trajectory]);
  const [hover, setHover] = useState<{ x: number; y: number; z: number; epoch: number | null } | null>(null);
  const [hoverPx, setHoverPx] = useState<{ x: number; y: number } | null>(null);

  const stats = useMemo(() => {
    const flatX = X.flat();
    const flatY = Y.flat();
    const flatZ = lossGrid.flat().filter((v) => Number.isFinite(v));
    const zMin = Math.min(...flatZ);
    const zMax = Math.max(...flatZ);
    // Auto-log if spans many orders of magnitude
    const autoUseLog = zMin > 0 && zMax / Math.max(zMin, 1e-12) > 1e4;
    return {
      xMin: Math.min(...flatX),
      xMax: Math.max(...flatX),
      yMin: Math.min(...flatY),
      yMax: Math.max(...flatY),
      zMin,
      zMax,
      autoUseLog,
    };
  }, [X, Y, lossGrid]);

  useEffect(() => {
    setUseLog(stats.autoUseLog);
  }, [stats.autoUseLog]);

  // Legend drag handlers (to avoid blocking the top-right corner) - optimized with requestAnimationFrame
  useEffect(() => {
    let rafId: number | null = null;
    
    const onMove = (e: MouseEvent) => {
      if (!isDraggingLegendRef.current) return;
      
      // Throttle updates with requestAnimationFrame
      if (rafId !== null) return;
      
      rafId = requestAnimationFrame(() => {
        rafId = null;
        const dx = e.clientX - dragStart.current.x;
        const dy = e.clientY - dragStart.current.y;
        setLegendPos({ x: posStart.current.x + dx, y: posStart.current.y + dy });
      });
    };
    
    const onUp = () => {
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      if (isDraggingLegendRef.current) {
        isDraggingLegendRef.current = false;
        setIsDraggingLegend(false);
      }
    };
    
    window.addEventListener('mousemove', onMove, { passive: true });
    window.addEventListener('mouseup', onUp);
    return () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, []);

  const startLegendDrag = (e: React.MouseEvent) => {
    isDraggingLegendRef.current = true;
    setIsDraggingLegend(true);
    dragStart.current = { x: e.clientX, y: e.clientY };
    posStart.current = legendPos;
    e.preventDefault();
  };

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

      const rows = lossGrid.length;
      const cols = lossGrid[0]?.length || 0;
      if (!rows || !cols) return;

      // Draw heatmap (nearest-neighbor scaled image for speed)
      const img = ctx.createImageData(cols, rows);
      const zEps = 1e-12;
      const zT = (z: number) => (useLog ? Math.log10(Math.max(z, 0) + zEps) : z);
      const zMinT = zT(stats.zMin);
      const zMaxT = zT(stats.zMax);
      const zRangeT = zMaxT - zMinT || 1;
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const z = lossGrid[i][j];
          const t = (zT(z) - zMinT) / zRangeT;
          const c = getViridisColor(t);
          const idx = (i * cols + j) * 4;
          img.data[idx + 0] = c.r;
          img.data[idx + 1] = c.g;
          img.data[idx + 2] = c.b;
          img.data[idx + 3] = 255;
        }
      }

      // Paint to an offscreen canvas, then scale to viewer size
      const off = document.createElement('canvas');
      off.width = cols;
      off.height = rows;
      const offCtx = off.getContext('2d');
      if (!offCtx) return;
      offCtx.putImageData(img, 0, 0);

      // A bit smoother (but still readable) for users: keep smoothing on
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, w, h);

      // Overlay contour lines (marching-squares)
      // We draw contours in transformed space (log or linear) so they match the color mapping.
      const levels = 10;
      const isoVals = Array.from({ length: levels }, (_, k) => zMinT + ((k + 1) / (levels + 1)) * zRangeT);
      const pxFor = (gx: number, gy: number) => {
        // grid coordinates gx in [0, cols-1], gy in [0, rows-1]
        const u = gx / Math.max(1, cols - 1);
        const v = gy / Math.max(1, rows - 1);
        return { x: u * w, y: (1 - v) * h };
      };

      const valAt = (gy: number, gx: number) => zT(lossGrid[gy][gx]);

      ctx.save();
      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(255,255,255,0.28)';
      for (const iso of isoVals) {
        ctx.beginPath();
        for (let gy = 0; gy < rows - 1; gy++) {
          for (let gx = 0; gx < cols - 1; gx++) {
            const v00 = valAt(gy, gx);
            const v10 = valAt(gy, gx + 1);
            const v01 = valAt(gy + 1, gx);
            const v11 = valAt(gy + 1, gx + 1);

            const p00 = pxFor(gx, gy);
            const p10 = pxFor(gx + 1, gy);
            const p01 = pxFor(gx, gy + 1);
            const p11 = pxFor(gx + 1, gy + 1);

            // Case index (bit set if corner >= iso)
            const c0 = v00 >= iso ? 1 : 0;
            const c1 = v10 >= iso ? 1 : 0;
            const c2 = v11 >= iso ? 1 : 0;
            const c3 = v01 >= iso ? 1 : 0;
            const idx = (c0 << 0) | (c1 << 1) | (c2 << 2) | (c3 << 3);
            if (idx === 0 || idx === 15) continue;

            // Edge midpoints (interpolated)
            const e0 = interp(p00, p10, v00, v10, iso); // top
            const e1 = interp(p10, p11, v10, v11, iso); // right
            const e2 = interp(p01, p11, v01, v11, iso); // bottom
            const e3 = interp(p00, p01, v00, v01, iso); // left

            // Standard marching squares segments
            // Ambiguous cases (5,10): pick a consistent pairing using average.
            const avg = (v00 + v10 + v01 + v11) / 4;
            const connect = (a: Pt, b: Pt) => {
              ctx.moveTo(a.x, a.y);
              ctx.lineTo(b.x, b.y);
            };

            switch (idx) {
              case 1:
              case 14:
                connect(e3, e0);
                break;
              case 2:
              case 13:
                connect(e0, e1);
                break;
              case 3:
              case 12:
                connect(e3, e1);
                break;
              case 4:
              case 11:
                connect(e1, e2);
                break;
              case 6:
              case 9:
                connect(e0, e2);
                break;
              case 7:
              case 8:
                connect(e3, e2);
                break;
              case 5:
                // two segments
                if (avg >= iso) {
                  connect(e3, e0);
                  connect(e1, e2);
                } else {
                  connect(e0, e1);
                  connect(e2, e3);
                }
                break;
              case 10:
                if (avg >= iso) {
                  connect(e0, e1);
                  connect(e2, e3);
                } else {
                  connect(e3, e0);
                  connect(e1, e2);
                }
                break;
              default:
                break;
            }
          }
        }
        ctx.stroke();
      }
      ctx.restore();

      // Overlay trajectory if available (filtered by viewEpoch)
      if (filteredTrajectory && filteredTrajectory.traj_1 && filteredTrajectory.traj_1.length >= 2) {
        const toPx = (x: number, y: number) => {
          // Map from data coords to canvas coords
          const u = (x - stats.xMin) / (stats.xMax - stats.xMin || 1);
          const v = (y - stats.yMin) / (stats.yMax - stats.yMin || 1);
          // Clamp to canvas bounds
          const px = Math.max(0, Math.min(w, u * w));
          const py = Math.max(0, Math.min(h, (1 - v) * h));
          return { px, py };
        };

        // Check if trajectory points are within bounds
        const trajXMin = Math.min(...filteredTrajectory.traj_1);
        const trajXMax = Math.max(...filteredTrajectory.traj_1);
        const trajYMin = Math.min(...filteredTrajectory.traj_2);
        const trajYMax = Math.max(...filteredTrajectory.traj_2);
        
        console.log('[LossLandscape2D] Rendering trajectory:', {
          length: filteredTrajectory.traj_1.length,
          viewEpoch,
          trajRange: { x: [trajXMin, trajXMax], y: [trajYMin, trajYMax] },
          gridRange: { x: [stats.xMin, stats.xMax], y: [stats.yMin, stats.yMax] },
          firstPoint: [filteredTrajectory.traj_1[0], filteredTrajectory.traj_2[0]],
          lastPoint: [filteredTrajectory.traj_1[filteredTrajectory.traj_1.length - 1], filteredTrajectory.traj_2[filteredTrajectory.traj_2.length - 1]],
        });

        ctx.save();
        
        // Draw trajectory line with thicker, more visible stroke
        ctx.lineWidth = 4;
        ctx.strokeStyle = 'rgba(255, 80, 80, 1.0)';
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.shadowColor = 'rgba(255, 80, 80, 0.5)';
        ctx.shadowBlur = 4;
        
        ctx.beginPath();
        const p0 = toPx(filteredTrajectory.traj_1[0], filteredTrajectory.traj_2[0]);
        console.log('[LossLandscape2D] First point mapped:', { 
          data: [filteredTrajectory.traj_1[0], filteredTrajectory.traj_2[0]], 
          px: p0,
          canvasSize: { w, h },
        });
        ctx.moveTo(p0.px, p0.py);
        
        // Draw all intermediate points
        const allPoints: Array<{ px: number; py: number; i: number }> = [{ px: p0.px, py: p0.py, i: 0 }];
        for (let i = 1; i < filteredTrajectory.traj_1.length; i++) {
          const p = toPx(filteredTrajectory.traj_1[i], filteredTrajectory.traj_2[i]);
          ctx.lineTo(p.px, p.py);
          allPoints.push({ px: p.px, py: p.py, i });
        }
        ctx.stroke();
        ctx.shadowBlur = 0; // Reset shadow
        
        // Draw intermediate points as small dots
        // Sample points to show (every N-th point to avoid clutter)
        const total = allPoints.length;
        const step = total <= 20 ? 1 : Math.ceil(total / 20);
        const sampledIndices = new Set<number>();
        sampledIndices.add(0); // Always show first
        for (let i = step; i < total - 1; i += step) {
          sampledIndices.add(i);
        }
        sampledIndices.add(total - 1); // Always show last
        
        for (const idx of sampledIndices) {
          const point = allPoints[idx];
          const progress = idx / (total - 1);
          
          // Color gradient: green -> yellow -> red (same as 3D)
          let r: number, g: number, b: number;
          if (progress < 0.5) {
            // green to yellow
            r = 0.2 + (1.0 - 0.2) * (progress * 2);
            g = 0.8 + (1.0 - 0.8) * (progress * 2);
            b = 0.2;
          } else {
            // yellow to red
            r = 1.0;
            g = 1.0 - (1.0 - 0.2) * ((progress - 0.5) * 2);
            b = 0.2;
          }
          
          const isStart = idx === 0;
          const isEnd = idx === total - 1;
          const radius = isStart || isEnd ? 5 : 3;
          
          ctx.fillStyle = `rgba(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)}, 0.9)`;
          ctx.beginPath();
          ctx.arc(point.px, point.py, radius, 0, Math.PI * 2);
          ctx.fill();
          
          // Add stroke for visibility
          ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        // Start / end markers (larger, more prominent)
        const start = allPoints[0];
        const end = allPoints[allPoints.length - 1];

        const drawDot = (p: { px: number; py: number }, color: string, size: number) => {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(p.px, p.py, size, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = 'rgba(0,0,0,0.6)';
          ctx.lineWidth = 2.5;
          ctx.stroke();
        };
        drawDot(start, 'rgba(80, 220, 140, 1)', 6);
        drawDot(end, 'rgba(255, 80, 80, 1)', 6);
        
        ctx.restore();
      }

      // Overlay highlighted trajectory points (e.g., those belonging to current γ slice slab)
      if (trajectoryHighlight && trajectoryHighlight.traj_1 && trajectoryHighlight.traj_1.length > 0) {
        const toPx = (x: number, y: number) => {
          const u = (x - stats.xMin) / (stats.xMax - stats.xMin || 1);
          const v = (y - stats.yMin) / (stats.yMax - stats.yMin || 1);
          const px = Math.max(0, Math.min(w, u * w));
          const py = Math.max(0, Math.min(h, (1 - v) * h));
          return { px, py };
        };

        ctx.save();
        for (let i = 0; i < trajectoryHighlight.traj_1.length; i++) {
          const ep = trajectoryHighlight.epochs?.[i] ?? i;
          if (ep > viewEpoch) continue;

          const p = toPx(trajectoryHighlight.traj_1[i], trajectoryHighlight.traj_2[i]);
          
          let opacity = 0.95;
          let radius = 4.5;
          let strokeOpacity = 0.7;
          let isClosest = false;

          // If we have 3D slice info, scale appearance by proximity to slice center
          if (
            sliceGamma !== undefined && 
            sliceThreshold !== undefined && 
            trajectoryHighlight.traj_3 && 
            trajectoryHighlight.traj_3[i] !== undefined
          ) {
            const g = trajectoryHighlight.traj_3[i];
            const dist = Math.abs(g - sliceGamma);
            // normalized distance 0..1 within the threshold slab
            // (points outside threshold shouldn't be here, but clamp just in case)
            const normDist = Math.max(0, Math.min(1, dist / sliceThreshold));
            
            // Closer points: larger, more opaque, maybe brighter
            // Distant points: smaller, faded
            const proximity = 1.0 - normDist; // 1.0 = on slice, 0.0 = at edge
            
            radius = 3.0 + proximity * 3.5; // 3.0 -> 6.5
            opacity = 0.4 + proximity * 0.6; // 0.4 -> 1.0
            strokeOpacity = 0.3 + proximity * 0.7;
            
            if (normDist < 0.1) isClosest = true;
          }

          if (isClosest) {
            // Draw star for points extremely close to the slice
            ctx.fillStyle = `rgba(255, 215, 0, ${opacity})`; // gold color
            drawStar(ctx, p.px, p.py, 5, radius * 1.8, radius * 0.8); // Increased size
            ctx.fill();

            // Add glowing ring
            ctx.strokeStyle = `rgba(255, 255, 255, ${strokeOpacity})`;
            ctx.lineWidth = 2.0;
            ctx.stroke();

            // Outer glow
            ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
            ctx.shadowBlur = 4;
            ctx.stroke();
            ctx.shadowBlur = 0;
          } else {
            // Use DIAMOND shape for other in-slab points to distinguish from projected circles
            ctx.fillStyle = `rgba(0, 255, 255, ${opacity})`; // Cyan/Aqua for high visibility
            
            ctx.beginPath();
            ctx.moveTo(p.px, p.py - radius * 1.2);
            ctx.lineTo(p.px + radius * 1.2, p.py);
            ctx.lineTo(p.px, p.py + radius * 1.2);
            ctx.lineTo(p.px - radius * 1.2, p.py);
            ctx.closePath();
            ctx.fill();

            // Double stroke for maximum contrast on any background
            // 1. White inner stroke
            ctx.strokeStyle = `rgba(255, 255, 255, ${Math.min(1, strokeOpacity + 0.2)})`;
            ctx.lineWidth = 2.0;
            ctx.stroke();
            
            // 2. Black outer stroke
            ctx.strokeStyle = `rgba(0, 0, 0, ${strokeOpacity})`;
            ctx.lineWidth = 1.0;
            ctx.stroke();
          }
        }
        ctx.restore();
      }

      // Axes labels with background (defaults α/β, but slice views may pass β/γ or α/γ)
      ctx.save();
      
      const xLabelText = xLabel ?? 'α';
      const planeLabelText = planeLabel ?? 'β';

      // X axis label
      const alphaText = `${xLabelText}: ${stats.xMin.toFixed(2)} → ${stats.xMax.toFixed(2)}`;
      ctx.font = 'bold 12px ui-sans-serif, system-ui';
      const alphaWidth = ctx.measureText(alphaText).width;
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(14, 8, alphaWidth + 12, 20);
      ctx.fillStyle = 'white';
      ctx.fillText(alphaText, 20, 22);
      
      // Plane axis label
      const betaText = `${planeLabelText}: ${stats.yMin.toFixed(2)} → ${stats.yMax.toFixed(2)}`;
      const betaWidth = ctx.measureText(betaText).width;
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(14, 34, betaWidth + 12, 20);
      ctx.fillStyle = 'white';
      ctx.fillText(betaText, 20, 48);
      
      ctx.restore();
    };

    const ro = new ResizeObserver(() => render());
    ro.observe(wrap);
    render();
    return () => ro.disconnect();
  }, [lossGrid, stats, filteredTrajectory, useLog, trajectoryHighlight, sliceGamma, sliceThreshold, xLabel, planeLabel, viewEpoch]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const onMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const w = rect.width;
      const h = rect.height;
      if (w <= 1 || h <= 1) return;

      // Map to data coords
      const u = Math.min(1, Math.max(0, px / w));
      const v = Math.min(1, Math.max(0, 1 - py / h));
      const x = stats.xMin + u * (stats.xMax - stats.xMin);
      const y = stats.yMin + v * (stats.yMax - stats.yMin);

      // nearest grid index
      const rows = lossGrid.length;
      const cols = lossGrid[0]?.length || 0;
      const gi = Math.min(rows - 1, Math.max(0, Math.round(v * (rows - 1))));
      const gj = Math.min(cols - 1, Math.max(0, Math.round(u * (cols - 1))));
      const z = lossGrid[gi][gj];

      // Find nearest trajectory point
      let nearestEpoch: number | null = null;
      
      if (filteredTrajectory && filteredTrajectory.traj_1 && filteredTrajectory.traj_1.length > 0) {
        let minDistSq = Infinity;
        // Threshold in pixels (squared) for snapping
        // We need to convert data distance to pixel distance or vice versa.
        // Doing it in data space is easier but "pixel threshold" varies.
        // Let's check pixel distance.
        const snapDistPxSq = 15 * 15; // 15px radius

        const len = Math.min(filteredTrajectory.traj_1.length, filteredTrajectory.traj_2.length);
        for (let i = 0; i < len; i++) {
          const tx = filteredTrajectory.traj_1[i];
          const ty = filteredTrajectory.traj_2[i];
          
          // Map to pixels
          const tu = (tx - stats.xMin) / (stats.xMax - stats.xMin || 1);
          const tv = (ty - stats.yMin) / (stats.yMax - stats.yMin || 1);
          const tpx = tu * w;
          const tpy = (1 - tv) * h;

          const dx = px - tpx;
          const dy = py - tpy;
          const dSq = dx * dx + dy * dy;

          if (dSq < snapDistPxSq && dSq < minDistSq) {
            minDistSq = dSq;
            nearestEpoch = filteredTrajectory.epochs[i] ?? i;
            // Optionally we could look up exact loss if we had it in trajectory
          }
        }
      }

      setHover({ x, y, z, epoch: nearestEpoch });
      setHoverPx({ x: px, y: py });
    };
    const onLeave = () => {
      setHover(null);
      setHoverPx(null);
    };

    canvas.addEventListener('mousemove', onMove);
    canvas.addEventListener('mouseleave', onLeave);
    return () => {
      canvas.removeEventListener('mousemove', onMove);
      canvas.removeEventListener('mouseleave', onLeave);
    };
  }, [lossGrid, stats, filteredTrajectory]);

  return (
    <div ref={wrapRef} style={{ width: '100%', height: '100%', position: 'relative' }}>
      <canvas ref={canvasRef} />
      {hover && hoverPx ? (
        <div
          style={{
            position: 'absolute',
            left: hoverPx.x + 8,
            top: hoverPx.y + 8,
            padding: '14px 16px',
            borderRadius: 14,
            border: '2px solid rgba(251, 191, 36, 0.5)',
            background: ui.tooltipBg,
            backdropFilter: 'blur(12px)',
            color: ui.tooltipText,
            fontSize: 12,
            lineHeight: 1.7,
            minWidth: 200,
            pointerEvents: 'none',
            boxShadow: isDark
              ? '0 8px 24px rgba(0,0,0,0.5), 0 0 20px rgba(251, 191, 36, 0.2)'
              : '0 8px 24px rgba(15,23,42,0.12), 0 0 18px rgba(251, 191, 36, 0.18)',
            zIndex: 1000,
          }}
        >
          <div style={{ 
            fontWeight: 800, 
            marginBottom: 10, 
            fontSize: 11, 
            opacity: 0.9, 
            textTransform: 'uppercase', 
            letterSpacing: '0.08em',
            color: '#fbbf24',
          }}>
            📍 {t.currentPosition}
          </div>
          <div style={{ display: 'grid', gap: 6, fontFamily: 'monospace', fontSize: 12 }}>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between',
              padding: '4px 0',
              borderBottom: `1px solid ${ui.surfaceBorder}`,
            }}>
              <span style={{ opacity: 0.8, fontWeight: 600 }}>{xLabel ?? 'α'}:</span> 
              <span style={{ fontWeight: 700, color: '#93c5fd' }}>{hover.x.toFixed(4)}</span>
            </div>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between',
              padding: '4px 0',
              borderBottom: '1px solid rgba(255,255,255,0.1)',
            }}>
              <span style={{ opacity: 0.8, fontWeight: 600 }}>{planeLabel ?? 'β'}:</span> 
              <span style={{ fontWeight: 700, color: '#a7f3d0' }}>{hover.y.toFixed(4)}</span>
            </div>
            {hover.epoch !== null && (
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between',
                padding: '4px 0',
                borderBottom: '1px solid rgba(255,255,255,0.1)',
              }}>
                <span style={{ opacity: 0.8, fontWeight: 600 }}>{t.epoch}:</span> 
                <span style={{ fontWeight: 700, color: '#fbbf24' }}>{hover.epoch}</span>
              </div>
            )}
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between',
              padding: '4px 0',
              marginTop: 2,
            }}>
              <span style={{ opacity: 0.8, fontWeight: 600 }}>{t.loss}:</span> 
              <span style={{ 
                fontWeight: 700, 
                color: Number.isFinite(hover.z) && hover.z < (stats.zMin + stats.zMax) / 2 ? '#51cf66' : '#ff6b6b',
              }}>
                {Number.isFinite(hover.z) ? hover.z.toExponential(3) : String(hover.z)}
              </span>
            </div>
          </div>
        </div>
      ) : null}
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
          cursor: isDraggingLegend ? 'grabbing' : 'grab',
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
          <span>{t.lossLandscape2D}</span>
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
                  const steps = 20; // More steps for smoother gradient
                  const colors: string[] = [];
                  for (let i = 0; i <= steps; i++) {
                    // Ensure exact boundaries: t=0 for first, t=1 for last
                    const t = i === 0 ? 0 : i === steps ? 1 : i / steps;
                    const c = getViridisColor(t);
                    const rgb = `rgb(${c.r},${c.g},${c.b})`;
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
              {t.hoverForHint}
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
          
          {trajectory && (
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

          {trajectory && maxEpoch > 0 && (
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
          borderTop: '1px solid rgba(255,255,255,0.1)',
          fontSize: 10,
          opacity: 0.6,
          lineHeight: 1.4,
        }}>
          💡 <strong>{t.tipHover}</strong>
        </div>
      </div>
    </div>
  );
}


