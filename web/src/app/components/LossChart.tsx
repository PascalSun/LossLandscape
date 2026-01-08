'use client';

import { useRef, useEffect, useMemo } from 'react';

interface LossChartProps {
  epochs: number[];
  trainLosses?: (number | null)[];
  landscapeLosses?: number[];
  valLosses?: (number | null)[];
  title?: string;
  height?: number;
}

export default function LossChart({
  epochs,
  trainLosses,
  landscapeLosses,
  valLosses,
  title = 'Loss Curves',
  height = 200,
}: LossChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const chartData = useMemo(() => {
    if (!epochs || epochs.length === 0) return null;

    const data: Array<{
      epoch: number;
      trainLoss: number | null;
      landscapeLoss: number | null;
      valLoss: number | null;
    }> = [];

    for (let i = 0; i < epochs.length; i++) {
      data.push({
        epoch: epochs[i],
        trainLoss: trainLosses && i < trainLosses.length ? trainLosses[i] : null,
        landscapeLoss: landscapeLosses && i < landscapeLosses.length ? landscapeLosses[i] : null,
        valLoss: valLosses && i < valLosses.length ? valLosses[i] : null,
      });
    }

    // Filter out invalid data points
    const validData = data.filter(d => 
      d.trainLoss !== null || d.landscapeLoss !== null || d.valLoss !== null
    );

    if (validData.length === 0) return null;

    // Calculate min/max for each series
    const trainValues = validData.map(d => d.trainLoss).filter(v => v !== null) as number[];
    const landscapeValues = validData.map(d => d.landscapeLoss).filter(v => v !== null) as number[];
    const valValues = validData.map(d => d.valLoss).filter(v => v !== null) as number[];

    const allValues = [...trainValues, ...landscapeValues, ...valValues];
    const minLoss = Math.min(...allValues);
    const maxLoss = Math.max(...allValues);
    const lossRange = maxLoss - minLoss || 1;

    const minEpoch = Math.min(...validData.map(d => d.epoch));
    const maxEpoch = Math.max(...validData.map(d => d.epoch));
    const epochRange = maxEpoch - minEpoch || 1;

    return {
      data: validData,
      minEpoch,
      maxEpoch,
      epochRange,
      minLoss,
      maxLoss,
      lossRange,
      hasTrain: trainValues.length > 0,
      hasLandscape: landscapeValues.length > 0,
      hasVal: valValues.length > 0,
    };
  }, [epochs, trainLosses, landscapeLosses, valLosses]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !chartData) return;

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

      ctx.clearRect(0, 0, width, h);

      const padding = { top: 20, right: 40, bottom: 30, left: 50 };
      const chartWidth = width - padding.left - padding.right;
      const chartHeight = h - padding.top - padding.bottom;

      const { data, minEpoch, maxEpoch, epochRange, minLoss, maxLoss, lossRange } = chartData;

      // Draw grid
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
      ctx.lineWidth = 1;

      // Horizontal grid lines (loss)
      const numHLines = 5;
      for (let i = 0; i <= numHLines; i++) {
        const y = padding.top + (chartHeight / numHLines) * i;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
      }

      // Vertical grid lines (epoch)
      const numVLines = 5;
      for (let i = 0; i <= numVLines; i++) {
        const x = padding.left + (chartWidth / numVLines) * i;
        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, h - padding.bottom);
        ctx.stroke();
      }

      // Draw axes
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
      ctx.lineWidth = 2;

      // X axis
      ctx.beginPath();
      ctx.moveTo(padding.left, h - padding.bottom);
      ctx.lineTo(width - padding.right, h - padding.bottom);
      ctx.stroke();

      // Y axis
      ctx.beginPath();
      ctx.moveTo(padding.left, padding.top);
      ctx.lineTo(padding.left, h - padding.bottom);
      ctx.stroke();

      // Draw axis labels
      ctx.fillStyle = 'var(--text-secondary)';
      ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';

      // X axis labels (epochs)
      for (let i = 0; i <= numVLines; i++) {
        const epoch = minEpoch + (epochRange / numVLines) * i;
        const x = padding.left + (chartWidth / numVLines) * i;
        ctx.fillText(epoch.toFixed(0), x, h - padding.bottom + 5);
      }

      // Y axis labels (loss)
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (let i = 0; i <= numHLines; i++) {
        const loss = maxLoss - (lossRange / numHLines) * i;
        const y = padding.top + (chartHeight / numHLines) * i;
        ctx.fillText(loss.toFixed(4), padding.left - 8, y);
      }

      // Draw lines
      const toX = (epoch: number) => padding.left + ((epoch - minEpoch) / epochRange) * chartWidth;
      const toY = (loss: number) => padding.top + ((maxLoss - loss) / lossRange) * chartHeight;

      // Train loss line
      if (chartData.hasTrain) {
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 2;
        ctx.beginPath();
        let firstPoint = true;
        for (const point of data) {
          if (point.trainLoss !== null) {
            const x = toX(point.epoch);
            const y = toY(point.trainLoss);
            if (firstPoint) {
              ctx.moveTo(x, y);
              firstPoint = false;
            } else {
              ctx.lineTo(x, y);
            }
          }
        }
        ctx.stroke();

        // Draw points
        ctx.fillStyle = '#3b82f6';
        for (const point of data) {
          if (point.trainLoss !== null) {
            const x = toX(point.epoch);
            const y = toY(point.trainLoss);
            ctx.beginPath();
            ctx.arc(x, y, 2.5, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }

      // Landscape loss line
      if (chartData.hasLandscape) {
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2;
        ctx.beginPath();
        let firstPoint = true;
        for (const point of data) {
          if (point.landscapeLoss !== null) {
            const x = toX(point.epoch);
            const y = toY(point.landscapeLoss);
            if (firstPoint) {
              ctx.moveTo(x, y);
              firstPoint = false;
            } else {
              ctx.lineTo(x, y);
            }
          }
        }
        ctx.stroke();

        // Draw points
        ctx.fillStyle = '#10b981';
        for (const point of data) {
          if (point.landscapeLoss !== null) {
            const x = toX(point.epoch);
            const y = toY(point.landscapeLoss);
            ctx.beginPath();
            ctx.arc(x, y, 2.5, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }

      // Val loss line
      if (chartData.hasVal) {
        ctx.strokeStyle = '#f97316';
        ctx.lineWidth = 2;
        ctx.beginPath();
        let firstPoint = true;
        for (const point of data) {
          if (point.valLoss !== null) {
            const x = toX(point.epoch);
            const y = toY(point.valLoss);
            if (firstPoint) {
              ctx.moveTo(x, y);
              firstPoint = false;
            } else {
              ctx.lineTo(x, y);
            }
          }
        }
        ctx.stroke();

        // Draw points
        ctx.fillStyle = '#f97316';
        for (const point of data) {
          if (point.valLoss !== null) {
            const x = toX(point.epoch);
            const y = toY(point.valLoss);
            ctx.beginPath();
            ctx.arc(x, y, 2.5, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }

      // Draw legend
      const legendItems: Array<{ label: string; color: string; hasData: boolean }> = [
        { label: 'Train Loss', color: '#3b82f6', hasData: chartData.hasTrain },
        { label: 'Landscape Loss', color: '#10b981', hasData: chartData.hasLandscape },
        { label: 'Val Loss', color: '#f97316', hasData: chartData.hasVal },
      ].filter(item => item.hasData);

      if (legendItems.length > 0) {
        ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        const legendX = width - padding.right - 10;
        let legendY = padding.top + 10;
        for (const item of legendItems) {
          // Draw colored line
          ctx.strokeStyle = item.color;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(legendX - 30, legendY);
          ctx.lineTo(legendX - 10, legendY);
          ctx.stroke();

          // Draw label
          ctx.fillStyle = 'var(--text-primary)';
          ctx.fillText(item.label, legendX - 5, legendY);
          legendY += 18;
        }
      }

      // Draw title
      ctx.fillStyle = 'var(--text-primary)';
      ctx.font = 'bold 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(title, width / 2, 5);
    };

    render();
    const resizeObserver = new ResizeObserver(render);
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
    };
  }, [chartData, height]);

  if (!chartData) {
    return (
      <div ref={containerRef} style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
        No data available
      </div>
    );
  }

  return (
    <div ref={containerRef} style={{ width: '100%', height, position: 'relative' }}>
      <canvas ref={canvasRef} style={{ display: 'block', width: '100%', height: '100%' }} />
    </div>
  );
}
