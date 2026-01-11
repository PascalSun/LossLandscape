/* eslint-disable @typescript-eslint/no-explicit-any */
'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  ComposedChart,
  ReferenceLine
} from 'recharts';
import { useTheme } from '../theme';
import { useI18n } from '../i18n';

export type MetricChartPoint = number | null | undefined;
export type MetricChartScale = 'linear' | 'log10';

export interface MetricChartSeries {
  label: string;
  color: string;
  values: MetricChartPoint[];
  dashed?: boolean;
}

interface MetricChartProps {
  title: string;
  color: string;
  epochs: number[];
  values: MetricChartPoint[];
  height?: number | string;
  valueLabel?: string;
  fillArea?: boolean;
  yScale?: MetricChartScale;
  extraSeries?: MetricChartSeries[];
  highlightEpoch?: number;
}

export default function MetricChart({
  title,
  color,
  epochs,
  values,
  height = 160,
  valueLabel = 'loss',
  fillArea = true,
  yScale = 'linear',
  extraSeries = [],
  highlightEpoch,
}: MetricChartProps) {
  const { theme } = useTheme();
  const { t } = useI18n();
  const isDark = theme === 'dark';

  // Guard against ResponsiveContainer measuring before layout settles
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [measured, setMeasured] = useState<{ width: number; height: number }>({ width: 0, height: 0 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const cr = entry.contentRect;
      const w = Math.max(0, Math.floor(cr.width));
      const h = Math.max(0, Math.floor(cr.height));
      setMeasured((prev) => {
        // If the element becomes temporarily hidden/collapsed (e.g. tab switch),
        // avoid pushing size to 0 which makes Recharts warn.
        if ((w === 0 || h === 0) && prev.width > 0 && prev.height > 0) return prev;
        if (prev.width === w && prev.height === h) return prev;
        return { width: w, height: h };
      });
    });

    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const chartData = useMemo(() => {
    if (!epochs || epochs.length === 0) return [];
    
    return epochs.map((epoch, i) => {
      const item: any = { epoch };
      
      // Main value
      const val = values[i];
      // Recharts log scale handles <= 0 poorly (it doesn't plot them).
      // We pass them as-is, Recharts will filter out invalid log values automatically or we can null them.
      // Better to null them to be explicit.
      if (typeof val === 'number' && Number.isFinite(val)) {
         if (yScale === 'log10' && val <= 0) {
             item.main = null; 
         } else {
             item.main = val;
         }
      } else {
        item.main = null;
      }

      // Extras
      extraSeries.forEach((s, sIdx) => {
        const sVal = s.values[i];
        if (typeof sVal === 'number' && Number.isFinite(sVal)) {
             if (yScale === 'log10' && sVal <= 0) {
                 item[`extra_${sIdx}`] = null;
             } else {
                 item[`extra_${sIdx}`] = sVal;
             }
        } else {
            item[`extra_${sIdx}`] = null;
        }
      });
      return item;
    });
  }, [epochs, values, extraSeries, yScale]);

  // Unique gradient ID to avoid conflicts - use stable ID based on title
  const gradientId = useMemo(() => {
    const titleHash = title.replace(/\s+/g, '_');
    // Use a hash of the title instead of random to ensure stability
    let hash = 0;
    for (let i = 0; i < titleHash.length; i++) {
      const char = titleHash.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return `grad_${titleHash}_${Math.abs(hash).toString(36)}`;
  }, [title]);

  if (!chartData.length) {
    return (
        <div style={{ height, width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: isDark ? '#555' : '#aaa', fontSize: 12, border: `1px solid ${isDark?'#333':'#eee'}`, borderRadius: 8 }}>
            {t.noData}
        </div>
    );
  }
  const minHeight = typeof height === 'number' ? height : 160;
  const canRenderChart = measured.width > 0 && measured.height > 0;

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height,
        // Ensure ResponsiveContainer never sees width/height -1
        minWidth: 0,
        minHeight,
        position: 'relative',
      }}
      className="relative"
    >
      <div className="absolute top-0 left-4 text-xs font-semibold z-10" style={{ color: isDark ? '#e2e8f0' : '#334155' }}>
        {title}
      </div>
      {canRenderChart ? (
        <ResponsiveContainer
          width="100%"
          height="100%"
          debounce={50}
          initialDimension={{ width: 1, height: 1 }}
        >
          <ComposedChart data={chartData} margin={{ top: 20, right: 10, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={color} stopOpacity={0.2} />
                <stop offset="95%" stopColor={color} stopOpacity={0} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#333' : '#eee'} vertical={false} />

            <XAxis
              dataKey="epoch"
              type="number"
              domain={['dataMin', 'dataMax']}
              tick={{ fontSize: 10, fill: isDark ? '#94a3b8' : '#64748b' }}
              tickLine={false}
              axisLine={false}
              minTickGap={30}
            />

            <YAxis
              scale={yScale === 'log10' ? 'log' : 'linear'}
              domain={['auto', 'auto']}
              tick={{ fontSize: 10, fill: isDark ? '#94a3b8' : '#64748b' }}
              tickFormatter={(val: number) => {
                if (Math.abs(val) >= 1000 || (Math.abs(val) < 0.001 && val !== 0)) {
                  return val.toExponential(1);
                }
                return val.toString();
              }}
              tickLine={false}
              axisLine={false}
              width={45}
              allowDataOverflow={true} // prevent scale breaking if log has negative numbers (though we nulled them)
            />

            <Tooltip
              contentStyle={{
                backgroundColor: isDark ? 'rgba(26, 31, 38, 0.95)' : 'rgba(255, 255, 255, 0.95)',
                borderColor: isDark ? '#333' : '#eee',
                fontSize: '12px',
                borderRadius: '6px',
                boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
              }}
              labelStyle={{ color: isDark ? '#cbd5e1' : '#64748b', marginBottom: '4px', fontWeight: 600 }}
              itemStyle={{ padding: 0 }}
              formatter={(value: any, name: any) => {
                const val = Number(value);
                const formatted =
                  Math.abs(val) < 0.0001 || Math.abs(val) > 1000 ? val.toExponential(4) : val.toFixed(4);

                if (name === 'main') return [formatted, valueLabel];
                if (typeof name === 'string' && name.startsWith('extra_')) {
                  const idx = parseInt(name.split('_')[1]);
                  return [formatted, extraSeries[idx].label];
                }
                return [formatted, name];
              }}
              labelFormatter={(label) => `${t.epoch} ${label}`}
            />

            {fillArea && (
              <Area
                type="monotone"
                dataKey="main"
                stroke="none"
                fill={`url(#${gradientId})`}
                connectNulls
                isAnimationActive={false}
              />
            )}

            <Line
              type="monotone"
              dataKey="main"
              stroke={color}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
              connectNulls
              name="main"
              isAnimationActive={false}
            />

            {highlightEpoch !== undefined && (
              <ReferenceLine x={highlightEpoch} stroke={isDark ? '#fff' : '#000'} strokeDasharray="3 3" />
            )}

            {extraSeries.map((s, idx) => (
              <Line
                key={idx}
                type="monotone"
                dataKey={`extra_${idx}`}
                stroke={s.color}
                strokeWidth={s.dashed ? 1.5 : 2}
                strokeDasharray={s.dashed ? '5 5' : undefined}
                dot={false}
                connectNulls
                name={`extra_${idx}`}
                isAnimationActive={false}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      ) : (
        <div
          style={{
            height: '100%',
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: isDark ? '#555' : '#aaa',
            fontSize: 12,
            border: `1px solid ${isDark ? '#333' : '#eee'}`,
            borderRadius: 8,
          }}
        >
          {t.loading}
        </div>
      )}
    </div>
  );
}
