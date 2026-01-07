'use client';

import React, { useMemo } from 'react';

type HoverCardProps = {
  x: number;
  y: number;
  children: React.ReactNode;
  visible: boolean;
  maxWidth?: number;
  margin?: number;
  /** 
   * 'default' keeps tooltip away from the cursor a bit,
   * 'tight' keeps it close to the cursor (for data-point hovers).
   */
  behavior?: 'default' | 'tight';
};

/**
 * Lightweight, clamped hover card used by both 2D/3D views.
 * Expects viewport coordinates (clientX/clientY).
 */
export function HoverCard({
  x,
  y,
  children,
  visible,
  maxWidth = 260,
  margin = 12,
  behavior = 'default',
}: HoverCardProps) {
  const { pos, arrow } = useMemo(() => {
    if (!visible) {
      return {
        pos: { left: -9999, top: -9999 },
        arrow: { side: 'top' as 'top' | 'bottom' | 'left' | 'right', offset: 0 },
      };
    }
    const vw = typeof window !== 'undefined' ? window.innerWidth : 0;
    const vh = typeof window !== 'undefined' ? window.innerHeight : 0;
    const estHeight = 140; // approximate tooltip height

    // For tight behavior, keep tooltip at a fixed small offset from the mouse,
    // without any fancy repositioning logic. This ensures it visually "sticks"
    // to the cursor in 2D/3D views.
    if (behavior === 'tight') {
      const dx = 8;
      const dy = 8;
      // No clamping except minimal to keep it on screen; we avoid shifting direction.
      const rawLeft = x + dx;
      const rawTop = y + dy;
      const left = Math.max(0, Math.min(rawLeft, vw - maxWidth));
      const top = Math.max(0, Math.min(rawTop, vh - estHeight));

      return {
        pos: { left, top },
        arrow: { side: 'top' as const, offset: 24 },
      };
    }

    const radius = 40; // keep this radius around cursor free of card for default mode

    type Side = 'top' | 'bottom' | 'left' | 'right';

    const dxRight = 24;
    const dyDown = 24;

    const candidates: { left: number; top: number; side: Side }[] = [
      // Prefer above-right
      { left: x + dxRight, top: y - estHeight - dyDown, side: 'bottom' },
      // Above-left
      { left: x - maxWidth - dxRight, top: y - estHeight - dyDown, side: 'bottom' },
      // Below-right
      { left: x + dxRight, top: y + dyDown, side: 'top' },
      // Below-left
      { left: x - maxWidth - dxRight, top: y + dyDown, side: 'top' },
    ];

    const fits = (c: { left: number; top: number }) =>
      c.left >= margin &&
      c.left + maxWidth <= vw - margin &&
      c.top >= margin &&
      c.top + estHeight <= vh - margin;

    const avoidsCursor = (c: { left: number; top: number }) => {
      const cx = Math.max(c.left, Math.min(x, c.left + maxWidth));
      const cy = Math.max(c.top, Math.min(y, c.top + estHeight));
      const dx = cx - x;
      const dy = cy - y;
      return dx * dx + dy * dy >= radius * radius;
    };

    let chosen = candidates.find((c) => fits(c) && avoidsCursor(c));
    if (!chosen) {
      chosen = candidates.find(fits) ?? candidates[0];
    }

    const left = Math.min(Math.max(chosen.left, margin), Math.max(margin, vw - maxWidth - margin));
    const top = Math.min(Math.max(chosen.top, margin), Math.max(margin, vh - estHeight - margin));

    const side: Side = chosen.side;
    const offset =
      side === 'top' || side === 'bottom'
        ? Math.max(12, Math.min(maxWidth - 12, x - left))
        : Math.max(12, Math.min(estHeight - 12, y - top));

    return {
      pos: { left, top },
      arrow: { side, offset },
    };
  }, [visible, x, y, maxWidth, margin]);

  if (!visible) return null;

  return (
    <div
      style={{
        position: 'fixed',
        left: pos.left,
        top: pos.top,
        padding: '14px 16px',
        borderRadius: 14,
        border: '2px solid rgba(251, 191, 36, 0.5)',
        background: 'rgba(0,0,0,0.9)',
        backdropFilter: 'blur(12px)',
        color: 'white',
        fontSize: 12,
        lineHeight: 1.7,
        minWidth: 200,
        maxWidth,
        pointerEvents: 'none',
        boxShadow: '0 8px 24px rgba(0,0,0,0.6), 0 0 20px rgba(251, 191, 36, 0.3)',
        zIndex: 1000,
      }}
    >
      {/* Arrow */}
      <div
        style={{
          position: 'absolute',
          width: 0,
          height: 0,
          ...(arrow.side === 'top'
            ? {
                left: arrow.offset,
                top: -8,
                borderLeft: '8px solid transparent',
                borderRight: '8px solid transparent',
                borderBottom: '8px solid rgba(0,0,0,0.9)',
              }
            : arrow.side === 'bottom'
            ? {
                left: arrow.offset,
                bottom: -8,
                borderLeft: '8px solid transparent',
                borderRight: '8px solid transparent',
                borderTop: '8px solid rgba(0,0,0,0.9)',
              }
            : arrow.side === 'left'
            ? {
                top: arrow.offset,
                left: -8,
                borderTop: '8px solid transparent',
                borderBottom: '8px solid transparent',
                borderRight: '8px solid rgba(0,0,0,0.9)',
              }
            : {
                top: arrow.offset,
                right: -8,
                borderTop: '8px solid transparent',
                borderBottom: '8px solid transparent',
                borderLeft: '8px solid rgba(0,0,0,0.9)',
              }),
        }}
      />
      {children}
    </div>
  );
}

