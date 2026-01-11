import React, { useState, useRef, useEffect, ReactNode } from 'react';

interface DraggablePanelProps {
  children: ReactNode;
  initialX?: number;
  initialY?: number;
  style?: React.CSSProperties;
  className?: string;
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
}

export default function DraggablePanel({ 
  children, 
  initialX = 0, 
  initialY = 0, 
  style = {}, 
  className = '',
  position = 'bottom-right'
}: DraggablePanelProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  
  // We keep track of position in state for render, but also use ref for drag logic
  const [currentPos, setCurrentPos] = useState({ x: initialX, y: initialY });
  const currentPosRef = useRef({ x: initialX, y: initialY });
  const isDragging = useRef(false);
  const dragStart = useRef({ x: 0, y: 0 });
  const posStart = useRef({ x: 0, y: 0 });
  const [isGrabbing, setIsGrabbing] = useState(false);

  // Determine base positioning styles
  const getPositionStyles = () => {
    switch (position) {
      case 'top-left': return { top: 20, left: 20 };
      case 'top-right': return { top: 20, right: 20 };
      case 'bottom-left': return { bottom: 20, left: 20 };
      case 'bottom-right': return { bottom: 20, right: 20 };
      default: return { bottom: 20, right: 20 };
    }
  };

  useEffect(() => {
    let rafId: number | null = null;

    const onMove = (e: MouseEvent) => {
      if (!isDragging.current || !panelRef.current) return;

      if (rafId !== null) return;

      rafId = requestAnimationFrame(() => {
        rafId = null;
        const dx = e.clientX - dragStart.current.x;
        const dy = e.clientY - dragStart.current.y;
        
        const newPos = {
          x: posStart.current.x + dx,
          y: posStart.current.y + dy
        };
        
        currentPosRef.current = newPos;
        setCurrentPos(newPos);
      });
    };

    const onUp = () => {
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      if (isDragging.current) {
        isDragging.current = false;
        setIsGrabbing(false);
      }
    };

    // Use capture phase for move to ensure we get events even if other elements try to consume them
    window.addEventListener('mousemove', onMove, { passive: true, capture: true });
    window.addEventListener('mouseup', onUp);

    return () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
      window.removeEventListener('mousemove', onMove, { capture: true });
      window.removeEventListener('mouseup', onUp);
    };
  }, []);

  const onMouseDown = (e: React.MouseEvent) => {
    // Only left click
    if (e.button !== 0) return;
    
    e.stopPropagation(); // Prevent event bubbling
    isDragging.current = true;
    setIsGrabbing(true);
    dragStart.current = { x: e.clientX, y: e.clientY };
    posStart.current = { ...currentPosRef.current };
  };

  return (
    <div
      ref={panelRef}
      className={className}
      style={{
        position: 'absolute',
        ...getPositionStyles(),
        transform: `translate(${currentPos.x}px, ${currentPos.y}px)`,
        cursor: isGrabbing ? 'grabbing' : 'grab',
        touchAction: 'none', // Prevent touch scrolling while dragging
        willChange: isGrabbing ? 'transform' : 'auto', // Hardware acceleration hint when dragging
        zIndex: 20,
        ...style,
      }}
      onMouseDown={onMouseDown}
    >
      {children}
    </div>
  );
}
