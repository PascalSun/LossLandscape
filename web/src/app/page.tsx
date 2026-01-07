'use client';

import { useEffect, useState, useRef, useMemo } from 'react';
import { createPortal } from 'react-dom';
import dynamic from 'next/dynamic';
import { useI18n } from './i18n';
import * as Tooltip from '@radix-ui/react-tooltip';

// Dynamically import React Three Fiber components to avoid SSR issues
const LossLandscape3D = dynamic(() => import('./components/LossLandscape3D'), {
  ssr: false,
  loading: () => <div>Loading 3D visualization...</div>,
});

const LossLandscape2D = dynamic(() => import('./components/LossLandscape2D'), {
  ssr: false,
  loading: () => <div>Loading 2D visualization...</div>,
});

const LossVolumeSlice2D = dynamic(() => import('./components/LossVolumeSlice2D'), {
  ssr: false,
  loading: () => <div>Loading volume slice...</div>,
});

const LossVolumeRender3D = dynamic(() => import('./components/LossVolumeRender3D'), {
  ssr: false,
  loading: () => <div>Loading volume render...</div>,
});

type GenerateResponse =
  | { success: true; data: any; id: number }
  | { error: string };

export default function Page() {
  const { t } = useI18n();
  
  const [selectedRun, setSelectedRun] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // Viewer controls
  const [viewMode, setViewMode] = useState<'3d-surface' | '3d-slice' | '3d-volume'>('3d-surface');
  const [hoveredView, setHoveredView] = useState<string | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ top: number; left: number } | null>(null);
  const buttonRefs = useRef<Map<string, HTMLButtonElement>>(new Map());
  const [sliceIndex, setSliceIndex] = useState(0);
  const [sliceMode, setSliceMode] = useState<'2d' | '3d'>('3d');
  const [sliceAxis, setSliceAxis] = useState<'gamma' | 'alpha' | 'beta'>('gamma');
  const [surfaceMode, setSurfaceMode] = useState<'2d' | '3d'>('3d');
  
  // Draggable card positions
  const [surfaceCardPos, setSurfaceCardPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [sliceCardPos, setSliceCardPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDraggingSurfaceCard, setIsDraggingSurfaceCard] = useState(false);
  const [isDraggingSliceCard, setIsDraggingSliceCard] = useState(false);
  const surfaceCardDragStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const sliceCardDragStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const surfaceCardPosStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const sliceCardPosStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  
  // Available runs
  const [availableRuns, setAvailableRuns] = useState<string[]>([]);
  const [listLoading, setListLoading] = useState(false);

  async function refreshHistory() {
    setHistoryLoading(true);
    try {
      const res = await fetch('/api/landscapes');
      const json = await res.json();
      if (!res.ok) throw new Error(json?.error || 'Failed to load history');
      const historyList = Array.isArray(json) ? json : [];
      console.log('[page] Loaded history:', {
        count: historyList.length,
        items: historyList.map((h: any) => ({
          id: h.id,
          config_path: h.config_path,
          run_dir: h.run_dir,
          mode: h.mode,
          hasConfigPath: !!h.config_path,
          hasRunDir: !!h.run_dir,
          willShowConfigButton: !!(h.config_path || h.run_dir),
        })),
      });
      setHistory(historyList);
    } catch (e: any) {
      console.error('Failed to load history:', e);
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  }

  async function refreshRuns() {
    setListLoading(true);
    try {
      const res = await fetch('/api/list-runs');
      const json = await res.json();
      if (!res.ok) throw new Error(json?.error || 'Failed to load runs');
      setAvailableRuns(json.runs || []);
      
      // Set default run if the current one is not valid
      if (json.runs?.length > 0) {
        if (!selectedRun || !json.runs.includes(selectedRun)) {
          setSelectedRun(json.runs[0]);
        }
      }
    } catch (e: any) {
      console.error(e);
    } finally {
      setListLoading(false);
    }
  }

  async function loadFromRun(runPath: string) {
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const res = await fetch('/api/load-from-run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ runPath }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json?.error || 'Failed to load from run');
      
      // Normalize trajectory data
      const normalizedData = {
        ...json.data,
        trajectory_1: json.data?.trajectory_1 || json.data?.trajectory_data?.traj_1,
        trajectory_2: json.data?.trajectory_2 || json.data?.trajectory_data?.traj_2,
        trajectory_3: json.data?.trajectory_3 || json.data?.trajectory_data?.traj_3,
        trajectory_epochs: json.data?.trajectory_epochs || json.data?.trajectory_data?.epochs,
        run_dir: runPath,
      };
      
      setData(normalizedData);
      // Set view mode based on data
      if (normalizedData.loss_grid_3d) {
        setViewMode('3d-surface');
      } else {
        setViewMode('2d');
      }
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }


  useEffect(() => {
    refreshRuns();
  }, []);

  useEffect(() => {
    if (selectedRun) {
      loadFromRun(selectedRun);
    }
  }, [selectedRun]);

  // Ensure each load defaults to 3D surface
  useEffect(() => {
    if (!data) return;
    setViewMode('3d-surface');
  }, [data?.loss_grid_3d, data?.loss_grid_2d]);

  // Card drag handlers
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (isDraggingSurfaceCard) {
        const dx = e.clientX - surfaceCardDragStart.current.x;
        const dy = e.clientY - surfaceCardDragStart.current.y;
        setSurfaceCardPos({ x: surfaceCardPosStart.current.x + dx, y: surfaceCardPosStart.current.y + dy });
      }
      if (isDraggingSliceCard) {
        const dx = e.clientX - sliceCardDragStart.current.x;
        const dy = e.clientY - sliceCardDragStart.current.y;
        setSliceCardPos({ x: sliceCardPosStart.current.x + dx, y: sliceCardPosStart.current.y + dy });
      }
    };
    const onUp = () => {
      if (isDraggingSurfaceCard) setIsDraggingSurfaceCard(false);
      if (isDraggingSliceCard) setIsDraggingSliceCard(false);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [isDraggingSurfaceCard, isDraggingSliceCard]);

  const startSurfaceCardDrag = (e: React.MouseEvent) => {
    setIsDraggingSurfaceCard(true);
    surfaceCardDragStart.current = { x: e.clientX, y: e.clientY };
    surfaceCardPosStart.current = surfaceCardPos;
    e.preventDefault();
  };

  const startSliceCardDrag = (e: React.MouseEvent) => {
    setIsDraggingSliceCard(true);
    sliceCardDragStart.current = { x: e.clientX, y: e.clientY };
    sliceCardPosStart.current = sliceCardPos;
    e.preventDefault();
  };

  const renderSidebarContent = () => {
    return (
      <div>
        <div style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 16, lineHeight: 1.6 }}>
          {t.loadFromRunHint}
        </div>

        <div className="formGroup">
          <label className="formLabel">
            <span>{t.runDir}</span>
            <button
              className="btn btn-sm btn-secondary"
              onClick={refreshRuns}
              disabled={listLoading || loading}
              style={{ fontSize: 14, padding: '4px 8px', lineHeight: 1 }}
              title={t.refresh}
            >
              🔄
            </button>
          </label>
          <select
            className="formSelect"
            value={selectedRun}
            onChange={(e) => setSelectedRun(e.target.value)}
            disabled={loading || listLoading}
          >
            {availableRuns.length === 0 && <option value="">{t.noRunsFound}</option>}
            {availableRuns.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </div>

        {error && <div className="errorBox" style={{ marginTop: 16 }}>{error}</div>}
        
        {loading && (
          <div style={{ marginTop: 16, textAlign: 'center', color: 'var(--text-muted)' }}>
            <div className="spinner" style={{ width: 20, height: 20, margin: '0 auto 8px' }} />
            {t.loading}
          </div>
        )}
      </div>
    );
  };


  const sliceMeta = useMemo(() => {
    if (!data?.loss_grid_3d) return { zVals: [] as number[], nz: 0 };
    const nz = data.loss_grid_3d[0]?.[0]?.length || 0;
    if (nz <= 0) return { zVals: [] as number[], nz: 0 };

    const maybeZ = data.Z;
    if (Array.isArray(maybeZ) && maybeZ.length > 0) {
      const first = maybeZ[0];
      if (typeof first === 'number' || typeof first === 'string') {
        const arr = (maybeZ as any[]).map((v) => Number(v)).filter((v) => Number.isFinite(v));
        if (arr.length === nz) return { zVals: arr, nz };
      }
      if (Array.isArray(first)) {
        const zLine = (maybeZ[0]?.[0] ?? []) as any[];
        const arr = zLine.map((v) => Number(v)).filter((v) => Number.isFinite(v));
        if (arr.length === nz) return { zVals: arr, nz };
      }
    }

    // Fallback: simple 0..nz-1
    return { zVals: Array.from({ length: nz }, (_, i) => i), nz };
  }, [data?.loss_grid_3d, data?.Z]);

  // Reset slice index when data changes
  useEffect(() => {
    if (!sliceMeta.nz) return;
    setSliceIndex(Math.floor(sliceMeta.nz / 2));
  }, [sliceMeta.nz]);

  const renderViewerControls = () => {
    if (!data?.loss_grid_2d) return null;
    
    // Build available view modes based on data
    // Order: Surface (2D/3D toggle) -> Slice (2D/3D toggle) -> 3D volume
    const availableViews: Array<{key: typeof viewMode, label: string, requires3D: boolean, desc: string}> = [
      { key: '3d-surface', label: t.viewSurface, requires3D: false, desc: `${t.view2DDesc} / ${t.view3DSurfaceDesc}` },
    ];
    
    if (data.loss_grid_3d) {
      availableViews.push(
        { key: '3d-slice', label: t.viewSlice, requires3D: true, desc: t.view3DSliceDesc },
        { key: '3d-volume', label: t.viewVolume, requires3D: true, desc: t.view3DVolumeDesc }
      );
    }
    
    return (
      <div style={{ position: 'relative' }}>
        <Tooltip.Provider delayDuration={120}>
          <div className="viewModeTabs">
            {availableViews.map((v) => (
              <Tooltip.Root key={v.key}>
                <Tooltip.Trigger asChild>
                  <button
                    className={`viewModeTab ${viewMode === v.key ? 'active' : ''}`}
                    onClick={() => setViewMode(v.key)}
                  >
                    {v.label}
                  </button>
                </Tooltip.Trigger>
                <Tooltip.Portal>
                  <Tooltip.Content
                    side="top"
                    align="center"
                    sideOffset={6}
                    style={{
                      padding: '8px 12px',
                      borderRadius: 8,
                      background: 'rgba(0,0,0,0.92)',
                      border: '1px solid rgba(251,191,36,0.7)',
                      color: 'white',
                      fontSize: 11,
                      lineHeight: 1.5,
                      maxWidth: 260,
                      boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
                      zIndex: 2000,
                    }}
                  >
                    <div
                      style={{
                        fontWeight: 700,
                        marginBottom: 4,
                        textTransform: 'uppercase',
                        letterSpacing: '0.06em',
                        fontSize: 10,
                        color: '#fbbf24',
                      }}
                    >
                      {v.label}
                    </div>
                    <div style={{ opacity: 0.9 }}>{v.desc}</div>
                    <Tooltip.Arrow
                      width={10}
                      height={6}
                      style={{ fill: 'rgba(0,0,0,0.92)' }}
                    />
                  </Tooltip.Content>
                </Tooltip.Portal>
              </Tooltip.Root>
            ))}
          </div>
        </Tooltip.Provider>
      </div>
    );
  };

  return (
    <div className="mainWorkspace">
      <div className="sidebar">
        <div className="sidebarCard">
          
          <div className="sidebarContent">
            {renderSidebarContent()}
          </div>
        </div>
      </div>

      <div className="contentArea">
        <div className="viewerToolbar">
          <div className="viewerInfo">
            <span className="viewerTitle">
              {data ? (data.run_dir || t.viewerTitle) : t.viewerTitle}
            </span>
            {data && (
              <div className="badge">
                {data.loss_grid_3d ? '3D' : '2D'} · {t.grid} {data.grid_size || data.gridSize || 'N/A'}
              </div>
            )}
          </div>
          
          {renderViewerControls()}
        </div>

        <div className="viewerCanvas">
          {loading && (
            <div className="loadingOverlay">
              <div className="spinner" />
              <div style={{ fontSize: 14, fontWeight: 500 }}>{t.loadingOverlay}</div>
            </div>
          )}

          {data?.X?.length && (data?.loss_grid_2d?.length || data?.loss_grid_3d?.length) ? (
             // Render based on viewMode instead of mode
             viewMode === '3d-surface' ? (
                surfaceMode === '2d' ? (
                  <LossLandscape2D
                    X={data.X}
                    Y={data.Y}
                    lossGrid={data.loss_grid_2d}
                    baselineLoss={data.baseline_loss}
                    trajectory={
                      data.trajectory_1 && data.trajectory_1.length > 0
                        ? (() => {
                            const traj = {
                              traj_1: data.trajectory_1,
                              traj_2: data.trajectory_2,
                              epochs: data.trajectory_epochs || [],
                            };
                            return traj;
                          })()
                        : undefined
                    }
                  />
                ) : (
                  <LossLandscape3D
                    X={data.X}
                    Y={data.Y}
                    lossGrid={data.loss_grid_2d}
                    baselineLoss={data.baseline_loss}
                    trajectory={
                      data.trajectory_1 && data.trajectory_1.length > 0
                        ? (() => {
                            const traj = {
                              traj_1: data.trajectory_1,
                              traj_2: data.trajectory_2,
                              traj_3: data.trajectory_3,
                              epochs: data.trajectory_epochs || [],
                            };
                            return traj;
                          })()
                        : undefined
                    }
                  />
                )
            ) : viewMode === '3d-slice' && data.loss_grid_3d ? (
                sliceMode === '2d'
                  ? (
                    sliceAxis === 'gamma'
                      ? (
                  <LossVolumeSlice2D
                    X={data.X}
                    Y={data.Y}
                    Z={data.Z}
                    lossGrid3d={data.loss_grid_3d}
                    k={sliceIndex}
                    onKChange={setSliceIndex}
                    xLabel="α"
                    planeLabel="β"
                    trajectory={
                      data.trajectory_1 && data.trajectory_1.length > 0
                        ? {
                            traj_1: data.trajectory_1,
                            traj_2: data.trajectory_2,
                            traj_3: data.trajectory_3,
                            epochs: data.trajectory_epochs || [],
                          }
                        : undefined
                    }
                  />
                      )
                      : (
                        <LossLandscape2D
                          X={
                            (() => {
                              const loss3d = data.loss_grid_3d;
                              const nx = loss3d.length;
                              const ny = loss3d[0]?.length || 0;
                              const nz = loss3d[0]?.[0]?.length || 0;
                              const k = Math.max(0, sliceIndex);

                              if (sliceAxis === 'alpha') {
                                const i = Math.min(k, nx - 1);
                                const rows = ny;
                                const cols = nz;
                                const grid: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));
                                for (let j = 0; j < rows; j++) {
                                  const beta = data.Y?.[i]?.[j] ?? 0;
                                  for (let kk = 0; kk < cols; kk++) {
                                    grid[j][kk] = beta;
                                  }
                                }
                                return grid;
                              } else {
                                const j = Math.min(k, ny - 1);
                                const rows = nx;
                                const cols = nz;
                                const grid: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));
                                for (let i = 0; i < rows; i++) {
                                  const alpha = data.X?.[i]?.[j] ?? 0;
                                  for (let kk = 0; kk < cols; kk++) {
                                    grid[i][kk] = alpha;
                                  }
                                }
                                return grid;
                              }
                            })()
                          }
                          Y={
                            (() => {
                              const loss3d = data.loss_grid_3d;
                              const nz = loss3d[0]?.[0]?.length || 0;
                              const { zVals } = sliceMeta;
                              const cols = nz;
                              if (!cols) return [[]];

                              if (sliceAxis === 'alpha') {
                                const ny = loss3d[0]?.length || 0;
                                const rows = ny;
                                const grid: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));
                                for (let j = 0; j < rows; j++) {
                                  for (let kk = 0; kk < cols; kk++) {
                                    grid[j][kk] = zVals[kk] ?? kk;
                                  }
                                }
                                return grid;
                              } else {
                                const nx = loss3d.length;
                                const rows = nx;
                                const grid: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));
                                for (let i = 0; i < rows; i++) {
                                  for (let kk = 0; kk < cols; kk++) {
                                    grid[i][kk] = zVals[kk] ?? kk;
                                  }
                                }
                                return grid;
                              }
                            })()
                          }
                          lossGrid={
                            (() => {
                              const loss3d = data.loss_grid_3d;
                              const nx = loss3d.length;
                              const ny = loss3d[0]?.length || 0;
                              const nz = loss3d[0]?.[0]?.length || 0;
                              const k = Math.max(0, sliceIndex);

                              if (sliceAxis === 'alpha') {
                                const i = Math.min(k, nx - 1);
                                const rows = ny;
                                const cols = nz;
                                const out: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));
                                for (let j = 0; j < rows; j++) {
                                  for (let kk = 0; kk < cols; kk++) {
                                    out[j][kk] = loss3d[i]?.[j]?.[kk] ?? 0;
                                  }
                                }
                                return out;
                              } else {
                                const j = Math.min(k, ny - 1);
                                const rows = nx;
                                const cols = nz;
                                const out: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));
                                for (let i = 0; i < rows; i++) {
                                  for (let kk = 0; kk < cols; kk++) {
                                    out[i][kk] = loss3d[i]?.[j]?.[kk] ?? 0;
                                  }
                                }
                                return out;
                              }
                            })()
                          }
                          xLabel={sliceAxis === 'alpha' ? 'β' : 'α'}
                          planeLabel="γ"
                        />
                      )
                  )
                  : (
                    sliceAxis === 'gamma'
                      ? (
                        <LossLandscape3D
                          X={data.X}
                          Y={data.Y}
                          lossGrid={
                            (() => {
                              const rows = data.loss_grid_3d.length;
                              const cols = data.loss_grid_3d[0]?.length || 0;
                              const k = Math.min(Math.max(sliceIndex, 0), sliceMeta.nz ? sliceMeta.nz - 1 : 0);
                              const out: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));
                              for (let i = 0; i < rows; i++) {
                                for (let j = 0; j < cols; j++) {
                                  out[i][j] = data.loss_grid_3d[i][j][k] ?? 0;
                                }
                              }
                              return out;
                            })()
                          }
                          baselineLoss={data.baseline_loss}
                          trajectory={
                            data.trajectory_1 && data.trajectory_1.length > 0
                              ? {
                                  traj_1: data.trajectory_1,
                                  traj_2: data.trajectory_2,
                                  traj_3: data.trajectory_3,
                                  epochs: data.trajectory_epochs || [],
                                }
                              : undefined
                          }
                        />
                      )
                      : (
                        <LossLandscape3D
                    X={
                            (() => {
                              const loss3d = data.loss_grid_3d;
                              const nx = loss3d.length;
                              const ny = loss3d[0]?.length || 0;
                              const nz = loss3d[0]?.[0]?.length || 0;
                              const k = Math.max(0, sliceIndex);

                              if (sliceAxis === 'alpha') {
                                const i = Math.min(k, nx - 1);
                                const rows = ny;
                                const cols = nz;
                                const grid: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));
                                for (let j = 0; j < rows; j++) {
                                  const beta = data.Y?.[i]?.[j] ?? 0;
                                  for (let kk = 0; kk < cols; kk++) {
                                    grid[j][kk] = beta;
                                  }
                                }
                                return grid;
                              } else {
                                const j = Math.min(k, ny - 1);
                                const rows = nx;
                                const cols = nz;
                                const grid: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));
                                for (let i = 0; i < rows; i++) {
                                  const alpha = data.X?.[i]?.[j] ?? 0;
                                  for (let kk = 0; kk < cols; kk++) {
                                    grid[i][kk] = alpha;
                                  }
                                }
                                return grid;
                              }
                            })()
                          }
                          Y={
                            (() => {
                              const loss3d = data.loss_grid_3d;
                              const nz = loss3d[0]?.[0]?.length || 0;
                              const { zVals } = sliceMeta;
                              const cols = nz;
                              if (!cols) return [[]];

                              if (sliceAxis === 'alpha') {
                                const ny = loss3d[0]?.length || 0;
                                const rows = ny;
                                const grid: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));
                                for (let j = 0; j < rows; j++) {
                                  for (let kk = 0; kk < cols; kk++) {
                                    grid[j][kk] = zVals[kk] ?? kk;
                                  }
                                }
                                return grid;
                              } else {
                                const nx = loss3d.length;
                                const rows = nx;
                                const grid: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));
                                for (let i = 0; i < rows; i++) {
                                  for (let kk = 0; kk < cols; kk++) {
                                    grid[i][kk] = zVals[kk] ?? kk;
                                  }
                                }
                                return grid;
                              }
                            })()
                          }
                          lossGrid={
                            (() => {
                              const loss3d = data.loss_grid_3d;
                              const nx = loss3d.length;
                              const ny = loss3d[0]?.length || 0;
                              const nz = loss3d[0]?.[0]?.length || 0;
                              const k = Math.max(0, sliceIndex);

                              if (sliceAxis === 'alpha') {
                                const i = Math.min(k, nx - 1);
                                const rows = ny;
                                const cols = nz;
                                const out: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));
                                for (let j = 0; j < rows; j++) {
                                  for (let kk = 0; kk < cols; kk++) {
                                    out[j][kk] = loss3d[i]?.[j]?.[kk] ?? 0;
                                  }
                                }
                                return out;
                              } else {
                                const j = Math.min(k, ny - 1);
                                const rows = nx;
                                const cols = nz;
                                const out: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));
                                for (let i = 0; i < rows; i++) {
                                  for (let kk = 0; kk < cols; kk++) {
                                    out[i][kk] = loss3d[i]?.[j]?.[kk] ?? 0;
                                  }
                                }
                                return out;
                              }
                            })()
                          }
                          baselineLoss={data.baseline_loss}
                          xLabel={sliceAxis === 'alpha' ? 'β' : 'α'}
                          planeLabel="γ"
                        />
                      )
                  )
            ) : viewMode === '3d-volume' && data.loss_grid_3d ? (
                <LossVolumeRender3D
                  X={data.X}
                  Y={data.Y}
                  Z={data.Z}
                  lossGrid2d={data.loss_grid_2d}
                  lossGrid3d={data.loss_grid_3d}
                  baselineLoss={data.baseline_loss}
                  trajectory={data.trajectory_1 && data.trajectory_2 ? {
                    traj_1: data.trajectory_1,
                    traj_2: data.trajectory_2,
                    traj_3: data.trajectory_3,
                    epochs: data.trajectory_epochs || [],
                  } : undefined}
                />
            ) : (
              // Fallback to 3D surface if view mode is invalid
              <LossLandscape3D
                X={data.X}
                Y={data.Y}
                lossGrid={data.loss_grid_2d}
                baselineLoss={data.baseline_loss}
                trajectory={
                  data.trajectory_1
                    ? {
                        traj_1: data.trajectory_1,
                        traj_2: data.trajectory_2,
                        traj_3: data.trajectory_3,
                        epochs: data.trajectory_epochs || [],
                      }
                    : undefined
                }
              />
            )
          ) : (
            <div className="emptyState">
              <div className="emptyStateIcon">📊</div>
              <div>
                <h3 className="emptyStateTitle">{t.noDataTitle}</h3>
                <p className="emptyStateDesc">{t.noDataDesc}</p>
              </div>
            </div>
          )}

          {/* Surface view 2D/3D toggle, bottom-right */}
          {viewMode === '3d-surface' && (
            <div
              style={{
                position: 'absolute',
                right: 20,
                bottom: 20,
                transform: `translate(${surfaceCardPos.x}px, ${surfaceCardPos.y}px)`,
                display: 'flex',
                flexDirection: 'column',
                gap: 8,
                padding: '10px 12px',
                borderRadius: 14,
                border: '1px solid rgba(255,255,255,0.25)',
                background: 'rgba(0,0,0,0.75)',
                backdropFilter: 'blur(10px)',
                color: 'white',
                fontSize: 12,
                boxShadow: '0 6px 20px rgba(0,0,0,0.4)',
                pointerEvents: 'auto',
                zIndex: 20,
                width: 180,
                minHeight: 80, // keep height stable when toggling 2D/3D
                cursor: isDraggingSurfaceCard ? 'grabbing' : 'grab',
              }}
              onMouseDown={startSurfaceCardDrag}
            >
              <div style={{ fontWeight: 700, fontSize: 13 }}>{t.viewSurface}</div>
              <div style={{ display: 'flex', gap: 6 }}>
                <button
                  type="button"
                  onClick={() => setSurfaceMode('2d')}
                  style={{
                    padding: '4px 8px',
                    borderRadius: 999,
                    border: surfaceMode === '2d' ? '2px solid #fbbf24' : '1px solid rgba(255,255,255,0.3)',
                    background: surfaceMode === '2d' ? 'rgba(251,191,36,0.2)' : 'rgba(0,0,0,0.3)',
                    color: 'white',
                    fontSize: 10,
                    fontWeight: 600,
                    cursor: 'pointer',
                    flex: 1,
                  }}
                >
                  2D
                </button>
                <button
                  type="button"
                  onClick={() => setSurfaceMode('3d')}
                  style={{
                    padding: '4px 8px',
                    borderRadius: 999,
                    border: surfaceMode === '3d' ? '2px solid #fbbf24' : '1px solid rgba(255,255,255,0.3)',
                    background: surfaceMode === '3d' ? 'rgba(251,191,36,0.2)' : 'rgba(0,0,0,0.3)',
                    color: 'white',
                    fontSize: 10,
                    fontWeight: 600,
                    cursor: 'pointer',
                    flex: 1,
                  }}
                >
                  3D
                </button>
              </div>
            </div>
          )}

          {viewMode === '3d-slice' && (
            <>
              {/* Right-side axis + mode controls */}
              <div
                style={{
                  position: 'absolute',
                  right: 20,
                  bottom: 80, // sit just above the bottom slice slider
                  transform: `translate(${sliceCardPos.x}px, ${sliceCardPos.y}px)`,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 10,
                  padding: '10px 12px',
                  borderRadius: 14,
                  border: '1px solid rgba(255,255,255,0.2)',
                  background: 'rgba(0,0,0,0.7)',
                  backdropFilter: 'blur(10px)',
                  color: 'white',
                  fontSize: 12,
                  boxShadow: '0 6px 20px rgba(0,0,0,0.4)',
                  pointerEvents: 'auto',
                  zIndex: 20,
                  width: 180,
                  cursor: isDraggingSliceCard ? 'grabbing' : 'grab',
                }}
                onMouseDown={startSliceCardDrag}
              >
                <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 2 }}>{t.viewSlice}</div>
                <div style={{ opacity: 0.8, fontSize: 11, fontFamily: 'monospace', marginBottom: 6 }}>
                  {sliceAxis === 'gamma'
                    ? (sliceMeta.zVals[sliceIndex] !== undefined
                        ? `${t.gamma} = ${sliceMeta.zVals[sliceIndex].toFixed(3)}`
                        : `${t.gamma} index = ${sliceIndex}`)
                    : sliceAxis === 'alpha'
                      ? `α index = ${sliceIndex}`
                      : `β index = ${sliceIndex}`}
                </div>
                <div style={{ display: 'flex', gap: 6 }}>
                  <button
                    type="button"
                    onClick={() => setSliceMode('2d')}
                    style={{
                      padding: '4px 8px',
                      borderRadius: 999,
                      border: sliceMode === '2d' ? '2px solid #fbbf24' : '1px solid rgba(255,255,255,0.3)',
                      background: sliceMode === '2d' ? 'rgba(251,191,36,0.2)' : 'rgba(0,0,0,0.3)',
                      color: 'white',
                      fontSize: 10,
                      fontWeight: 600,
                      cursor: 'pointer',
                      flex: 1,
                    }}
                  >
                    2D
                  </button>
                  <button
                    type="button"
                    onClick={() => setSliceMode('3d')}
                    style={{
                      padding: '4px 8px',
                      borderRadius: 999,
                      border: sliceMode === '3d' ? '2px solid #fbbf24' : '1px solid rgba(255,255,255,0.3)',
                      background: sliceMode === '3d' ? 'rgba(251,191,36,0.2)' : 'rgba(0,0,0,0.3)',
                      color: 'white',
                      fontSize: 10,
                      fontWeight: 600,
                      cursor: 'pointer',
                      flex: 1,
                    }}
                  >
                    3D
                  </button>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 8, fontSize: 11, opacity: 0.85 }}>
                  <span style={{ fontWeight: 600 }}>Axis</span>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <button
                      type="button"
                      onClick={() => setSliceAxis('gamma')}
                      style={{
                        padding: '2px 8px',
                        borderRadius: 999,
                        border: sliceAxis === 'gamma' ? '2px solid #6ee7b7' : '1px solid rgba(255,255,255,0.3)',
                        background: sliceAxis === 'gamma' ? 'rgba(110,231,183,0.18)' : 'rgba(0,0,0,0.3)',
                        color: 'white',
                        fontSize: 10,
                        fontWeight: 600,
                        cursor: 'pointer',
                      }}
                    >
                      γ
                    </button>
                    <button
                      type="button"
                      onClick={() => setSliceAxis('alpha')}
                      style={{
                        padding: '2px 8px',
                        borderRadius: 999,
                        border: sliceAxis === 'alpha' ? '2px solid #60a5fa' : '1px solid rgba(255,255,255,0.3)',
                        background: sliceAxis === 'alpha' ? 'rgba(96,165,250,0.18)' : 'rgba(0,0,0,0.3)',
                        color: 'white',
                        fontSize: 10,
                        fontWeight: 600,
                        cursor: 'pointer',
                      }}
                    >
                      α-slice
                    </button>
                    <button
                      type="button"
                      onClick={() => setSliceAxis('beta')}
                      style={{
                        padding: '2px 8px',
                        borderRadius: 999,
                        border: sliceAxis === 'beta' ? '2px solid #fb7185' : '1px solid rgba(255,255,255,0.3)',
                        background: sliceAxis === 'beta' ? 'rgba(251,113,133,0.18)' : 'rgba(0,0,0,0.3)',
                        color: 'white',
                        fontSize: 10,
                        fontWeight: 600,
                        cursor: 'pointer',
                      }}
                    >
                      β-slice
                    </button>
                  </div>
                </div>
              </div>

              {/* Bottom slice index slider */}
              <div
                style={{
                  position: 'absolute',
                  left: 20,
                  right: 20,
                  bottom: 20,
                  display: 'flex',
                  gap: 16,
                  alignItems: 'center',
                  padding: '10px 14px',
                  borderRadius: 16,
                  border: '1px solid rgba(255,255,255,0.25)',
                  background: 'rgba(0,0,0,0.7)',
                  backdropFilter: 'blur(10px)',
                  color: 'white',
                  fontSize: 12,
                    boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
                    pointerEvents: 'auto',
                    zIndex: 20,
                }}
              >
                <div style={{ minWidth: 90, fontWeight: 700, fontSize: 12 }}>Slice:</div>
                <input
                  type="range"
                  min={0}
                  max={sliceMeta.nz - 1}
                  step={1}
                  value={Math.min(sliceIndex, sliceMeta.nz - 1)}
                  onChange={(e) => setSliceIndex(parseInt(e.target.value, 10))}
                  style={{
                    flex: 1,
                    accentColor: '#fbbf24',
                  }}
                />
                <div
                  style={{
                    minWidth: 80,
                    textAlign: 'right',
                    opacity: 0.9,
                    fontFamily: 'monospace',
                    fontSize: 12,
                  }}
                >
                  {sliceIndex} / {sliceMeta.nz - 1}
                </div>
              </div>
            </>
          )}
        </div>
      </div>

    </div>
  );
}
