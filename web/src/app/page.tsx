'use client';

import { useEffect, useState, useRef, useMemo, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { useI18n } from './i18n';
import { useTheme } from './theme';
import * as Tooltip from '@radix-ui/react-tooltip';
import MetricChart from './components/MetricChart';
import Modal from './components/Modal';
import DraggablePanel from './components/DraggablePanel';

// Dynamically import React Three Fiber components to avoid SSR issues
const LossLandscape3D = dynamic(() => import('./components/LossLandscape3D'), {
  ssr: false,
  loading: () => <div>Loading 3D visualization...</div>,
});

const LossLandscape2D = dynamic(() => import('./components/LossLandscape2D'), {
  ssr: false,
  loading: () => <div>Loading 2D visualization...</div>,
});

const LossLandscape1D = dynamic(() => import('./components/LossLandscape1D'), {
  ssr: false,
  loading: () => <div>Loading 1D visualization...</div>,
});

const LossVolumeSlice2D = dynamic(() => import('./components/LossVolumeSlice2D'), {
  ssr: false,
  loading: () => <div>Loading volume slice...</div>,
});

const LossVolumeRender3D = dynamic(() => import('./components/LossVolumeRender3D'), {
  ssr: false,
  loading: () => <div>Loading volume render...</div>,
});

const HessianAnalysis = dynamic(() => import('./components/HessianAnalysis'), {
  ssr: false,
  loading: () => <div>Loading Hessian analysis...</div>,
});

export default function Page() {
  const { t } = useI18n();
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  
  // Theme-aware color helper
  const getThemeColor = (lightColor: string, darkColor: string) => {
    return isDark ? darkColor : lightColor;
  };
  
  const [selectedRun, setSelectedRun] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // Viewer controls
  const [viewMode, setViewMode] = useState<'1d' | '2d' | '3d' | 'metadata' | 'hessian'>('2d');
  const [view3DRenderMode, setView3DRenderMode] = useState<'slice' | 'volume'>('slice');
  const [sliceIndex, setSliceIndex] = useState(0);
  const [sliceMode, setSliceMode] = useState<'2d' | '3d'>('3d');
  const [sliceAxis, setSliceAxis] = useState<'gamma' | 'alpha' | 'beta'>('gamma');
  const [surfaceMode, setSurfaceMode] = useState<'2d' | '3d'>('3d');
  
  // Draggable card positions
  const [surfaceCardPos, setSurfaceCardPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [sliceCardPos, setSliceCardPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [renderModeCardPos, setRenderModeCardPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  // Use refs for drag state to avoid re-binding event listeners
  const isDraggingSurfaceCard = useRef(false);
  const isDraggingSliceCard = useRef(false);
  const isDraggingRenderModeCard = useRef(false);
  const surfaceCardDragStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const sliceCardDragStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const renderModeCardDragStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const surfaceCardPosStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const sliceCardPosStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const renderModeCardPosStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  // Track cursor state for UI feedback (only update on drag start/end)
  const [isDraggingAnyCard, setIsDraggingAnyCard] = useState(false);
  
  // Available runs
  const [availableRuns, setAvailableRuns] = useState<string[]>([]);
  const [listLoading, setListLoading] = useState(false);
  
  // Sidebar tabs
  const [activeTab, setActiveTab] = useState<'run' | 'History' | 'upload'>('run');
  
  // History
  const [history, setHistory] = useState<any[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  
  // Modal state
  const [modalState, setModalState] = useState<{
    isOpen: boolean;
    type: 'confirm' | 'alert';
    title: string;
    message: string;
    onConfirm?: () => void;
  }>({
    isOpen: false,
    type: 'alert',
    title: '',
    message: '',
  });
  
  // Metadata section expand/collapse state - all sections default to collapsed
  const [expandedMetadataSections, setExpandedMetadataSections] = useState<Set<string>>(new Set());

  // Upload state
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

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

  async function deleteHistoryItem(id: number) {
    setModalState({
      isOpen: true,
      type: 'confirm',
      title: t.deleteLandscape,
      message: t.deleteConfirm,
      onConfirm: async () => {
        setDeletingId(id);
        try {
          const res = await fetch(`/api/landscape/${id}`, {
            method: 'DELETE',
          });

          const json = await res.json();
          if (!res.ok) {
            throw new Error(json?.error || 'Failed to delete');
          }

          // Refresh history list
          await refreshHistory();
        } catch (e: any) {
          setModalState({
            isOpen: true,
            type: 'alert',
            title: t.deleteFailed,
            message: `${t.deleteFailedMessage}: ${e?.message || String(e)}`,
          });
        } finally {
          setDeletingId(null);
        }
      },
    });
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
      
      // Handle export in progress (202 Accepted)
      if (res.status === 202) {
        const retryAfter = json.retry_after || 5000;
        const message = json.message || 'Export is in progress. Please wait...';
        setError(`${message} (Will retry in ${retryAfter / 1000}s)`);
        
        // Auto-retry after the suggested delay
        setTimeout(() => {
          loadFromRun(runPath);
        }, retryAfter);
        return;
      }
      
      if (!res.ok) {
        // Provide more user-friendly error messages
        let errorMessage = json?.error || 'Failed to load from run';
        if (json?.message) {
          errorMessage = json.message;
        } else if (json?.status === 'failed' && json?.error_details) {
          errorMessage = `Export failed: ${json.error_details}`;
        } else if (res.status === 422) {
          errorMessage = json?.message || 'The landscape file appears to be incomplete or corrupted.';
        }
        throw new Error(errorMessage);
      }
      
      // Normalize trajectory data
      const normalizedData = {
        ...json.data,
        trajectory_1: json.data?.trajectory_1 || json.data?.trajectory_data?.traj_1,
        trajectory_2: json.data?.trajectory_2 || json.data?.trajectory_data?.traj_2,
        trajectory_3: json.data?.trajectory_3 || json.data?.trajectory_data?.traj_3,
        trajectory_epochs: json.data?.trajectory_epochs || json.data?.trajectory_data?.epochs,
        trajectory_losses: json.data?.trajectory_losses || json.data?.trajectory_data?.losses,
        trajectory_val_losses: json.data?.trajectory_val_losses || json.data?.trajectory_data?.val_losses,
        run_dir: runPath,
      };
      
      setData(normalizedData);
      // Set view mode based on data
      if (normalizedData.loss_grid_3d) {
        setViewMode('2d');
      } else {
        setViewMode('2d');
        setSurfaceMode('2d');
      }
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  async function loadFromId(id: number) {
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const res = await fetch(`/api/landscape/${id}`);
      const json = await res.json();
      
      if (!res.ok) {
        throw new Error(json?.error || 'Failed to load landscape');
      }
      
      // Normalize trajectory data
      const normalizedData = {
        ...json,
        trajectory_1: json?.trajectory_1 || json?.trajectory_data?.traj_1,
        trajectory_2: json?.trajectory_2 || json?.trajectory_data?.traj_2,
        trajectory_3: json?.trajectory_3 || json?.trajectory_data?.traj_3,
        trajectory_epochs: json?.trajectory_epochs || json?.trajectory_data?.epochs,
        trajectory_losses: json?.trajectory_losses || json?.trajectory_data?.losses,
        trajectory_val_losses: json?.trajectory_val_losses || json?.trajectory_data?.val_losses,
        run_dir: json?.run_dir,
      };
      
      setData(normalizedData);
      // Set view mode based on data
      if (normalizedData.loss_grid_3d) {
        setViewMode('2d');
      } else {
        setViewMode('2d');
        setSurfaceMode('2d');
      }
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }


  useEffect(() => {
    refreshRuns();
    refreshHistory();
  }, []);

  // Ensure each load defaults to 2D view
  useEffect(() => {
    if (!data) return;
    setViewMode('2d');
  }, [data?.loss_grid_3d, data?.loss_grid_2d]);

  // Optimize 2D/3D toggle handlers with useCallback and stopPropagation
  const handleSetSurfaceMode2D = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setSurfaceMode('2d');
  }, []);

  const handleSetSurfaceMode3D = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setSurfaceMode('3d');
  }, []);

  // Memoize button styles to avoid recreating on every render - match top-right component style
  const buttonBaseStyle = useMemo(() => ({
    padding: '6px 14px' as const,
    borderRadius: 8,
    fontSize: 12,
    fontWeight: 600,
    cursor: 'pointer' as const,
    flex: 1,
    transition: 'all 0.2s' as const, // Add smooth transition like top-right component
  }), []);

  const button2DStyle = useMemo(() => ({
    ...buttonBaseStyle,
    border: surfaceMode === '2d' ? '2px solid var(--accent)' : `1px solid ${isDark ? 'rgba(255,255,255,0.22)' : 'rgba(15,23,42,0.12)'}`,
    background: surfaceMode === '2d' ? 'rgba(249,115,22,0.15)' : (isDark ? 'rgba(255,255,255,0.06)' : 'rgba(15,23,42,0.04)'),
    color: isDark ? 'rgba(255,255,255,0.95)' : '#0f172a',
  }), [surfaceMode, buttonBaseStyle, isDark]);

  const button3DStyle = useMemo(() => ({
    ...buttonBaseStyle,
    border: surfaceMode === '3d' ? '2px solid var(--accent)' : `1px solid ${isDark ? 'rgba(255,255,255,0.22)' : 'rgba(15,23,42,0.12)'}`,
    background: surfaceMode === '3d' ? 'rgba(249,115,22,0.15)' : (isDark ? 'rgba(255,255,255,0.06)' : 'rgba(15,23,42,0.04)'),
    color: isDark ? 'rgba(255,255,255,0.95)' : '#0f172a',
  }), [surfaceMode, buttonBaseStyle, isDark]);

  // Memoize card visual style (layout styles handled by DraggablePanel)
  const cardVisualStyle = useMemo(() => ({
    display: 'flex',
    flexDirection: 'column' as const,
    gap: 10,
    padding: '14px 16px',
    borderRadius: 16,
    border: `1px solid ${isDark ? 'rgba(255,255,255,0.22)' : 'rgba(15,23,42,0.12)'}`,
    background: isDark ? 'rgba(0,0,0,0.72)' : 'rgba(255,255,255,0.86)',
    backdropFilter: 'blur(12px)',
    color: isDark ? 'rgba(255,255,255,0.95)' : '#0f172a',
    fontSize: 13,
    boxShadow: isDark ? '0 12px 40px rgba(0,0,0,0.5)' : '0 12px 40px rgba(15,23,42,0.12)',
    width: 300,
    minHeight: 80,
  }), [isDark]);

  const renderRunTab = () => {
    return (
      <div>

        <div className="formGroup" style={{ marginBottom: 16 }}>
          <label className="formLabel">
            <span>{t.runDir}</span>
            <button
              className={`refreshButton ${listLoading ? 'refreshing' : ''}`}
              onClick={refreshRuns}
              disabled={listLoading || loading}
              title={t.refresh}
            >
              <span className="refreshIcon">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path d="M1 4v6h6" />
                  <path d="M23 20v-6h-6" />
                  <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15" />
                </svg>
              </span>
            </button>
          </label>
        </div>

        {listLoading ? (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '20px 0' }}>
            <div className="spinner" style={{ width: 20, height: 20, margin: '0 auto 8px' }} />
            {t.loading}
          </div>
        ) : availableRuns.length === 0 ? (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '20px 0' }}>
            {t.noRunsFound}
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {availableRuns.map((r) => (
              <div
                key={r}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '10px 12px',
                  borderRadius: 8,
                  border: selectedRun === r ? '1px solid var(--accent)' : '1px solid var(--border)',
                  background: selectedRun === r ? 'rgba(249,115,22,0.1)' : 'transparent',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                }}
                onClick={() => setSelectedRun(r)}
              >
                <span style={{ fontSize: 14, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {r}
                </span>
                <button
                  className="btn btn-sm btn-primary"
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedRun(r);
                    loadFromRun(r);
                  }}
                  disabled={loading}
                  style={{ fontSize: 13, padding: '4px 12px', marginLeft: 8 }}
                >
                  {t.import}
                </button>
              </div>
            ))}
          </div>
        )}

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

  const renderHistoryTab = () => {
    return (
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          <div style={{ fontSize: 14, color: 'var(--text-muted)' }}>
            {t.previouslyGenerated}
          </div>
          <button
            className={`refreshButton ${historyLoading ? 'refreshing' : ''}`}
            onClick={refreshHistory}
            disabled={historyLoading}
            title={t.refresh}
          >
            <span className="refreshIcon">
              <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M1 4v6h6" />
                <path d="M23 20v-6h-6" />
                <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15" />
              </svg>
            </span>
          </button>
        </div>

        {historyLoading ? (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '20px 0' }}>
            <div className="spinner" style={{ width: 20, height: 20, margin: '0 auto 8px' }} />
            {t.loading}
          </div>
        ) : history.length === 0 ? (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '20px 0' }}>
            {t.noHistoryFound}
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {history.map((h: any) => {
              const getSourceIcon = (source?: string) => {
                switch (source) {
                  case 'upload': return '📤';
                  case 'run': return '📁';
                  case 'scan': return '🔍';
                  case 'import': return '📥';
                  case 'generate': return '⚡';
                  default: return '📊';
                }
              };

              const getSourceColor = (source?: string) => {
                switch (source) {
                  case 'upload': return '#3b82f6';
                  case 'run': return '#10b981';
                  case 'scan': return '#8b5cf6';
                  case 'import': return '#06b6d4';
                  case 'generate': return '#f59e0b';
                  default: return 'var(--text-muted)';
                }
              };

              return (
                <div
                  key={h.id}
                  style={{
                    padding: '16px',
                    borderRadius: 12,
                    border: '1px solid var(--border)',
                    background: 'var(--bg-history-item)',
                    transition: 'all 0.2s ease',
                    boxShadow: 'var(--shadow)',
                    position: 'relative',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = 'var(--accent)';
                    e.currentTarget.style.background = 'var(--bg-history-item-hover)';
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = 'var(--shadow-lg)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = 'var(--border)';
                    e.currentTarget.style.background = 'var(--bg-history-item)';
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = 'var(--shadow)';
                  }}
                >
                  {/* Header with ID and Delete button */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{
                        width: 32,
                        height: 32,
                        borderRadius: 8,
                        background: 'var(--accent-gradient)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 16,
                        fontWeight: 700,
                        color: '#fff',
                        flexShrink: 0,
                      }}>
                        #{h.id}
                      </div>
                      {h.import_source && (
                        <div style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 4,
                          padding: '4px 10px',
                          borderRadius: 6,
                          background: `${getSourceColor(h.import_source)}15`,
                          border: `1px solid ${getSourceColor(h.import_source)}40`,
                          fontSize: 12,
                          fontWeight: 600,
                          color: getSourceColor(h.import_source),
                        }}>
                          <span>{getSourceIcon(h.import_source)}</span>
                          <span style={{ textTransform: 'capitalize' }}>{h.import_source}</span>
                        </div>
                      )}
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteHistoryItem(h.id);
                      }}
                      disabled={deletingId === h.id}
                      style={{
                        width: 28,
                        height: 28,
                        padding: 0,
                        borderRadius: 6,
                        border: '1px solid rgba(239, 68, 68, 0.3)',
                        background: deletingId === h.id 
                          ? 'rgba(239, 68, 68, 0.2)' 
                          : 'transparent',
                        color: '#ef4444',
                        fontSize: 16,
                        cursor: deletingId === h.id ? 'not-allowed' : 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        transition: 'all 0.2s ease',
                        flexShrink: 0,
                      }}
                      onMouseEnter={(e) => {
                        if (deletingId !== h.id) {
                          e.currentTarget.style.background = 'rgba(239, 68, 68, 0.15)';
                          e.currentTarget.style.borderColor = '#ef4444';
                          e.currentTarget.style.transform = 'scale(1.1)';
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (deletingId !== h.id) {
                          e.currentTarget.style.background = 'transparent';
                          e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.3)';
                          e.currentTarget.style.transform = 'scale(1)';
                        }
                      }}
                      title={t.delete}
                    >
                      {deletingId === h.id ? (
                        <div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />
                      ) : (
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M3 6h18" />
                          <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
                          <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                          <line x1="10" y1="11" x2="10" y2="17" />
                          <line x1="14" y1="11" x2="14" y2="17" />
                        </svg>
                      )}
                    </button>
                  </div>

                  {/* Info section */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 12 }}>
                    {h.run_dir && (
                      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
                        <span style={{ fontSize: 14, opacity: 0.7, flexShrink: 0, marginTop: 2 }}>📁</span>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 2, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                            {t.runDirectory}
                          </div>
                          <div style={{ fontSize: 13, color: 'var(--text-primary)', wordBreak: 'break-all' }}>
                            {h.run_dir}
                          </div>
                        </div>
                      </div>
                    )}
                    {h.import_filename && (
                      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
                        <span style={{ fontSize: 14, opacity: 0.7, flexShrink: 0, marginTop: 2 }}>📄</span>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 2, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                            {t.file}
                          </div>
                          <div style={{ fontSize: 13, color: 'var(--text-primary)', wordBreak: 'break-all' }}>
                            {h.import_filename}
                          </div>
                        </div>
                      </div>
                    )}
                    {(h.imported_at || h.created_at) && (
                      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
                        <span style={{ fontSize: 14, opacity: 0.7, flexShrink: 0, marginTop: 2 }}>🕒</span>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 2, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                            {h.imported_at ? t.imported : t.created}
                          </div>
                          <div style={{ fontSize: 13, color: 'var(--text-primary)' }}>
                            {h.imported_at 
                              ? new Date(h.imported_at).toLocaleString(undefined, { 
                                  year: 'numeric', 
                                  month: 'short', 
                                  day: 'numeric', 
                                  hour: '2-digit', 
                                  minute: '2-digit' 
                                })
                              : new Date(h.created_at).toLocaleString(undefined, { 
                                  year: 'numeric', 
                                  month: 'short', 
                                  day: 'numeric', 
                                  hour: '2-digit', 
                                  minute: '2-digit' 
                                })
                            }
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Load button */}
                  <button
                    className="btn btn-sm btn-primary"
                    onClick={() => {
                      // Prefer loading from database by ID (standalone, no file needed)
                      // Only fall back to loadFromRun if no ID is available
                      if (h.id) {
                        loadFromId(h.id);
                      } else if (h.run_dir) {
                        loadFromRun(h.run_dir);
                      }
                    }}
                    disabled={loading || (!h.run_dir && !h.id)}
                    style={{ 
                      fontSize: 13, 
                      padding: '8px 16px', 
                      width: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: 6,
                    }}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="7 10 12 15 17 10" />
                      <line x1="12" y1="15" x2="12" y2="3" />
                    </svg>
                    {t.load}
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  function handleFileSelect(file: File) {
    const fileName = file.name.toLowerCase();
    const isJson = fileName.endsWith('.json');

    if (!isJson) {
      setUploadError(t.fileMustBeJson);
      return;
    }

    setUploadFile(file);
    setUploadError(null);
    setUploadSuccess(false);
  }

  async function handleUpload() {
    if (!uploadFile) {
      setUploadError(t.pleaseSelectFile);
      return;
    }

    setUploading(true);
    setUploadError(null);
    setUploadSuccess(false);

    try {
      const formData = new FormData();
      formData.append('file', uploadFile);

      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const json = await res.json();

      if (!res.ok) {
        let errorMessage = json?.error || t.uploadFailed;
        if (json?.validationErrors && Array.isArray(json.validationErrors)) {
          errorMessage += ': ' + json.validationErrors.join('; ');
        }
        throw new Error(errorMessage);
      }

      setUploadSuccess(true);
      setUploadFile(null);
      
      // Refresh history to show the new import
      await refreshHistory();

      // Optionally load the uploaded data
      if (json.id) {
        setTimeout(() => {
          loadFromId(json.id);
        }, 500);
      }
    } catch (e: any) {
      setUploadError(e?.message || String(e));
    } finally {
      setUploading(false);
    }
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (!uploading) {
      setIsDragging(true);
    }
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (uploading) {
      return;
    }

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  }

  const renderUploadTab = () => {
    return (
      <div>
        <div style={{ fontSize: 14, color: 'var(--text-muted)', marginBottom: 16, lineHeight: 1.7 }}>
          {t.uploadLandscapeHint}
        </div>

        {/* Drag and drop area */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          style={{
            padding: '40px 20px',
            borderRadius: 12,
            border: `2px dashed ${isDragging ? 'var(--accent)' : 'var(--border)'}`,
            background: isDragging
              ? 'rgba(249, 115, 22, 0.1)'
              : uploadFile
              ? 'var(--bg-history-item)'
              : 'var(--bg-input)',
            textAlign: 'center',
            cursor: uploading ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s ease',
            marginBottom: 16,
            position: 'relative',
          }}
          onClick={() => {
            if (!uploading) {
              document.getElementById('file-input')?.click();
            }
          }}
        >
          <input
            id="file-input"
            type="file"
            accept=".json"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) {
                handleFileSelect(file);
              }
            }}
            disabled={uploading}
            style={{ display: 'none' }}
          />
          {uploadFile ? (
            <div>
              <div style={{ fontSize: 16, marginBottom: 8, color: 'var(--text-primary)' }}>
                📄 {uploadFile.name}
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                {t.size}: {(uploadFile.size / 1024 / 1024).toFixed(2)} MB
              </div>
              <div
                style={{
                  marginTop: 12,
                  fontSize: 12,
                  color: 'var(--text-muted)',
                  textDecoration: 'underline',
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  setUploadFile(null);
                  setUploadError(null);
                  setUploadSuccess(false);
                }}
              >
                {t.clickToChangeFile}
              </div>
            </div>
          ) : (
            <div>
              <div style={{ fontSize: 48, marginBottom: 16 }}>📤</div>
              <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 8, color: 'var(--text-primary)' }}>
                {isDragging ? t.dropFileHere : t.dragDropFile}
              </div>
              <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
                {t.orClickToBrowse}
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>
                {t.supportsJsonOnly}
              </div>
            </div>
          )}
        </div>


        {uploadError && (
          <div
            style={{
              padding: '12px',
              borderRadius: 8,
              border: '1px solid #ef4444',
              background: 'rgba(239, 68, 68, 0.1)',
              color: '#ef4444',
              fontSize: 13,
              marginBottom: 16,
            }}
          >
            {uploadError}
          </div>
        )}

        {uploadSuccess && (
          <div
            style={{
              padding: '12px',
              borderRadius: 8,
              border: '1px solid #10b981',
              background: 'rgba(16, 185, 129, 0.1)',
              color: '#10b981',
              fontSize: 13,
              marginBottom: 16,
            }}
          >
            ✓ {t.fileUploadedSuccess}
          </div>
        )}

        <button
          className="btn btn-primary"
          onClick={handleUpload}
          disabled={!uploadFile || uploading}
          style={{
            width: '100%',
            opacity: (!uploadFile || uploading) ? 0.6 : 1,
            cursor: (!uploadFile || uploading) ? 'not-allowed' : 'pointer',
          }}
        >
          {uploading ? (
            <>
              <span className="spinner" style={{ width: 14, height: 14, marginRight: 8 }} />
              {t.uploadingAndImporting}
            </>
          ) : (
            t.uploadAndImport
          )}
        </button>
      </div>
    );
  };

  const renderMetadataTab = () => {
    if (!data || !data.metadata) {
      return null;
    }

    const metadata = data.metadata;
    
    const toggleSection = (title: string) => {
      setExpandedMetadataSections(prev => {
        const newSet = new Set(prev);
        if (newSet.has(title)) {
          newSet.delete(title);
        } else {
          newSet.add(title);
        }
        return newSet;
      });
    };

    // Section icons and colors mapping
    const sectionConfig: Record<string, { icon: string; color: string; gradient: string }> = {
      'Experiment': { icon: '🧪', color: '#8b5cf6', gradient: 'linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%)' },
      'Dataset': { icon: '📊', color: '#06b6d4', gradient: 'linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(6, 182, 212, 0.05) 100%)' },
      'Model': { icon: '🤖', color: '#ec4899', gradient: 'linear-gradient(135deg, rgba(236, 72, 153, 0.1) 0%, rgba(236, 72, 153, 0.05) 100%)' },
      'Training': { icon: '🎯', color: '#f59e0b', gradient: 'linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%)' },
      'Training Statistics': { icon: '📈', color: '#10b981', gradient: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%)' },
      'Loss Function': { icon: '📉', color: '#ef4444', gradient: 'linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%)' },
      'Landscape Generation': { icon: '🗺️', color: '#3b82f6', gradient: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%)' },
      'System Information': { icon: '💻', color: '#6366f1', gradient: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(99, 102, 241, 0.05) 100%)' },
    };

    const renderValue = (value: any, depth: number = 0): any => {
      if (value === null || value === undefined) {
        return <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>null</span>;
      }
      
      if (typeof value === 'boolean') {
        return (
          <span style={{ 
            color: value ? '#10b981' : '#ef4444',
            fontWeight: 600,
            padding: '4px 10px',
            borderRadius: 6,
            background: value ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.12)',
            border: `1px solid ${value ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)'}`,
            display: 'inline-block',
            fontSize: 12
          }}>
            {String(value).toUpperCase()}
          </span>
        );
      }
      
      if (typeof value === 'number') {
        return (
          <span style={{ 
            color: '#3b82f6',
            fontWeight: 600,
            fontFamily: '"SF Mono", "Monaco", "Inconsolata", "Roboto Mono", monospace',
            fontSize: 13,
            padding: '2px 6px',
                background: getThemeColor('rgba(59, 130, 246, 0.08)', 'rgba(148, 163, 184, 0.1)'),
            borderRadius: 4
          }}>
            {value.toLocaleString()}
          </span>
        );
      }
      
      if (typeof value === 'string') {
        // Check if it's a long string that should be formatted
        if (value.length > 100 && depth === 0) {
          return (
            <div style={{ marginTop: 6 }}>
              <div style={{ 
                whiteSpace: 'pre-wrap', 
                wordBreak: 'break-word', 
                fontSize: 14, 
                lineHeight: 1.9,
                color: 'var(--text-primary)',
                padding: '14px 16px',
                background: getThemeColor('rgba(59, 130, 246, 0.04)', 'rgba(148, 163, 184, 0.06)'),
                borderRadius: 8,
                border: '1px solid var(--border)',
                fontFamily: 'inherit'
              }}>
                {value}
              </div>
            </div>
          );
        }
        return (
          <span style={{ 
            color: '#10b981',
            fontWeight: 500,
            padding: '2px 4px',
            background: 'rgba(16, 185, 129, 0.08)',
            borderRadius: 4
          }}>
            "{value}"
          </span>
        );
      }
      
      if (Array.isArray(value)) {
        if (value.length === 0) {
          return (
            <span style={{ 
              color: 'var(--text-muted)', 
              fontStyle: 'italic',
              padding: '4px 8px',
              background: getThemeColor('rgba(59, 130, 246, 0.03)', 'rgba(148, 163, 184, 0.05)'),
              borderRadius: 4
            }}>
              {t.emptyArray}
            </span>
          );
        }
        // For large arrays, show summary
        if (value.length > 10 && depth > 1) {
          return (
            <span style={{ 
              color: 'var(--text-muted)',
              padding: '4px 8px',
              background: getThemeColor('rgba(59, 130, 246, 0.03)', 'rgba(148, 163, 184, 0.05)'),
              borderRadius: 4,
              fontStyle: 'italic'
            }}>
              {t.arrayItems}({value.length} {t.items})
            </span>
          );
        }
        return (
          <div style={{ marginLeft: 8, marginTop: 6 }}>
            {value.map((item: any, idx: number) => (
              <div key={`array-item-${idx}`} style={{ 
                marginBottom: 8,
                padding: '8px 12px',
                background: idx % 2 === 0 ? getThemeColor('rgba(59, 130, 246, 0.03)', 'rgba(148, 163, 184, 0.05)') : 'transparent',
                borderRadius: 6,
                borderLeft: '3px solid var(--accent)'
              }}>
                <span style={{ 
                  color: 'var(--accent)', 
                  fontSize: 11, 
                  fontWeight: 600,
                  marginRight: 8
                }}>
                  [{idx}]:
                </span>
                {renderValue(item, depth + 1)}
              </div>
            ))}
          </div>
        );
      }
      
      if (typeof value === 'object') {
        const keys = Object.keys(value);
        if (keys.length === 0) {
          return (
            <span style={{ 
              color: 'var(--text-muted)', 
              fontStyle: 'italic',
              padding: '6px 12px',
              background: getThemeColor('rgba(59, 130, 246, 0.04)', 'rgba(148, 163, 184, 0.06)'),
              borderRadius: 6,
              fontSize: 12
            }}>
              {'{}'}
            </span>
          );
        }
        // For top-level fields (depth === 0), display as rows
        if (depth === 0) {
          return (
            <div style={{ marginLeft: 0, marginTop: 12, display: 'grid', gap: 12 }}>
              {keys.map((key, idx) => (
                <div key={`obj-key-${key}-${depth}`} style={{ 
                  padding: '16px 18px',
                  background: depth === 0 
                    ? (idx % 2 === 0 ? getThemeColor('rgba(59, 130, 246, 0.03)', 'rgba(148, 163, 184, 0.05)') : getThemeColor('rgba(59, 130, 246, 0.02)', 'rgba(148, 163, 184, 0.03)'))
                    : getThemeColor('rgba(59, 130, 246, 0.03)', 'rgba(148, 163, 184, 0.05)'),
                  borderRadius: 10,
                  border: '1px solid var(--border)',
                  transition: 'all 0.2s ease',
                  position: 'relative'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = depth === 0 
                    ? 'rgba(249, 115, 22, 0.05)' 
                    : getThemeColor('rgba(59, 130, 246, 0.05)', 'rgba(148, 163, 184, 0.08)');
                  e.currentTarget.style.borderColor = depth === 0 
                    ? 'rgba(249, 115, 22, 0.3)' 
                    : 'var(--border)';
                  e.currentTarget.style.transform = 'translateX(4px)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = depth === 0 
                    ? (idx % 2 === 0 ? getThemeColor('rgba(59, 130, 246, 0.03)', 'rgba(148, 163, 184, 0.05)') : getThemeColor('rgba(59, 130, 246, 0.02)', 'rgba(148, 163, 184, 0.03)'))
                    : getThemeColor('rgba(59, 130, 246, 0.03)', 'rgba(148, 163, 184, 0.05)');
                  e.currentTarget.style.borderColor = 'var(--border)';
                  e.currentTarget.style.transform = 'translateX(0)';
                }}
                >
                  {depth === 0 && (
                    <div style={{
                      position: 'absolute',
                      left: 0,
                      top: 0,
                      bottom: 0,
                      width: 3,
                      background: 'linear-gradient(180deg, var(--accent) 0%, #fb923c 100%)',
                      borderRadius: '10px 0 0 10px'
                    }} />
                  )}
                  <div style={{ 
                    fontWeight: 600, 
                    color: depth === 0 ? 'var(--accent)' : 'var(--text-secondary)', 
                    marginBottom: 10,
                    fontSize: 14,
                    textTransform: 'capitalize',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    paddingLeft: depth === 0 ? 8 : 0
                  }}>
                    <span style={{ 
                      fontSize: 12,
                      color: depth === 0 ? 'var(--accent)' : 'var(--text-muted)',
                      fontWeight: 700
                    }}>▶</span>
                    {key.replace(/_/g, ' ')}
                  </div>
                  <div style={{ marginLeft: depth === 0 ? 20 : 16, marginTop: 4 }}>
                    {renderValue(value[key], depth + 1)}
                  </div>
                </div>
              ))}
            </div>
          );
        }
        
        // For nested objects (depth > 0), use the original layout
        return (
          <div style={{ marginLeft: 0, marginTop: 12, display: 'grid', gap: 12 }}>
            {keys.map((key, idx) => (
              <div key={`obj-key-${key}-${depth}`} style={{ 
                padding: '16px 18px',
                background: getThemeColor('rgba(59, 130, 246, 0.03)', 'rgba(148, 163, 184, 0.05)'),
                borderRadius: 10,
                border: '1px solid var(--border)',
                transition: 'all 0.2s ease',
                position: 'relative'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = getThemeColor('rgba(59, 130, 246, 0.05)', 'rgba(148, 163, 184, 0.08)');
                e.currentTarget.style.borderColor = 'var(--border)';
                e.currentTarget.style.transform = 'translateX(4px)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = getThemeColor('rgba(59, 130, 246, 0.03)', 'rgba(148, 163, 184, 0.05)');
                e.currentTarget.style.borderColor = 'var(--border)';
                e.currentTarget.style.transform = 'translateX(0)';
              }}
              >
                <div style={{ 
                  fontWeight: 600, 
                  color: 'var(--text-secondary)', 
                  marginBottom: 10,
                  fontSize: 14,
                  textTransform: 'capitalize',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8
                }}>
                  <span style={{ 
                    fontSize: 12,
                    color: 'var(--text-muted)',
                    fontWeight: 700
                  }}>▶</span>
                  {key.replace(/_/g, ' ')}
                </div>
                <div style={{ marginLeft: 16, marginTop: 4 }}>
                  {renderValue(value[key], depth + 1)}
                </div>
              </div>
            ))}
          </div>
        );
      }
      
      return <span>{String(value)}</span>;
    };

    const getSectionTitle = (key: string): string => {
      const titleMap: Record<string, string> = {
        'Experiment': t.sectionExperiment,
        'Dataset': t.sectionDataset,
        'Model': t.sectionModel,
        'Training': t.sectionTraining,
        'Training Statistics': t.sectionTrainingStatistics,
        'Loss Function': t.sectionLossFunction,
        'Landscape Generation': t.sectionLandscapeGeneration,
        'System Information': t.sectionSystemInformation,
      };
      return titleMap[key] || (key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' '));
    };

    const renderSection = (title: string, content: any) => {
      if (!content) return null;
      
      const config = sectionConfig[title] || { icon: '📌', color: '#6b7280', gradient: 'linear-gradient(135deg, rgba(107, 114, 128, 0.1) 0%, rgba(107, 114, 128, 0.05) 100%)' };
      const isExpanded = expandedMetadataSections.has(title);
      const translatedTitle = getSectionTitle(title);
      
      return (
        <div style={{ 
          marginBottom: 28,
          padding: 0,
          background: 'var(--bg-card)',
          borderRadius: 16,
          border: '1px solid var(--border)',
          boxShadow: '0 2px 12px rgba(59, 130, 246, 0.05)',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          position: 'relative',
          overflow: 'hidden'
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.transform = 'translateY(-4px)';
          e.currentTarget.style.boxShadow = isDark ? '0 8px 24px rgba(0, 0, 0, 0.4)' : '0 8px 24px rgba(59, 130, 246, 0.1)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = 'translateY(0)';
          e.currentTarget.style.boxShadow = isDark ? '0 2px 12px rgba(0, 0, 0, 0.25)' : '0 2px 12px rgba(59, 130, 246, 0.05)';
        }}
        >
          {/* Colored header bar - clickable */}
          <div 
            style={{
              padding: '20px 24px',
              background: config.gradient,
              borderBottom: isExpanded ? `2px solid ${config.color}40` : 'none',
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              cursor: 'pointer',
              userSelect: 'none',
              transition: 'background 0.2s ease'
            }}
            onClick={() => toggleSection(title)}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = `${config.color}15`;
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = config.gradient;
            }}
          >
            <div style={{
              width: 40,
              height: 40,
              borderRadius: 10,
              background: `${config.color}15`,
              border: `2px solid ${config.color}30`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 20,
              flexShrink: 0,
              transition: 'transform 0.2s ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'scale(1.1)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'scale(1)';
            }}
            >
              {config.icon}
            </div>
            <h3 style={{ 
              fontSize: 18, 
              fontWeight: 700, 
              color: 'var(--text-primary)', 
              margin: 0,
              flex: 1
            }}>
              {translatedTitle}
            </h3>
            <div style={{
              width: 28,
              height: 28,
              borderRadius: 8,
              background: `${config.color}20`,
              border: `1.5px solid ${config.color}50`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
              transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
              flexShrink: 0
            }}>
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M5 4L9 7L5 10" stroke={config.color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
          </div>
          
          {/* Collapsible content */}
          <div style={{ 
            padding: isExpanded ? '24px' : '0 24px',
            fontSize: 14, 
            lineHeight: 1.85, 
            color: 'var(--text-primary)',
            maxHeight: isExpanded ? '10000px' : '0',
            overflow: 'hidden',
            transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
            opacity: isExpanded ? 1 : 0,
            transform: isExpanded ? 'translateY(0)' : 'translateY(-10px)'
          }}>
            {isExpanded && renderValue(content)}
          </div>
        </div>
      );
    };

    return (
      <div style={{ 
        padding: '32px 40px', 
        maxWidth: '1400px',
        margin: '0 auto',
        minHeight: '100%',
        background: 'var(--bg-card)'
      }}>
        {/* Header with gradient background */}
        <div style={{ 
          marginBottom: 40,
          padding: '32px 40px',
          background: 'linear-gradient(135deg, rgba(249, 115, 22, 0.1) 0%, rgba(249, 115, 22, 0.05) 50%, transparent 100%)',
          borderRadius: 20,
          border: '1px solid rgba(249, 115, 22, 0.2)',
          position: 'relative',
          overflow: 'hidden'
        }}>
          <div style={{
            position: 'absolute',
            top: -50,
            right: -50,
            width: 200,
            height: 200,
            background: 'radial-gradient(circle, rgba(249, 115, 22, 0.15) 0%, transparent 70%)',
            borderRadius: '50%'
          }} />
          <div style={{ 
            fontSize: 32, 
            fontWeight: 800, 
            color: 'var(--text-primary)', 
            marginBottom: 8,
            lineHeight: 1.2,
            display: 'flex',
            alignItems: 'center',
            gap: 16,
            position: 'relative',
            zIndex: 1
          }}>
            <div style={{
              width: 56,
              height: 56,
              borderRadius: 16,
              background: 'linear-gradient(135deg, #f97316 0%, #fb923c 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 28,
              boxShadow: '0 4px 12px rgba(249, 115, 22, 0.3)'
            }}>
              📋
            </div>
            <div>
              <div>{t.runMetadata}</div>
              <div style={{ 
                fontSize: 14, 
                fontWeight: 400,
                color: 'var(--text-muted)', 
                marginTop: 4
              }}>
                {t.comprehensiveInfo}
              </div>
            </div>
          </div>
        </div>

        {/* Summary Card - Hero Section */}
        {metadata.summary && (
          <div style={{ 
            marginBottom: 40, 
            padding: '28px 32px', 
            background: 'linear-gradient(135deg, rgba(249, 115, 22, 0.12) 0%, rgba(249, 115, 22, 0.06) 100%)',
            borderRadius: 18,
            border: '2px solid rgba(249, 115, 22, 0.2)',
            boxShadow: '0 8px 24px rgba(249, 115, 22, 0.15)',
            position: 'relative',
            overflow: 'hidden'
          }}>
            <div style={{
              position: 'absolute',
              top: -60,
              right: -60,
              width: 180,
              height: 180,
              background: 'radial-gradient(circle, rgba(249, 115, 22, 0.2) 0%, transparent 70%)',
              borderRadius: '50%'
            }} />
            <div style={{ 
              fontSize: 18, 
              fontWeight: 700, 
              marginBottom: 16, 
              color: 'var(--accent)',
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              position: 'relative',
              zIndex: 1
            }}>
              <span style={{ fontSize: 24 }}>📝</span>
              <span>{t.summary}</span>
            </div>
            <div style={{ 
              fontSize: 15, 
              lineHeight: 1.85, 
              color: 'var(--text-primary)',
              position: 'relative',
              zIndex: 1,
              fontWeight: 400
            }}>
              {metadata.summary}
            </div>
          </div>
        )}

        {/* Overview */}
        <div style={{ marginBottom: 32 }}>
          <div style={{
            fontSize: 18,
            fontWeight: 700,
            color: 'var(--text-primary)',
            marginBottom: 20,
            paddingBottom: 12,
            borderBottom: '2px solid var(--border)',
            display: 'flex',
            alignItems: 'center',
            gap: 8
          }}>
            <span style={{ fontSize: 20 }}>📋</span>
            <span>{t.overview}</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
            {renderSection('Experiment', metadata.experiment)}
            {renderSection('Training Statistics', metadata.training_statistics)}
          </div>
        </div>

        {/* Factors Directly Affecting Loss Landscape */}
        <div style={{ marginBottom: 32 }}>
          <div style={{
            fontSize: 18,
            fontWeight: 700,
            color: 'var(--text-primary)',
            marginBottom: 20,
            paddingBottom: 12,
            borderBottom: '2px solid var(--border)',
            display: 'flex',
            alignItems: 'center',
            gap: 8
          }}>
            <span style={{ fontSize: 20 }}>🎯</span>
            <span>{t.factorsAffectingLoss}</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
            {renderSection('Training', metadata.training)}
            {renderSection('Dataset', metadata.dataset)}
            {renderSection('Model', metadata.model)}
            {renderSection('Loss Function', metadata.loss_function)}
          </div>
        </div>

        {/* Meta Information */}
        <div style={{ marginBottom: 32 }}>
          <div style={{
            fontSize: 18,
            fontWeight: 700,
            color: 'var(--text-primary)',
            marginBottom: 20,
            paddingBottom: 12,
            borderBottom: '2px solid var(--border)',
            display: 'flex',
            alignItems: 'center',
            gap: 8
          }}>
            <span style={{ fontSize: 20 }}>ℹ️</span>
            <span>Meta Information</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
            {renderSection('Landscape Generation', metadata.landscape_generation)}
            {renderSection('System Information', metadata.system)}
            {/* Render any other top-level keys - add to Meta Information section */}
            {Object.keys(metadata).map((key) => {
              const knownKeys = ['summary', 'experiment', 'dataset', 'model', 'training', 
                                'training_statistics', 'loss_function', 'landscape_generation', 'system'];
              if (knownKeys.includes(key)) return null;
              return (
                <div key={key}>
                  {renderSection(key.charAt(0).toUpperCase() + key.slice(1), metadata[key])}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  };

  const renderSidebarContent = () => {
    switch (activeTab) {
      case 'run':
        return renderRunTab();
      case 'History':
        return renderHistoryTab();
      case 'upload':
        return renderUploadTab();
      default:
        return renderRunTab();
    }
  };


  const sliceMeta = useMemo(() => {
    if (!data?.loss_grid_3d) return { zVals: [] as number[], nz: 0 };
    const nz = data.loss_grid_3d[0]?.[0]?.length || 0;
    if (nz <= 0) return { zVals: [] as number[], nz: 0 };

    // Prefer explicit 1D volume axis if present (newer exports).
    const volumeZ = data.volume_z;
    if (Array.isArray(volumeZ) && volumeZ.length === nz) {
      const arr = (volumeZ as any[]).map((v) => Number(v)).filter((v) => Number.isFinite(v));
      if (arr.length === nz) return { zVals: arr, nz };
    }

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
  }, [data?.loss_grid_3d, data?.Z, data?.volume_z]);

  // Derive consistent 1D axes for volume slicing.
  // IMPORTANT: surface X/Y may have a different resolution than volume (e.g. 30x30 vs 16x16),
  // so we must NOT use surface grids to label/plot volume slices.
  const volumeAxes = useMemo(() => {
    if (!data?.loss_grid_3d) return { xVals: [] as number[], yVals: [] as number[], zVals: [] as number[] };
    const nx = data.loss_grid_3d.length || 0;
    const ny = data.loss_grid_3d[0]?.length || 0;
    const nz = data.loss_grid_3d[0]?.[0]?.length || 0;

    const asNumArr = (arr: any, n: number) => {
      if (!Array.isArray(arr) || arr.length !== n) return null;
      const out = (arr as any[]).map((v) => Number(v));
      return out.every((v) => Number.isFinite(v)) ? out : null;
    };

    const xFromExport = asNumArr(data.volume_x, nx);
    const yFromExport = asNumArr(data.volume_y, ny);
    const zFromExport = asNumArr(data.volume_z, nz);

    const linspace = (a: number, b: number, n: number) => {
      if (n <= 1) return [a];
      const out: number[] = [];
      for (let i = 0; i < n; i++) out.push(a + (i / (n - 1)) * (b - a));
      return out;
    };

    // Fallback for older exports: match volume resolution using min/max from surface grids.
    const flatX = Array.isArray(data.X) ? (data.X as any[]).flat().map(Number).filter(Number.isFinite) : [];
    const flatY = Array.isArray(data.Y) ? (data.Y as any[]).flat().map(Number).filter(Number.isFinite) : [];
    const xMin = flatX.length ? Math.min(...flatX) : -1;
    const xMax = flatX.length ? Math.max(...flatX) : 1;
    const yMin = flatY.length ? Math.min(...flatY) : -1;
    const yMax = flatY.length ? Math.max(...flatY) : 1;

    const xVals = xFromExport ?? linspace(xMin, xMax, nx);
    const yVals = yFromExport ?? linspace(yMin, yMax, ny);
    const zVals = zFromExport ?? sliceMeta.zVals;

    return { xVals, yVals, zVals };
  }, [data?.loss_grid_3d, data?.volume_x, data?.volume_y, data?.volume_z, data?.X, data?.Y, sliceMeta.zVals]);

  // Reset slice index when data changes
  useEffect(() => {
    if (!sliceMeta.nz) return;
    setSliceIndex(Math.floor(sliceMeta.nz / 2));
  }, [sliceMeta.nz]);

  // Slice trajectory semantics:
  // - **Projection**: always show the full training path projected onto the current slice plane.
  // - **In-slab highlight**: optionally highlight points that actually lie inside the current slice slab.
  //
  // This matches the intuition that a plane "view" should show the trajectory trend, while the true
  // plane intersection can legitimately be empty for many slice indices.
  const sliceTrajectory = useMemo(() => {
    if (!data?.trajectory_1 || !data.trajectory_1.length) return undefined;
    if (!data?.loss_grid_3d) return undefined;

    const trajA: number[] = data.trajectory_1;
    const trajB: number[] = data.trajectory_2 || [];
    const trajG: number[] = data.trajectory_3 || [];
    const epochs: number[] = data.trajectory_epochs || [];

    const halfThicknessAt = (vals: number[], idx: number) => {
      if (!vals || vals.length <= 1) return 0;
      const k = Math.max(0, Math.min(idx, vals.length - 1));
      if (k <= 0) return Math.abs(vals[1] - vals[0]) / 2;
      if (k >= vals.length - 1) return Math.abs(vals[vals.length - 1] - vals[vals.length - 2]) / 2;
      const left = Math.abs(vals[k] - vals[k - 1]);
      const right = Math.abs(vals[k + 1] - vals[k]);
      return Math.min(left, right) / 2;
    };

    const { xVals, yVals, zVals } = volumeAxes;
    const eps = 1e-12;

    // Projection: full path onto the selected plane.
    // We keep traj_3 optional for highlight/tooltip logic downstream.
    const projected = {
      traj_1: [] as number[],
      traj_2: [] as number[],
      traj_3: [] as number[],
      epochs: [] as number[],
    };

    // In-slab highlight: subset of points that lie near the selected slice coordinate.
    const highlight = {
      traj_1: [] as number[],
      traj_2: [] as number[],
      traj_3: [] as number[],
      epochs: [] as number[],
    };

    const n = Math.min(trajA.length, trajB.length || trajA.length, trajG.length || trajA.length);
    const defaultGamma =
      zVals && zVals.length > 0 ? (zVals[0] + zVals[zVals.length - 1]) / 2 : 0;

    if (sliceAxis === 'gamma') {
      const k = Math.max(0, Math.min(sliceIndex, Math.max(0, zVals.length - 1)));
      const gamma0 = zVals[k] ?? 0;
      const half = halfThicknessAt(zVals, k) + eps;

      // On γ-slice, plane axes are α (traj_1) and β (traj_2).
      // Projection uses all points (ignoring gamma), highlight uses the gamma slab.
      for (let i = 0; i < n; i++) {
        const a = trajA[i];
        const b = trajB[i];
        const g = trajG[i] ?? defaultGamma;
        if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(g)) continue;
        projected.traj_1.push(a);
        projected.traj_2.push(b);
        projected.traj_3.push(g);
        projected.epochs.push(epochs[i] ?? i);

        if (!trajG.length) continue;
        if (Math.abs(g - gamma0) <= half) {
          highlight.traj_1.push(a);
          highlight.traj_2.push(b);
          highlight.traj_3.push(g);
          highlight.epochs.push(epochs[i] ?? i);
        }
      }

      return {
        projected: projected.traj_1.length >= 2 ? projected : undefined,
        highlight: highlight.traj_1.length > 0 ? highlight : undefined,
      };
    }

    if (sliceAxis === 'alpha') {
      const i0 = Math.max(0, Math.min(sliceIndex, Math.max(0, xVals.length - 1)));
      const alpha0 = xVals[i0] ?? 0;
      const half = halfThicknessAt(xVals, i0) + eps;

      // On α-slice, plane axes are β (x-axis) and γ (y-axis).
      // Projection uses all points mapped to (β,γ), highlight uses the α slab.
      for (let i = 0; i < n; i++) {
        const a = trajA[i];
        const b = trajB[i];
        const g = trajG[i] ?? defaultGamma;
        if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(g)) continue;
        projected.traj_1.push(b);
        projected.traj_2.push(g);
        projected.traj_3.push(g);
        projected.epochs.push(epochs[i] ?? i);

        if (Math.abs(a - alpha0) <= half) {
          highlight.traj_1.push(b);
          highlight.traj_2.push(g);
          highlight.traj_3.push(g);
          highlight.epochs.push(epochs[i] ?? i);
        }
      }

      return {
        projected: projected.traj_1.length >= 2 ? projected : undefined,
        highlight: highlight.traj_1.length > 0 ? highlight : undefined,
      };
    }

    // sliceAxis === 'beta'
    const j0 = Math.max(0, Math.min(sliceIndex, Math.max(0, yVals.length - 1)));
    const beta0 = yVals[j0] ?? 0;
    const half = halfThicknessAt(yVals, j0) + eps;

    // On β-slice, plane axes are α (x-axis) and γ (y-axis).
    // Projection uses all points mapped to (α,γ), highlight uses the β slab.
    for (let i = 0; i < n; i++) {
      const a = trajA[i];
      const b = trajB[i];
      const g = trajG[i] ?? defaultGamma;
      if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(g)) continue;
      projected.traj_1.push(a);
      projected.traj_2.push(g);
      projected.traj_3.push(g);
      projected.epochs.push(epochs[i] ?? i);

      if (Math.abs(b - beta0) <= half) {
        highlight.traj_1.push(a);
        highlight.traj_2.push(g);
        highlight.traj_3.push(g);
        highlight.epochs.push(epochs[i] ?? i);
      }
    }

    return {
      projected: projected.traj_1.length >= 2 ? projected : undefined,
      highlight: highlight.traj_1.length > 0 ? highlight : undefined,
    };
  }, [data, sliceAxis, sliceIndex, volumeAxes]);

  const renderViewerControls = () => {
    if (!data) return null;
    
    // Build available view modes based on data
    // Order: 1D -> 2D -> 3D (with render mode selector) -> Metadata
    const availableViews: Array<{key: typeof viewMode, label: string, requires3D: boolean, desc: string}> = [];
    
    if (data.loss_line_1d && data.loss_line_1d.length > 0) {
      availableViews.push(
        { key: '1d', label: t.view1D, requires3D: false, desc: t.view1DDesc }
      );
    }
    
    if (data.loss_grid_2d) {
      availableViews.push(
        { key: '2d', label: t.viewSurface, requires3D: false, desc: `${t.view2DDesc} / ${t.view3DSurfaceDesc}` }
      );
    }
    
    if (data.loss_grid_3d) {
      availableViews.push(
        { key: '3d', label: t.viewSlice, requires3D: true, desc: view3DRenderMode === 'slice' ? t.view3DSliceDesc : t.view3DVolumeDesc }
      );
    }
    
    // Add Hessian view if data exists
    if (data.hessian) {
      const isSnapshot = Array.isArray(data?.hessian?.epochs) && data.hessian.epochs.length <= 1;
      availableViews.push(
        { key: 'hessian', label: t.viewHessian, requires3D: false, desc: isSnapshot ? t.viewHessianSnapshot : t.viewHessianMetrics }
      );
    }
    
    // Always add metadata view if data exists
    if (data.metadata) {
      availableViews.push(
        { key: 'metadata', label: t.viewMetadata, requires3D: false, desc: t.viewMetadataDesc }
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
                    onClick={() => {
                      setViewMode(v.key);
                      if (v.key === '3d' && view3DRenderMode === 'volume') {
                        // Keep current render mode when switching to 3D
                      } else if (v.key === '3d') {
                        // Default to slice when first entering 3D mode
                        setView3DRenderMode('slice');
                      }
                    }}
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
                      background: 'var(--bg-glass-active)',
                      border: '1px solid rgba(249,115,22,0.7)',
                      color: 'var(--text-primary)',
                      fontSize: 12,
                      lineHeight: 1.5,
                      maxWidth: 260,
                      boxShadow: getThemeColor('0 4px 16px rgba(59, 130, 246, 0.1)', '0 4px 16px rgba(0, 0, 0, 0.3)'),
                      zIndex: 2000,
                    }}
                  >
                    <div
                      style={{
                        fontWeight: 700,
                        marginBottom: 4,
                        textTransform: 'uppercase',
                        letterSpacing: '0.06em',
                        fontSize: 11,
                        color: '#f97316',
                      }}
                    >
                      {v.label}
                    </div>
                    <div style={{ opacity: 0.9 }}>{v.desc}</div>
                    <Tooltip.Arrow
                      width={10}
                      height={6}
                      style={{ fill: 'var(--bg-glass-active)' }}
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
          <div className="sidebarTabs">
            <button
              className={`sidebarTab ${activeTab === 'run' ? 'active' : ''}`}
              onClick={() => setActiveTab('run')}
            >
              {t.tabRun}
            </button>
            <button
              className={`sidebarTab ${activeTab === 'History' ? 'active' : ''}`}
              onClick={() => setActiveTab('History')}
            >
              {t.tabHistory}
            </button>
            <button
              className={`sidebarTab ${activeTab === 'upload' ? 'active' : ''}`}
              onClick={() => setActiveTab('upload')}
            >
              {t.tabUpload}
            </button>
          </div>
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
                {data.loss_grid_3d ? '3D' : '2D'} · {t.grid} {data.grid_size || data.gridSize || t.notAvailable}
              </div>
            )}
          </div>
          
          {renderViewerControls()}
        </div>

        <div style={{ display: 'flex', flex: 1, minHeight: 0, gap: 12 }}>
          {/* Left side: Loss charts */}
          {(() => {
            if (!data || !data.trajectory_1 || data.trajectory_1.length === 0 || viewMode === 'metadata') {
              return null;
            }
            
            // Calculate landscape losses from trajectory (bilinear interpolation on grid axes).
            // IMPORTANT: return null (not 0) when out of bounds, otherwise charts look like all-zeros.
            const interpolateLoss = (
              x: number,
              y: number,
              X: number[][],
              Y: number[][],
              lossGrid: number[][]
            ): number | null => {
              const nx = lossGrid.length;
              const ny = lossGrid[0]?.length || 0;
              if (nx === 0 || ny === 0) return null;

              const xRow0 = X[0] || [];
              const xCol0 = X.map((r) => r?.[0]).filter((v) => typeof v === 'number') as number[];
              const yRow0 = Y[0] || [];
              const yCol0 = Y.map((r) => r?.[0]).filter((v) => typeof v === 'number') as number[];

              const range = (arr: number[]) => (arr.length ? Math.max(...arr) - Math.min(...arr) : 0);
              const pickAxis = (a: number[], b: number[]) => (range(a) >= range(b) ? a : b);
              const xAxisCand = pickAxis(xCol0, xRow0 as number[]);
              const yAxisCand = pickAxis(yCol0, yRow0 as number[]);

              // Determine orientation against lossGrid shape
              const xAxis = xAxisCand.slice();
              const yAxis = yAxisCand.slice();

              const xLen = xAxis.length;
              const yLen = yAxis.length;
              if (!(xLen > 1 && yLen > 1)) return null;

              const axisInc = (arr: number[]) => arr[arr.length - 1] >= arr[0];
              const xInc = axisInc(xAxis);
              const yInc = axisInc(yAxis);

              const xMin = xInc ? xAxis[0] : xAxis[xLen - 1];
              const xMax = xInc ? xAxis[xLen - 1] : xAxis[0];
              const yMin = yInc ? yAxis[0] : yAxis[yLen - 1];
              const yMax = yInc ? yAxis[yLen - 1] : yAxis[0];

              if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
              if (x < xMin || x > xMax || y < yMin || y > yMax) return null;

              const lowerIndex = (arr: number[], v: number, inc: boolean) => {
                // returns i such that arr[i] <= v <= arr[i+1] (for inc) or reversed for dec
                let lo = 0;
                let hi = arr.length - 2;
                while (lo <= hi) {
                  const mid = (lo + hi) >> 1;
                  const a = arr[mid];
                  const b = arr[mid + 1];
                  if (inc) {
                    if (v < a) hi = mid - 1;
                    else if (v > b) lo = mid + 1;
                    else return mid;
                  } else {
                    if (v > a) hi = mid - 1;
                    else if (v < b) lo = mid + 1;
                    else return mid;
                  }
                }
                return Math.max(0, Math.min(arr.length - 2, lo));
              };

              const ix = lowerIndex(xAxis, x, xInc);
              const iy = lowerIndex(yAxis, y, yInc);
              const x0 = xAxis[ix];
              const x1 = xAxis[ix + 1];
              const y0 = yAxis[iy];
              const y1 = yAxis[iy + 1];

              const tx = (x1 - x0) !== 0 ? (x - x0) / (x1 - x0) : 0;
              const ty = (y1 - y0) !== 0 ? (y - y0) / (y1 - y0) : 0;

              // Map to lossGrid indices (handle either [x][y] or [y][x])
              const useXY = xLen === nx && yLen === ny;
              const useYX = yLen === nx && xLen === ny;
              if (!useXY && !useYX) {
                // Fallback: nearest neighbor on normalized coords
                const i = Math.max(0, Math.min(nx - 1, Math.round((ix + tx) * (nx - 1) / (xLen - 1))));
                const j = Math.max(0, Math.min(ny - 1, Math.round((iy + ty) * (ny - 1) / (yLen - 1))));
                const v = lossGrid[i]?.[j];
                return Number.isFinite(v) ? v : null;
              }

              const g = (a: number, b: number) => {
                const v = useXY ? lossGrid[a]?.[b] : lossGrid[b]?.[a];
                return Number.isFinite(v) ? (v as number) : null;
              };

              const v00 = g(ix, iy);
              const v10 = g(ix + 1, iy);
              const v01 = g(ix, iy + 1);
              const v11 = g(ix + 1, iy + 1);
              if (v00 === null || v10 === null || v01 === null || v11 === null) return null;

              const v0 = v00 * (1 - tx) + v10 * tx;
              const v1 = v01 * (1 - tx) + v11 * tx;
              const out = v0 * (1 - ty) + v1 * ty;
              return Number.isFinite(out) ? out : null;
            };
            
            const landscapeLosses: (number | null)[] = [];
            if (data.trajectory_1 && data.loss_grid_2d && data.X && data.Y) {
              for (let i = 0; i < data.trajectory_1.length; i++) {
                const alpha = data.trajectory_1[i];
                const beta = data.trajectory_2[i];
                const loss = interpolateLoss(alpha, beta, data.X, data.Y, data.loss_grid_2d);
                landscapeLosses.push(loss);
              }
            }

            const epochs: number[] = data.trajectory_epochs || [];
            const trainLossesRaw = (data.trajectory_data?.losses || data.trajectory_losses || []) as any[];
            const valLossesRaw = (data.trajectory_data?.val_losses || data.trajectory_val_losses || []) as any[];

            const toFiniteOrNull = (v: any): number | null => {
              const n = typeof v === 'number' ? v : Number(v);
              return Number.isFinite(n) ? n : null;
            };

            const trainLosses: (number | null)[] = epochs.map((_, i) =>
              i < trainLossesRaw.length ? toFiniteOrNull(trainLossesRaw[i]) : null
            );

            const valLosses: (number | null)[] = epochs.map((_, i) =>
              i < valLossesRaw.length ? toFiniteOrNull(valLossesRaw[i]) : null
            );

            const landscapeSeries: (number | null)[] = epochs.map((_, i) =>
              i < landscapeLosses.length ? (landscapeLosses[i] ?? null) : null
            );

            const hasTrain = trainLosses.some((v) => v !== null);
            const hasVal = valLosses.some((v) => v !== null);
            const hasLandscape = landscapeSeries.some((v) => v !== null);
            
            return (
              <div style={{
                width: 300, // Slightly narrower for dashboard feel
                display: 'flex',
                flexDirection: 'column',
                gap: 16,
                flexShrink: 0,
                background: 'var(--bg-sidebar-tabs)', // Adapts to dark mode
                borderRight: '1px solid var(--border)',
                padding: 16,
                overflowY: 'auto'
              }}>
                <div style={{ paddingBottom: 8, borderBottom: '1px solid var(--border)' }}>
                  <h3 style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>{t.metrics}</h3>
                  <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '4px 0 0' }}>{t.trainingProgression}</p>
                </div>

                {hasTrain && (
                  <MetricChart
                    title={t.trainLoss}
                    color="#3b82f6" // blue-500
                    epochs={epochs}
                    values={trainLosses}
                    height={140}
                    valueLabel={t.valueLabelTrain}
                  />
                )}

                {hasVal && (
                  <MetricChart
                    title={t.validationLoss}
                    color="#f59e0b" // amber-500
                    epochs={epochs}
                    values={valLosses}
                    height={140}
                    valueLabel={t.valueLabelVal}
                  />
                )}

                {hasLandscape && (
                  <MetricChart
                    title={t.landscapeLoss}
                    color="#10b981" // emerald-500
                    epochs={epochs}
                    values={landscapeSeries}
                    height={140}
                    valueLabel={t.valueLabelLand}
                  />
                )}
              </div>
            );
          })()}
          
          {/* Right side: Main viewer */}
          <div className="viewerCanvas" style={{ flex: 1, minWidth: 0, position: 'relative' }}>
          {loading && (
            <div className="loadingOverlay">
              <div className="spinner" />
              <div style={{ fontSize: 14, fontWeight: 500 }}>{t.loadingOverlay}</div>
            </div>
          )}

          {/* 3D Render Mode Selector - Bottom Left - optimized with DraggablePanel */}
          {viewMode === '3d' && data?.loss_grid_3d && (
            <DraggablePanel
              position="bottom-left"
              style={{ ...cardVisualStyle, minWidth: 220 }}
            >
              <div style={{ fontWeight: 800, fontSize: 14, marginBottom: 2, letterSpacing: '0.02em' }}>
                {t.view3DRenderMode}
              </div>
              <div style={{ display: 'flex', gap: 6 }}>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setView3DRenderMode('slice');
                  }}
                  style={{
                    padding: '6px 14px',
                    borderRadius: 8,
                    border: view3DRenderMode === 'slice' ? '2px solid var(--accent)' : `1px solid ${isDark ? 'rgba(255,255,255,0.22)' : 'rgba(15,23,42,0.12)'}`,
                    background: view3DRenderMode === 'slice' ? 'rgba(249,115,22,0.15)' : (isDark ? 'rgba(255,255,255,0.06)' : 'rgba(15,23,42,0.04)'),
                    color: isDark ? 'rgba(255,255,255,0.95)' : '#0f172a',
                    fontSize: 12,
                    fontWeight: 600,
                    cursor: 'pointer',
                    flex: 1,
                    transition: 'all 0.2s',
                  }}
                >
                  {t.view3DSlice}
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setView3DRenderMode('volume');
                  }}
                  style={{
                    padding: '6px 14px',
                    borderRadius: 8,
                    border: view3DRenderMode === 'volume' ? '2px solid var(--accent)' : `1px solid ${isDark ? 'rgba(255,255,255,0.22)' : 'rgba(15,23,42,0.12)'}`,
                    background: view3DRenderMode === 'volume' ? 'rgba(249,115,22,0.15)' : (isDark ? 'rgba(255,255,255,0.06)' : 'rgba(15,23,42,0.04)'),
                    color: isDark ? 'rgba(255,255,255,0.95)' : '#0f172a',
                    fontSize: 12,
                    fontWeight: 600,
                    cursor: 'pointer',
                    flex: 1,
                    transition: 'all 0.2s',
                  }}
                >
                  {t.view3DVolume}
                </button>
              </div>
            </DraggablePanel>
          )}

          {viewMode === 'metadata' ? (
            // Metadata view
            data && data.metadata ? (
              <div style={{ 
                padding: '24px', 
                height: '100%', 
                overflowY: 'auto',
                background: 'var(--bg-card)'
              }}>
                {renderMetadataTab()}
              </div>
            ) : (
              <div className="emptyState">
                <div className="emptyStateIcon">📋</div>
                <div>
                  <h3 className="emptyStateTitle">{t.noMetadataAvailable}</h3>
                  <p className="emptyStateDesc">{t.noMetadataInfo}</p>
                </div>
              </div>
            )
          ) : viewMode === 'hessian' ? (
            // Hessian view
            data && data.hessian ? (
              <div style={{ 
                height: '100%',
                minHeight: 0,
                background: 'var(--bg-card)',
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden',
              }}>
                <HessianAnalysis data={data.hessian} metadata={data.metadata} />
              </div>
            ) : (
              <div className="emptyState">
                <div className="emptyStateIcon">📉</div>
                <div>
                  <h3 className="emptyStateTitle">{t.noHessianDataAvailable}</h3>
                  <p className="emptyStateDesc">{t.noHessianMetrics}</p>
                </div>
              </div>
            )
          ) : data?.X?.length && (data?.loss_grid_2d?.length || data?.loss_grid_3d?.length || data?.loss_line_1d?.length) ? (
             // Render based on viewMode instead of mode
             viewMode === '1d' ? (
                // 1D View - Full screen
                (() => {
                  const has1D = data?.loss_line_1d && data.loss_line_1d.length > 0;
                  const x1d = data?.X_1d || (data?.X && Array.isArray(data.X) && data.X.length > 0 && typeof data.X[0] === 'number' ? data.X : null);
                  
                  return has1D && x1d ? (
                    <LossLandscape1D
                      X={x1d as number[]}
                      lossLine={data.loss_line_1d}
                      baselineLoss={data.baseline_loss_1d || data.baseline_loss}
                      trajectory={
                        data.trajectory_1 && data.trajectory_1.length > 0
                          ? (() => {
                              const traj = {
                                traj_1: data.trajectory_1,
                                epochs: data.trajectory_epochs || [],
                              };
                              return traj;
                            })()
                          : undefined
                      }
                      trajectoryHighlight={
                        data.trajectory_1 && data.trajectory_1.length > 0
                          ? (() => {
                              const traj = {
                                traj_1: data.trajectory_1,
                                epochs: data.trajectory_epochs || [],
                              };
                              return traj;
                            })()
                          : undefined
                      }
                    />
                  ) : (
                    <div className="emptyState">
                      <div className="emptyStateIcon">📊</div>
                      <div>
                        <h3 className="emptyStateTitle">{t.no1DDataAvailable}</h3>
                        <p className="emptyStateDesc">{t.no1DLossData}</p>
                      </div>
                    </div>
                  );
                })()
             ) : viewMode === '2d' ? (
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
                              losses: data.trajectory_data?.losses || data.trajectory_losses,
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
                              losses: data.trajectory_data?.losses || data.trajectory_losses,
                            };
                            return traj;
                          })()
                        : undefined
                    }
                  />
                )
            ) : viewMode === '3d' && view3DRenderMode === 'slice' && data.loss_grid_3d ? (
                sliceMode === '2d'
                  ? (
                    sliceAxis === 'gamma'
                      ? (
                  <LossVolumeSlice2D
                    X={(() => {
                      // Prefer volume axes for slice grids (volume resolution can differ from surface).
                      const { xVals, yVals } = volumeAxes;
                      if (!xVals.length || !yVals.length) return data.X;
                      return Array.from({ length: xVals.length }, (_, i) =>
                        Array.from({ length: yVals.length }, (_, j) => xVals[i])
                      );
                    })()}
                    Y={(() => {
                      const { xVals, yVals } = volumeAxes;
                      if (!xVals.length || !yVals.length) return data.Y;
                      return Array.from({ length: xVals.length }, (_, i) =>
                        Array.from({ length: yVals.length }, (_, j) => yVals[j])
                      );
                    })()}
                    Z={volumeAxes.zVals}
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
                              const { xVals, yVals, zVals } = volumeAxes;

                              if (sliceAxis === 'alpha') {
                                const i = Math.min(k, nx - 1);
                                const rows = ny;
                                const cols = nz;
                                const grid: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));
                                for (let j = 0; j < rows; j++) {
                                  const beta = yVals[j] ?? 0;
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
                                  const alpha = xVals[i] ?? 0;
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
                              const { zVals } = volumeAxes;
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
                          trajectory={sliceTrajectory?.projected}
                          trajectoryHighlight={sliceTrajectory?.highlight}
                        />
                      )
                  )
                  : (
                    sliceAxis === 'gamma'
                      ? (
                        <LossLandscape3D
                          X={(() => {
                            const { xVals, yVals } = volumeAxes;
                            if (!xVals.length || !yVals.length) return data.X;
                            return Array.from({ length: xVals.length }, (_, i) =>
                              Array.from({ length: yVals.length }, (_, j) => xVals[i])
                            );
                          })()}
                          Y={(() => {
                            const { xVals, yVals } = volumeAxes;
                            if (!xVals.length || !yVals.length) return data.Y;
                            return Array.from({ length: xVals.length }, (_, i) =>
                              Array.from({ length: yVals.length }, (_, j) => yVals[j])
                            );
                          })()}
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
                          trajectory={sliceTrajectory?.projected}
                          trajectoryHighlight={sliceTrajectory?.highlight}
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
                              const { xVals, yVals } = volumeAxes;

                              if (sliceAxis === 'alpha') {
                                const i = Math.min(k, nx - 1);
                                const rows = ny;
                                const cols = nz;
                                const grid: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));
                                for (let j = 0; j < rows; j++) {
                                  const beta = yVals[j] ?? 0;
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
                                  const alpha = xVals[i] ?? 0;
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
                              const { zVals } = volumeAxes;
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
                          trajectory={sliceTrajectory?.projected}
                          trajectoryHighlight={sliceTrajectory?.highlight}
                        />
                      )
                  )
            ) : viewMode === '3d' && view3DRenderMode === 'volume' && data.loss_grid_3d ? (
                <LossVolumeRender3D
                  X={(() => {
                    const { xVals, yVals } = volumeAxes;
                    if (!xVals.length || !yVals.length) return data.X;
                    return Array.from({ length: xVals.length }, (_, i) =>
                      Array.from({ length: yVals.length }, (_, j) => xVals[i])
                    );
                  })()}
                  Y={(() => {
                    const { xVals, yVals } = volumeAxes;
                    if (!xVals.length || !yVals.length) return data.Y;
                    return Array.from({ length: xVals.length }, (_, i) =>
                      Array.from({ length: yVals.length }, (_, j) => yVals[j])
                    );
                  })()}
                  Z={volumeAxes.zVals}
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
                        losses: data.trajectory_data?.losses || data.trajectory_losses,
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

          {/* Surface view 2D/3D toggle, bottom-right - optimized with DraggablePanel */}
          {viewMode === '2d' && (
            <DraggablePanel
              position="bottom-right"
              style={cardVisualStyle}
            >
              <div style={{ fontWeight: 800, fontSize: 14, marginBottom: 2, letterSpacing: '0.02em' }}>{t.viewSurface}</div>
              <div style={{ display: 'flex', gap: 6 }}>
                <button
                  type="button"
                  onClick={handleSetSurfaceMode2D}
                  style={button2DStyle}
                >
                  2D
                </button>
                <button
                  type="button"
                  onClick={handleSetSurfaceMode3D}
                  style={button3DStyle}
                >
                  3D
                </button>
              </div>
            </DraggablePanel>
          )}

          {viewMode === '3d' && view3DRenderMode === 'slice' && (
            <>
              {/* Right-side axis + mode controls */}
              <DraggablePanel
                position="bottom-right"
                style={cardVisualStyle}
              >
                <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 2 }}>{t.viewSlice} - {view3DRenderMode === 'slice' ? t.view3DSlice : t.view3DVolume}</div>
                
                <div style={{ display: 'flex', gap: 6 }}>
                  <button
                    type="button"
                    onClick={() => setSliceMode('2d')}
                    style={{
                      padding: '4px 8px',
                      borderRadius: 999,
                      border: sliceMode === '2d' ? '2px solid var(--accent)' : '1px solid var(--border)',
                      background: sliceMode === '2d' ? 'rgba(249,115,22,0.15)' : 'var(--bg-lang-toggle)',
                      color: 'var(--text-primary)',
                      fontSize: 11,
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
                      border: sliceMode === '3d' ? '2px solid var(--accent)' : '1px solid var(--border)',
                      background: sliceMode === '3d' ? 'rgba(249,115,22,0.15)' : 'var(--bg-lang-toggle)',
                      color: 'var(--text-primary)',
                      fontSize: 11,
                      fontWeight: 600,
                      cursor: 'pointer',
                      flex: 1,
                    }}
                  >
                    3D
                  </button>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12, opacity: 0.85 }}>
                  <span style={{ fontWeight: 600 }}>{t.axis}</span>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <button
                      type="button"
                      onClick={() => setSliceAxis('gamma')}
                      style={{
                        padding: '2px 8px',
                        borderRadius: 999,
                        border: sliceAxis === 'gamma' ? '2px solid #6ee7b7' : '1px solid var(--border)',
                      background: sliceAxis === 'gamma' ? 'rgba(110,231,183,0.15)' : 'var(--bg-lang-toggle)',
                      color: 'var(--text-primary)',
                      fontSize: 11,
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
                        border: sliceAxis === 'alpha' ? '2px solid #60a5fa' : '1px solid var(--border)',
                        background: sliceAxis === 'alpha' ? 'rgba(96,165,250,0.15)' : 'var(--bg-lang-toggle)',
                        color: 'var(--text-primary)',
                        fontSize: 11,
                        fontWeight: 600,
                        cursor: 'pointer',
                      }}
                    >
                      {t.alphaSlice}
                    </button>
                    <button
                      type="button"
                      onClick={() => setSliceAxis('beta')}
                      style={{
                        padding: '2px 8px',
                        borderRadius: 999,
                        border: sliceAxis === 'beta' ? '2px solid #fb7185' : '1px solid var(--border)',
                        background: sliceAxis === 'beta' ? 'rgba(251,113,133,0.15)' : 'var(--bg-lang-toggle)',
                        color: 'var(--text-primary)',
                        fontSize: 11,
                        fontWeight: 600,
                        cursor: 'pointer',
                      }}
                    >
                      {t.betaSlice}
                    </button>
                  </div>
                </div>
                
                <div style={{ height: 1, background: 'var(--border)', margin: '2px 0' }} />
                
                <div onMouseDown={(e) => e.stopPropagation()}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 11, opacity: 0.8, fontFamily: 'monospace' }}>
                    <span style={{ fontWeight: 600 }}>{t.sliceIndex}</span>
                    <span>{sliceIndex} / {sliceMeta.nz - 1}</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={sliceMeta.nz - 1}
                    step={1}
                    value={Math.min(sliceIndex, sliceMeta.nz - 1)}
                    onChange={(e) => setSliceIndex(parseInt(e.target.value, 10))}
                    style={{
                      width: '100%',
                      accentColor: '#f97316',
                      cursor: 'pointer',
                      marginBottom: 4,
                    }}
                  />
                  <div style={{ fontSize: 10, opacity: 0.7, fontFamily: 'monospace', textAlign: 'center', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {sliceAxis === 'gamma'
                      ? (sliceMeta.zVals[sliceIndex] !== undefined
                          ? `${t.gamma} = ${sliceMeta.zVals[sliceIndex].toFixed(3)}`
                          : `${t.gammaIdx}=${sliceIndex}`)
                      : sliceAxis === 'alpha'
                        ? `${t.alphaIdx}=${sliceIndex}`
                        : `${t.betaIdx}=${sliceIndex}`}
                  </div>
                </div>
              </DraggablePanel>

            </>
          )}
          </div>
        </div>
      </div>
      
      {/* Modal for confirmations and alerts */}
      <Modal
        isOpen={modalState.isOpen}
        onClose={() => setModalState({ ...modalState, isOpen: false })}
        type={modalState.type}
        title={modalState.title}
        message={modalState.message}
        confirmText={modalState.type === 'confirm' ? t.delete : t.modalOK}
        cancelText={t.modalCancel}
        onConfirm={modalState.onConfirm}
        confirmButtonStyle={
          modalState.type === 'confirm'
            ? { background: '#ef4444' }
            : undefined
        }
      />
    </div>
  );
}
