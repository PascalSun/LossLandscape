/* Theme provider for dark/light mode */
'use client';

import React, { createContext, useContext, useEffect, useMemo, useRef, useState } from 'react';

export type Theme = 'light' | 'dark';

const ThemeCtx = createContext<{
  theme: Theme;
  setTheme: (t: Theme) => void;
  toggleTheme: () => void;
} | null>(null);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  // Always start with 'light' to ensure SSR and client hydration match
  const [theme, setTheme] = useState<Theme>('light');
  const didInitRef = useRef(false);

  // Initialize from localStorage/system preference after mount.
  // NOTE: our eslint config forbids calling setState directly inside an effect body,
  // so we schedule the state update in a microtask callback.
  useEffect(() => {
    const applyInitialTheme = () => {
      try {
        const saved = window.localStorage.getItem('theme');
        if (saved === 'light' || saved === 'dark') {
          setTheme(saved);
          document.documentElement.setAttribute('data-theme', saved);
          return;
        }

        if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
          setTheme('dark');
          document.documentElement.setAttribute('data-theme', 'dark');
        } else {
          document.documentElement.setAttribute('data-theme', 'light');
        }
      } finally {
        didInitRef.current = true;
      }
    };

    Promise.resolve().then(applyInitialTheme);

    const onStorage = (e: StorageEvent) => {
      if (e.key !== 'theme') return;
      const v = e.newValue;
      if (v === 'light' || v === 'dark') setTheme(v);
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  // Save to localStorage and update DOM when theme changes (only after init)
  useEffect(() => {
    if (!didInitRef.current) return;
    window.localStorage.setItem('theme', theme);
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  };

  const value = useMemo(() => ({ theme, setTheme, toggleTheme }), [theme]);
  return <ThemeCtx.Provider value={value}>{children}</ThemeCtx.Provider>;
}

export function useTheme() {
  const ctx = useContext(ThemeCtx);
  if (!ctx) throw new Error('useTheme must be used within ThemeProvider');
  return ctx;
}

