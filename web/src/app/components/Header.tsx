'use client';

import { useI18n } from '../i18n';
import { useTheme } from '../theme';
import Image from 'next/image';

export default function Header() {
  const { locale, setLocale, t } = useI18n();
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="appHeader">
      <div className="logoArea">
        <div className="appLogo">
          <Image
            src="/assets/logo/android-chrome-192x192.png"
            alt="LearnableKGE Logo"
            width={48}
            height={48}
            priority
            unoptimized
          />
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline' }}>
          <span className="appTitle">{t.appName}</span>
          <span className="appTagline">{t.tagline}</span>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <button
          type="button"
          onClick={toggleTheme}
          className="themeToggle"
          title={theme === 'light' ? '切换到深色模式' : 'Switch to dark mode'}
          aria-label={theme === 'light' ? '切换到深色模式' : 'Switch to dark mode'}
        >
          {theme === 'light' ? '🌙' : '☀️'}
        </button>
        <div className="langToggle">
          <button
            type="button"
            className={locale === 'zh' ? 'active' : ''}
            onClick={() => setLocale('zh')}
          >
            中文
          </button>
          <button
            type="button"
            className={locale === 'en' ? 'active' : ''}
            onClick={() => setLocale('en')}
          >
            English
          </button>
        </div>
      </div>
    </header>
  );
}
