'use client';

import { useI18n } from '../i18n';
import Image from 'next/image';

export default function Header() {
  const { locale, setLocale, t } = useI18n();

  return (
    <header className="appHeader">
      <div className="logoArea">
        <div className="appLogo">
          <Image
            src="/assets/logo/android-chrome-192x192.png"
            alt="LearnableKGE Logo"
            width={40}
            height={40}
            priority
            unoptimized
          />
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline' }}>
          <span className="appTitle">{t.appName}</span>
          <span className="appTagline">{t.tagline}</span>
        </div>
      </div>

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
    </header>
  );
}
