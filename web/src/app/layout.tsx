import type { Metadata } from 'next';
import './globals.css';
import Header from './components/Header';
import { I18nProvider } from './i18n';
import { ThemeProvider } from './theme';

export const metadata: Metadata = {
  title: 'Dr.Sun LearnableKGE',
  description: '2D/3D loss landscape visualization with trajectories for Physics-Informed Neural Networks',
  icons: {
    icon: [
      { url: '/assets/logo/favicon-16x16.png', sizes: '16x16', type: 'image/png' },
      { url: '/assets/logo/favicon-32x32.png', sizes: '32x32', type: 'image/png' },
      { url: '/assets/logo/favicon.ico', sizes: 'any' },
    ],
    apple: [
      { url: '/assets/logo/apple-touch-icon.png', sizes: '180x180', type: 'image/png' },
    ],
    other: [
      { rel: 'android-chrome', url: '/assets/logo/android-chrome-192x192.png', sizes: '192x192' },
      { rel: 'android-chrome', url: '/assets/logo/android-chrome-512x512.png', sizes: '512x512' },
    ],
  },
  manifest: '/assets/logo/site.webmanifest',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <ThemeProvider>
          <I18nProvider>
            <div className="appShell">
              <Header />
              {children}
            </div>
          </I18nProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
