/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // DuckDB is a native Node dependency; keep it external so Next doesn't try to bundle it.
  serverExternalPackages: ['duckdb'],
  // Turbopack configuration (Next.js 16+)
  turbopack: {
    // Externalize packages that should not be bundled
    resolveAlias: {
      // Keep duckdb external
    },
  },
  // Ensure React Three Fiber and related packages are not bundled for server
  experimental: {
    serverActions: {
      bodySizeLimit: '2mb',
    },
  },
}

module.exports = nextConfig

