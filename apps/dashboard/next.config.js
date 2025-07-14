/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Enable Webpack 5 support (default in Next.js 12+)
  webpack5: true,

  // Environment variables can also be loaded from .env files
  env: {
    API_BASE_URL: process.env.API_BASE_URL || 'http://localhost:8000/api',
  },

  // Enable image optimization for external domains if needed
  images: {
    domains: ['example.com', 'cdn.example.com'],
  },

  // Internationalization (i18n) example setup, optional
  i18n: {
    locales: ['en', 'es', 'fr'],
    defaultLocale: 'en',
  },

  // Future-proof settings for Next.js features
  experimental: {
    scrollRestoration: true,
  },

  // Customize Webpack config if needed
  webpack(config, { isServer }) {
    // Example: Add custom alias
    config.resolve.alias['@components'] = require('path').resolve(__dirname, 'components');
    return config;
  },
};

module.exports = nextConfig;
