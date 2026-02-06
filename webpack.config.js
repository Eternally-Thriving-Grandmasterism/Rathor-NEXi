// webpack.config.js – ML-Optimized Webpack 5 Production Config v1
// React 18 + TensorFlow.js + MediaPipe + Framer Motion + PWA + extreme ML bundle optimization
// MIT License – Autonomicity Games Inc. 2026

const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const CssMinimizerPlugin = require('css-minimizer-webpack-plugin');
const TerserPlugin = require('terser-webpack-plugin');
const CompressionPlugin = require('compression-webpack-plugin');
const { WebpackManifestPlugin } = require('webpack-manifest-plugin');
const WorkboxWebpackPlugin = require('workbox-webpack-plugin');
const CopyPlugin = require('copy-webpack-plugin');
const webpack = require('webpack');

module.exports = (env, argv) => {
  const isProduction = argv.mode === 'production';

  return {
    mode: isProduction ? 'production' : 'development',
    devtool: isProduction ? 'source-map' : 'eval-cheap-module-source-map',
    entry: './src/main.tsx',
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: isProduction ? 'assets/js/[name].[contenthash:8].js' : 'assets/js/[name].js',
      chunkFilename: isProduction ? 'assets/js/[name].[contenthash:8].chunk.js' : 'assets/js/[name].chunk.js',
      publicPath: '/Rathor-NEXi/', // Critical for GitHub Pages
      clean: true,
      assetModuleFilename: 'assets/static/[name].[hash:8][ext]',
    },

    resolve: {
      extensions: ['.tsx', '.ts', '.jsx', '.js', '.json'],
      alias: {
        '@': path.resolve(__dirname, 'src'),
      },
      fallback: {
        // tfjs polyfills
        buffer: require.resolve('buffer/'),
        process: require.resolve('process/browser'),
        crypto: require.resolve('crypto-browserify'),
        stream: require.resolve('stream-browserify'),
        path: require.resolve('path-browserify'),
      },
    },

    module: {
      rules: [
        {
          test: /\.(ts|tsx|js|jsx)$/,
          exclude: /node_modules/,
          use: [
            {
              loader: 'babel-loader',
              options: {
                presets: [
                  '@babel/preset-env',
                  ['@babel/preset-react', { runtime: 'automatic' }],
                  '@babel/preset-typescript',
                ],
                plugins: ['@babel/plugin-transform-runtime'],
                cacheDirectory: true,
              },
            },
          ],
        },
        {
          test: /\.css$/,
          use: [
            isProduction ? MiniCssExtractPlugin.loader : 'style-loader',
            {
              loader: 'css-loader',
              options: { importLoaders: 1, modules: false },
            },
            'postcss-loader',
          ],
        },
        {
          test: /\.(png|jpe?g|gif|webp|svg|ico|wasm)$/,
          type: 'asset',
          parser: {
            dataUrlCondition: {
              maxSize: 8 * 1024, // 8 KB inline limit
            },
          },
        },
        {
          test: /\.(woff2?|eot|ttf|otf)$/,
          type: 'asset/resource',
        },
      ],
    },

    optimization: {
      minimize: isProduction,
      minimizer: [
        new TerserPlugin({
          parallel: true,
          extractComments: false,
          terserOptions: {
            compress: {
              drop_console: false,
              passes: 3,
              pure_funcs: ['console.debug'],
              pure_getters: true,
              unsafe: true,
              unsafe_comps: true,
              unsafe_math: true,
              unsafe_methods: true,
              unsafe_undefined: true,
            },
            mangle: true,
            format: {
              comments: false,
            },
          },
        }),
        new CssMinimizerPlugin(),
      ],
      splitChunks: {
        chunks: 'all',
        minSize: 15000,
        maxSize: 0,
        minChunks: 1,
        cacheGroups: {
          // ML-heavy chunks (tfjs + backends)
          tfjs: {
            test: /[\\/]node_modules[\\/](@tensorflow)[\\/]/,
            name: 'tfjs',
            priority: 15,
            chunks: 'all',
            enforce: true,
          },

          // MediaPipe WASM & models
          mediapipe: {
            test: /[\\/]node_modules[\\/](@mediapipe)[\\/]/,
            name: 'mediapipe',
            priority: 14,
            chunks: 'all',
            enforce: true,
          },

          // Core vendor (React + motion)
          vendor: {
            test: /[\\/]node_modules[\\/](react|react-dom|framer-motion)[\\/]/,
            name: 'vendor-core',
            priority: 12,
            chunks: 'initial',
          },

          // Other vendors
          defaultVendors: {
            test: /[\\/]node_modules[\\/]/,
            priority: -10,
            reuseExistingChunk: true,
          },

          default: {
            minChunks: 2,
            priority: -20,
            reuseExistingChunk: true,
          },
        },
      },
      runtimeChunk: {
        name: 'runtime',
      },
    },

    plugins: [
      new HtmlWebpackPlugin({
        template: './index.html',
        filename: 'index.html',
        minify: isProduction && {
          collapseWhitespace: true,
          removeComments: true,
          removeRedundantAttributes: true,
          removeScriptTypeAttributes: true,
          removeStyleLinkTypeAttributes: true,
          useShortDoctype: true,
        },
      }),

      isProduction && new MiniCssExtractPlugin({
        filename: 'assets/css/[name].[contenthash:8].css',
        chunkFilename: 'assets/css/[name].[contenthash:8].chunk.css',
      }),

      new CleanWebpackPlugin(),

      // Brotli + Gzip compression
      new CompressionPlugin({
        algorithm: 'brotliCompress',
        test: /\.(js|css|html|svg|wasm|json)$/,
        threshold: 10240,
        minRatio: 0.8,
        deleteOriginalAssets: false,
      }),

      new CompressionPlugin({
        algorithm: 'gzip',
        test: /\.(js|css|html|svg|wasm|json)$/,
        threshold: 10240,
        minRatio: 0.8,
        deleteOriginalAssets: false,
      }),

      new WebpackManifestPlugin({
        fileName: 'asset-manifest.json',
      }),

      // PWA service worker
      new WorkboxWebpackPlugin.GenerateSW({
        clientsClaim: true,
        skipWaiting: true,
        cleanupOutdatedCaches: true,
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/cdn\.jsdelivr\.net\/.*/i,
            handler: 'CacheFirst',
            options: {
              cacheName: 'cdn-assets',
              expiration: { maxEntries: 50, maxAgeSeconds: 2592000 },
            },
          },
          {
            urlPattern: /\.(?:png|jpg|jpeg|svg|gif|webp|ico|wasm|json)$/,
            handler: 'CacheFirst',
            options: {
              cacheName: 'static-resources',
              expiration: { maxEntries: 200, maxAgeSeconds: 2592000 },
            },
          },
        ],
        navigateFallback: '/index.html',
        navigateFallbackDenylist: [/^\/api\//],
      }),

      new CopyPlugin({
        patterns: [
          { from: 'public/manifest.json', to: 'manifest.json' },
          { from: 'public/pwa-*.png', to: 'assets/[name][ext]' },
        ],
      }),

      // Define environment variables
      new webpack.DefinePlugin({
        'process.env.NODE_ENV': JSON.stringify(isProduction ? 'production' : 'development'),
      }),
    ].filter(Boolean),

    performance: {
      hints: isProduction && 'warning',
      maxEntrypointSize: 2500000,
      maxAssetSize: 1500000,
    },

    devServer: {
      port: 3000,
      open: true,
      hot: true,
      historyApiFallback: true,
      compress: true,
      static: {
        directory: path.join(__dirname, 'public'),
      },
    },

    externals: {
      // Optional: exclude heavy tfjs deps from bundle if you want to load from CDN
      // '@tensorflow/tfjs': 'tf',
    },
  };
};      ],
    },

    optimization: {
      minimize: isProduction,
      minimizer: [
        new TerserPlugin({
          parallel: true,
          extractComments: false,
          terserOptions: {
            compress: {
              drop_console: false,
              passes: 3,
              pure_funcs: ['console.debug'],
              pure_getters: true,
              unsafe: true,
              unsafe_comps: true,
              unsafe_math: true,
              unsafe_methods: true,
              unsafe_undefined: true,
            },
            mangle: true,
            format: {
              comments: false,
            },
          },
        }),
        new CssMinimizerPlugin(),
      ],
      splitChunks: {
        chunks: 'all',
        minSize: 20000,
        maxSize: 0,
        minChunks: 1,
        cacheGroups: {
          vendor: {
            test: /[\\/]node_modules[\\/](react|react-dom|framer-motion|@tensorflow|tensorflow|@mediapipe)[\\/]/,
            name: 'vendor',
            priority: 10,
            chunks: 'initial',
          },
          tfjs: {
            test: /[\\/]node_modules[\\/](@tensorflow)[\\/]/,
            name: 'tfjs',
            priority: 12,
            chunks: 'all',
          },
          mediapipe: {
            test: /[\\/]node_modules[\\/](@mediapipe)[\\/]/,
            name: 'mediapipe',
            priority: 11,
            chunks: 'all',
          },
          defaultVendors: {
            test: /[\\/]node_modules[\\/]/,
            priority: -10,
            reuseExistingChunk: true,
          },
          default: {
            minChunks: 2,
            priority: -20,
            reuseExistingChunk: true,
          },
        },
      },
      runtimeChunk: {
        name: 'runtime',
      },
    },

    plugins: [
      new HtmlWebpackPlugin({
        template: './index.html',
        filename: 'index.html',
        minify: isProduction && {
          collapseWhitespace: true,
          removeComments: true,
          removeRedundantAttributes: true,
          removeScriptTypeAttributes: true,
          removeStyleLinkTypeAttributes: true,
          useShortDoctype: true,
        },
      }),

      isProduction && new MiniCssExtractPlugin({
        filename: 'assets/css/[name].[contenthash:8].css',
        chunkFilename: 'assets/css/[name].[contenthash:8].chunk.css',
      }),

      new CleanWebpackPlugin(),

      new CompressionPlugin({
        algorithm: 'brotliCompress',
        test: /\.(js|css|html|svg|woff2?)$/,
        threshold: 10240,
        minRatio: 0.8,
        deleteOriginalAssets: false,
      }),

      new CompressionPlugin({
        algorithm: 'gzip',
        test: /\.(js|css|html|svg|woff2?)$/,
        threshold: 10240,
        minRatio: 0.8,
        deleteOriginalAssets: false,
      }),

      new WebpackManifestPlugin({
        fileName: 'asset-manifest.json',
        generate: (seed, files, entrypoints) => {
          const manifestFiles = files.reduce((manifest, file) => {
            manifest[file.name] = file.path;
            return manifest;
          }, seed);
          const entrypointFiles = entrypoints.main.filter(
            (fileName) => !fileName.endsWith('.map')
          );

          return {
            files: manifestFiles,
            entrypoints: entrypointFiles,
          };
        },
      }),

      new WorkboxWebpackPlugin.GenerateSW({
        clientsClaim: true,
        skipWaiting: true,
        cleanupOutdatedCaches: true,
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/cdn\.jsdelivr\.net\/.*/i,
            handler: 'CacheFirst',
            options: {
              cacheName: 'cdn-assets',
              expiration: { maxEntries: 50, maxAgeSeconds: 2592000 },
            },
          },
          {
            urlPattern: /\.(?:png|jpg|jpeg|svg|gif|webp|ico)$/,
            handler: 'CacheFirst',
            options: {
              cacheName: 'images',
              expiration: { maxEntries: 100, maxAgeSeconds: 2592000 },
            },
          },
        ],
        navigateFallback: '/index.html',
        navigateFallbackDenylist: [/^\/api\//],
      }),

      new CopyPlugin({
        patterns: [
          { from: 'public/manifest.json', to: 'manifest.json' },
          { from: 'public/pwa-*.png', to: 'assets/[name][ext]' },
        ],
      }),
    ].filter(Boolean),

    performance: {
      hints: isProduction && 'warning',
      maxEntrypointSize: 2000000,
      maxAssetSize: 1000000,
    },

    devServer: {
      port: 3000,
      open: true,
      hot: true,
      historyApiFallback: true,
      compress: true,
    },
  };
};
