import { defineConfig } from 'vite';

// assets/ を公開ディレクトリにして /avatar.vrm と /humanoid.xml をそのまま配信する。
// avatar.vrm は Git 管理対象外（PLAN §11.1）で、無い環境では簡易人形が使われる。
export default defineConfig({
  publicDir: '../assets',
  server: { host: '127.0.0.1', port: 5173 },
});
