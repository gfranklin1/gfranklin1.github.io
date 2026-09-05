import { defineConfig } from 'astro/config';

// Served from the root of gfranklin1.github.io, so no `base`. Asset URLs go
// through BASE_URL in the pages, which resolves to '/' here — pointing this at
// a custom domain later means changing `site` and adding public/CNAME, nothing
// else.
export default defineConfig({
  site: 'https://gfranklin1.github.io',
  output: 'static',
  devToolbar: { enabled: false },
  build: { inlineStylesheets: 'auto' },
});
