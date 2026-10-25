import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const api = `http://127.0.0.1:${process.env.PLACEMENT_API_PORT || 8765}`;

export default defineConfig({
  base: process.env.VITE_BASE || "/",   // GitHub Pages serves a project site from /<repository>/
  plugins: [react()],
  server: {
    proxy: { "/api": api, "/media": api },
    // Generated data and large runtime files: watching thousands of them on a network drive crashed the dev server during an export.
    watch: { ignored: ["**/public/static-api/**", "**/public/models/**", "**/public/ort/**", "**/public/pyodide/**"] },
  },
});
