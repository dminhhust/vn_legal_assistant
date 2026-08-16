import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev-server proxy: the browser talks to `/api/*` and Vite forwards it
// to the local FastAPI backend, mirroring the nginx `/api` proxy used
// in production (frontend/Dockerfile + deploy/entrypoint.sh). This
// keeps the app origin-agnostic — no CORS involved in either mode.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8080",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
