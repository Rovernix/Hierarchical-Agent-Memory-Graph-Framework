import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist-plugin",
    emptyOutDir: true,
    lib: {
      entry: resolve(import.meta.dirname, "src/plugin/index.js"),
      name: "HamgfCmgPlugin",
      formats: ["es"],
      fileName: "hamgf-cmg-plugin",
    },
    rollupOptions: {
      external: ["react", "react-dom", "cytoscape"],
    },
  },
});
