import { defineConfig } from "vite";

// Webflow Cloud overwrites `base` from the environment mount path.
// Do not set it here.
export default defineConfig({
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
