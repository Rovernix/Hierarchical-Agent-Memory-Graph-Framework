import { CmgMemoryPlugin } from "./plugin/CmgMemoryPlugin.jsx";
import { DEMO_GRAPH } from "./lib/demoGraph.js";

export function App() {
  const baseUrl = import.meta.env.VITE_HAMGF_API_URL || "http://127.0.0.1:8000";

  return (
    <main className="page-shell">
      <header className="brand-bar">
        <div className="brand-mark" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <div>
          <p className="eyebrow">HIERARCHICAL AGENT MEMORY GRAPH</p>
          <h1>Memory Atlas</h1>
        </div>
        <div className="phase-chip">PHASE 03 · LIVE</div>
      </header>
      <CmgMemoryPlugin baseUrl={baseUrl} fallbackGraph={DEMO_GRAPH} refreshMs={1800} />
      <footer className="page-footer">
        CMG Visual Audit Plugin · NetworkX source · React + Cytoscape
      </footer>
    </main>
  );
}
