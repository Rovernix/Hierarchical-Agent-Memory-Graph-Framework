import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const mode = process.argv[2] || 'dev';
const extra = process.argv.slice(3);
if (!['dev', 'start', 'backend'].includes(mode)) throw new Error(`Unknown mode: ${mode}`);
const executable = process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python';
const candidates = [
  path.join(root, '.venv', executable),
  path.join(root, '..', '.venv', executable),
  path.join(root, '..', '.venv-studio', executable),
];
const python = process.env.HAMGF_STUDIO_PYTHON || candidates.find(existsSync)
  || (process.platform === 'win32' ? 'python' : 'python3');
let backend;
let vite;
let stopping = false;
let watchdog;
let forcedExit;

function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  clearInterval(watchdog);
  if (vite && vite.exitCode === null) vite.kill();
  // The bootstrap translates pipe closure into the backend's graceful shutdown.
  if (backend && backend.exitCode === null) backend.stdin.end();
  forcedExit = setTimeout(() => {
    if (backend && backend.exitCode === null) backend.kill();
    process.exit(code);
  }, 5000);
  const children = [backend, vite].filter(child => child && child.exitCode === null);
  if (!children.length) process.exit(code);
  let pending = children.length;
  for (const child of children) child.once('exit', () => {
    if (--pending === 0) { clearTimeout(forcedExit); process.exit(code); }
  });
}
process.on('SIGINT', () => stop());
process.on('SIGTERM', () => stop());
if (process.platform === 'win32') process.on('SIGBREAK', () => stop());
process.on('exit', () => {
  if (vite && vite.exitCode === null) vite.kill();
  if (backend && backend.exitCode === null) backend.kill();
});
// hamgf-chat may terminate its npm parent directly on Windows.
const ownerPid = process.ppid;
watchdog = setInterval(() => {
  try { process.kill(ownerPid, 0); } catch { stop(); }
}, 1000);

function startBackend() {
  return new Promise((resolve, reject) => {
    const args = [path.join(root, 'scripts', 'studio_bootstrap.py'), '--watch-stdin'];
    if (mode === 'dev') args.push('--port', '0', '--no-browser');
    else if (mode === 'backend') args.push('--no-browser', ...extra);
    else args.push(...extra);
    if (process.env.HAMGF_STUDIO_DATA_DIR) args.push('--data-dir', process.env.HAMGF_STUDIO_DATA_DIR);
    backend = spawn(python, args, { cwd: root, windowsHide: true, stdio: ['pipe', 'pipe', 'inherit'],
      env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' } });
    let output = '';
    const timeout = setTimeout(() => reject(new Error('Studio backend startup timed out.')), 20000);
    backend.on('error', error => { clearTimeout(timeout); reject(error); });
    backend.stdout.on('data', chunk => {
      process.stdout.write(chunk);
      output = (output + chunk.toString()).slice(-4096);
      const match = output.match(/HAMGF Studio: (http:\/\/127\.0\.0\.1:\d+)/);
      if (match) { clearTimeout(timeout); resolve(match[1]); }
    });
    backend.on('exit', code => {
      clearTimeout(timeout);
      if (!stopping) {
        reject(new Error(`Studio backend exited (${code}). Install the parent HAMGF package and its dependencies, or set HAMGF_STUDIO_PYTHON.`));
        stop(code || 0);
      }
    });
  });
}

try {
  let target = mode === 'dev' && process.env.HAMGF_STUDIO_API_URL;
  if (target) {
    const url = new URL(target);
    if (url.protocol !== 'http:' || !['127.0.0.1', 'localhost'].includes(url.hostname) || url.username || url.password) {
      throw new Error('HAMGF_STUDIO_API_URL must be a local HTTP Studio service.');
    }
    target = url.origin;
  } else target = await startBackend();
  if (mode === 'dev') {
    const viteCli = path.join(root, 'node_modules', 'vite', 'bin', 'vite.js');
    if (!existsSync(viteCli)) throw new Error('Run npm ci in frontend first.');
    vite = spawn(process.execPath, [viteCli, ...extra], { cwd: root, windowsHide: true,
      stdio: 'inherit', env: { ...process.env, HAMGF_STUDIO_API_URL: target } });
    vite.on('error', error => { console.error(error.message); stop(1); });
    vite.on('exit', code => { if (!stopping) stop(code || 0); });
  }
} catch (error) {
  console.error(error.message);
  stop(1);
}
