import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const plugin = readFileSync(new URL("../src/plugin/CmgMemoryPlugin.jsx", import.meta.url), "utf8");
const css = readFileSync(new URL("../src/styles.css", import.meta.url), "utf8");
const app = readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");

test("renders breadth-first memory levels from left to right with faster zoom", () => {
  assert.match(plugin, /transform:\s*\(_node, position\)\s*=>\s*\(\{\s*x:\s*position\.y,\s*y:\s*position\.x\s*\}\)/);
  const sensitivity = Number(plugin.match(/wheelSensitivity:\s*([\d.]+)/)?.[1]);
  const initialMultiplier = Number(plugin.match(/cy\.zoom\(\{ level:.*cy\.zoom\(\) \* ([\d.]+)/)?.[1]);
  assert.ok(sensitivity >= 0.4, `wheel sensitivity ${sensitivity} is too low`);
  assert.ok(initialMultiplier > 1, `initial zoom multiplier ${initialMultiplier} must enlarge the fitted view`);
});

test("uses larger information rail typography without changing the monochrome layout", () => {
  assert.match(css, /\.node-title-row h3\s*\{[^}]*font-size:\s*20px/si);
  assert.match(app, /VITE_HAMGF_API_URL \|\| "http:\/\/127\.0\.0\.1:8000"/);
  assert.doesNotMatch(app, /VITE_HAMGF_API_URL \|\| "http:\/\/127\.0\.0\.1:5173"/);
  assert.match(css, /\.node-content\s*\{[^}]*font-size:\s*14px/si);
  assert.match(css, /\.metadata-list div\s*\{[^}]*font-size:\s*11px/si);
  assert.match(css, /\.stat strong\s*\{[^}]*24px/si);
  assert.match(css, /\.stat strong\s*\{(?=[^}]*font-family:\s*"Microsoft YaHei",\s*"微软雅黑",\s*sans-serif)(?=[^}]*font-weight:\s*700)[^}]*\}/si);
  assert.match(css, /\.panel-index\s*\{(?=[^}]*font-family:\s*"Microsoft YaHei",\s*"微软雅黑",\s*sans-serif)(?=[^}]*font-weight:\s*700)[^}]*\}/si);
  assert.match(css, /\.score b\s*\{(?=[^}]*font-family:\s*"Microsoft YaHei",\s*"微软雅黑",\s*sans-serif)(?=[^}]*font-weight:\s*700)[^}]*\}/si);
  assert.match(css, /\.timeline p span\s*\{(?=[^}]*font-family:\s*"Microsoft YaHei",\s*"微软雅黑",\s*sans-serif)(?=[^}]*font-weight:\s*700)[^}]*\}/si);
  assert.match(css, /\.timeline small\s*\{(?=[^}]*font-family:\s*"Microsoft YaHei",\s*"微软雅黑",\s*sans-serif)(?=[^}]*font-weight:\s*700)[^}]*\}/si);
});
