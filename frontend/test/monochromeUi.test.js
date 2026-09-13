import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const css = readFileSync(new URL("../src/styles.css", import.meta.url), "utf8");
const plugin = readFileSync(new URL("../src/plugin/CmgMemoryPlugin.jsx", import.meta.url), "utf8");

function colorChannels(source) {
  return [...source.matchAll(/#([0-9a-f]{3,8})\b/gi)].map((match) => {
    let value = match[1];
    if (value.length === 3 || value.length === 4) {
      value = value.split("").map((part) => part + part).join("");
    }
    return { raw: match[0], red: value.slice(0, 2), green: value.slice(2, 4), blue: value.slice(4, 6) };
  });
}

test("uses only grayscale channels in CSS and Cytoscape styles", () => {
  const colors = [...colorChannels(css), ...colorChannels(plugin)];
  const colored = colors.filter(({ red, green, blue }) => red !== green || green !== blue);
  assert.deepEqual(colored, []);
  assert.doesNotMatch(css, /--(?:blue|mint|violet|amber|rose)\b/);
});

test("uses one continuous workspace instead of separate panel cards", () => {
  assert.match(css, /\.workspace\s*\{[^}]*border:\s*1px solid/si);
  assert.match(css, /\.graph-panel,\s*\.detail-panel\s*\{[^}]*border:\s*0/si);
  assert.deepEqual([...css.matchAll(/box-shadow:\s*([^;]+)/g)].map((match) => match[1].trim()), ["none"]);
  assert.doesNotMatch(plugin, /accent=/);
});
