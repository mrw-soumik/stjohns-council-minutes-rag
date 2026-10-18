// Automated browser check for the static demo (docs/).
//
// Serves docs/ locally, drives the installed Chrome with puppeteer-core and:
//   * saves screenshots at 1440 px (desktop) and 390 px (mobile device emulation),
//   * records console errors and page errors,
//   * checks the keyboard path (skip link, search field, results),
//   * runs axe-core (WCAG 2.0/2.1 A and AA rules) in light and dark colour schemes,
//   * measures search time after the model has loaded (NFR-03),
//   * compares query embeddings computed in the browser with transformers.js in Node.
//
// Usage: node scripts/screenshots.mjs [--chrome <path>] [--out docs/screenshots]
import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import puppeteer from "puppeteer-core";

const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const CHROME = opt(
  "--chrome",
  process.env.CHROME_PATH ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
);
const OUT = opt("--out", "docs/screenshots");
async function freePort() {
  const { createServer } = await import("node:net");
  return new Promise((resolve) => {
    const srv = createServer();
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}
const PORT = Number(opt("--port", String(await freePort())));
const BASE = `http://127.0.0.1:${PORT}/`;

const server = spawn("python3", ["-m", "http.server", String(PORT), "--bind", "127.0.0.1"], {
  cwd: "docs",
  stdio: "ignore",
});
process.on("exit", () => server.kill());
await new Promise((r) => setTimeout(r, 800));

const report = { base: BASE, chrome: CHROME, console: [], pageErrors: [], checks: {} };
const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ["--no-first-run", "--no-default-browser-check"],
});

async function openPage(viewport, { mobile = false, dark = false } = {}) {
  const page = await browser.newPage();
  if (mobile) {
    await page.emulate({
      viewport: { width: 390, height: 844, deviceScaleFactor: 3, isMobile: true, hasTouch: true },
      userAgent:
        "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 " +
        "(KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1",
    });
  } else {
    await page.setViewport(viewport);
  }
  await page.emulateMediaFeatures([
    { name: "prefers-color-scheme", value: dark ? "dark" : "light" },
  ]);
  page.on("console", (msg) => {
    if (["error", "warn"].includes(msg.type())) {
      report.console.push({ type: msg.type(), text: msg.text().slice(0, 300) });
    }
  });
  page.on("pageerror", (err) => report.pageErrors.push(String(err).slice(0, 300)));
  return page;
}

async function waitReady(page) {
  await page.waitForFunction(() => window.__demo && window.__demo.ready, { timeout: 30000 });
  await page.evaluate(() => window.__demo.ready);
  await page.waitForFunction(() => window.__demo.modelReady, { timeout: 120000 });
  return page.evaluate(() => window.__demo.modelReady);
}

async function axe(page, label) {
  await page.addScriptTag({ path: "node_modules/axe-core/axe.min.js" });
  const result = await page.evaluate(async () => {
    const r = await window.axe.run(document, {
      runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] },
    });
    return {
      violations: r.violations.map((v) => ({ id: v.id, impact: v.impact, nodes: v.nodes.length })),
      passes: r.passes.length,
      incomplete: r.incomplete.map((v) => v.id),
    };
  });
  report.checks[`axe_${label}`] = result;
}

try {
  await mkdir(OUT, { recursive: true });

  // ---- desktop, light ----------------------------------------------------------
  const desk = await openPage({ width: 1440, height: 900 });
  await desk.goto(BASE, { waitUntil: "networkidle0", timeout: 60000 });
  const modelOk = await waitReady(desk);
  report.checks.model_loaded = modelOk;

  // keyboard path: first Tab reaches the skip link, Enter moves focus into <main>
  await desk.keyboard.press("Tab");
  const firstFocus = await desk.evaluate(() => document.activeElement.className);
  await desk.keyboard.press("Enter");
  const afterSkip = await desk.evaluate(() => document.activeElement.id);
  await desk.focus("#q");
  await desk.keyboard.type("Who voted against the 2026 budget?");
  await desk.keyboard.press("Enter");
  await desk.waitForFunction(() => document.querySelectorAll("#results li").length > 0);
  report.checks.keyboard = { first_tab: firstFocus, after_skip_link: afterSkip };

  // PA-06: time from navigation start until keyword search and the model are ready
  report.checks.timings_ms = await desk.evaluate(() => window.__demo.timings);

  // PA-10: disclaimer, independence notice, coverage dates and model names are visible
  const visible = await desk.evaluate(() => document.body.innerText);
  const meta = JSON.parse(await readFile("docs/data/index.json", "utf8"));
  const showcase = JSON.parse(await readFile("docs/data/showcase.json", "utf8"));
  const longDate = (iso) =>
    new Date(`${iso}T12:00:00`).toLocaleDateString("en-CA", { year: "numeric", month: "long", day: "numeric" });
  report.checks.ui_notices = {
    independence_notice: visible.includes("Independent portfolio project"),
    disclaimer: visible.includes("Not affiliated with or endorsed by the City of St. John's"),
    coverage_dates:
      visible.includes(longDate(meta.coverage.from)) && visible.includes(longDate(meta.coverage.to)),
    search_model_name: visible.includes("bge-small-en-v1.5"),
    answer_model_name: showcase.model ? visible.includes(showcase.model) : "no showcase yet",
  };

  // timing: hybrid searches after the model has loaded (embedding + ranking)
  const timings = [];
  for (const q of [
    "Who voted against the 2026 budget?",
    "heritage designation 157-163 Water Street",
    "Did the City change the name of its June holiday?",
    "parking relief for a subsidiary dwelling unit",
    "Tely 10 road closures",
  ]) {
    await desk.evaluate((query) => window.__demo.search(query, { mode: "hybrid" }), q);
    timings.push(await desk.evaluate(() => window.__demo.lastSearchMs));
  }
  report.checks.search_ms = timings;

  // embedding parity in the real browser (WASM) against transformers.js in Node
  const parity = JSON.parse(await readFile(".cache/parity/js_vectors.json", "utf8"));
  const sims = [];
  for (let i = 0; i < 6; i++) {
    const browserVec = await desk.evaluate((t) => window.__demo.embed(t), parity.texts[i]);
    const nodeVec = parity.vectors.q8[i];
    let dot = 0;
    let a = 0;
    let b = 0;
    for (let d = 0; d < nodeVec.length; d++) {
      dot += browserVec[d] * nodeVec[d];
      a += browserVec[d] ** 2;
      b += nodeVec[d] ** 2;
    }
    sims.push(dot / Math.sqrt(a * b));
  }
  report.checks.browser_vs_node_q8_cosine = { min: Math.min(...sims), mean: sims.reduce((x, y) => x + y) / sims.length };

  await desk.evaluate(() =>
    window.__demo.search("Who voted against the 2026 budget?", { mode: "hybrid", from: "", to: "" }),
  );
  // README screenshot: search panel plus the first results
  await desk.setViewport({ width: 1440, height: 1500 });
  await desk.evaluate(() => document.activeElement.blur());
  await desk.evaluate(() => window.scrollTo(0, 0));
  await desk.screenshot({ path: `${OUT}/desktop.png`, fullPage: false });
  await axe(desk, "desktop_light");

  // ---- desktop, dark (accessibility scan only) -----------------------------------
  const dark = await openPage({ width: 1440, height: 900 }, { dark: true });
  await dark.goto(`${BASE}?q=Tely+10+road+closures`, { waitUntil: "networkidle0", timeout: 60000 });
  await waitReady(dark);
  await dark.waitForFunction(() => document.querySelectorAll("#results li").length > 0);
  await axe(dark, "desktop_dark");

  // ---- mobile 390 px ---------------------------------------------------------------
  const mob = await openPage(null, { mobile: true });
  await mob.goto(`${BASE}?q=Did+the+City+change+the+name+of+its+June+holiday%3F`, {
    waitUntil: "networkidle0",
    timeout: 60000,
  });
  await waitReady(mob);
  await mob.waitForFunction(() => document.querySelectorAll("#results li").length > 0);
  await mob.evaluate(() => window.__demo.search(document.getElementById("q").value));
  const overflow = await mob.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
  );
  report.checks.mobile_horizontal_overflow = overflow;
  await mob.evaluate(() => document.activeElement.blur());
  await mob.screenshot({ path: `${OUT}/mobile.png`, fullPage: false });
  await mob.evaluate(() => document.getElementById("results-heading").scrollIntoView());
  await mob.screenshot({ path: `${OUT}/mobile-results.png`, fullPage: false });
  await axe(mob, "mobile_light");
} finally {
  await browser.close();
  server.kill();
}

await writeFile(`${OUT}/browser-check.json`, JSON.stringify(report, null, 1) + "\n");
console.log(JSON.stringify(report, null, 1));
const failed =
  report.pageErrors.length > 0 ||
  report.console.some((c) => c.type === "error") ||
  Object.entries(report.checks).some(([k, v]) => k.startsWith("axe_") && v.violations.length > 0) ||
  Object.values(report.checks.ui_notices ?? {}).some((v) => v === false);
process.exit(failed ? 1 : 0);
