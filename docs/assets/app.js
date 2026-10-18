// UI for the static demo: loads the index, loads the embedding model with
// transformers.js (pinned), runs searches and renders results. All text from the
// index is inserted with textContent, never as HTML.
import { SearchIndex } from "./search.js";

const TRANSFORMERS_URL =
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist/transformers.min.js";
const MODEL_ID = "Xenova/bge-small-en-v1.5";
const MODEL_DTYPE = "q8"; // onnx/model_quantized.onnx; same retrieval quality as fp32 in our eval
const RESULTS_K = 10;

const $ = (id) => document.getElementById(id);
const els = {
  form: $("search-form"),
  q: $("q"),
  from: $("from"),
  to: $("to"),
  mode: $("mode"),
  clear: $("clear"),
  status: $("model-status"),
  progress: $("model-progress"),
  progressBar: $("model-progress-bar"),
  summary: $("results-summary"),
  results: $("results"),
  coverage: $("coverage-text"),
  showcaseLabel: $("showcase-label"),
  showcaseList: $("showcase-list"),
};

const state = {
  index: null,
  embed: null, // async (text) => Float32Array, once the model is ready
  modelState: "loading", // loading | ready | failed
  lastSearchDegraded: false,
};

const longDate = (iso) =>
  new Date(`${iso}T12:00:00`).toLocaleDateString("en-CA", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function setStatus(text, kind = "") {
  els.status.textContent = text;
  els.status.className = `status ${kind}`.trim();
}

// ---------------------------------------------------------------- model --
async function loadModel(revision) {
  const files = new Map();
  const onProgress = (event) => {
    if (event.status === "progress" && event.file && event.total) {
      files.set(event.file, { loaded: event.loaded, total: event.total });
      let loaded = 0;
      let total = 0;
      for (const f of files.values()) {
        loaded += f.loaded;
        total += f.total;
      }
      const pct = total ? Math.round((100 * loaded) / total) : 0;
      els.progress.hidden = false;
      els.progressBar.style.width = `${pct}%`;
      const mb = (n) => (n / 1e6).toFixed(0);
      els.status.textContent =
        `Downloading the search model: ${mb(loaded)} of ${mb(total)} MB. ` +
        "Keyword search already works.";
    }
  };
  const { pipeline, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false;
  env.useBrowserCache = true;
  const extractor = await pipeline("feature-extraction", MODEL_ID, {
    revision,
    dtype: MODEL_DTYPE,
    device: "wasm",
    progress_callback: onProgress,
  });
  return async (text) => {
    const output = await extractor(text, { pooling: "cls", normalize: true });
    return output.data;
  };
}

// --------------------------------------------------------------- search --
async function runSearch({ updateUrl = true } = {}) {
  const query = els.q.value.trim();
  if (!state.index) return;
  if (!query) {
    els.summary.textContent = "Type a question or keywords to search.";
    els.q.focus();
    return;
  }
  const dateFrom = els.from.value || null;
  const dateTo = els.to.value || null;
  if (dateFrom && dateTo && dateFrom > dateTo) {
    els.summary.textContent = "The start date must be on or before the end date.";
    els.from.focus();
    return;
  }
  let mode = els.mode.value;
  let note = "";
  if (mode !== "bm25" && state.modelState !== "ready") {
    note =
      state.modelState === "failed"
        ? " The meaning-based model could not be loaded, so these are keyword results."
        : " The meaning-based model is still loading, so these are keyword results for now.";
    mode = "bm25";
    state.lastSearchDegraded = state.modelState === "loading";
  } else {
    state.lastSearchDegraded = false;
  }

  const started = performance.now();
  let vector = null;
  if (mode !== "bm25") {
    const instruction = state.index.meta.embedding.query_instruction;
    vector = await state.embed(instruction + query);
  }
  const hits = state.index.search(query, vector, { k: RESULTS_K, mode, dateFrom, dateTo });
  const ms = Math.round(performance.now() - started);
  render(hits, query, mode, ms, note);
  window.__demo.lastSearchMs = ms;

  if (updateUrl) {
    const params = new URLSearchParams();
    params.set("q", query);
    if (dateFrom) params.set("from", dateFrom);
    if (dateTo) params.set("to", dateTo);
    if (els.mode.value !== "hybrid") params.set("mode", els.mode.value);
    history.replaceState(null, "", `?${params}`);
  }
  return hits;
}

const MODE_LABEL = { hybrid: "hybrid search", bm25: "keyword search", dense: "meaning-based search" };

function render(hits, query, mode, ms, note) {
  els.results.replaceChildren();
  const searched = state.index.n.toLocaleString("en-CA");
  if (!hits.length) {
    els.summary.textContent =
      `No passages matched \u201c${query}\u201d with these filters (${MODE_LABEL[mode]}).` + note;
    return;
  }

  // Group passages from the same agenda item into one card, in rank order.
  const groups = new Map();
  for (const hit of hits) {
    const chunk = state.index.chunks[hit.idx];
    const key = `${chunk.m}|${chunk.anchor}|${chunk.ref}|${chunk.kind}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(hit);
  }
  const items = groups.size === 1 ? "1 agenda item" : `${groups.size} agenda items`;
  els.summary.textContent =
    `${hits.length} passages from ${items} for \u201c${query}\u201d (${MODE_LABEL[mode]} of ` +
    `${searched} passages, ${ms} ms).` + note;

  let n = 0;
  for (const group of groups.values()) {
    n += 1;
    const chunk = state.index.chunks[group[0].idx];
    const meeting = state.index.meetings[chunk.m];
    const titleId = `result-${n}`;
    const itemUrl = meeting.url && chunk.anchor ? `${meeting.url}#${chunk.anchor}` : meeting.url;
    const isAttendance = chunk.kind === "attendance";

    const links = el("p", { class: "result-links" });
    if (itemUrl) {
      links.append(
        el(
          "a",
          { href: itemUrl },
          isAttendance ? "Open the official minutes" : "Open this item in the official minutes",
        ),
      );
    }
    if (meeting.pdf) links.append(el("a", { href: meeting.pdf }, "Official minutes (PDF)"));

    const article = el(
      "article",
      { "aria-labelledby": titleId },
      el(
        "p",
        { class: "result-meta" },
        el("time", { datetime: meeting.date }, longDate(meeting.date)),
        ` \u00b7 ${meeting.name}`,
      ),
      el(
        "h3",
        { id: titleId },
        isAttendance ? null : el("span", { class: "ref" }, `Item ${chunk.ref}`),
        isAttendance ? "Meeting details and attendance" : chunk.title,
      ),
      chunk.path ? el("p", { class: "result-path" }, chunk.path.replaceAll(" > ", " \u203a ")) : null,
    );
    for (const hit of group) {
      const part = state.index.chunks[hit.idx];
      // Only decisions carry a quote: the opening words of the motion and its outcome.
      if (part.excerpt) {
        const quote = el("blockquote", { class: "excerpt", cite: itemUrl || null }, part.excerpt);
        if (part.cut) quote.append(el("span", { class: "more", "aria-label": "excerpt continues" }, " \u2026"));
        article.append(quote);
      }
      if (part.outcome) article.append(el("p", { class: "result-outcome" }, `Recorded result: ${part.outcome}`));
      const why = [];
      if (hit.bm25Rank) why.push(`keyword rank ${hit.bm25Rank}`);
      if (hit.denseRank) why.push(`meaning rank ${hit.denseRank}`);
      if (why.length) {
        const lead = part.excerpt ? "Matched by" : "Passage not quoted; matched by";
        const cls = part.excerpt ? "result-why" : "result-why unquoted";
        article.append(el("p", { class: cls }, `${lead} ${why.join(" and ")}.`));
      }
    }
    article.append(links);
    els.results.append(el("li", { class: "result" }, article));
  }
}

// ------------------------------------------------------------- showcase --
async function loadShowcase() {
  let data;
  try {
    const res = await fetch("data/showcase.json", { cache: "no-cache" });
    if (!res.ok) return;
    data = await res.json();
  } catch {
    return;
  }
  if (!data || !Array.isArray(data.items) || !data.items.length) return;
  els.showcaseLabel.textContent =
    `Answers generated offline by ${data.model} on ${longDate(data.generated_on)}; ` +
    "search runs live in your browser. Citations link to the official minutes.";
  for (const item of data.items) {
    const block = el("article", { class: "qa" }, el("h3", {}, item.question));
    if (item.status === "answered") {
      block.append(el("p", { class: "answer" }, item.answer));
      const list = el("ol", { "aria-label": "Sources for this answer" });
      for (const c of item.citations) {
        list.append(
          el(
            "li",
            {},
            el("a", { href: c.url }, `${longDate(c.meeting_date)}, item ${c.item_ref}: ${c.item_title}`),
          ),
        );
      }
      block.append(list);
    } else {
      block.append(el("p", { class: "answer not-found" }, item.answer));
    }
    if (item.auto_check) {
      block.append(
        el(
          "p",
          { class: "grade" },
          `Automatic check: ${item.auto_check} (required facts and a citation to the source ` +
            "passage, checked against the test set).",
        ),
      );
    }
    if (item.review_note) block.append(el("p", { class: "review-note" }, item.review_note));
    els.showcaseList.append(block);
  }
}

// ----------------------------------------------------------------- init --
async function init() {
  const params = new URLSearchParams(location.search);
  els.q.value = params.get("q") ?? "";
  els.from.value = params.get("from") ?? "";
  els.to.value = params.get("to") ?? "";
  if (["hybrid", "bm25", "dense"].includes(params.get("mode"))) els.mode.value = params.get("mode");

  try {
    state.index = await SearchIndex.load(new URL("data/", document.baseURI));
  } catch (error) {
    setStatus(`The search index could not be loaded (${error.message}).`, "warn");
    return;
  }
  window.__demo.timings.indexReadyMs = Math.round(performance.now());
  const cov = state.index.meta.coverage;
  els.coverage.textContent =
    `${cov.meetings} Regular Council Meetings, ${longDate(cov.from)} to ${longDate(cov.to)} ` +
    `(${state.index.n.toLocaleString("en-CA")} passages).`;
  els.from.min = els.to.min = cov.from;
  els.from.max = els.to.max = cov.to;
  setStatus("Keyword search is ready. Loading the meaning-based search model\u2026");

  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    runSearch();
  });
  els.clear.addEventListener("click", () => {
    els.from.value = "";
    els.to.value = "";
    els.mode.value = "hybrid";
    if (els.q.value.trim()) runSearch();
  });
  for (const chip of document.querySelectorAll(".chip")) {
    chip.addEventListener("click", () => {
      els.q.value = chip.dataset.q;
      runSearch();
    });
  }
  loadShowcase();
  if (els.q.value.trim()) runSearch({ updateUrl: false });

  window.__demo.search = (query, options = {}) => {
    els.q.value = query;
    if (options.from !== undefined) els.from.value = options.from ?? "";
    if (options.to !== undefined) els.to.value = options.to ?? "";
    if (options.mode) els.mode.value = options.mode;
    return runSearch();
  };

  window.__demo.modelReady = loadModel(state.index.meta.embedding.revision)
    .then((embed) => {
      state.embed = embed;
      state.modelState = "ready";
      window.__demo.timings.modelReadyMs = Math.round(performance.now());
      window.__demo.embed = (text) => embed(text).then((v) => Array.from(v));
      els.progress.hidden = true;
      setStatus(
        `Ready: hybrid search with ${MODEL_ID.split("/").pop()} (keywords and meaning) runs in your browser.`,
        "ok",
      );
      if (state.lastSearchDegraded && els.q.value.trim()) runSearch({ updateUrl: false });
      return true;
    })
    .catch((error) => {
      state.modelState = "failed";
      els.progress.hidden = true;
      console.warn("Model load failed:", error);
      setStatus(
        "The meaning-based model could not be loaded (offline or blocked). Keyword search still works.",
        "warn",
      );
      return false;
    });
  return true;
}

// Hooks for the automated browser check (scripts/screenshots.mjs).
window.__demo = { ready: null, modelReady: null, lastSearchMs: null, timings: {} };
window.__demo.ready = init();
