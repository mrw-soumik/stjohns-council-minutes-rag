// Search core for the static demo. No DOM access, so Node can import it for the
// parity check against the Python implementation (scripts/check_search_parity.mjs).
//
// Mirrors src/council_rag/text.py (tokeniser) and src/council_rag/demo_export.py
// (DemoIndex): Okapi BM25 over hashed terms, cosine similarity over int8 vectors,
// reciprocal rank fusion, and a meeting-date filter applied before ranking.

export const STOPWORDS = new Set(
  (
    "a about above after again all also am an and any are as at be been before being " +
    "below between both but by can could did do does doing down during each few for " +
    "from further had has have having he her here hers herself him himself his how i " +
    "if in into is it its itself just me more most my myself no nor not now of off on " +
    "once only or other our ours ourselves out over own same she should so some such " +
    "than that the their theirs them themselves then there these they this those " +
    "through to too under until up very was we were what when where which while who " +
    "whom why will with would you your yours yourself yourselves"
  ).split(" "),
);

const MARKS = /[\u0300-\u036f]/g;
const HYPHEN = /([a-z])-(?=[a-z])/g;
const TOKEN = /[a-z0-9]+/g;

export function normalize(text) {
  return text.toLowerCase().normalize("NFKD").replace(MARKS, "").replace(HYPHEN, "$1");
}

export function sStem(word) {
  const n = word.length;
  if (n > 3 && word.endsWith("ies") && !word.endsWith("eies") && !word.endsWith("aies")) {
    return word.slice(0, -3) + "y";
  }
  if (n > 3 && word.endsWith("es") && !["aes", "ees", "oes"].some((s) => word.endsWith(s))) {
    return word.slice(0, -1);
  }
  if (n > 2 && word.endsWith("s") && !word.endsWith("us") && !word.endsWith("ss")) {
    return word.slice(0, -1);
  }
  return word;
}

export function tokenize(text) {
  const out = [];
  for (const token of normalize(text).match(TOKEN) ?? []) {
    if (STOPWORDS.has(token)) continue;
    if (token.length === 1 && !/[0-9]/.test(token)) continue;
    out.push(sStem(token));
  }
  return out;
}

export function fnv1a32(text) {
  const bytes = new TextEncoder().encode(text);
  let h = 0x811c9dc5;
  for (const b of bytes) {
    h ^= b;
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

// Fuse ranked lists of chunk numbers; ties keep first-appearance order.
export function rrfFuse(lists, k = 60, weights = null) {
  const scores = new Map();
  const firstSeen = new Map();
  let order = 0;
  lists.forEach((list, li) => {
    const w = weights ? weights[li] : 1.0;
    list.forEach((idx, r) => {
      scores.set(idx, (scores.get(idx) ?? 0) + w / (k + r + 1));
      if (!firstSeen.has(idx)) firstSeen.set(idx, order++);
    });
  });
  return [...scores.entries()]
    .sort((a, b) => b[1] - a[1] || firstSeen.get(a[0]) - firstSeen.get(b[0]))
    .map(([idx, score]) => ({ idx, score }));
}

function lowerBound(sorted, value) {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (sorted[mid] < value) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function rankIndices(scores, mask, depth, positiveOnly) {
  const idx = [];
  for (let i = 0; i < scores.length; i++) {
    if (!mask[i]) continue;
    if (positiveOnly && !(scores[i] > 0)) continue;
    idx.push(i);
  }
  idx.sort((a, b) => scores[b] - scores[a] || a - b); // stable: lower index first on ties
  return idx.slice(0, depth);
}

export class SearchIndex {
  constructor(meta, vectorsBuffer, postingsBuffer) {
    this.meta = meta;
    this.chunks = meta.chunks;
    this.meetings = meta.meetings;
    this.n = this.chunks.length;
    this.dim = meta.embedding.dim;
    this.vectors = new Int8Array(vectorsBuffer, 0, this.n * this.dim);
    this.scales = new Float32Array(vectorsBuffer, meta.files.vectors.scales_offset, this.n);
    const sec = meta.files.postings.sections;
    const view = (Type, name) => new Type(postingsBuffer, sec[name][0], sec[name][1]);
    this.hashes = view(Uint32Array, "hashes");
    this.offsets = view(Uint32Array, "offsets");
    this.docs = view(Uint16Array, "docs");
    this.doclen = view(Uint16Array, "doclen");
    this.tfs = view(Uint8Array, "tfs");
    this.dates = this.chunks.map((c) => this.meetings[c.m].date);
    const { k1, b, avgdl } = meta.bm25;
    this.k1 = k1;
    this.norm = new Float64Array(this.n);
    for (let i = 0; i < this.n; i++) {
      this.norm[i] = k1 * (1 - b + (b * this.doclen[i]) / Math.max(avgdl, 1e-9));
    }
  }

  static async load(baseUrl, fetchImpl = globalThis.fetch) {
    const get = async (path, kind) => {
      const res = await fetchImpl(new URL(path, baseUrl));
      if (!res.ok) throw new Error(`Could not load ${path} (HTTP ${res.status})`);
      return kind === "json" ? res.json() : res.arrayBuffer();
    };
    const meta = await get("index.json", "json");
    const [vectors, postings] = await Promise.all([
      get(meta.files.vectors.path, "bin"),
      get(meta.files.postings.path, "bin"),
    ]);
    return new SearchIndex(meta, vectors, postings);
  }

  mask(dateFrom, dateTo) {
    const m = new Uint8Array(this.n);
    for (let i = 0; i < this.n; i++) {
      const d = this.dates[i];
      m[i] = (!dateFrom || d >= dateFrom) && (!dateTo || d <= dateTo) ? 1 : 0;
    }
    return m;
  }

  bm25Scores(query) {
    const scores = new Float32Array(this.n);
    for (const term of tokenize(query)) {
      const h = fnv1a32(term);
      const j = lowerBound(this.hashes, h);
      if (j >= this.hashes.length || this.hashes[j] !== h) continue;
      const lo = this.offsets[j];
      const hi = this.offsets[j + 1];
      const df = hi - lo;
      const idf = Math.log(1 + (this.n - df + 0.5) / (df + 0.5));
      for (let p = lo; p < hi; p++) {
        const doc = this.docs[p];
        const tf = this.tfs[p];
        scores[doc] += (idf * tf * (this.k1 + 1)) / (tf + this.norm[doc]);
      }
    }
    return scores;
  }

  denseScores(queryVector) {
    const scores = new Float32Array(this.n);
    const { dim, vectors } = this;
    for (let i = 0; i < this.n; i++) {
      let dot = 0;
      const base = i * dim;
      for (let d = 0; d < dim; d++) dot += vectors[base + d] * queryVector[d];
      scores[i] = dot * this.scales[i];
    }
    return scores;
  }

  // mode: "hybrid" | "bm25" | "dense". Returns [{idx, score, bm25Rank, denseRank}].
  search(query, queryVector, { k = 10, mode = "hybrid", dateFrom = null, dateTo = null } = {}) {
    const mask = this.mask(dateFrom, dateTo);
    const depth = Math.max(k, this.meta.rrf.candidates);
    let bm25 = [];
    let dense = [];
    let bm25Scores = null;
    let denseScores = null;
    if (mode === "hybrid" || mode === "bm25") {
      bm25Scores = this.bm25Scores(query);
      bm25 = rankIndices(bm25Scores, mask, depth, true);
    }
    if (mode === "hybrid" || mode === "dense") {
      if (!queryVector) throw new Error("dense search needs a query vector");
      denseScores = this.denseScores(queryVector);
      dense = rankIndices(denseScores, mask, depth, false);
    }
    const bm25Rank = new Map(bm25.map((idx, r) => [idx, r + 1]));
    const denseRank = new Map(dense.map((idx, r) => [idx, r + 1]));
    let ranked;
    if (mode === "bm25") ranked = bm25.map((idx) => ({ idx, score: bm25Scores[idx] }));
    else if (mode === "dense") ranked = dense.map((idx) => ({ idx, score: denseScores[idx] }));
    else ranked = rrfFuse([bm25, dense], this.meta.rrf.k);
    return ranked.slice(0, k).map((r) => ({
      ...r,
      bm25Rank: bm25Rank.get(r.idx) ?? null,
      denseRank: denseRank.get(r.idx) ?? null,
    }));
  }
}
