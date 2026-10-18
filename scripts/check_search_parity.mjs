// Ranks the cases written by scripts/check_search_parity.py with docs/assets/search.js
// (the code the browser runs) and reports agreement with the Python rankings.
//
// Usage: node scripts/check_search_parity.mjs <cases.json> [docs/data]
import { readFile } from "node:fs/promises";
import { fnv1a32, SearchIndex, tokenize } from "../docs/assets/search.js";

const [casesPath, dataDir = "docs/data"] = process.argv.slice(2);
const cases = JSON.parse(await readFile(casesPath, "utf8"));
const read = (name) => readFile(`${dataDir}/${name}`);
const meta = JSON.parse(await read("index.json"));
const toBuffer = (buf) => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const index = new SearchIndex(
  meta,
  toBuffer(await read(meta.files.vectors.path)),
  toBuffer(await read(meta.files.postings.path)),
);

let failures = 0;
for (const c of cases.tokenizer) {
  const got = JSON.stringify(tokenize(c.text));
  if (got !== JSON.stringify(c.tokens)) {
    failures++;
    console.log(`tokenizer mismatch for ${JSON.stringify(c.text)}: ${got}`);
  }
}
for (const [text, value] of cases.fnv1a) {
  if (fnv1a32(text) !== value) {
    failures++;
    console.log(`fnv1a32 mismatch for ${JSON.stringify(text)}`);
  }
}

const summary = {};
for (const mode of ["bm25", "dense", "hybrid"]) {
  let same = 0;
  let top1 = 0;
  for (const q of cases.queries) {
    const vector = Float32Array.from(q.vector);
    const hits = index.search(q.question, vector, {
      k: 10,
      mode,
      dateFrom: q.date_from,
      dateTo: q.date_to,
    });
    const js = hits.map((h) => index.chunks[h.idx].id);
    const py = q.python[mode];
    if (JSON.stringify(js) === JSON.stringify(py)) same++;
    if (js[0] === py[0]) top1++;
  }
  summary[mode] = { identical_top10: same, same_top1: top1, n: cases.queries.length };
}
console.log(JSON.stringify(summary));
if (failures) {
  console.log(`${failures} tokenizer/hash mismatch(es)`);
  process.exit(1);
}
