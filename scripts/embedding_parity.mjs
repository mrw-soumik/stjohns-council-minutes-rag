// Embeds the parity sample texts with transformers.js (the library the browser demo uses)
// and writes the vectors to JSON for scripts/embedding_parity.py to compare.
//
// Usage: node scripts/embedding_parity.mjs <out.json> <revision> <instruction>
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { env, pipeline } from "@huggingface/transformers";

const [outPath, revision, instruction] = process.argv.slice(2);
if (!outPath || !revision) {
  console.error("usage: node scripts/embedding_parity.mjs <out.json> <revision> <instruction>");
  process.exit(2);
}
env.cacheDir = ".cache/transformers-js";

const samples = JSON.parse(await readFile("scripts/parity_texts.json", "utf8"));
const texts = [
  ...samples.queries.map((q) => `${instruction ?? ""}${q}`),
  ...samples.documents,
];

const result = { library: "@huggingface/transformers", revision, vectors: {} };
for (const dtype of ["fp32", "q8"]) {
  const extractor = await pipeline("feature-extraction", "Xenova/bge-small-en-v1.5", {
    revision,
    dtype,
  });
  const vectors = [];
  for (const text of texts) {
    // one text at a time, exactly as the browser embeds a query
    const output = await extractor(text, { pooling: "cls", normalize: true });
    vectors.push(Array.from(output.data));
  }
  result.vectors[dtype] = vectors;
  await extractor.dispose?.();
}
result.texts = texts;
await mkdir(dirname(outPath), { recursive: true });
await writeFile(outPath, JSON.stringify(result));
console.log(`wrote ${texts.length} x 2 vectors to ${outPath}`);
