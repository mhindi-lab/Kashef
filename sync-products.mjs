/**
 * Copyright (c) 2026 Monzer (mhindi-lab). All rights reserved.
 * Proprietary — see LICENSE. No permission to copy, reuse, or
 * redistribute any part of this file.
 * ---------------------------------------------------------------
 * Kashef — live store sync (server-side, no CORS issues)
 * ---------------------------------------------------------------
 * Fetches every store's public /products.json feed directly from
 * Node (not a browser tab), paginates through the FULL catalog,
 * maps each product into Kashef's schema, and writes
 * synced-products.json next to kashef_9.html. It also bakes the
 * same product list directly into kashef_9.html (between the
 * SYNCED_PRODUCTS markers) so the site works standalone — just
 * opening the file — with zero server needed. It also strips any
 * placeholder brand photo that isn't a real logo, falling back to
 * the honest colored-initials badge.
 *
 * MODEL-FREE IMAGES (site policy: never show a human model)
 * ---------------------------------------------------------------
 * Every product's display image (p.image) is guaranteed to contain
 * no person. For each product, in order:
 *
 *   1. GALLERY SCAN — check the product's photo gallery (up to
 *      MAX_GALLERY_CHECK images) with the local segmentation model
 *      and pick the first photo with no person in it (no face,
 *      hair, arms, or legs detected). Most products have a
 *      flat-lay/back/detail shot, so this covers the majority.
 *   2. STRICT CUTOUT — if every photo has a person, generate a
 *      ghost-mannequin garment cutout (person removed, white
 *      background) that must pass all quality checks.
 *   3. BEST-EFFORT CUTOUT — if the strict cutout fails quality
 *      checks, accept the best cutout we can make anyway. A rough
 *      garment cutout beats an empty tile, and it never shows a
 *      person.
 *
 * p.image is OVERWRITTEN with the chosen safe image, so the site
 * (which renders p.image directly) needs no changes. The original
 * first photo is preserved in p.originalImage for caching, and
 * p.cutoutImage is still set when a cutout was used. Verdicts are
 * cached in synced-products.json via p.imagesHash, so unchanged
 * products are never re-checked on later runs.
 *
 * USAGE
 *   node sync-products.mjs
 *
 * Requires Node 18+ (built-in fetch). To check your version:
 *   node -v
 *
 * ENV KNOBS
 *   CUTOUT_LIMIT        max products to run segmentation on this
 *                       run (blank = unlimited). Unprocessed
 *                       products keep their previous image until a
 *                       later run reaches them.
 *   CUTOUT_MIN_SOLIDITY strict-mode solidity cutoff (default 0.9)
 *
 * Edit the STORES list below to add/remove stores.
 * ---------------------------------------------------------------
 */

import { writeFile, readFile, mkdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { pipeline } from "@xenova/transformers";
import sharp from "sharp";

const STORES = [
  { brand: "Mzaco", url: "https://mzaco-eg.com" },
  { brand: "27", url: "https://twentysevenegy.myshopify.com" },
  { brand: "Locco", url: "https://loccoeg.com" },
  { brand: "Mavin", url: "https://www.mavin-wear.com" },
  { brand: "Be-Indie", url: "https://be-indie.com" },
  { brand: "TruCult", url: "https://trucult.co" },
  { brand: "Marsy", url: "https://marsy.shop" },
];

const OUTPUT_FILE = "synced-products.json";
const PAGE_SIZE = 250; // Shopify's max per page

const HTML_FILE = "kashef_9.html";
const HTML_MARKER_START = "const SYNCED_PRODUCTS = [";
const HTML_MARKER_END = "];\n/* END SYNCED_PRODUCTS */";

const BRANDS_WITHOUT_REAL_LOGOS = [];

const CATEGORIES = [
  { name: "Women's Fashion", icon: "dress", c1: "#D8B4A0", c2: "#BE8E76" },
  { name: "Men's Fashion", icon: "shirt", c1: "#3B434C", c2: "#20262D" },
  { name: "Shoes", icon: "shoe", c1: "#C99B6A", c2: "#A5794E" },
  { name: "Bags", icon: "bag", c1: "#8B9574", c2: "#69735A" },
  { name: "Accessories", icon: "ring", c1: "#C9A15D", c2: "#A57F41" },
  { name: "Makeup", icon: "makeup", c1: "#B98CA6", c2: "#95688A" },
  { name: "Hair Care", icon: "hair", c1: "#0E6B62", c2: "#0A4F49" },
  { name: "Skin Care", icon: "skin", c1: "#CBB794", c2: "#A9946F" },
  { name: "Perfumes", icon: "perfume", c1: "#5B4B6A", c2: "#3E3350" },
];

function normalize(word) {
  return (word || "").toLowerCase().replace(/[^a-z]/g, "");
}

const KEYWORD_MAP = {
  dress: "Women's Fashion", skirt: "Women's Fashion", blouse: "Women's Fashion",
  shirt: "Men's Fashion", tee: "Men's Fashion", tshirt: "Men's Fashion", hoodie: "Men's Fashion",
  sweatpants: "Men's Fashion", sweater: "Men's Fashion", jacket: "Men's Fashion", crewneck: "Men's Fashion",
  short: "Men's Fashion", shorts: "Men's Fashion", pants: "Men's Fashion", polo: "Men's Fashion",
  shoe: "Shoes", shoes: "Shoes", sneaker: "Shoes", sneakers: "Shoes",
  bag: "Bags", bags: "Bags", backpack: "Bags",
  ring: "Accessories", cap: "Accessories", hat: "Accessories", belt: "Accessories", jewelry: "Accessories",
  makeup: "Makeup", lipstick: "Makeup",
  hair: "Hair Care", skin: "Skin Care", skincare: "Skin Care",
  perfume: "Perfumes", fragrance: "Perfumes",
};

function guessCategoryFromText(text) {
  const words = (text || "").toLowerCase().split(/[\s,/]+/).map(normalize).filter(Boolean);
  for (const w of words) { if (KEYWORD_MAP[w]) { return CATEGORIES.find((c) => c.name === KEYWORD_MAP[w]); } }
  for (const w of words) { const match = CATEGORIES.find((c) => c.icon === w); if (match) return match; }
  return CATEGORIES.find((c) => c.name === "Accessories") || CATEGORIES[0];
}

let _embedder = null;
async function getEmbedder() {
  if (!_embedder) { _embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { quantized: true }); }
  return _embedder;
}

async function embedText(text) {
  const embedder = await getEmbedder();
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data).map((n) => Math.round(n * 10000) / 10000);
}

function cosineSim(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

const CATEGORY_LABELS = {
  "Women's Fashion": "women's clothing: dresses, skirts, blouses, gowns",
  "Men's Fashion": "men's clothing: shirts, suits, formal wear",
  "Shoes": "shoes, sneakers, footwear",
  "Bags": "bags, backpacks, totes, handbags",
  "Accessories": "accessories: jewelry, rings, watches, belts, caps",
  "Makeup": "makeup and cosmetics: lipstick, foundation",
  "Hair Care": "hair care: shampoo, conditioner, styling products",
  "Skin Care": "skin care: moisturizer, serum, cleanser",
  "Perfumes": "perfumes, fragrances, cologne",
};

let _categoryEmbeddings = null;
async function getCategoryEmbeddings() {
  if (!_categoryEmbeddings) {
    _categoryEmbeddings = {};
    for (const [name, label] of Object.entries(CATEGORY_LABELS)) { _categoryEmbeddings[name] = await embedText(label); }
  }
  return _categoryEmbeddings;
}

const KASHEF_WOMEN_ONLY_WORDS = ["skirt","skirts","crop","gown","gowns","blouse","blouses","romper","rompers","jumpsuit","jumpsuits","bra","bralette","legging","leggings","abaya","abayas","hijab","hijabs","skort","skorts"];
const KASHEF_DRESS_EXCEPTION_RE = /\bdress(es)?\b(?!\s*(shirt|shirts|pant|pants|shoe|shoes|code|sock|socks|watch|watches))/;
const KASHEF_CLOTHING_WORDS = ["shirt","shirts","tee","tees","tshirt","hoodie","hoodies","sweatpants","sweatshirt","sweatshirts","sweater","sweaters","jacket","jackets","crewneck","crewnecks","short","shorts","pant","pants","jean","jeans","trouser","trousers","cargo","polo","polos","tracksuit","tracksuits","jogger","joggers","top","tops","blazer","blazers","coat","coats","vest","vests","cardigan","cardigans","sock","socks","henley"];
const KASHEF_MEN_ONLY_RE = /\b(compression|muscle\s*(fit|tee|shirt)|cut\s*shirt)\b/;
const KASHEF_SHOES_WORDS = ["shoe","shoes","sneaker","sneakers","boot","boots","sandal","sandals","slipper","slippers","slide","slides","footwear","heel","heels","flat","flats"];
const KASHEF_BAGS_WORDS = ["bag","bags","backpack","backpacks","tote","totes","handbag","handbags","purse","purses","wallet","wallets","clutch","clutches"];
const KASHEF_ACCESSORIES_WORDS = ["jewelry","jewellery","ring","rings","necklace","necklaces","bracelet","bracelets","earring","earrings","watch","watches","belt","belts","cap","caps","beanie","beanies","sunglasses","hat","hats"];
const KASHEF_MAKEUP_WORDS = ["makeup","lipstick","lipsticks","foundation","mascara","eyeliner","blush","concealer","eyeshadow","lipgloss"];
const KASHEF_HAIRCARE_WORDS = ["shampoo","conditioner","hairspray"];
const KASHEF_SKINCARE_WORDS = ["moisturizer","cleanser","sunscreen","skincare","toner"];
const KASHEF_PERFUME_WORDS = ["perfume","perfumes","cologne","fragrance","fragrances","parfum","edp","edt"];
const KASHEF_WOMEN_SIGNAL = ["women","womens","woman","ladies","lady","girls","girl","female"];
const KASHEF_MEN_SIGNAL = ["men","mens","man","guys","guy","boys","boy","male"];

async function classifyWithAI(text) {
  const clean = (text || "").trim();
  if (!clean) { return { cat: CATEGORIES.find((c) => c.name === "Accessories") || CATEGORIES[0], emb: null, unisex: false }; }
  const emb = await embedText(clean);
  const lower = clean.toLowerCase().replace(/'s\b/g, "s");
  const words = lower.split(/[^a-z]+/).filter(Boolean);
  const wordSet = new Set(words);
  const has = (arr) => arr.some((w) => wordSet.has(w));
  const hasWomenSignal = has(KASHEF_WOMEN_SIGNAL);
  const hasMenSignal = has(KASHEF_MEN_SIGNAL);

  if (has(KASHEF_WOMEN_ONLY_WORDS) || KASHEF_DRESS_EXCEPTION_RE.test(lower)) {
    const women = CATEGORIES.find((c) => c.name === "Women's Fashion");
    return { cat: women || CATEGORIES[0], emb, unisex: false };
  }
  if (KASHEF_MEN_ONLY_RE.test(lower)) {
    const men = CATEGORIES.find((c) => c.name === "Men's Fashion");
    return { cat: men || CATEGORIES[0], emb, unisex: false };
  }
  if (has(KASHEF_CLOTHING_WORDS)) {
    if (hasWomenSignal) { const women = CATEGORIES.find((c) => c.name === "Women's Fashion"); return { cat: women || CATEGORIES[0], emb, unisex: false }; }
    if (hasMenSignal) { const men = CATEGORIES.find((c) => c.name === "Men's Fashion"); return { cat: men || CATEGORIES[0], emb, unisex: false }; }
    const men = CATEGORIES.find((c) => c.name === "Men's Fashion");
    return { cat: men || CATEGORIES[0], emb, unisex: true };
  }
  const directMatches = [
    ["Shoes", KASHEF_SHOES_WORDS], ["Bags", KASHEF_BAGS_WORDS], ["Accessories", KASHEF_ACCESSORIES_WORDS],
    ["Makeup", KASHEF_MAKEUP_WORDS], ["Hair Care", KASHEF_HAIRCARE_WORDS], ["Skin Care", KASHEF_SKINCARE_WORDS],
    ["Perfumes", KASHEF_PERFUME_WORDS],
  ];
  for (const [name, list] of directMatches) {
    if (has(list)) { const cat = CATEGORIES.find((c) => c.name === name); return { cat: cat || CATEGORIES[0], emb, unisex: false }; }
  }

  const catEmbeddings = await getCategoryEmbeddings();
  let best = null;
  let bestSim = -1;
  for (const [name, catEmb] of Object.entries(catEmbeddings)) {
    const sim = cosineSim(emb, catEmb);
    if (sim > bestSim) { bestSim = sim; best = name; }
  }
  const cat = CATEGORIES.find((c) => c.name === best) || CATEGORIES[0];
  return { cat, emb, unisex: false };
}

function stripHtml(html) {
  return (html || "").replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
}

async function fetchAllProducts(storeUrl) {
  const all = [];
  let page = 1;
  while (true) {
    const res = await fetch(
      `${storeUrl}/products.json?limit=${PAGE_SIZE}&page=${page}`,
      { headers: { "User-Agent": "Mozilla/5.0 (Kashef sync bot)" } }
    );
    if (!res.ok) { if (page === 1) { throw new Error(`${storeUrl} responded with ${res.status}`); } break; }
    const data = await res.json();
    const products = data.products || [];
    if (!products.length) break;
    all.push(...products);
    if (products.length < PAGE_SIZE) break;
    page++;
    if (page > 20) break;
  }
  return all;
}

// ---------------------------------------------------------------
// MODEL-FREE IMAGE PIPELINE
// The site policy is that no product photo may show a person. The
// same local segmentation model (Xenova/segformer_b0_clothes) is
// used for two jobs:
//   • detectPerson(url) — does this photo contain a person?
//     (checks coverage of Face / Hair / arms / legs labels)
//   • generateCutout(url) — segment just the product out of a
//     model photo onto a white background ("ghost mannequin").
// Runs entirely locally/free via @xenova/transformers — no paid API.
//
// p.image IS overwritten with the chosen safe image (this is the
// point: the site renders p.image directly, so overwriting it is
// what guarantees no person ever appears). The original photo URL
// is kept in p.originalImage, and p.imagesHash caches the decision
// so unchanged products are never re-processed.
// ---------------------------------------------------------------

const CUTOUT_DIR = "cutouts";
const CUTOUT_MIN_COVERAGE = 0.03; // strict: mask must cover at least 3% of the image
const CUTOUT_MAX_COVERAGE = 0.55; // ...and at most 55% (a bigger mask is probably wrong)
const CUTOUT_MIN_FILL_RATIO = 0.68; // painted px / bounding-box area — catches big holes
const CUTOUT_RELAXED_MIN_COVERAGE = 0.005; // best-effort mode: only reject near-empty masks
const PROCESS_LIMIT = process.env.CUTOUT_LIMIT ? parseInt(process.env.CUTOUT_LIMIT, 10) : Infinity;
const MAX_GALLERY_CHECK = 5; // how many gallery photos to scan for a person-free one
const MAX_CUTOUT_SOURCES = 2; // how many photos to try strict cutouts on

// Person indicators among segformer_b0_clothes labels. If these
// cover more than PERSON_MIN_COVERAGE of the image, a person (or
// realistic mannequin — treated the same, to be safe) is present.
const PERSON_LABELS = new Set(["Face", "Hair", "Left-arm", "Right-arm", "Left-leg", "Right-leg"]);
const PERSON_MIN_COVERAGE = 0.005; // 0.5% of the image

// Which segmentation labels count as "the product" per category,
// so cutouts work for shoes/bags/accessories too, not just garments.
const GARMENT_LABELS = ["Upper-clothes", "Dress", "Pants", "Skirt", "Scarf", "Hat"];
const CUTOUT_LABELS_BY_CAT = {
  "Men's Fashion": GARMENT_LABELS,
  "Women's Fashion": GARMENT_LABELS,
  "Shoes": ["Left-shoe", "Right-shoe"],
  "Bags": ["Bag"],
  "Accessories": ["Hat", "Scarf", "Sunglasses", "Belt", "Bag"],
};
const CUTOUT_LABELS_DEFAULT = [...new Set([...GARMENT_LABELS, "Left-shoe", "Right-shoe", "Bag", "Sunglasses", "Belt"])];

let _segmenter = null;
async function getSegmenter() {
  if (!_segmenter) {
    _segmenter = await pipeline("image-segmentation", "Xenova/segformer_b0_clothes", { quantized: true });
  }
  return _segmenter;
}

async function segmentImage(imageUrl, timeoutMs = 20000) {
  const segmenter = await getSegmenter();
  const timeout = new Promise((_, reject) => setTimeout(() => reject(new Error("segmentation timeout")), timeoutMs));
  return Promise.race([segmenter(imageUrl), timeout]);
}

function maskCoverage(mask) {
  const mdata = mask.data;
  let count = 0;
  for (let i = 0; i < mdata.length; i++) if (mdata[i] > 128) count++;
  return count / mdata.length;
}

// Returns true (person present), false (no person), or null (couldn't
// check). Callers must treat null as UNSAFE — never show an unverified
// photo.
async function detectPerson(imageUrl) {
  if (!imageUrl) return null;
  try {
    const output = await segmentImage(imageUrl);
    let personCoverage = 0;
    for (const seg of output) {
      if (PERSON_LABELS.has(seg.label)) personCoverage += maskCoverage(seg.mask);
    }
    return personCoverage > PERSON_MIN_COVERAGE;
  } catch (err) {
    console.warn("Person check failed for", imageUrl, err && err.message);
    return null;
  }
}

function imageSlug(imageUrl) {
  return createHash("md5").update(imageUrl).digest("hex").slice(0, 16);
}

function imagesHash(urls) {
  return createHash("md5").update((urls || []).join("|")).digest("hex").slice(0, 16);
}

// ---- Mask cleanup helpers ----
// Real photos reviewed after the first two live test runs showed the raw
// segmentation mask leaves gaps in two different ways: (1) small local
// notches (a neckline gap where the model missed a strip of collar), which
// a modest morphological closing fixes fine, and (2) longer structural gaps
// (a strip of visible torso between a loose sleeve and the body, running
// most of a side seam) that need a much bigger closing radius to bridge.
// We (a) use a generous closing radius, then (b) measure how "solid"
// (convex) the resulting shape is; in STRICT mode anything too irregular
// is rejected, while BEST-EFFORT mode ships it anyway (site policy: a
// rough cutout beats showing a person or an empty tile).

// Keep every connected component at least `keepRatio` the size of the
// largest one (so a pair of shoes keeps both shoes), drop the noise.
function mainComponents(bin, w, h, keepRatio = 0.25) {
  const labels = new Int32Array(w * h).fill(-1);
  const sizes = [];
  let label = 0;
  const stack = [];
  for (let start = 0; start < w * h; start++) {
    if (!bin[start] || labels[start] !== -1) continue;
    let size = 0;
    stack.length = 0;
    stack.push(start);
    labels[start] = label;
    while (stack.length) {
      const idx = stack.pop();
      size++;
      const x = idx % w;
      const y = (idx / w) | 0;
      if (x > 0 && bin[idx - 1] && labels[idx - 1] === -1) { labels[idx - 1] = label; stack.push(idx - 1); }
      if (x < w - 1 && bin[idx + 1] && labels[idx + 1] === -1) { labels[idx + 1] = label; stack.push(idx + 1); }
      if (y > 0 && bin[idx - w] && labels[idx - w] === -1) { labels[idx - w] = label; stack.push(idx - w); }
      if (y < h - 1 && bin[idx + w] && labels[idx + w] === -1) { labels[idx + w] = label; stack.push(idx + w); }
    }
    sizes.push(size);
    label++;
  }
  if (!sizes.length) return new Uint8Array(w * h);
  const maxSize = Math.max(...sizes);
  const keep = new Set();
  sizes.forEach((s, l) => { if (s >= maxSize * keepRatio) keep.add(l); });
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = labels[i] !== -1 && keep.has(labels[i]) ? 1 : 0;
  return out;
}

function dilate(src, w, h, r) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 0;
      for (let dy = -r; dy <= r && !v; dy++) {
        for (let dx = -r; dx <= r && !v; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < w && ny >= 0 && ny < h && src[ny * w + nx]) v = 1;
        }
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

function erode(src, w, h, r) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 1;
      for (let dy = -r; dy <= r && v; dy++) {
        for (let dx = -r; dx <= r && v; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= w || ny < 0 || ny >= h || !src[ny * w + nx]) v = 0;
        }
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

// Fill gaps (neckline notches, and longer side-seam strips) without eating
// real edges too much.
function closeSmallGaps(bin, w, h, r) {
  return erode(dilate(bin, w, h, r), w, h, r);
}

const CUTOUT_CLOSE_RADIUS = 8; // fills gaps/notches, including longer structural strips
const CUTOUT_SHAVE_RADIUS = 2; // trims a thin ring off the outer edge to remove
// jagged/torn-looking boundaries and background bleed

function cleanupMask(bin, mw, mh) {
  const main = mainComponents(bin, mw, mh);
  const closed = closeSmallGaps(main, mw, mh, CUTOUT_CLOSE_RADIUS);
  const shaved = erode(closed, mw, mh, CUTOUT_SHAVE_RADIUS);
  return shaved;
}

// Convex-hull area of a mask, computed on a subsampled point grid (fast and
// close enough — we only need this for a solidity ratio, not exact geometry).
function convexHullArea(mask, w, h) {
  const pts = [];
  const step = 2;
  for (let y = 0; y < h; y += step) {
    for (let x = 0; x < w; x += step) {
      if (mask[y * w + x]) pts.push([x, y]);
    }
  }
  if (pts.length < 3) return 0;
  pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  const hull = lower.slice(0, -1).concat(upper.slice(0, -1));
  let area = 0;
  for (let i = 0; i < hull.length; i++) {
    const [x1, y1] = hull[i];
    const [x2, y2] = hull[(i + 1) % hull.length];
    area += x1 * y2 - x2 * y1;
  }
  return Math.abs(area) / 2;
}

// Strict-mode solidity cutoff (mask area / hull area). Configurable via env
// for easy tuning without a code change.
const CUTOUT_MIN_SOLIDITY = process.env.CUTOUT_MIN_SOLIDITY ? parseFloat(process.env.CUTOUT_MIN_SOLIDITY) : 0.9;

// Cache of previous cutouts (original photo URL -> cutout path) so we never
// regenerate one we already have.
async function loadCutoutCache() {
  try {
    const raw = await readFile(OUTPUT_FILE, "utf-8");
    const prev = JSON.parse(raw);
    const cache = new Map();
    for (const p of prev) {
      const key = p.originalImage || p.image;
      if (key && p.cutoutImage) cache.set(key, p.cutoutImage);
    }
    return cache;
  } catch {
    return new Map();
  }
}

// Cache of previous image decisions (product link -> chosen safe image),
// valid as long as the product's photo gallery hasn't changed (imagesHash).
async function loadSelectionCache() {
  try {
    const raw = await readFile(OUTPUT_FILE, "utf-8");
    const prev = JSON.parse(raw);
    const cache = new Map();
    for (const p of prev) {
      if (p.link && p.imagesHash && p.imageSource) {
        cache.set(p.link, {
          imagesHash: p.imagesHash,
          image: p.image,
          cutoutImage: p.cutoutImage || null,
          imageSource: p.imageSource,
        });
      }
    }
    return cache;
  } catch {
    return new Map();
  }
}

// Generate a product cutout from a photo. In strict mode all quality checks
// apply and null is returned on failure. In relaxed (best-effort) mode we
// ship the best mask we can find — the only hard requirement is that a
// product mask exists at all and isn't practically empty.
async function generateCutout(imageUrl, category, { relaxed = false } = {}) {
  if (!imageUrl) return null;
  try {
    const output = await segmentImage(imageUrl);
    const candidateLabels = CUTOUT_LABELS_BY_CAT[category] || CUTOUT_LABELS_DEFAULT;

    // Union all product labels into one mask (so e.g. a pair of shoes, or
    // a top + skirt outfit, stays complete).
    let mw = 0, mh = 0;
    let union = null;
    for (const label of candidateLabels) {
      const seg = output.find((o) => o.label === label);
      if (!seg) continue;
      const mdata = seg.mask.data;
      if (!union) { mw = seg.mask.width; mh = seg.mask.height; union = new Uint8Array(mw * mh); }
      for (let i = 0; i < mdata.length; i++) if (mdata[i] > 128) union[i] = 1;
    }
    if (!union) return null;

    let rawCoverage = 0;
    for (let i = 0; i < union.length; i++) rawCoverage += union[i];
    rawCoverage /= union.length;
    const minCoverage = relaxed ? CUTOUT_RELAXED_MIN_COVERAGE : CUTOUT_MIN_COVERAGE;
    if (rawCoverage < minCoverage) return null;
    if (!relaxed && rawCoverage > CUTOUT_MAX_COVERAGE) return null;

    const cleanedBin = cleanupMask(union, mw, mh);

    let count = 0;
    let minX = mw, maxX = 0, minY = mh, maxY = 0;
    for (let y = 0; y < mh; y++) {
      for (let x = 0; x < mw; x++) {
        if (cleanedBin[y * mw + x]) {
          count++;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    const coverage = count / (mw * mh);
    if (coverage < minCoverage) return null;
    if (!relaxed) {
      if (coverage > CUTOUT_MAX_COVERAGE) return null;
      const bboxArea = Math.max(1, (maxX - minX + 1) * (maxY - minY + 1));
      const fillRatio = count / bboxArea;
      if (fillRatio < CUTOUT_MIN_FILL_RATIO) return null;
      const hullArea = convexHullArea(cleanedBin, mw, mh);
      const solidity = hullArea > 0 ? count / hullArea : 0;
      console.log(`  Cutout candidate solidity=${solidity.toFixed(3)} fillRatio=${fillRatio.toFixed(3)} coverage=${coverage.toFixed(3)} for ${imageUrl}`);
      if (solidity < CUTOUT_MIN_SOLIDITY) {
        console.log(`  Rejected (solidity ${solidity.toFixed(3)} < ${CUTOUT_MIN_SOLIDITY}) — will retry in best-effort mode`);
        return null;
      }
    }

    const res = await fetch(imageUrl, { headers: { "User-Agent": "Mozilla/5.0 (Kashef sync bot)" } });
    if (!res.ok) return null;
    const buf = Buffer.from(await res.arrayBuffer());
    const { data: srcRgba } = await sharp(buf)
      .resize(mw, mh, { fit: "fill" })
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const out = Buffer.alloc(mw * mh * 4);
    for (let i = 0; i < mw * mh; i++) {
      const si = i * 4;
      if (cleanedBin[i]) {
        out[si] = srcRgba[si];
        out[si + 1] = srcRgba[si + 1];
        out[si + 2] = srcRgba[si + 2];
        out[si + 3] = 255;
      } else {
        out[si] = 255;
        out[si + 1] = 255;
        out[si + 2] = 255;
        out[si + 3] = 255;
      }
    }

    await mkdir(CUTOUT_DIR, { recursive: true });
    const outPath = `${CUTOUT_DIR}/${imageSlug(imageUrl)}.jpg`;
    await sharp(out, { raw: { width: mw, height: mh, channels: 4 } })
      .jpeg({ quality: 88 })
      .toFile(outPath);

    return outPath;
  } catch (err) {
    console.warn("Cutout generation failed for", imageUrl, err && err.message);
    return null;
  }
}

// Decide the model-free display image for one product.
// Returns { image, cutoutImage, imageSource } where imageSource is one of:
//   "gallery"        — a person-free photo straight from the store
//   "cutout"         — strict-quality ghost-mannequin cutout
//   "cutout-rough"   — best-effort cutout (quality checks relaxed)
//   "none"           — nothing usable found (site shows the category icon)
async function chooseSafeImage(gallery, category, cutoutCache) {
  // 1. GALLERY SCAN — first photo with no person wins.
  for (const url of gallery.slice(0, MAX_GALLERY_CHECK)) {
    const hasPerson = await detectPerson(url);
    if (hasPerson === false) {
      return { image: url, cutoutImage: null, imageSource: "gallery" };
    }
    // true or null (couldn't verify) → keep looking; never show unverified.
  }

  // 2. STRICT CUTOUT — reuse a cached one if we have it.
  for (const url of gallery.slice(0, MAX_CUTOUT_SOURCES)) {
    if (cutoutCache.has(url)) {
      const path = cutoutCache.get(url);
      return { image: path, cutoutImage: path, imageSource: "cutout" };
    }
  }
  for (const url of gallery.slice(0, MAX_CUTOUT_SOURCES)) {
    const path = await generateCutout(url, category, { relaxed: false });
    if (path) return { image: path, cutoutImage: path, imageSource: "cutout" };
  }

  // 3. BEST-EFFORT CUTOUT — a rough cutout beats an empty tile, and it
  //    never shows a person.
  for (const url of gallery.slice(0, MAX_CUTOUT_SOURCES)) {
    const path = await generateCutout(url, category, { relaxed: true });
    if (path) return { image: path, cutoutImage: path, imageSource: "cutout-rough" };
  }

  // Nothing usable — fall back to the category icon tile rather than ever
  // showing a person. (Expected to be extremely rare: it means every photo
  // had a person AND segmentation found no product mask in any of them.)
  return { image: "", cutoutImage: null, imageSource: "none" };
}

async function mapProduct(sp, storeUrl, brandName) {
  const { cat, emb, unisex } = await classifyWithAI(
    `${sp.product_type || ""} ${(sp.tags || []).join ? (sp.tags || []).join(" ") : sp.tags || ""} ${sp.title || ""}`
  );
  const variants = sp.variants || [];
  const firstVariant = variants[0] || {};
  const inStock = variants.some((v) => v.available !== false);
  const onSale = variants.some(
    (v) => v.compare_at_price && parseFloat(v.compare_at_price) > parseFloat(v.price)
  );
  const sizeOption = (sp.options || []).find((o) => /size/i.test(o.name));
  const colorOption = (sp.options || []).find((o) => /colou?r/i.test(o.name));
  const gallery = (sp.images || []).map((im) => im && im.src).filter(Boolean);

  const product = {
    name: sp.title,
    brand: brandName || sp.vendor || "Unknown Brand",
    cat: cat.name,
    emb: emb,
    unisex: unisex,
    price: `${Math.round(parseFloat(firstVariant.price || 0))} EGP`,
    icon: cat.icon,
    c1: cat.c1,
    c2: cat.c2,
    platform: "website",
    image: gallery[0] || "",
    originalImage: gallery[0] || "",
    imagesHash: imagesHash(gallery),
    imageSource: "unprocessed",
    link: `${storeUrl}/products/${sp.handle}`,
    desc: stripHtml(sp.body_html).slice(0, 140),
    sizes: sizeOption ? [...new Set(sizeOption.values)] : ["One Size"],
    color: colorOption ? colorOption.values[0] : "",
    onSale,
    inStock,
    source: { type: "shopify", url: storeUrl },
    lastSynced: new Date().toLocaleString("en-GB", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }),
  };
  return { product, gallery };
}

function stripPlaceholderLogos(html) {
  let updated = html;
  let count = 0;
  for (const name of BRANDS_WITHOUT_REAL_LOGOS) {
    const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`(\\{name:"${escaped}"[^}]*?), photo:"[^"]*"(\\})`);
    if (re.test(updated)) { updated = updated.replace(re, "$1$2"); count++; }
  }
  return { html: updated, count };
}

async function main() {
  const allMapped = [];
  const summary = [];
  const cutoutCache = await loadCutoutCache();
  const selectionCache = await loadSelectionCache();
  let budget = PROCESS_LIMIT; // products that may run segmentation this run
  const stats = { gallery: 0, cutout: 0, cutoutRough: 0, none: 0, cached: 0, deferred: 0 };

  for (const store of STORES) {
    const storeUrl = store.url.replace(/\/$/, "");
    try {
      const products = await fetchAllProducts(storeUrl);
      const mappedPairs = await Promise.all(products.map((sp) => mapProduct(sp, storeUrl, store.brand)));

      let _progress = 0;
      for (const { product: p, gallery } of mappedPairs) {
        _progress++;

        if (!gallery.length) {
          p.image = "";
          p.imageSource = "none";
          stats.none++;
        } else {
          const cached = selectionCache.get(p.link);
          if (cached && cached.imagesHash === p.imagesHash && cached.imageSource !== "unprocessed" && cached.imageSource !== "none") {
            // Same photos as last run — reuse the previous safe choice.
            p.image = cached.image;
            if (cached.cutoutImage) p.cutoutImage = cached.cutoutImage;
            p.imageSource = cached.imageSource;
            stats.cached++;
          } else if (budget > 0) {
            budget--;
            const choice = await chooseSafeImage(gallery, p.cat, cutoutCache);
            p.image = choice.image;
            if (choice.cutoutImage) p.cutoutImage = choice.cutoutImage;
            p.imageSource = choice.imageSource;
            if (choice.imageSource === "gallery") stats.gallery++;
            else if (choice.imageSource === "cutout") stats.cutout++;
            else if (choice.imageSource === "cutout-rough") stats.cutoutRough++;
            else stats.none++;
          } else {
            // Out of budget this run. NEVER show an unverified photo:
            // reuse any previous safe image, else show the icon tile until
            // a later run (they run every 6 hours) processes this product.
            if (cached && cached.image && cached.imageSource !== "unprocessed" && cached.imageSource !== "none") {
              p.image = cached.image;
              if (cached.cutoutImage) p.cutoutImage = cached.cutoutImage;
              p.imageSource = cached.imageSource;
            } else {
              p.image = "";
              p.imageSource = "unprocessed";
            }
            stats.deferred++;
          }
        }

        if (_progress % 25 === 0 || _progress === mappedPairs.length) {
          console.log(`  Safe images: ${_progress}/${mappedPairs.length} for ${store.brand}`);
        }
      }

      const mapped = mappedPairs.map((mp) => mp.product);
      allMapped.push(...mapped);
      const inStockCount = mapped.filter((p) => p.inStock).length;
      summary.push(`${store.brand}: ${mapped.length} products (${inStockCount} in stock, ${mapped.length - inStockCount} sold out)`);
    } catch (err) {
      summary.push(`${store.brand}: FAILED — ${err.message}`);
    }
  }

  const seen = new Set();
  const deduped = allMapped.filter((p) => {
    if (seen.has(p.link)) return false;
    seen.add(p.link);
    return true;
  });

  await writeFile(OUTPUT_FILE, JSON.stringify(deduped, null, 2));

  let bakedIntoHtml = false;
  let logosFixed = 0;
  try {
    let html = await readFile(HTML_FILE, "utf8");

    const stripped = stripPlaceholderLogos(html);
    html = stripped.html;
    logosFixed = stripped.count;

    const startIdx = html.indexOf(HTML_MARKER_START);
    const endIdx = startIdx === -1 ? -1 : html.indexOf(HTML_MARKER_END, startIdx);
    if (startIdx !== -1 && endIdx !== -1) {
      const before = html.slice(0, startIdx + HTML_MARKER_START.length);
      const after = html.slice(endIdx);
      const arrayBody = JSON.stringify(deduped, null, 2).replace(/^\[/, "").replace(/\]$/, "");
      const newHtml = `${before}\n${arrayBody.trim()}\n${after}`;
      await writeFile(HTML_FILE, newHtml);
      bakedIntoHtml = true;
    } else {
      console.warn(`Warning: couldn't find the SYNCED_PRODUCTS markers in ${HTML_FILE} — skipped baking data into the HTML. synced-products.json was still written normally.`);
    }
  } catch (err) {
    console.warn(`Warning: couldn't update ${HTML_FILE} (${err.message}). synced-products.json was still written normally.`);
  }

  console.log("\n--- Kashef sync summary ---");
  summary.forEach((line) => console.log(line));
  console.log(`\nWrote ${deduped.length} total products to ${OUTPUT_FILE}`);
  console.log("Model-free images —");
  console.log(`  person-free gallery photo: ${stats.gallery}`);
  console.log(`  ghost-mannequin cutout:    ${stats.cutout}`);
  console.log(`  best-effort cutout:        ${stats.cutoutRough}`);
  console.log(`  reused from cache:         ${stats.cached}`);
  console.log(`  deferred to a later run:   ${stats.deferred}`);
  console.log(`  no usable image (icon):    ${stats.none}`);
  if (bakedIntoHtml) {
    console.log(`Also baked ${deduped.length} products directly into ${HTML_FILE} — it now works standalone, no server needed.`);
  }
  if (logosFixed) {
    console.log(`Removed ${logosFixed} placeholder brand photo(s) that weren't real logos — those brands now show initials instead.`);
  }
}

main().catch((err) => {
  console.error("Sync failed:", err);
  process.exit(1);
});
