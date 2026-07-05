/**
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
 * USAGE
 * node sync-products.mjs
 *
 * Requires Node 18+ (built-in fetch). To check your version:
 * node -v
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
// Garment cutout ("ghost mannequin" style): segments just the
// clothing item out of a model photo and composites it on a plain
// white background, so the product photo shows the garment itself
// rather than the person wearing it. Runs entirely locally/free via
// @xenova/transformers (Xenova/segformer_b0_clothes) — no paid API.
//
// SAFETY: this NEVER overwrites p.image. It only sets a separate
// p.cutoutImage field when the result passes quality checks. If
// generation fails, times out, or the mask looks unreliable (too
// small, too large, or full of gaps from something like a hand
// resting on the garment), p.cutoutImage is simply left unset and
// the site falls back to the original photo — see productCard()
// and openProductModal() in kashef_9.html for the fallback logic.
// ---------------------------------------------------------------

const CUTOUT_DIR = "cutouts";
const CUTOUT_CANDIDATE_LABELS = ["Upper-clothes", "Dress", "Pants", "Skirt"];
const CUTOUT_MIN_COVERAGE = 0.03; // mask must cover at least 3% of the image
const CUTOUT_MAX_COVERAGE = 0.55; // ...and at most 55% (a bigger mask is probably wrong)
const CUTOUT_MIN_FILL_RATIO = 0.68; // painted px / bounding-box area — catches big holes
                                     // left by things like a hand resting on the garment
const CUTOUT_LIMIT = process.env.CUTOUT_LIMIT ? parseInt(process.env.CUTOUT_LIMIT, 10) : Infinity;
const CUTOUT_ELIGIBLE_CATS = new Set(["Men's Fashion", "Women's Fashion"]);

let _segmenter = null;
async function getSegmenter() {
  if (!_segmenter) {
    _segmenter = await pipeline("image-segmentation", "Xenova/segformer_b0_clothes", { quantized: true });
  }
  return _segmenter;
}

function imageSlug(imageUrl) {
  return createHash("md5").update(imageUrl).digest("hex").slice(0, 16);
}

// ---- Mask cleanup helpers ----
// Real photos reviewed after the first live run showed two recurring
// problems the raw segmentation mask doesn't catch on its own: (1) a
// jagged, torn-paper-looking edge (worst on shorts/pants), and (2) small
// gaps at the neckline where the mask misses a strip of real garment.
// These three passes fix both, using nothing but the mask itself — no
// extra model, still zero cost.
function largestComponent(bin, w, h) {
  const labels = new Int32Array(w * h).fill(-1);
  let bestLabel = -1;
  let bestSize = 0;
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
    if (size > bestSize) { bestSize = size; bestLabel = label; }
    label++;
  }
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = labels[i] === bestLabel ? 1 : 0;
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

// Fill small gaps (like a neckline notch) without eating real edges.
function closeSmallGaps(bin, w, h, r) {
  return erode(dilate(bin, w, h, r), w, h, r);
}

const CUTOUT_CLOSE_RADIUS = 8; // fills small internal gaps/notches
const CUTOUT_SHAVE_RADIUS = 2; // trims a thin ring off the outer edge to remove
                                // jagged/torn-looking boundaries and background bleed

function cleanupMask(mdata, mw, mh) {
  let rawBin = new Uint8Array(mw * mh);
  for (let i = 0; i < mdata.length; i++) rawBin[i] = mdata[i] > 128 ? 1 : 0;
  const largest = largestComponent(rawBin, mw, mh);
  const closed = closeSmallGaps(largest, mw, mh, CUTOUT_CLOSE_RADIUS);
  const shaved = erode(closed, mw, mh, CUTOUT_SHAVE_RADIUS);
  return shaved;
}

async function loadCutoutCache() {
  try {
    const raw = await readFile(OUTPUT_FILE, "utf-8");
    const prev = JSON.parse(raw);
    const cache = new Map();
    for (const p of prev) {
      if (p.image && p.cutoutImage) cache.set(p.image, p.cutoutImage);
    }
    return cache;
  } catch {
    return new Map();
  }
}

async function generateCutout(imageUrl) {
  if (!imageUrl) return null;
  try {
    const segmenter = await getSegmenter();
    const timeout = new Promise((_, reject) => setTimeout(() => reject(new Error("cutout timeout")), 20000));
    const output = await Promise.race([segmenter(imageUrl), timeout]);

    let best = null;
    for (const label of CUTOUT_CANDIDATE_LABELS) {
      const candidate = output.find((o) => o.label === label);
      if (!candidate) continue;
      const mdata = candidate.mask.data;
      let count = 0;
      for (let i = 0; i < mdata.length; i++) if (mdata[i] > 128) count++;
      const coverage = count / mdata.length;
      if (coverage < CUTOUT_MIN_COVERAGE || coverage > CUTOUT_MAX_COVERAGE) continue;
      if (!best || coverage > best.coverage) best = { label, mask: candidate.mask, coverage, count };
    }
    if (!best) return null;

    const { mask } = best;
    const mw = mask.width, mh = mask.height;
    const cleanedBin = cleanupMask(mask.data, mw, mh);

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
    if (coverage < CUTOUT_MIN_COVERAGE || coverage > CUTOUT_MAX_COVERAGE) return null;
    const bboxArea = Math.max(1, (maxX - minX + 1) * (maxY - minY + 1));
    const fillRatio = count / bboxArea;
    if (fillRatio < CUTOUT_MIN_FILL_RATIO) return null;

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

  return {
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
    image: (sp.images && sp.images[0] && sp.images[0].src) || "",
    link: `${storeUrl}/products/${sp.handle}`,
    desc: stripHtml(sp.body_html).slice(0, 140),
    sizes: sizeOption ? [...new Set(sizeOption.values)] : ["One Size"],
    color: colorOption ? colorOption.values[0] : "",
    onSale,
    inStock,
    source: { type: "shopify", url: storeUrl },
    lastSynced: new Date().toLocaleString("en-GB", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }),
  };
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
  let cutoutBudget = CUTOUT_LIMIT;
  let cutoutsGenerated = 0;
  let cutoutsSkipped = 0;
  let cutoutsReused = 0;

  for (const store of STORES) {
    const storeUrl = store.url.replace(/\/$/, "");
    try {
      const products = await fetchAllProducts(storeUrl);
      const mapped = await Promise.all(products.map((sp) => mapProduct(sp, storeUrl, store.brand)));

      let _cutoutProgress = 0;
      for (const p of mapped) {
        _cutoutProgress++;
        if (CUTOUT_ELIGIBLE_CATS.has(p.cat) && p.image) {
          if (cutoutCache.has(p.image)) {
            p.cutoutImage = cutoutCache.get(p.image);
            cutoutsReused++;
          } else if (cutoutBudget > 0) {
            const path = await generateCutout(p.image);
            if (path) { p.cutoutImage = path; cutoutsGenerated++; } else { cutoutsSkipped++; }
            cutoutBudget--;
          }
        }
        if (_cutoutProgress % 25 === 0 || _cutoutProgress === mapped.length) {
          console.log(`  Cutouts: ${_cutoutProgress}/${mapped.length} for ${store.brand}`);
        }
      }

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
  console.log(`Cutouts — new: ${cutoutsGenerated}, reused from cache: ${cutoutsReused}, skipped (failed quality check): ${cutoutsSkipped}`);
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
