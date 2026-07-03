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
 *   node sync-products.mjs
 *
 * Requires Node 18+ (built-in fetch). To check your version:
 *   node -v
 *
 * Edit the STORES list below to add/remove stores.
 * ---------------------------------------------------------------
 */

import { writeFile, readFile } from "node:fs/promises";
import { pipeline } from "@xenova/transformers";

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

// Also bake the same product list directly into the site's HTML, between
// these markers, so opening the file works with zero setup — no server,
// no fetch — not just when it happens to be hosted somewhere with
// synced-products.json sitting next to it. See kashef_9.html for the
// matching SYNCED_PRODUCTS block and how it's used on page load.
const HTML_FILE = "kashef_9.html";
const HTML_MARKER_START = "const SYNCED_PRODUCTS = [";
const HTML_MARKER_END = "];\n/* END SYNCED_PRODUCTS */";

// Brands whose BRANDS-array `photo` is currently a stand-in (a product shot
// or banner), not an actual uploaded logo — their real logo only exists as
// an Instagram profile picture, which can't be pulled programmatically.
// Strip the stand-in so the site falls back to its honest colored-initials
// badge instead of implying these are real logos.
const BRANDS_WITHOUT_REAL_LOGOS = [];

// Keep this in sync with the CATEGORIES array in kashef_9.html.
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
  dress: "Women's Fashion",
  skirt: "Women's Fashion",
  blouse: "Women's Fashion",
  shirt: "Men's Fashion",
  tee: "Men's Fashion",
  tshirt: "Men's Fashion",
  hoodie: "Men's Fashion",
  sweatpants: "Men's Fashion",
  sweater: "Men's Fashion",
  jacket: "Men's Fashion",
  crewneck: "Men's Fashion",
  short: "Men's Fashion",
  shorts: "Men's Fashion",
  pants: "Men's Fashion",
  polo: "Men's Fashion",
  shoe: "Shoes",
  shoes: "Shoes",
  sneaker: "Shoes",
  sneakers: "Shoes",
  bag: "Bags",
  bags: "Bags",
  backpack: "Bags",
  ring: "Accessories",
  cap: "Accessories",
  hat: "Accessories",
  belt: "Accessories",
  jewelry: "Accessories",
  makeup: "Makeup",
  lipstick: "Makeup",
  hair: "Hair Care",
  skin: "Skin Care",
  skincare: "Skin Care",
  perfume: "Perfumes",
  fragrance: "Perfumes",
};

function guessCategoryFromText(text) {
  const words = (text || "")
    .toLowerCase()
    .split(/[\s,/]+/)
    .map(normalize)
    .filter(Boolean);
  for (const w of words) {
    if (KEYWORD_MAP[w]) {
      return CATEGORIES.find((c) => c.name === KEYWORD_MAP[w]);
    }
  }
  for (const w of words) {
    const match = CATEGORIES.find((c) => c.icon === w);
    if (match) return match;
  }
  return CATEGORIES.find((c) => c.name === "Accessories") || CATEGORIES[0];
}

let _embedder = null;
async function getEmbedder() {
  if (!_embedder) {
    _embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { quantized: true });
  }
  return _embedder;
}

async function embedText(text) {
  const embedder = await getEmbedder();
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data).map((n) => Math.round(n * 10000) / 10000);
}

function cosineSim(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

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
    for (const [name, label] of Object.entries(CATEGORY_LABELS)) {
      _categoryEmbeddings[name] = await embedText(label);
    }
  }
  return _categoryEmbeddings;
}

const KASHEF_WOMEN_ONLY_WORDS = ["skirt","skirts","crop","gown","gowns","blouse","blouses","romper","rompers","jumpsuit","jumpsuits","bra","bralette","legging","leggings","abaya","abayas","hijab","hijabs","skort","skorts"];
const KASHEF_DRESS_EXCEPTION_RE = /\bdress(es)?\b(?!\s*(shirt|shirts|pant|pants|shoe|shoes|code|sock|socks|watch|watches))/;
const KASHEF_CLOTHING_WORDS = ["shirt","shirts","tee","tees","tshirt","hoodie","hoodies","sweatpants","sweatshirt","sweatshirts","sweater","sweaters","jacket","jackets","crewneck","crewnecks","short","shorts","pant","pants","polo","polos","tracksuit","tracksuits","jogger","joggers","top","tops","blazer","blazers","coat","coats","vest","vests","cardigan","cardigans","sock","socks"];
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
  if (!clean) {
    return { cat: CATEGORIES.find((c) => c.name === "Accessories") || CATEGORIES[0], emb: null, unisex: false };
  }
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
  if (has(KASHEF_CLOTHING_WORDS)) {
    const catName = hasWomenSignal ? "Women's Fashion" : hasMenSignal ? "Men's Fashion" : "Men's Fashion";
    const cat = CATEGORIES.find((c) => c.name === catName);
    return { cat: cat || CATEGORIES[0], emb, unisex: wordSet.has("unisex") };
  }
  const directMatches = [
    ["Shoes", KASHEF_SHOES_WORDS],
    ["Bags", KASHEF_BAGS_WORDS],
    ["Accessories", KASHEF_ACCESSORIES_WORDS],
    ["Makeup", KASHEF_MAKEUP_WORDS],
    ["Hair Care", KASHEF_HAIRCARE_WORDS],
    ["Skin Care", KASHEF_SKINCARE_WORDS],
    ["Perfumes", KASHEF_PERFUME_WORDS],
  ];
  for (const [name, list] of directMatches) {
    if (has(list)) {
      const cat = CATEGORIES.find((c) => c.name === name);
      return { cat: cat || CATEGORIES[0], emb, unisex: false };
    }
  }

  const catEmbeddings = await getCategoryEmbeddings();
  let best = null;
  let bestSim = -1;
  for (const [name, catEmb] of Object.entries(catEmbeddings)) {
    const sim = cosineSim(emb, catEmb);
    if (sim > bestSim) {
      bestSim = sim;
      best = name;
    }
  }
  const cat = CATEGORIES.find((c) => c.name === best) || CATEGORIES[0];
  return { cat, emb, unisex: false };
}

function stripHtml(html) {
  return (html || "")
    .replace(/<[^>]*>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

async function fetchAllProducts(storeUrl) {
  const all = [];
  let page = 1;
  while (true) {
    const res = await fetch(
      `${storeUrl}/products.json?limit=${PAGE_SIZE}&page=${page}`,
      { headers: { "User-Agent": "Mozilla/5.0 (Kashef sync bot)" } }
    );
    if (!res.ok) {
      if (page === 1) {
        throw new Error(`${storeUrl} responded with ${res.status}`);
      }
      break;
    }
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

async function mapProduct(sp, storeUrl, brandName) {
  const { cat, emb, unisex } = await classifyWithAI(
    `${sp.product_type || ""} ${(sp.tags || []).join ? (sp.tags || []).join(" ") : sp.tags || ""} ${sp.title || ""}`
  );
  const variants = sp.variants || [];
  const firstVariant = variants[0] || {};
  const inStock = variants.some((v) => v.available !== false);
  const onSale = variants.some(
    (v) =>
      v.compare_at_price &&
      parseFloat(v.compare_at_price) > parseFloat(v.price)
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
    lastSynced: new Date().toLocaleString("en-GB", {
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    }),
  };
}

function stripPlaceholderLogos(html) {
  let updated = html;
  let count = 0;
  for (const name of BRANDS_WITHOUT_REAL_LOGOS) {
    const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`(\\{name:"${escaped}"[^}]*?), photo:"[^"]*"(\\})`);
    if (re.test(updated)) {
      updated = updated.replace(re, "$1$2");
      count++;
    }
  }
  return { html: updated, count };
}

async function main() {
  const allMapped = [];
  const summary = [];

  for (const store of STORES) {
    const storeUrl = store.url.replace(/\/$/, "");
    try {
      const products = await fetchAllProducts(storeUrl);
      const mapped = await Promise.all(products.map((sp) => mapProduct(sp, storeUrl, store.brand)));
      allMapped.push(...mapped);
      const inStockCount = mapped.filter((p) => p.inStock).length;
      summary.push(
        `${store.brand}: ${mapped.length} products (${inStockCount} in stock, ${
          mapped.length - inStockCount
        } sold out)`
      );
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
      console.warn(
        `Warning: couldn't find the SYNCED_PRODUCTS markers in ${HTML_FILE} — skipped baking data into the HTML. synced-products.json was still written normally.`
      );
    }
  } catch (err) {
    console.warn(`Warning: couldn't update ${HTML_FILE} (${err.message}). synced-products.json was still written normally.`);
  }

  console.log("\n--- Kashef sync summary ---");
  summary.forEach((line) => console.log(line));
  console.log(`\nWrote ${deduped.length} total products to ${OUTPUT_FILE}`);
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
