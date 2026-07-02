/**
 * Kashef — live store sync (server-side, no CORS issues)
 * ---------------------------------------------------------------
 * Fetches every store's public /products.json feed directly from
 * Node (not a browser tab), paginates through the FULL catalog,
 * maps each product into Kashef's schema, and writes
 * synced-products.json next to kashef_9.html.
 *
 * kashef_9.html already auto-loads synced-products.json on every
 * page visit (see loadSyncedProducts() in the HTML) — so once this
 * file exists in the same folder as the site, the site is live and
 * up to date automatically. No code changes needed in the HTML.
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

import { writeFile } from "node:fs/promises";

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

// Same word->category guessing rule as the in-browser version, plus a
// couple of practical keyword extensions since real store tags/types
// are messier than the single-word demo case.
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
      break; // later pages 404ing just means we're past the end
    }
    const data = await res.json();
    const products = data.products || [];
    if (!products.length) break;
    all.push(...products);
    if (products.length < PAGE_SIZE) break; // last page
    page++;
    if (page > 20) break; // sanity cap, ~5000 products
  }
  return all;
}

function mapProduct(sp, storeUrl, brandName) {
  const cat = guessCategoryFromText(
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

async function main() {
  const allMapped = [];
  const summary = [];

  for (const store of STORES) {
    const storeUrl = store.url.replace(/\/$/, "");
    try {
      const products = await fetchAllProducts(storeUrl);
      const mapped = products.map((sp) => mapProduct(sp, storeUrl, store.brand));
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

  // De-dupe by link, in case a product shows up twice across pages.
  const seen = new Set();
  const deduped = allMapped.filter((p) => {
    if (seen.has(p.link)) return false;
    seen.add(p.link);
    return true;
  });

  await writeFile(OUTPUT_FILE, JSON.stringify(deduped, null, 2));

  console.log("\n--- Kashef sync summary ---");
  summary.forEach((line) => console.log(line));
  console.log(`\nWrote ${deduped.length} total products to ${OUTPUT_FILE}`);
  console.log("Put this file next to kashef_9.html and the site will pick it up automatically.\n");
}

main().catch((err) => {
  console.error("Sync failed:", err);
  process.exit(1);
});
