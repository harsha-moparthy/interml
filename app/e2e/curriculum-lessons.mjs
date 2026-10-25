import { chromium } from "playwright-core";
import assert from "node:assert/strict";

const base = `http://127.0.0.1:${process.env.WEB_PORT || 5173}`;
const selected = [16,17,27,30,54,57,66,77,92,100,108,113,122,132,137,152,162,168,181,182,207,209,233,241,248,249,267,287,294,308,316,328,331,339,354,361,374,379,391,397,409,413,420,429,433,439,461,462,469,476,485,499,503,509,515,519,525,528,542,545,549,555,557,562,567,573,577,581,588,592,593,598];
const samples = [16,66,137,181,241,248,316,328,331,391,439,519,573,598];
const browser = await chromium.launch({ executablePath: "/mnt/nas/harshamttl/.cache/ms-playwright/chromium-1148/chrome-linux/chrome", args: ["--no-sandbox"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));

await page.goto(base);
await page.getByRole("heading", { level: 1 }).waitFor();
await page.getByText("600 questions in original order").waitFor();
await page.screenshot({ path: "planning/design/site-index.png", fullPage: false });

for (const id of selected) {
  const q = await (await fetch(`${base}/api/exercises/${id}`)).json();
  const lessons = q.articles.filter((a) => a.purpose === "deep-lesson" && a.review_state === "complete");
  assert.equal(lessons.length, 1, `q${id} has one complete deep lesson`);
  const article = await (await fetch(`${base}/api/articles/${lessons[0].id}`)).json();
  assert.equal(article.exercises.some((e) => e.id === id), true);
  assert.equal(Object.keys(article.demonstrations).length >= 1, true);
  assert.equal(Object.values(article.demonstrations)[0].steps.length >= 4, true);
  assert.equal(Object.keys(article.checks).length >= 2, true);
  assert.equal(Object.keys(article.media).length >= 1, true);
  assert.equal(Object.values(article.demonstrations).some((d) => d.component === "frames"), false, `q${id} does not use the retired animation`);
  if (id !== 30) assert.equal(Object.values(article.demonstrations).some((d) => d.component === "concept-lab"), true, `q${id} uses the gold-standard concept lab`);
}

for (const id of samples) {
  const q = await (await fetch(`${base}/api/exercises/${id}`)).json();
  const lesson = q.articles.find((a) => a.purpose === "deep-lesson");
  await page.goto(`${base}/#/article/${lesson.id}`);
  await page.getByRole("heading", { level: 1 }).waitFor();
  const demo = page.locator("section.concept-demo").first();
  assert.equal(await demo.locator(".cl-steps button").count() >= 3, true);
  await demo.getByRole("button", { name: /Next/ }).click();
  assert.match(await demo.locator(".cl-caption h4").innerText(), /Step 2/);
  await demo.getByRole("button", { name: /Play/ }).click();
  await page.waitForTimeout(1750);
  assert.equal(await demo.getByRole("button", { name: /Pause/ }).count() + await demo.getByRole("button", { name: /Play/ }).count() > 0, true);
  assert.equal(await page.locator("figure img").count() >= 1, true);
  assert.equal(await page.locator(".check").count() >= 2, true);
}

for (const slug of [
  "lesson-q316-multi-head-attention",
  "lesson-q328-cross-attention",
  "lesson-q331-rotary-positional-embeddings-rope",
  "lesson-q391-kv-cache-for-efficient-autoregressive-attention",
]) {
  await page.goto(`${base}/#/article/${slug}`);
  const demo = page.locator("section.concept-demo").first();
  await demo.getByRole("button", { name: "Idea first" }).waitFor();
  assert.equal(await demo.locator(".cs-story").count(), 1);
  await demo.getByRole("button", { name: "Inspect vectors" }).click();
  assert.equal(await demo.locator(".cl-plane").count() >= 1, true);
  if (slug.includes("q328")) await demo.locator(".cl-buttons").getByRole("button", { name: /Next/ }).click();
  if (slug.includes("q316") || slug.includes("q328")) {
    assert.equal(await demo.locator(".cl-vector-bar").count() >= 1, true);
  }
}

await page.goto(`${base}/#/article/lesson-q391-kv-cache-for-efficient-autoregressive-attention`);
await page.getByRole("heading", { name: "KV Cache for Efficient Autoregressive Attention" }).waitFor();
await page.screenshot({ path: "planning/design/curriculum-kv.png", fullPage: true });
await page.setViewportSize({ width: 390, height: 844 });
await page.reload();
await page.locator("section.demo").scrollIntoViewIfNeeded();
assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
await page.screenshot({ path: "planning/design/curriculum-kv-mobile.png", fullPage: false });
assert.deepEqual(errors, []);
console.log(`PASS: ${selected.length} lesson APIs and ${samples.length} representative animated article routes; mobile has no page overflow; no browser errors.`);
await browser.close();
