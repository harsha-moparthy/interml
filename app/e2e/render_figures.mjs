// Render still images from a 3D scene: stage screenshots at chosen times, saved as PNG under app/media/<article>/.
// Usage: node render_figures.mjs <article-id> <demo-index> name:time [name:time ...]     (needs the site running on WEB_PORT)
import { chromium } from "playwright-core";
import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const PORT = process.env.WEB_PORT || 5173;
const [, , article, which, ...shots] = process.argv;
const media = join(dirname(fileURLToPath(import.meta.url)), "..", "media", article);
mkdirSync(media, { recursive: true });
const b = await chromium.launch({ executablePath: "/mnt/nas/harshamttl/.cache/ms-playwright/chromium-1148/chrome-linux/chrome", args: ["--no-sandbox", "--use-angle=swiftshader", "--use-gl=angle", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const p = await b.newPage({ viewport: { width: 1200, height: 900 }, deviceScaleFactor: 2 });
await p.goto(`http://127.0.0.1:${PORT}/#/lesson-db/${article}`);
const sc = p.locator("section.scenedemo").nth(Number(which));
await sc.scrollIntoViewIfNeeded();
await p.waitForFunction((w) => document.querySelectorAll(".stage")[w]?.dataset.ready === "ready", Number(which), { timeout: 60000 });
await p.addStyleTag({ content: ".stagehint, .stagetip { display:none !important; }" });
for (const s of shots) {
  const [name, t] = s.split(":");
  await sc.locator("input.scrub").fill(String(Math.round(Number(t) * 100)));
  await p.waitForTimeout(700);
  await sc.locator(".stage").screenshot({ path: join(media, `${name}.png`) });
  console.log("wrote", join(media, `${name}.png`));
}
await b.close();
