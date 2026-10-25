// Browser check for the 3D timeline player. Runs against the real lessons (needs WebGL; SwiftShader works).
import { chromium } from "playwright-core";
const PORT = process.env.WEB_PORT || 5173;
const exe = "/mnt/nas/harshamttl/.cache/ms-playwright/chromium-1148/chrome-linux/chrome";
const b = await chromium.launch({ executablePath: exe, args: ["--no-sandbox", "--use-angle=swiftshader", "--use-gl=angle", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const p = await b.newPage({ viewport: { width: 1000, height: 900 } });
const errs = []; p.on("pageerror", (e) => errs.push(String(e))); p.on("console", (m) => m.type() === "error" && errs.push(m.text()));
const ok = (c, msg) => { console.log((c ? "PASS " : "FAIL ") + msg); if (!c) process.exitCode = 1; };
await p.goto(`http://127.0.0.1:${PORT}/#/article/lesson-q030-pairwise-differences`);
const sc = p.locator("section.scenedemo").first();
const stage = sc.locator(".stage");
await sc.scrollIntoViewIfNeeded();
await p.waitForFunction(() => document.querySelector(".stage")?.dataset.ready === "ready", null, { timeout: 60000 });
const time = async () => (await sc.locator(".time").innerText()).split(" / ")[0];
const secs = async () => { const [m, s] = (await time()).split(":").map(Number); return m * 60 + s; };
const scrub = async (t) => { await sc.locator("input.scrub").fill(String(Math.round(t * 100))); await p.waitForTimeout(350); };
const shot = () => stage.screenshot();
const chapter = async () => (await sc.locator(".steptext h4").innerText());
const project = (id) => stage.evaluate((el, i) => el.__scene.project(i), id);

ok(true, "3D scene is created when it scrolls into view");
ok((await chapter()).startsWith("Chapter 1 of 5"), "starts on chapter 1 with the inputs visible");
const idle = await shot();
ok(idle.length > 8000, "the 3D view draws something (not a blank canvas)");
ok((await stage.getAttribute("aria-label")).includes("Press Space to play"), "stage has an accessible name with instructions");
ok((await sc.locator(".steptext dd").first().innerText()).length > 20, "narration explains what changed");

// scrubbing changes the picture and the chapter
await scrub(15);
ok((await chapter()).startsWith("Chapter 3 of 5"), "scrubbing to 0:15 shows chapter 3, stretch");
const mid = await shot();
await scrub(30);
const end = await shot();
ok(Buffer.compare(mid, end) !== 0 && Buffer.compare(idle, mid) !== 0, "the picture changes as the time changes");
ok((await chapter()).startsWith("Chapter 5 of 5"), "0:30 is chapter 5");

// play and pause advance the time
await scrub(0);
await sc.getByRole("button", { name: "Play" }).click();
await p.waitForTimeout(2200);
const t1 = await secs();
ok(t1 >= 1, `Play advances the time (${t1} s after ~2 s)`);
await sc.getByRole("button", { name: "Pause" }).click();
const frozen = await secs(); await p.waitForTimeout(1200);
ok((await secs()) === frozen, "Pause stops the time");

// chapter button plays one chapter, then pauses at its end
await sc.locator(".chapters button").nth(1).click();
await p.waitForTimeout(500);
ok(await sc.getByRole("button", { name: "Pause" }).isVisible(), "a chapter button starts playing that chapter");
await p.waitForFunction(() => /Play/.test(document.querySelector(".transport .play")?.innerText || ""), null, { timeout: 15000 });
ok((await chapter()).startsWith("Chapter 2 of 5") && (await secs()) >= 9, "the chapter clip stops at the end of its chapter");

// speed
await scrub(0);
await sc.locator("label.speed select").selectOption("2");
await sc.getByRole("button", { name: "Play" }).click(); await p.waitForTimeout(2000);
const fast = await secs(); await sc.getByRole("button", { name: "Pause" }).click();
ok(fast >= 3, `2× speed moves faster (${fast} s after ~2 s)`);
await sc.locator("label.speed select").selectOption("1");

// click a result cube: highlight, links, trace
await scrub(30);
const pt = await project("out_1_2_1");
ok(pt && pt.visible, "result cube out[1,2,1] is visible at the end");
await p.mouse.click(pt.x, pt.y); await p.waitForTimeout(400);
const trace = await sc.locator(".trace").innerText();
ok(/out\[1, 2, 1\] = x\[1, 1\] − y\[2, 1\]/.test(trace) && /x_1_1, y_2_1/.test(trace), "clicking a result cube traces it to its x and y cubes");
const sel = await shot();
ok(Buffer.compare(sel, end) !== 0, "the selected cube and its sources are drawn differently");
await sc.getByRole("button", { name: "Clear selection" }).click();
ok((await sc.locator(".trace").count()) === 0, "Clear selection removes the trace");
// clicking a source cube links to its results
const px = await project("x_0_2");
await p.mouse.click(px.x, px.y); await p.waitForTimeout(300);
ok(/x\[0, 2\] = 3/.test(await sc.locator(".trace").innerText()), "clicking an x cube explains it and marks the results that read it");
await p.keyboard.press("Escape");

// camera views
const before = await shot();
await sc.getByRole("button", { name: /Front view/ }).click(); await p.waitForTimeout(400);
ok(Buffer.compare(before, await shot()) !== 0, "Front view moves the camera");
await sc.getByRole("button", { name: "Follow the animation" }).click(); await p.waitForTimeout(300);

// drag to rotate keeps the camera under the user's control
const box = await stage.boundingBox();
const still = await shot();
await p.mouse.move(box.x + 300, box.y + 200); await p.mouse.down(); await p.mouse.move(box.x + 420, box.y + 160, { steps: 6 }); await p.mouse.up(); await p.waitForTimeout(300);
ok(Buffer.compare(still, await shot()) !== 0, "dragging rotates the 3D box");
await scrub(31);
ok(await stage.evaluate((el) => el.__scene.userCamera), "after the user rotates, the animation no longer takes the camera");
await sc.getByRole("button", { name: "Follow the animation" }).click();

// keyboard
await stage.focus(); await scrub(0); await stage.focus();
await p.keyboard.press("ArrowRight"); await p.waitForTimeout(300);
ok((await chapter()).startsWith("Chapter 2 of 5"), "ArrowRight jumps to the next chapter");
await p.keyboard.press("ArrowLeft"); await p.waitForTimeout(300);
ok((await chapter()).startsWith("Chapter 1 of 5"), "ArrowLeft goes back one chapter");
const wasPlaying = await sc.getByRole("button", { name: "Pause" }).isVisible();   // a chapter clip may still be running
await p.keyboard.press("Space"); await p.waitForTimeout(500);
ok((await sc.getByRole("button", { name: "Pause" }).isVisible()) !== wasPlaying, "Space toggles play and pause");
await p.keyboard.press("Space"); await p.waitForTimeout(300);
ok((await sc.getByRole("button", { name: "Pause" }).isVisible()) === wasPlaying, "Space toggles back");
if (await sc.getByRole("button", { name: "Pause" }).isVisible()) await sc.getByRole("button", { name: "Pause" }).click();

// help and text version
await scrub(8);
await sc.getByRole("button", { name: "Help me understand this step" }).click();
ok((await sc.locator(".helppanel").count()) === 1, "step help opens for the current chapter");
await sc.locator("details.static summary").click();
ok((await sc.locator("details.static ol li").count()) === 5, "text version lists all 5 chapters");
await sc.locator(".picklist button", { hasText: "out_0_1_2" }).click();
ok(/out\[0, 1, 2\]/.test(await sc.locator(".trace").innerText()), "a number can be selected without the 3D view");

// parameter: 4 y points -> bigger box
await sc.locator(".inputs select").selectOption("4");
await p.waitForTimeout(600);
await scrub(30);
const big = await project("out_1_3_2");
ok(big && big.visible, "with 4 y points the box has a cube out[1,3,2]");

// reduced motion
await p.emulateMedia({ reducedMotion: "reduce" }); await p.waitForTimeout(300);
ok(await sc.getByRole("button", { name: "Play" }).isDisabled(), "Play is disabled when the system asks for reduced motion");
await sc.locator(".chapters button").nth(2).click(); await p.waitForTimeout(300);
ok((await chapter()).startsWith("Chapter 3 of 5") && (await secs()) >= 18, "chapter buttons jump to the end state without animating");
await p.emulateMedia({ reducedMotion: "no-preference" });

// dark theme
await p.emulateMedia({ colorScheme: "dark" }); await p.waitForTimeout(500);
const dark = await shot();
ok(dark.length > 5000, "the scene still draws in dark mode");
await p.emulateMedia({ colorScheme: "light" });

// the failure scene
const sc2 = p.locator("section.scenedemo").nth(1);
await sc2.scrollIntoViewIfNeeded();
await p.waitForFunction(() => document.querySelectorAll(".stage")[1]?.dataset.ready === "ready", null, { timeout: 60000 });
await sc2.locator("input.scrub").fill("1700"); await p.waitForTimeout(500);
ok((await sc2.locator(".steptext h4").innerText()).includes("The feature axis does not fit"), "the mismatch scene ends on the conflict chapter");
const miss = await sc2.locator(".stage").evaluate((el) => el.__scene.project("missing_1_1_2"));
ok(miss && miss.visible, "the unmatched feature layer shows red '?' cubes");

// mobile layout
await p.setViewportSize({ width: 390, height: 800 });
await sc.scrollIntoViewIfNeeded(); await p.waitForTimeout(500);
ok(!(await p.evaluate(() => document.documentElement.scrollWidth > innerWidth)), "no sideways scroll at 390 px");
ok((await stage.boundingBox()).width <= 390 && (await stage.boundingBox()).height >= 200, "the 3D stage fits a phone screen");
console.log("console errors:", errs.length ? errs : "none");
await b.close();
