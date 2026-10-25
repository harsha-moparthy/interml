// Browser check for demonstrations, checks, comparisons, and step help (T10). Needs the fixture lesson-030.
import { chromium } from "playwright-core";
const PORT = process.env.WEB_PORT || 5173;
const exe = "/mnt/nas/harshamttl/.cache/ms-playwright/chromium-1148/chrome-linux/chrome";
const b = await chromium.launch({ executablePath: exe, args: ["--no-sandbox"] });
const ctx = await b.newContext({ viewport: { width: 1000, height: 1000 } });
const p = await ctx.newPage();
const errs = []; p.on("pageerror", (e) => errs.push(String(e))); p.on("console", (m) => m.type() === "error" && errs.push(m.text()));
const ok = (c, msg) => { console.log((c ? "PASS " : "FAIL ") + msg); if (!c) process.exitCode = 1; };
const demo = p.locator("section.demo");
const stepText = () => demo.locator(".steptext h4").innerText();
await p.goto("http://127.0.0.1:" + PORT + "/#/article/lesson-030");
await demo.waitFor();

// start state and no autoplay
ok((await stepText()).startsWith("Step 1 of 4"), "demo starts at step 1");
await p.waitForTimeout(2200);
ok((await stepText()).startsWith("Step 1 of 4"), "nothing animates by itself");
ok((await demo.locator(".timeline li").count()) === 4, "labelled timeline has 4 steps");

// forward to the trace step
for (let n = 0; n < 3; n++) await demo.getByRole("button", { name: /Forward/ }).click();
ok((await stepText()).startsWith("Step 4 of 4"), "forward x3 reaches step 4");
ok((await demo.locator(".matrix").count()) === 3, "A, B and C matrices shown");

// linked selection: pick slice i=1 and C[1,2,1]
await demo.getByRole("button", { name: "1", exact: true }).first().click();
await demo.getByRole("button", { name: "C[2, 1] = 35" }).click();
const marks = async () => ({
  a: await demo.locator(".matrix:nth-child(1) td.hit").count(),
  b: await demo.locator(".matrix:nth-child(2) td.hit").count(),
  c: await demo.locator(".matrix:nth-child(3) td.hit").count(),
});
const m = await marks();
ok(m.a === 1 && m.b === 1 && m.c === 1, "one traced cell in each of A, B, C");
ok((await demo.locator(".matrix:nth-child(1) td.hit").innerText()).includes("5"), "A cell shows 5");
ok((await demo.locator(".matrix:nth-child(2) td.hit").innerText()).includes("30"), "B cell shows 30");
ok((await demo.locator(".eq").getAttribute("aria-label")).includes("equals 5 plus 30 equals 35"), "equation text shows 5 + 30 = 35");
ok((await demo.locator("code.on").innerText()).includes("C[i, j, k]"), "code line 3 highlighted");
ok((await demo.locator(".steptext dd").first().innerText()).includes("C[1, 2, 1] = 5 + 30 = 35"), "step text names the cell and values");

// backward restores earlier states; selection kept; reset restores defaults
await demo.getByRole("button", { name: /Back/ }).click();
ok((await stepText()).startsWith("Step 3 of 4"), "back goes to step 3");
await demo.getByRole("button", { name: /Forward/ }).click();
ok((await demo.locator(".eq").getAttribute("aria-label")).includes("equals 35"), "returning to step 4 restores the same trace");
await demo.getByRole("button", { name: "Reset" }).click();
ok((await stepText()).startsWith("Step 1 of 4"), "reset returns to step 1");
await demo.getByRole("button", { name: /Forward/ }).click(); await demo.getByRole("button", { name: /Forward/ }).click(); await demo.getByRole("button", { name: /Forward/ }).click();
ok((await demo.locator(".eq").getAttribute("aria-label")).includes("C[0,0,0] equals 1 plus 10 equals 11"), "reset restored the default selection C[0,0,0]");

// keyboard
await demo.getByRole("button", { name: "Reset" }).click(); await demo.getByRole("button", { name: "Reset" }).focus();
await p.keyboard.press("ArrowRight");
ok((await stepText()).startsWith("Step 2 of 4"), "ArrowRight steps forward");
await p.keyboard.press("ArrowLeft");
ok((await stepText()).startsWith("Step 1 of 4"), "ArrowLeft steps back");
await p.keyboard.press("ArrowRight"); await p.keyboard.press("ArrowRight"); await p.keyboard.press("Home");
ok((await stepText()).startsWith("Step 1 of 4"), "Home goes to step 1");

// step help preserves state; only relevant options
ok((await demo.getByRole("button", { name: "Help me understand this step" }).count()) === 1, "help available on step 1");
await demo.getByRole("button", { name: /Forward/ }).click();
await demo.getByRole("button", { name: "Help me understand this step" }).click();
ok((await demo.locator(".helppanel section").count()) === 2, "step 2 shows its 2 help items only");
ok((await demo.locator(".helppanel a").getAttribute("href")) === "#/article/broadcasting", "prerequisite link in help");
ok((await stepText()).startsWith("Step 2 of 4"), "opening help keeps the step");
await demo.getByRole("button", { name: /Forward/ }).click();
ok((await demo.getByRole("button", { name: "Help me understand this step" }).count()) === 0, "no help button where no help is authored (step 3)");

// edit inputs to an incompatible pair
await demo.getByLabel("B axis 2").selectOption("2");
ok((await demo.locator(".timeline li").count()) === 2, "incompatible edit leaves 2 steps");
ok((await stepText()).startsWith("Step 2 of 2"), "step clamps to the last valid step");
ok((await demo.locator(".steptext dd").first().innerText()).includes("Axis 1 from the right has sizes 3 and 2"), "exact conflicting axis named");
ok((await demo.locator("tr.conflict").innerText()).includes("CONFLICT"), "conflict row is labelled with text");
await demo.getByLabel("B axis 2").selectOption("1");
ok((await demo.locator(".timeline li").count()) === 4, "fixing the shape restores 4 steps");

// static alternative and reduced motion
await demo.locator("details.static summary").click();
ok((await demo.locator("details.static li").count()) === 4, "text version lists 4 steps");
await p.emulateMedia({ reducedMotion: "reduce" });
await p.waitForTimeout(200);
ok(await demo.getByRole("button", { name: "Play" }).isDisabled(), "Play is disabled under reduced motion");
ok((await demo.locator(".hint").innerText()).includes("reduced motion"), "reduced-motion note shown");
await p.emulateMedia({ reducedMotion: "no-preference" });
await p.waitForTimeout(200);
await demo.getByRole("button", { name: "Reset" }).click();
await demo.getByRole("button", { name: "Play" }).click();
await p.waitForTimeout(1900);
ok(!(await stepText()).startsWith("Step 1 of"), "Play advances the steps when motion is allowed");
await demo.getByRole("button", { name: "Pause" }).click();
const frozen = await stepText(); await p.waitForTimeout(1800);
ok((await stepText()) === frozen, "Pause stops it");

// prediction: answer hidden until requested
const pred = p.locator("form.check").nth(0);
ok(!(await p.content()).includes("Axis 1 pairs 1 with 4"), "prediction answer is not in the page before the learner asks");
await pred.getByLabel("Your answer").fill("(3, 1)");
await pred.getByRole("button", { name: "Check my answer" }).click();
ok((await pred.locator(".feedback").innerText()).includes("That is shape A"), "wrong prediction gets a specific explanation");
await pred.getByLabel("Your answer").fill("3 4");
await pred.getByRole("button", { name: "Check my answer" }).click();
ok((await pred.locator(".feedback").innerText()).startsWith("Correct"), "correct prediction accepted in another format");
await pred.getByRole("button", { name: "Show the answer" }).click();
ok((await pred.locator(".answer").innerText()).includes("Axis 1 pairs 1 with 4"), "Show the answer reveals the explanation");

// new case (number) and explain-back
const nc = p.locator("form.check").nth(1);
await nc.getByLabel("Your answer").fill("36"); await nc.getByRole("button", { name: "Check my answer" }).click();
ok((await nc.locator(".feedback").innerText()).includes("Axes of size 1 always read index 0"), "wrong number gets the default explanation");
await nc.getByLabel("Your answer").fill("35"); await nc.getByRole("button", { name: "Check my answer" }).click();
ok((await nc.locator(".feedback").innerText()).startsWith("Correct"), "35 accepted");
const eb = p.locator("form.check").nth(2);
ok((await eb.getByRole("button", { name: "Check my answer" }).count()) === 0, "explain-back has no right/wrong button");
await eb.getByRole("button", { name: "Show a model explanation" }).click();
ok((await eb.locator(".answer").innerText()).includes("not enlarged"), "model explanation shown on request");

// comparison
const cmp = p.locator("section.compare");
ok((await cmp.locator("article").count()) === 2, "comparison shows two cases");
ok((await cmp.locator("article").nth(0).innerText()).includes("(2, 3)"), "case 1 result shape (2, 3)");
ok((await cmp.locator("article").nth(1).innerText()).includes("error"), "case 2 is an error");
ok((await cmp.locator(".takeaway").innerText()).includes("conflict"), "takeaway states the cause");

// walkthrough: solution hidden until asked, notes tie lines to explanations
const wt = p.locator("section.walkthrough");
ok((await wt.locator(".cm-content").count()) === 0, "walkthrough solution is hidden by default");
await wt.getByRole("button", { name: "Show solution with notes" }).click();
await wt.locator(".cm-content").waitFor();
ok((await wt.locator(".notes li").count()) === 2, "walkthrough lists its 2 notes");
ok((await wt.locator(".cm-hl").count()) === 2, "both noted line ranges are highlighted in the code");
await wt.locator(".notes li").nth(1).getByRole("button").click();
ok((await wt.locator(".cm-hl-active").count()) === 1 && (await wt.locator(".cm-hl-active").innerText()).includes("[:, None, :]"), "choosing a note marks its exact line in the code");
await wt.locator(".cm-hl").first().click();
ok((await wt.locator(".cm-hl-active").innerText()).includes("def q030"), "clicking a highlighted line selects its note");
ok((await wt.locator(".notes button[aria-pressed=true]").innerText()).includes("line 4"), "the matching note shows as selected");
await wt.getByRole("button", { name: "Hide solution and notes" }).click();
ok((await wt.locator(".cm-content").count()) === 0, "Hide removes the code again");
await p.setViewportSize({ width: 1000, height: 900 });
await demo.scrollIntoViewIfNeeded();
await demo.screenshot({ path: process.argv[2] + "/demo.png" });
await p.setViewportSize({ width: 390, height: 800 });
await demo.screenshot({ path: process.argv[2] + "/demo-mobile.png" });
console.log("console errors:", errs.length ? errs : "none");
await b.close();
