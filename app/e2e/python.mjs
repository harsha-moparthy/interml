// Browser check for the Python playground (T11). Needs the fixture lesson-030 and `npm run setup:pyodide`.
import { chromium } from "playwright-core";
const PORT = process.env.WEB_PORT || 5173;
const exe = "/mnt/nas/harshamttl/.cache/ms-playwright/chromium-1148/chrome-linux/chrome";
const b = await chromium.launch({ executablePath: exe, args: ["--no-sandbox"] });
const ctx = await b.newContext({ viewport: { width: 1000, height: 1000 } });
const p = await ctx.newPage();
const errs = [], requests = [];
p.on("pageerror", (e) => errs.push(String(e)));
p.on("console", (m) => m.type() === "error" && errs.push(m.text()));
ctx.on("request", (r) => requests.push(r.url()));
const ok = (c, msg) => { console.log((c ? "PASS " : "FAIL ") + msg); if (!c) process.exitCode = 1; };
const url = "http://127.0.0.1:" + PORT + "/#/article/lesson-030";
await p.goto(url);
const pg = p.locator("section.playground").first();
await pg.waitFor();
const status = () => pg.locator(".pgstatus").innerText();
const output = async () => (await pg.locator("pre.output").count()) ? pg.locator("pre.output").innerText() : "";
const typeInto = async (loc, c, page = p) => { const ed = loc.locator(".cm-content"); await ed.click(); await page.keyboard.press("Control+a"); await page.keyboard.press("Delete"); await page.keyboard.insertText(c); };
const setCode = (c) => typeInto(pg, c);
const run = () => pg.getByRole("button", { name: "Run", exact: true }).click();
const until = (re, ms = 60000) => p.waitForFunction(([sel, src]) => new RegExp(src).test(document.querySelector(sel)?.innerText || ""), [".playground .pgstatus", re.source], { timeout: ms });

// nothing loads until the learner runs code
await p.waitForTimeout(800);
ok(!requests.some((u) => u.includes("pyodide.asm.wasm")), "runtime is not downloaded until Run is pressed");
ok((await status()).startsWith("Not started"), "idle status explains the first-run load");
ok((await pg.getByRole("button", { name: "Stop" }).isDisabled()), "Stop is disabled when nothing runs");
ok((await p.locator("section.playground").nth(1).innerText()).includes("Display only"), "PyTorch question is labelled Display only");
ok((await p.locator("section.playground").nth(1).getByRole("button", { name: "Run", exact: true }).count()) === 0, "display-only block has no Run button");
ok((await p.locator("section.playground").nth(1).innerText()).includes("PyTorch does not run in the browser runtime"), "label states the exact reason");

// first run: starter code raises inside the function -> prints nothing, checks fail
await run();
await until(/Loading|Running|Finished|Stopped/);
await until(/Finished|Stopped by an error/);
ok((await status()).startsWith("Finished"), "runtime loaded and ran the starter code");
const failing = await pg.locator(".checks-result").innerText();
ok(failing.includes("Checks: 2 of 2 failed") && failing.includes("NotImplementedError"), "starter fails both checks with a readable reason");
await setCode("def f():\n    raise ValueError('boom')\n\nf()");
await run(); await until(/Stopped by an error/);
const tb = await output();
ok(tb.includes('File "<exec>"') && tb.includes("ValueError: boom") && !tb.includes("_pyodide"), "traceback shows the learner's code, not Pyodide internals");
await pg.getByRole("button", { name: "Reset code and output" }).click();
await run(); await until(/Finished/);

// solve it
await setCode("import numpy as np\n\ndef pairwise_differences(x, y):\n    return np.asarray(x)[:, None, :] - np.asarray(y)[None, :, :]\n\nprint(pairwise_differences(np.ones((2, 3)), np.ones((4, 3))).shape)\n");
await run(); await until(/Finished/);
ok((await output()).trim() === "(2, 4, 3)", "print output shown: (2, 4, 3)");
ok((await pg.locator(".checks-result h4").innerText()).includes("all passed"), "checks all pass for a correct solution");
ok((await pg.locator(".checks-result li").count()) === 2, "each check listed by name");

// errors
await setCode("def f(:\n  pass");
await run(); await until(/Stopped by an error/);
ok((await output()).includes("SyntaxError"), "syntax error is reported");
await setCode("print(1/0)");
await run(); await until(/Stopped by an error/);
ok((await output()).includes("ZeroDivisionError"), "runtime error is reported");

// isolation between runs
await setCode("leak = 5"); await run(); await until(/Finished/);
await setCode("print(leak)"); await run(); await until(/Stopped by an error/);
ok((await output()).includes("NameError"), "each run starts with a fresh namespace");

// output cap
await setCode("print('a' * 30000)"); await run(); await until(/Finished/);
const big = await output();
ok(big.includes("output cut at 20000 characters") && big.length < 20200, "output is capped at 20000 characters");

// a single oversized array gives a readable error
await setCode("import numpy as np\nnp.zeros(2**30)"); await run(); await until(/Stopped by an error|Finished/);
ok(/ValueError|MemoryError/.test(await output()), "oversized array gives a readable error");
// real memory exhaustion: keep allocating until the browser refuses
await setCode("import numpy as np\nkeep = []\nwhile True:\n    keep.append(np.ones(2**24))"); await run();
await p.waitForFunction(() => /Stopped|Finished|time limit|crashed/.test(document.querySelector(".pgstatus")?.innerText || "") && !/^Running/.test(document.querySelector(".pgstatus")?.innerText || ""), null, { timeout: 40000 });
const memStatus = await status(); const memOut = await output();
console.log("   exhaustion status:", JSON.stringify(memStatus), "output tail:", JSON.stringify(memOut.slice(-100)));
ok(!/^Running|^Loading/.test(memStatus), "memory exhaustion ends the run with a visible status");
await setCode("print('still alive')"); await run(); await until(/Finished/, 60000);
ok((await output()).trim() === "still alive", "runtime works after memory exhaustion");

// timeout (fixture limit is 2 s) and recovery
await setCode("while True:\n    pass"); await run();
await until(/Stopped after/, 15000);
ok((await status()).includes("Stopped after the 2 s time limit"), "infinite loop stopped at the time limit");
ok(await pg.getByRole("button", { name: "Run", exact: true }).isEnabled(), "Run is available again after a timeout");
await setCode("print('after timeout')"); await run(); await until(/Finished/);
ok((await output()).trim() === "after timeout", "next run works on a fresh runtime");

// Stop button
await setCode("while True:\n    pass"); await run();
await until(/Running/);
await pg.getByRole("button", { name: "Stop" }).click();
await until(/^Stopped\. A fresh runtime/, 5000);
ok(true, "Stop ends a running program");
await setCode("print('after stop')"); await run(); await until(/Finished/);
ok((await output()).trim() === "after stop", "next run works after Stop");

// Ctrl+Enter
await setCode("print('keys')"); await pg.locator(".cm-content").click(); await p.keyboard.press("Control+Enter"); await until(/Finished/);
ok((await output()).trim() === "keys", "Ctrl+Enter runs the code");

// containment precautions
const before = requests.length;
await setCode("import js\nprint(js.fetch)"); await run(); await until(/Finished|Stopped by an error/);
const jf = (await output()).trim(); console.log("   js.fetch output:", JSON.stringify(jf));
ok(jf === "None" || jf === "undefined", "js.fetch is not available to learner code");
await setCode("from pyodide.http import pyfetch\nawait pyfetch('/api/health')"); await run(); await until(/Stopped by an error|Finished/);
ok((await status()).includes("error"), "pyfetch fails");
ok(!requests.slice(before).some((u) => u.includes("/api/health")), "no request reached the local service from learner code");
await setCode("import os\nprint(sorted(os.listdir('/')))"); await run(); await until(/Finished/);
ok(!(await output()).includes("mnt"), "filesystem is the virtual browser one, not the host's");

// Reset
await setCode("print('edited')"); await run(); await until(/Finished/);
await pg.getByRole("button", { name: "Reset code and output" }).click();
ok((await pg.locator(".cm-content").innerText()).includes("raise NotImplementedError"), "Reset restores the starter code");
ok((await output()) === "" && (await status()).startsWith("Not started"), "Reset clears output and status");

// runtime files missing -> clear message, then retry works
const page2 = await ctx.newPage();
await page2.route("**/pyodide/pyodide.js", (r) => r.abort());
await page2.goto(url);
const pg2 = page2.locator("section.playground").first(); await pg2.waitFor();
await pg2.getByRole("button", { name: "Run", exact: true }).click();
await page2.waitForFunction(() => /could not start/.test(document.querySelector(".pgstatus")?.innerText || ""), null, { timeout: 20000 });
ok((await pg2.locator(".pgstatus").innerText()).includes("npm run setup:pyodide"), "missing runtime shows how to fix it");
await page2.unroute("**/pyodide/pyodide.js");
await typeInto(pg2, "print('retry ok')", page2);
await pg2.getByRole("button", { name: "Run", exact: true }).click();
await page2.waitForFunction(() => /Finished/.test(document.querySelector(".pgstatus")?.innerText || ""), null, { timeout: 60000 });
ok((await pg2.locator("pre.output").innerText()).trim() === "retry ok", "retry succeeds after the files are available");


await p.bringToFront();
// pandas playground: packages load on demand, are not counted in the time limit, and containment still holds
const pd = p.locator("section.playground").nth(2);
await pd.scrollIntoViewIfNeeded();
ok((await pd.innerText()).includes("pandas 2.2.3") && (await pd.innerText()).includes("about 24 MB"), "pandas playground states its packages and download size");
ok((await pg.innerText()).includes("about 17 MB") && !(await pg.innerText()).includes("pandas 2.2.3"), "NumPy-only playground states the smaller size");
ok(!requests.some((u) => u.includes("pandas-2.2.3")), "pandas was not downloaded before it was needed");
const pdCode = "import pandas as pd\n\n\ndef add_group_mean(frame, group, value):\n    result = frame.copy()\n    result['group_mean'] = frame.groupby(group, dropna=False)[value].transform('mean')\n    return result\n\nprint(pd.__version__)\n";
await typeInto(pd, pdCode);
await pd.getByRole("button", { name: "Run", exact: true }).click();
await p.waitForFunction(() => [...document.querySelectorAll(".playground .pgstatus")][1]?.innerText.startsWith("Finished"), null, { timeout: 90000 });
ok((await pd.locator("pre.output").innerText()).trim() === "2.2.3", "pandas 2.2.3 runs in the browser");
ok(requests.some((u) => u.includes("pandas-2.2.3")), "pandas was downloaded on the first pandas run");
ok((await pd.locator(".checks-result h4").innerText()).includes("all passed"), "pandas checks pass for a correct solution (2 s limit, load not counted)");
await typeInto(pd, "def add_group_mean(frame, group, value):\n    r = frame.copy()\n    r['group_mean'] = frame[value].mean()\n    return r\n");
await pd.getByRole("button", { name: "Run", exact: true }).click();
await p.waitForFunction(() => [...document.querySelectorAll(".playground .pgstatus")][1]?.innerText.startsWith("Finished"), null, { timeout: 30000 });
const wrong = await pd.locator(".checks-result").innerText();
ok(wrong.includes("1 of 2 failed") || wrong.includes("2 of 2 failed"), "a wrong pandas solution fails the check (overall mean, not per group)");
ok(/\[3\.0, 4\.0, 3\.0, 4\.0, 3\.0\]/.test(wrong) || wrong.includes("Fail"), "the failing check reports a reason");
await typeInto(pd, "import js\nprint(js.fetch)"); await pd.getByRole("button", { name: "Run", exact: true }).click();
await p.waitForFunction(() => [...document.querySelectorAll(".playground .pgstatus")][1]?.innerText.startsWith("Finished"), null, { timeout: 30000 });
ok((await pd.locator("pre.output").innerText()).trim() === "None", "network is blocked again after packages loaded");

// runner behaviour with a fake worker: crash, never-replying worker, and init failure
const fake = await p.evaluate(async () => {
  const { createRunner } = await import("/src/py/runner.js");
  const results = {};
  const make = (mode) => () => {
    const w = { terminated: false, postMessage(m) {
      if (m.type === "init") setTimeout(() => (mode === "initfail" ? w.onmessage({ data: { type: "init-error", message: "no files" } }) : w.onmessage({ data: { type: "ready" } })), 5);
      if (m.type === "run" && mode === "crash") setTimeout(() => w.onerror({ message: "out of memory" }), 5);
    }, terminate() { this.terminated = true; } };
    return w;
  };
  let r = createRunner({ makeWorker: make("crash") });
  results.crash = (await r.run("x")).status; results.afterCrash = r.state;
  r = createRunner({ makeWorker: make("silent") });
  const t0 = performance.now(); results.timeout = (await r.run("x", { timeoutMs: 1000 })).status; results.timeoutMs = Math.round(performance.now() - t0);
  r = createRunner({ makeWorker: make("silent") });
  const pr = r.run("x"); await new Promise((s) => setTimeout(s, 50)); r.stop(); results.stop = (await pr).status;
  const makeLoad = (mode) => () => {
    const w = { postMessage(m) {
      if (m.type === "init") setTimeout(() => w.onmessage({ data: { type: "ready" } }), 5);
      if (m.type === "load") setTimeout(() => w.onmessage({ data: mode === "loadfail" ? { type: "load-error", message: "nope" } : { type: "loaded" } }), 1500);
      if (m.type === "run") setTimeout(() => w.onmessage({ data: { type: "result", id: m.id, ok: true, error: null, checks: [], ms: 1 } }), 5);
    }, terminate() {} };
    return w;
  };
  r = createRunner({ makeWorker: makeLoad("ok") });
  const slowLoad = await r.run("x", { timeoutMs: 1000, packages: ["pandas"] });
  results.loadNotTimed = slowLoad.status;
  const again = await r.run("x", { timeoutMs: 1000, packages: ["pandas"] });   // already loaded: no second wait
  results.loadedOnce = again.status;
  r = createRunner({ makeWorker: makeLoad("loadfail") });
  const lf = await r.run("x", { packages: ["pandas"] }); results.loadfail = lf.status + ":" + lf.message;
  r = createRunner({ makeWorker: make("initfail") });
  const bad = await r.run("x"); results.initfail = bad.status + ":" + bad.message;
  return results;
});
console.log("   fake worker:", JSON.stringify(fake));
ok(fake.crash === "crashed" && fake.afterCrash === "loading", "a worker that dies mid-run is reported and replaced");
ok(fake.timeout === "timeout" && fake.timeoutMs >= 1000 && fake.timeoutMs < 1500, "a silent worker is cut off at the limit");
ok(fake.stop === "stopped", "stop works before the worker answers");
ok(fake.initfail === "unavailable:no files", "init failure is reported with its message");
ok(fake.loadNotTimed === "done" && fake.loadedOnce === "done", "a 1.5 s package load does not count against a 1 s time limit");
ok(fake.loadfail === "unavailable:Could not load packages: nope", "a failed package load is reported");
await p.bringToFront();
await p.screenshot({ path: process.argv[2] + "/python.png", fullPage: false });
console.log("console errors:", errs.filter((e) => !/net::ERR_FAILED|Failed to load resource/.test(e)).length ? errs : "none");
await b.close();
