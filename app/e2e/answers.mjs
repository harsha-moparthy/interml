// Render every authored answer, checking citations, math, figures, and navigation.
import { chromium } from 'playwright-core';
import fs from 'node:fs';
const BASE=process.env.SITE || `http://127.0.0.1:${process.env.WEB_PORT||5173}/`;
const browser=await chromium.launch({executablePath:process.env.CHROME||'/mnt/nas/harshamttl/.cache/ms-playwright/chromium-1148/chrome-linux/chrome',args:['--no-sandbox']});
const page=await browser.newPage({viewport:{width:1100,height:850}});
const errors=[];page.on('pageerror',e=>errors.push(String(e)));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
const ok=(condition,message)=>{console.log(`${condition?'PASS':'FAIL'} ${message}`);if(!condition)process.exitCode=1;};
await page.goto(BASE+'#/question/601');
const total=1100+JSON.parse(fs.readFileSync(new URL("../content/interview_additions.json",import.meta.url))).length;
let failures=[];
for(let id=601;id<=total;id++){
 await page.evaluate(id=>{window.location.hash=`#/question/${id}`;},id);
 await page.waitForFunction(id=>document.querySelector('h1 .id')?.textContent===String(id)&&document.querySelector('.answer-guide')?.innerText.length>100,id);
 const result=await page.evaluate(()=>({visible:!document.querySelector('#solution-panel').hidden,resources:document.querySelectorAll('.readings li').length,citations:document.querySelectorAll('.answer-guide a[href^="https://"]').length,mathErrors:document.querySelectorAll('.answer-guide .katex-error').length}));
 if(!result.visible||result.resources!==5||!result.citations||result.mathErrors)failures.push({id,...result});
 if(id%50===0)console.log(`Checked through Q${id}`);
}
ok(failures.length===0,`all ${total-600} answers render with five readings and citations: ${JSON.stringify(failures)}`);
for(const id of [605,614,653,684,688,701,724,734,796,824,1080,1101]){
 await page.goto(BASE+`#/question/${id}`);await page.waitForSelector('.answer-figure img');
 await page.locator('.answer-figure').scrollIntoViewIfNeeded();
 await page.waitForFunction(()=>[...document.querySelectorAll('.answer-figure img')].every(i=>i.complete&&i.naturalWidth>0));
 ok(true,`Q${id} diagram loads`);
}
for(const id of [1102,1117]){
 await page.goto(BASE+`#/question/${id}`);await page.waitForSelector('.answer-guide');
 ok(await page.locator('pre.answer-code code').count()===1,`Q${id} preserves fenced code and line breaks`);
}
for(const [id,phrase] of [[796,'0.8113'],[824,'Hard negative'],[1100,'first day']]){
 await page.goto(BASE+`#/question/${id}`);await page.waitForSelector('.answer-guide');
 const answer=await page.locator('.answer-guide').innerText();
 ok(answer.toLowerCase().includes(phrase.toLowerCase())&&answer.length>1500,`Q${id} expanded explanation is served`);
}
await page.goto(BASE+'#/question/1080');await page.waitForSelector('.answer-guide');
ok(await page.locator('.task strong .katex').count()===3,'math inside emphasized question text renders');
await page.setViewportSize({width:390,height:844});
for(const id of [688,796,824,1053,1058,1080,1100,1101,1102,1119,1135,1136]){
 await page.goto(BASE+`#/question/${id}`);await page.waitForSelector('.answer-guide');
 ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1),`Q${id} mobile page fits viewport`);
}
if(process.argv[2]){fs.mkdirSync(process.argv[2],{recursive:true});await page.goto(BASE+'#/question/1080');await page.waitForSelector('.answer-guide');await page.screenshot({path:process.argv[2]+'/appendix-answer-mobile.png',fullPage:true});}
ok(errors.length===0,'no page or console errors');
console.log('console/page errors:',errors.length?errors:'none');
await browser.close();
