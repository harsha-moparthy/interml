// Content integrity and actual KaTeX parsing for all authored appendix answers.
import fs from 'node:fs';
import path from 'node:path';
import katex from '../../app/frontend/node_modules/katex/dist/katex.mjs';
const root=path.resolve(import.meta.dirname,'../..');
let equations=0, citations=0, figures=0;
for(let id=601;id<=1100;id++) {
 const item=JSON.parse(fs.readFileSync(`${root}/app/content/solutions/q${id}.json`));
 const assert=(condition,message)=>{if(!condition)throw Error(`Q${id}: ${message}`);};
 assert(item.id===id,'ID mismatch');
 assert(item.answer.split(/\s+/).length>=60,'answer missing or too short');
 assert(!/A complete (answer|solution) must|Answer focus:/.test(item.answer),'generic guide retained');
 assert(item.readings.length===5 && new Set(item.readings.map(r=>r.url)).size===5,'five distinct resources required');
 const links=[...item.answer.matchAll(/(?<!!)\[(\d)\]\((https:\/\/[^\s)]+)\)/g)];
 assert(links.length>0,'no inline citations');citations+=links.length;
 for(const [,rank,url] of links)assert(item.readings[Number(rank)-1]?.url===url,'citation and resource mismatch');
 for(const m of item.answer.matchAll(/\$\$([\s\S]*?)\$\$|\$([^$]+)\$/g)) {
  katex.renderToString(m[1]??m[2],{throwOnError:true,strict:'error',trust:false});equations++;
 }
 for(const [,name] of item.answer.matchAll(/!\[[^\]]+\]\(\/answer-diagrams\/([a-z0-9-]+\.svg)\)/g)) {
  assert(fs.existsSync(`${root}/app/frontend/public/answer-diagrams/${name}`),'missing figure');figures++;
 }
}
console.log(`PASS 500 authored answers; 2500 reading entries; ${citations} citations; ${equations} math expressions; ${figures} figure placements`);
