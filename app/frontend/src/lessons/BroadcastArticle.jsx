import { useEffect, useRef, useState } from 'react';
import { Tex } from '../Inline.jsx';
import { CodeView } from '../code/CodeView.jsx';
import { useFetch } from '../lib.js';
import { A, B, outputValue, broadcastShape, parseShape } from './broadcastModel.js';
import './broadcastArticle.css';

const chapters = [
  ['Start with the inputs.', 'A has two groups of three numbers. B has four numbers on its middle axis. Their shapes describe where each number lives. Nothing has been copied.'],
  ['Align the axes.', 'Read from the right: 3 matches 1, then 1 matches 4, then 2 matches 1. A size-one axis reuses its only entry. The result will have shape (2, 4, 3).'],
  ['Reuse values across positions.', 'A supplies the same row at each of the four middle-axis positions. B supplies the same scalar across three columns and both groups. The translucent tiles show reuse, not allocated copies.'],
  ['Add the corresponding values.', 'Each result cell reads one value from A and one from B. Select any result cell below to trace its input coordinates. The new result contains 24 numbers.'],
];
const palette = { a: ['#d2efec', '#13877e', '#a4d9d3'], b: ['#ffebc6', '#aa6500', '#efce8e'], c: ['#ebe2ff', '#7545c4', '#ccb6ee'] };
function Tile({ x, y, value, tone, active, ghost, onClick, label, hidden }) {
  const [fill, stroke, side] = palette[tone];
  return <g transform={`translate(${x},${y})`} className={`br-tile ${active ? 'selected' : ''}`} opacity={hidden ? 0 : ghost ? .5 : 1} onClick={onClick} role={onClick ? 'button' : undefined} tabIndex={onClick && !hidden ? 0 : undefined} aria-label={label} aria-pressed={onClick ? !!active : undefined} onKeyDown={e => { if (onClick && ['Enter',' '].includes(e.key)) { e.preventDefault(); onClick(); } }} style={{ pointerEvents: hidden ? 'none' : 'auto' }}>
    <path d="M0 0 L12 -10 L56 -10 L44 0 Z" fill={side} stroke={stroke} strokeWidth=".8"/>
    <path d="M44 0 L56 -10 L56 26 L44 36 Z" fill={side} stroke={stroke} strokeWidth=".8"/>
    <rect width="44" height="36" rx="2" fill={active ? stroke : fill} stroke={stroke} strokeWidth={active ? 2.5 : .8}/>
    <text x="22" y="23" textAnchor="middle" fill={active ? '#fff' : stroke} fontSize="14" fontWeight="600">{value}</text>
  </g>;
}
function TensorScene({ progress, selected, onSelect, angle, offset, flat }) {
  const phase = Math.min(3, Math.floor(progress));
  const expansion = Math.max(0, Math.min(1, progress - 1));
  const depthX = flat ? 0 : 9 + angle * .22, depthY = 40;
  const [si,sj,sk] = selected;
  const point = (base,i,j,k) => [base + k*51 + j*depthX, 95 + i*160 + j*depthY];
  const tiles=[];
  for (let i=0;i<2;i++) for (let j=3;j>=0;j--) for (let k=0;k<3;k++) {
    const [cx,cy]=point(847,i,j,k);
    tiles.push(<Tile key={`c${i}${j}${k}`} x={cx} y={cy} value={phase===3 ? outputValue(i,j,k,offset) : '·'} tone="c" active={phase===3 && i===si && j===sj && k===sk} onClick={()=>onSelect([i,j,k])} label={`Result [${i}, ${j}, ${k}] = ${outputValue(i,j,k,offset)}`} ghost={phase<1}/>);
    const [ax,ay]=point(76,i,j*expansion,k);
    tiles.push(<Tile key={`a${i}${j}${k}`} x={ax} y={ay} value={A[i][k]} tone="a" active={i===si && k===sk && (j===0 || (phase>=2 && j===sj))} ghost={j>0} hidden={j>0 && progress<=1} label={`A[${i},0,${k}]`} />);
    const [bx,by]=point(467,i*expansion,j,k*expansion);
    tiles.push(<Tile key={`b${i}${j}${k}`} x={bx} y={by} value={B[j]+offset} tone="b" active={j===sj && (i===0&&k===0 || phase>=2&&i===si&&k===sk)} ghost={i>0||k>0} hidden={(i>0||k>0)&&progress<=1} label={`B[0,${j},0]`}/>);
  }
  return <svg viewBox="0 0 1160 460" className="br-scene" aria-label="Broadcasting tensor diagram. Teal A, amber B, violet result. Select a result cell or use the coordinate controls below.">
    <defs><pattern id="br-dot" width="18" height="18" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".65" fill="#c8d6df"/></pattern></defs>
    <rect width="1160" height="460" fill="url(#br-dot)"/>
    <g className="br-tensor-label"><text x="76" y="35" fill="#08776e">A <tspan dx="14" fill="#23334d">(2, 1, 3)</tspan></text><text x="467" y="35" fill="#9c5c00">B <tspan dx="14" fill="#23334d">(1, 4, 1)</tspan></text><text x="847" y="35" fill="#7139ba">A + B <tspan dx="10" fill="#23334d">(2, 4, 3)</tspan></text></g>
    {tiles}
    <text x="365" y="220" fontSize="28" fill="#718096">+</text><text x="764" y="220" fontSize="28" fill="#718096">=</text>
    <g fontSize="13" fill="#526279"><text x="76" y="435">i: group · j: row · k: column</text><text x="467" y="435">Faded tiles = reused input values</text><text x="847" y="435">Select a violet cell to trace it</text></g>
  </svg>;
}
function BroadcastLab() {
  const [progress,setProgress]=useState(0), [playing,setPlaying]=useState(false), [selected,setSelected]=useState([1,2,2]);
  const [angle,setAngle]=useState(20),[flat,setFlat]=useState(false),[offset,setOffset]=useState(0),[help,setHelp]=useState(false);
  const reduced=useRef(window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  useEffect(()=>{ const mq=window.matchMedia('(prefers-reduced-motion: reduce)'); const listener=e=>{reduced.current=e.matches;if(e.matches)setPlaying(false);};mq.addEventListener('change',listener);return()=>mq.removeEventListener('change',listener);},[]);
  useEffect(()=>{
    if(!playing)return;
    let raf,last;
    const frame=t=>{if(last!==undefined)setProgress(p=>Math.min(3.99,p+Math.min(t-last,70)/2300));last=t;raf=requestAnimationFrame(frame);};
    raf=requestAnimationFrame(frame);return()=>cancelAnimationFrame(raf);
  },[playing]);
  useEffect(()=>{if(progress>=3.99)setPlaying(false);},[progress]);
  const phase=Math.min(3,Math.floor(progress)),[i,j,k]=selected;
  const seek=v=>{setPlaying(false);setProgress(v);};
  const reset=()=>{seek(0);setSelected([1,2,2]);setOffset(0);setAngle(20);setFlat(false);setHelp(false);};
  return <section className="br-lab" aria-label="Broadcasting interactive laboratory">
    <div className="br-stage"><TensorScene progress={progress} selected={selected} onSelect={s=>{setSelected(s);seek(3);}} angle={angle} offset={offset} flat={flat}/></div>
    <div className="br-transport">
      <div className="br-buttons"><button className="br-primary" onClick={()=>{if(reduced.current){seek(Math.min(3,phase+1));return;}if(progress>=3.99)setProgress(0);setPlaying(!playing);}}>{playing?'Ⅱ Pause':'▶ Play'}</button><button onClick={()=>seek(Math.max(0,phase-1))} disabled={phase===0}>← Back</button><button onClick={()=>seek(Math.min(3,phase+1))} disabled={phase===3}>Next →</button><button onClick={reset}>↺ Reset</button></div>
      <div className="br-steps">{['Inputs','Align','Expand','Add'].map((name,n)=><button key={name} aria-current={phase===n?'step':undefined} onClick={()=>seek(n)}><span>{n+1}</span>{name}</button>)}</div>
    </div>
    <input className="br-scrub" aria-label="Animation timeline" type="range" min="0" max="3.99" step=".01" value={progress} onChange={e=>seek(Number(e.target.value))}/>
    <div className="br-caption"><div><h3>{chapters[phase][0]}</h3><p>{chapters[phase][1]}</p></div><div className="br-calculation" aria-live="polite"><span className="br-mini">Trace one output cell</span><div className="br-equation"><div className="br-a"><code>A[{i}, 0, {k}]</code><strong>{A[i][k]}</strong></div><b>+</b><div className="br-b"><code>B[0, {j}, 0]</code><strong>{B[j]+offset}</strong></div><b>=</b><div className="br-c"><code>C[{i}, {j}, {k}]</code><strong data-testid="cell-result">{outputValue(i,j,k,offset)}</strong></div></div></div></div>
    <details className="br-explore"><summary>Explore the values and viewing angle</summary><div className="br-inputs">{['Group i','Row j','Column k'].map((name,axis)=><label key={name}>{name}<select aria-label={name} value={selected[axis]} onChange={e=>{const v=[...selected];v[axis]=Number(e.target.value);setSelected(v);seek(3);}}>{Array.from({length:[2,4,3][axis]},(_,n)=><option key={n}>{n}</option>)}</select></label>)}<label>Add to every B value: {offset}<input aria-label="B offset" type="range" min="-10" max="10" value={offset} onChange={e=>{setOffset(Number(e.target.value));seek(3);}}/></label><label>Depth angle<input aria-label="Depth angle" type="range" min="-20" max="60" value={angle} onChange={e=>setAngle(Number(e.target.value))}/></label><button onClick={()=>setFlat(!flat)} aria-pressed={flat}>{flat?'Use 3D view':'Use flat view'}</button></div></details>
    <button className="br-help" aria-expanded={help} onClick={()=>setHelp(!help)}>Help me understand this step {help?'−':'+'}</button>
    {help&&<div className="br-help-body"><strong>Follow coordinates, not copies.</strong><p>In C[i, j, k], i selects one of A’s two groups and k selects one of its three columns. A has only one middle row, so its row index is always 0. B has only one group and one column, so only j changes. For this cell, read A[{i}, 0, {k}] and B[0, {j}, 0].</p><p>The three axes describe array positions. They do not need to represent physical space.</p></div>}
  </section>;
}
function ShapeLab(){
  const [a,setA]=useState('2,1,3'),[b,setB]=useState('1,4,1');
  const aa=parseShape(a),bb=parseShape(b),result=aa&&bb?broadcastShape(aa,bb):{error:'Enter 1–5 comma-separated dimensions, each between 0 and 1000.'};
  return <div className="br-shape-lab"><div className="br-inputs"><label>A shape<input value={a} onChange={e=>setA(e.target.value)} aria-label="A shape"/></label><span>+</span><label>B shape<input value={b} onChange={e=>setB(e.target.value)} aria-label="B shape"/></label><button onClick={()=>{setA('2,3');setB('4,3');}}>Try a mismatch</button><button onClick={()=>{setA('2,1,3');setB('1,4,3');}}>Repair with axes</button></div><p className={result.error?'br-invalid':'br-valid'} role="status">{result.error?`Cannot broadcast. ${result.error}`:`Result shape: (${result.shape.join(', ')})`}</p><p>Read each aligned axis from the right. Sizes must be equal, or one must be 1. A missing leading axis acts as size 1. A size of 0 represents an empty axis.</p></div>;
}
function Prediction(){const [answer,setAnswer]=useState(null);return <div className="br-predict"><h3>Before you run it: what is the output shape?</h3><p><code>x.shape = (5, 2)</code> and <code>y.shape = (7, 2)</code>.<br/>You compute <code>x[:, None, :] - y[None, :, :]</code>.</p><div className="br-buttons">{['(5, 7, 2)','(5, 7)','(7, 5, 2)'].map(v=><button key={v} aria-pressed={answer===v} onClick={()=>setAnswer(v)}>{v}</button>)}</div>{answer&&<p role="status">{answer==='(5, 7, 2)'?'Exactly.':'Look at the inserted axes.'} The inputs become (5, 1, 2) and (1, 7, 2). There are 35 pairs, with two coordinate differences per pair. The result is (5, 7, 2).</p>}</div>;}
function PairLab(){const [i,setI]=useState(0),[j,setJ]=useState(0);const x=[[1,2],[4,6]],y=[[0,1],[3,3],[7,2]],d=x[i].map((v,k)=>v-y[j][k]);return <div className="br-pairs"><div className="br-pair-select"><div><h3 className="br-a">Choose an x point</h3>{x.map((v,n)=><button key={n} aria-pressed={i===n} onClick={()=>setI(n)}>x[{n}] = [{v.join(', ')}]</button>)}</div><div><h3 className="br-b">Choose a y point</h3>{y.map((v,n)=><button key={n} aria-pressed={j===n} onClick={()=>setJ(n)}>y[{n}] = [{v.join(', ')}]</button>)}</div></div><div className="br-pair-result" aria-live="polite"><code>D[{i}, {j}, :] = [{x[i].join(', ')}] − [{y[j].join(', ')}] = [{d.join(', ')}]</code><p>This is a displacement vector. Its Euclidean length is √({d.map(v=>`(${v})²`).join(' + ')}) ≈ {Math.hypot(...d).toFixed(3)}.</p></div><div className="br-pair-matrix">{x.map((v,ii)=><div key={ii}>{y.map((w,jj)=><button key={jj} aria-pressed={ii===i&&jj===j} aria-label={`Pair ${ii}, ${jj}`} onClick={()=>{setI(ii);setJ(jj);}}>[{v.map((n,k)=>n-w[k]).join(', ')}]</button>)}</div>)}</div><small>Rows select x. Columns select y. Every tile contains a two-coordinate difference.</small></div>;}
function SourceSolution(){const [show,setShow]=useState(false);const {data,error}=useFetch('/api/exercises/30/solution');return <div className="br-solution"><button aria-expanded={show} aria-controls="br-source" onClick={()=>setShow(!show)}>{show?'Hide':'Show'} original PDF solution</button>{show&&<div id="br-source">{error?<p role="alert">The local service is unavailable. Start the backend to load the original solution.</p>:data?<CodeView code={data.code} language={data.language} label="Original solution for question 030"/>:<p>Loading original solution…</p>}<p>The code comes from the existing solution database. This article adds explanations without rewriting it.</p></div>}</div>;}
export default function BroadcastArticle(){
  useEffect(()=>{window.scrollTo(0,0);document.body.classList.add('broadcast-page');return()=>document.body.classList.remove('broadcast-page');},[]);
  return <div className="br-root"><header className="br-header"><a className="br-brand" href="#/"><strong>ML</strong> / understood</a><a href="#/">All questions →</a></header><main className="br-main">
    <header className="br-hero"><h1>One rule. Every pair.</h1><p>Understand NumPy broadcasting by following the numbers.</p></header>
    <BroadcastLab/>
    <nav className="br-contents" aria-label="In this article">{[['rule','The rule'],['memory','What is reused'],['pairs','Every pair'],['practice','Try it yourself']].map(([id,name])=><a key={id} href={`#${id}`} onClick={e=>{e.preventDefault();document.getElementById(`br-${id}`).scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});}}>{name} ↘</a>)}</nav>
    <section className="br-section br-two-col" id="br-rule"><div><h2>Read shapes<br/>from the right.</h2><p>A shape tells you how many positions exist along each axis. In <code>(2, 1, 3)</code>, there are two groups, one row per group, and three numbers per row.</p><p>NumPy aligns the last axes first. At each position, the sizes must match, or one size must be <strong>1</strong>. A size-one axis always uses index zero while the other array moves along that axis.</p><p>This is why a single row can participate in many rows of a result. The operation still happens one output position at a time.</p></div><div className="br-rule-table"><h3>Three axes. Three checks.</h3><table><thead><tr><th>Input</th><th>Group i</th><th>Row j</th><th>Column k</th></tr></thead><tbody><tr className="br-a"><th>A</th><td>2</td><td>1</td><td>3</td></tr><tr className="br-b"><th>B</th><td>1</td><td>4</td><td>1</td></tr><tr className="br-c"><th>Result</th><td>2</td><td>4</td><td>3</td></tr></tbody></table><p>2 with 1 → 2<br/>1 with 4 → 4<br/>3 with 1 → 3</p><small>The arrows show the result axis size, not a multiplication.</small></div></section>
    <section className="br-section" id="br-memory"><div className="br-reading"><h2>Reuse the input.<br/>Create the output.</h2><p>The faded tiles in the animation are a way to see which input values are reused. NumPy does not need to create those expanded input copies. It can read the same input location for many output positions.</p><div className="br-math"><Tex tex={'C_{i,j,k}=A_{i,0,k}+B_{0,j,0}'} display/></div><p>For example, <code>C[1, 2, 2] = A[1, 0, 2] + B[0, 2, 0] = 6 + 30 = 36</code>. The zero coordinates identify the singleton axes.</p><aside className="br-note"><strong>Broadcasting saves input copies, not result storage.</strong><p>This addition creates 24 result values. A large pairwise operation can still allocate a very large output. For float64, an output of shape (10,000, 10,000, 3) alone occupies 2.4 billion bytes, about 2.24 GiB.</p></aside><p>Also distinguish broadcasting from repetition. <code>np.tile</code> explicitly builds repeated data. Inserting an axis with <code>None</code> changes the view’s shape; it does not add new data values.</p></div></section>
    <section className="br-section"><div className="br-reading"><h2>Make a shape fail.<br/>Then repair it.</h2><p>Try (2, 3) and (4, 3). The final size 3 matches. The preceding sizes 2 and 4 conflict. NumPy cannot guess that you want every pair: you must create distinct axes for the two collections.</p></div><ShapeLab/></section>
    <section className="br-section" id="br-pairs"><div className="br-reading"><h2>Now compare every point<br/>with every other point.</h2><p>This is what <a href="#/question/30">question 030: Pairwise Differences</a> asks you to implement. Given x with shape (n, d) and y with shape (m, d), return one d-dimensional difference vector for each pair.</p><p>Here n and m count points. The final axis d counts coordinates within a point. Both inputs must use the same coordinate dimension. Choose a pair and check the subtraction yourself.</p></div><PairLab/><div className="br-reading"><h3>The missing axes express your intent.</h3><CodeView code={'x[:, None, :]     # (n, 1, d): one x point across all y points\ny[None, :, :]     # (1, m, d): all y points for each x point\n\nD = x[:, None, :] - y[None, :, :]  # (n, m, d)'} language="Python" label="Pairwise broadcasting explained"/><p><code>:</code> keeps every entry on an axis. <code>None</code> inserts a size-one axis. In the result, i chooses x, j chooses y, and k chooses a coordinate.</p><div className="br-math"><Tex tex={'D_{i,j,k}=x_{i,k}-y_{j,k}'} display/></div><p>The signs matter: x − y points from y to x. Reversing the subtraction reverses the vector. It does not change its Euclidean length.</p><aside className="br-note"><strong>A difference vector is not a distance.</strong><p>To obtain pairwise Euclidean distances, square each coordinate, sum over the final axis, and take the square root: <code>np.sqrt((D ** 2).sum(axis=-1))</code>. This returns (n, m). It is an extension, not the requested output of question 030.</p></aside><h3>Why plain x − y does not mean “every pair”</h3><p>If both arrays have shape (n, d), ordinary subtraction pairs row 0 with row 0, row 1 with row 1, and so on. If n and m differ and neither is 1, the shapes conflict. Separate singleton axes create the independent n and m dimensions.</p><SourceSolution/></div></section>
    <section className="br-section" id="br-practice"><div className="br-reading"><h2>Try it without the animation.</h2><Prediction/><p>Next, explain why the result has three axes even though each input started with two. If you can identify the point from x, the point from y, and the coordinate within their difference, you have the whole mechanism.</p><a className="br-next" href="#/question/30">Open question 030 and implement it →</a></div></section>
    <footer className="br-footer"><p>Continue with <a href="#/article/arrays-and-shapes">array shapes</a> or <a href="#/article/lesson-q027-array-mutation">views and memory sharing</a>.</p><p>Technical reference: <a href="https://numpy.org/doc/stable/user/basics.broadcasting.html" target="_blank" rel="noreferrer">NumPy broadcasting documentation ↗</a>. Worked examples are computed locally. No model or remote API is involved.</p></footer>
  </main></div>;
}
