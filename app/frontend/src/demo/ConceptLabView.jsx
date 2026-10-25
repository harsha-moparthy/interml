import { useId, useState } from "react";
import { Inline, Tex } from "../Inline.jsx";
import { CodeView } from "../code/CodeView.jsx";
import { highlightSet } from "./frames.js";
import { ConceptStory, hasConceptStory } from "./ConceptStories.jsx";
import "./conceptLab.css";

export function ConceptInputs({ inputs, dispatch, demo }) {
  const params = demo.data.params || [];
  if (!params.length) return null;
  return (
    <fieldset className="cl-inputs">
      <legend>Change one condition</legend>
      <div className="cl-input-row">
        {params.map((p) => {
          const idx = Math.max(0, p.options.findIndex((o) => String(o.value) === String(inputs[p.name])));
          const set = (i) => dispatch({ type: "setInput", name: p.name, value: p.options[i].value });
          return p.options.length > 6 ? (
            <label key={p.name}><span>{p.label}</span><strong>{p.options[idx].label}</strong>
              <input type="range" min="0" max={p.options.length - 1} step="1" value={idx}
                onChange={(e) => set(Number(e.target.value))} aria-valuetext={p.options[idx].label} />
            </label>
          ) : (
            <label key={p.name}><span>{p.label}</span>
              <select value={idx} onChange={(e) => set(Number(e.target.value))}>
                {p.options.map((o, i) => <option key={String(o.value)} value={i}>{o.label}</option>)}
              </select>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}

const cell = (v) => (typeof v === "object" && v !== null ? v : { t: String(v) });
const cls = (id, hs, extra = "") => `${extra} ${id === hs.sel ? "is-selected" : hs.linked.has(id) ? "is-linked" : ""}`;
const toggle = (dispatch, current, id) => id && dispatch({ type: "select", selection: current === id ? null : id });
const fmt = (n) => Math.abs(n) >= 1000 || (Math.abs(n) < .01 && n !== 0) ? Number(n).toExponential(1) : String(Math.round(Number(n) * 1000) / 1000);

function VectorBar({ value, max }) {
  const n = Number(value);
  if (!Number.isFinite(n)) return <span className="cl-symbol">{value}</span>;
  const size = Math.max(4, Math.round(Math.abs(n) / (max || 1) * 44));
  return <span className={`cl-vector-bar ${n < 0 ? "negative" : "positive"}`} aria-hidden="true"><i style={{"--size": `${size}%`}} /></span>;
}

function LabGrid({ v, hs, dispatch, selection, index, vectorBars }) {
  const numeric = v.cells.flat().map((raw)=>Number(cell(raw).t)).filter(Number.isFinite);
  const max = Math.max(1e-9, ...numeric.map(Math.abs));
  return (
    <figure className={`cl-plane cl-grid tone-${index % 4} ${vectorBars ? "cl-vector-grid" : ""}`}>
      <figcaption><span className="cl-kicker">{vectorBars ? "VECTOR MAP" : "MATRIX"}</span><Inline text={v.title || v.caption || "Values"} /></figcaption>
      <div className="cl-grid-scroll"><table>
        {v.col_labels && <thead><tr>{v.row_labels && <th />}{v.col_labels.map((x, i) => <th key={i}>{x}</th>)}</tr></thead>}
        <tbody>{v.cells.map((row, i) => <tr key={i}>
          {v.row_labels && <th>{v.row_labels[i]}</th>}
          {row.map((raw, j) => { const c = cell(raw); const label = c.title || `${v.row_labels?.[i] || i}, ${v.col_labels?.[j] || j}: ${c.t}`; return (
            <td key={j} className={cls(c.id, hs, c.cls || "")} style={c.h != null ? { "--heat": c.h } : undefined}>
              {c.id ? <button type="button" title={vectorBars ? `Exact value: ${c.t}` : undefined} aria-label={label} aria-pressed={selection === c.id} onClick={() => toggle(dispatch, selection, c.id)}>{vectorBars ? <VectorBar value={c.t} max={max}/> : c.t}</button>
                : vectorBars ? <span title={`Exact value: ${c.t}`} aria-label={label}><VectorBar value={c.t} max={max}/></span> : <span>{c.t}</span>}
            </td>); })}
        </tr>)}</tbody>
      </table></div>
      {v.caption && v.title && <small><Inline text={v.caption} /></small>}
    </figure>
  );
}

function LabTokens({ v, hs, dispatch, selection, index }) {
  return (
    <figure className={`cl-plane cl-tokens tone-${index % 4}`}>
      <figcaption><span className="cl-kicker">SEQUENCE</span><Inline text={v.title || v.caption || "Objects in order"} /></figcaption>
      <ol>{v.items.map((x, i) => <li key={i} className={cls(x.id, hs, x.cls || "")}>
        {x.id ? <button type="button" aria-pressed={selection === x.id} onClick={() => toggle(dispatch, selection, x.id)}><b>{x.t}</b>{x.sub && <small>{x.sub}</small>}</button>
          : <span><b>{x.t}</b>{x.sub && <small>{x.sub}</small>}</span>}
      </li>)}</ol>
      {v.caption && v.title && <small><Inline text={v.caption} /></small>}
    </figure>
  );
}

function LabGraph({ v, hs, dispatch, selection, index }) {
  const uid = useId().replace(/:/g, "");
  const W = v.width || 620, H = v.height || 280, nw = v.node_w || 104, nh = v.node_h || 48;
  const by = Object.fromEntries(v.nodes.map((n) => [n.id, n]));
  const edge = (e) => {
    const a = by[e.from], b = by[e.to];
    if (!a || !b) return [0, 0, 0, 0];
    const dx = b.x - a.x, dy = b.y - a.y;
    const k = Math.min(Math.abs(nw / 2 / (dx || 1e-9)), Math.abs(nh / 2 / (dy || 1e-9)), 1);
    return [a.x + dx * k, a.y + dy * k, b.x - dx * k, b.y - dy * k];
  };
  return (
    <figure className={`cl-plane cl-graph tone-${index % 4}`}>
      <figcaption><span className="cl-kicker">DATA FLOW</span>{v.alt || "Follow the active objects"}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={v.alt || "Concept flow"}>
        <defs><marker id={`${uid}arrow`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto"><path d="M0 0L10 5L0 10z" /></marker></defs>
        {v.edges.map((e, i) => { const [x1,y1,x2,y2] = edge(e); return <g className="cl-edge" key={i}><line x1={x1} y1={y1} x2={x2} y2={y2} markerEnd={`url(#${uid}arrow)`}/>{e.label && <text x={(x1+x2)/2} y={(y1+y2)/2-7}>{e.label}</text>}</g>; })}
        {v.nodes.map((n, i) => <g key={n.id} transform={`translate(${n.x - nw/2} ${n.y - nh/2})`}
          className={`cl-node tone-${i % 4} ${cls(n.sel, hs, n.cls || "")}`}
          role={n.sel ? "button" : undefined} tabIndex={n.sel ? 0 : undefined} aria-pressed={n.sel ? selection === n.sel : undefined}
          aria-label={n.title || `${n.label}${n.sub ? `: ${n.sub}` : ""}`}
          onClick={() => toggle(dispatch, selection, n.sel)} onKeyDown={(e) => { if (n.sel && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); toggle(dispatch, selection, n.sel); } }}>
          <path d={`M8 0H${nw-8}L${nw} 8V${nh-8}L${nw-8} ${nh}H8L0 ${nh-8}V8Z`} />
          <text x={nw/2} y={n.sub ? nh/2-3 : nh/2+5} textAnchor="middle">{n.label}</text>
          {n.sub && <text className="cl-node-sub" x={nw/2} y={nh/2+14} textAnchor="middle">{n.sub}</text>}
        </g>)}
      </svg>
      {v.caption && <small><Inline text={v.caption} /></small>}
    </figure>
  );
}

function LabPlot({ v, index }) {
  const W = 680, H = 330, m = { l: 55, r: 22, t: 28, b: 48 };
  const [x0,x1] = v.x, [y0,y1] = v.y;
  const sx = (x) => m.l + (Number(x)-x0)/(x1-x0 || 1)*(W-m.l-m.r);
  const sy = (y) => H-m.b-(Number(y)-y0)/(y1-y0 || 1)*(H-m.t-m.b);
  const series = v.series || [];
  const path = (pts) => pts.map(([x,y],i) => `${i ? "L" : "M"}${sx(x)} ${sy(y)}`).join(" ");
  const ticks = (a,b) => Array.from({length:5},(_,i)=>a+(b-a)*i/4);
  return (
    <figure className={`cl-plane cl-plot tone-${index % 4}`}>
      <figcaption><span className="cl-kicker">MEASURE</span>{v.title || v.alt || "Read the change"}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={v.alt || v.title || "Plot"}>
        {ticks(y0,y1).map((t,i)=><g key={`y${i}`}><line className="cl-gridline" x1={m.l} x2={W-m.r} y1={sy(t)} y2={sy(t)}/><text x={m.l-8} y={sy(t)+4} textAnchor="end">{fmt(t)}</text></g>)}
        {ticks(x0,x1).map((t,i)=><g key={`x${i}`}><line className="cl-gridline" y1={m.t} y2={H-m.b} x1={sx(t)} x2={sx(t)}/><text x={sx(t)} y={H-m.b+18} textAnchor="middle">{fmt(t)}</text></g>)}
        {(v.hlines || []).map((l,i)=><line key={`h${i}`} className="cl-reference" x1={m.l} x2={W-m.r} y1={sy(l.y)} y2={sy(l.y)}/>)}
        {(v.vlines || []).map((l,i)=><line key={`v${i}`} className="cl-reference" y1={m.t} y2={H-m.b} x1={sx(l.x)} x2={sx(l.x)}/>)}
        {series.map((s,i) => s.kind === "bar" ? <g key={i}>{s.points.map(([x,y],j)=><rect key={j} className={`cl-series s${i}`} x={sx(x)-9} y={sy(Math.max(y,0))} width="18" height={Math.abs(sy(y)-sy(0))}/>)}</g>
          : s.kind === "scatter" ? <g key={i}>{s.points.map(([x,y],j)=><circle key={j} className={`cl-series s${i}`} cx={sx(x)} cy={sy(y)} r="5"/>)}</g>
          : <path key={i} className={`cl-series cl-line s${i}`} d={path(s.points)} />)}
        {v.xlabel && <text className="cl-axis" x={(m.l+W-m.r)/2} y={H-7} textAnchor="middle">{v.xlabel}</text>}
        {v.ylabel && <text className="cl-axis" transform={`translate(15 ${(m.t+H-m.b)/2}) rotate(-90)`} textAnchor="middle">{v.ylabel}</text>}
      </svg>
      <div className="cl-legend">{series.map((s,i)=><span key={i}><i className={`s${i}`}/>{s.label}</span>)}</div>
      {v.caption && <small><Inline text={v.caption} /></small>}
    </figure>
  );
}

function Visual({ v, index, hs, dispatch, selection, vectorBars }) {
  if (v.type === "grid") return <LabGrid {...{v,index,hs,dispatch,selection,vectorBars}}/>;
  if (v.type === "tokens") return <LabTokens {...{v,index,hs,dispatch,selection}}/>;
  if (v.type === "graph") return <LabGraph {...{v,index,hs,dispatch,selection}}/>;
  if (v.type === "plot") return <LabPlot {...{v,index}}/>;
  if (v.type === "equation") return <figure className={`cl-plane cl-equation tone-${index%4}`}><figcaption><span className="cl-kicker">EQUATION</span>Active relationship</figcaption><div role="math" aria-label={v.tex}><Tex display tex={v.by_sel?.[selection] || v.tex}/></div>{v.caption&&<small><Inline text={v.caption}/></small>}</figure>;
  return null;
}

export function ConceptLabView({ frame, state, dispatch, demo, demoId }) {
  const hasStory = hasConceptStory(demoId);
  const [mode, setMode] = useState(hasStory ? "idea" : "detail");
  const hs = highlightSet(frame, state.selection);
  const visuals = frame.views.filter((v) => !["text","code"].includes(v.type));
  const notes = frame.views.filter((v) => v.type === "text");
  const codes = frame.views.filter((v) => v.type === "code");
  return (
    <div className="cl-view" key={frame.id} data-frame={frame.id}>
      <div className="cl-stage">
        <div className="cl-stage-label"><span>LIVE CONCEPT MODEL</span><b>{frame.title}</b></div>
        {hasStory && <div className="cl-mode" role="group" aria-label="Explanation level"><button type="button" aria-pressed={mode==="idea"} onClick={()=>setMode("idea")}>Idea first</button><button type="button" aria-pressed={mode==="detail"} onClick={()=>setMode("detail")}>Inspect vectors</button></div>}
        <div className="cl-glow"/>
        {hasStory && mode === "idea" ? <div className="cl-visuals"><ConceptStory demoId={demoId} frameId={frame.id}/></div> : <div className={`cl-visuals ${visuals.length > 1 ? "is-multiple" : ""}`}>
          {(visuals.length ? visuals : notes).map((v,i) => v.type === "text"
            ? <p className="cl-empty" key={i}><Inline text={v.by_sel?.[state.selection] || v.text}/></p>
            : <Visual key={`${frame.id}-${i}`} v={v} index={i} hs={hs} dispatch={dispatch} selection={state.selection} vectorBars={hasStory}/>) }
        </div>}
      </div>
      {hasStory && mode === "idea"
        ? <div className="cl-observation"><span>What to notice</span><p>Track which object owns each role and where information moves. Select <strong>Inspect vectors</strong> only after the story is clear.</p></div>
        : notes.length > 0 && <div className="cl-observation"><span>What to notice</span>{notes.map((v,i)=><p key={i}><Inline text={v.by_sel?.[state.selection] || v.text}/></p>)}</div>}
      {codes.length > 0 && <div className="cl-code-dock">{codes.map((v,i)=><CodeView key={i} code={v.code} language={v.language || "Python"} label={v.label || "Source operation for this step"}
        highlights={(v.lines || []).map((l,n)=>({id:String(n),from:Array.isArray(l)?l[0]:l,to:Array.isArray(l)?l[1]:l,kind:"note"}))} scrollToActive={false} maxHeight="18rem"/>)}</div>}
      {state.selection && <button className="cl-selection" type="button" onClick={()=>dispatch({type:"select",selection:null})}>Tracing <code>{state.selection}</code><span>Clear ×</span></button>}
    </div>
  );
}
