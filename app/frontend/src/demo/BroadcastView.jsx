import { Tex } from "../Inline.jsx";
import { CODE, MAX_DIMS, MAX_SIZE, fmt, resultShape } from "./broadcast.js";

const STATUS_TEXT = { equal: "equal", "stretch A": "A has size 1 → stretches", "stretch B": "B has size 1 → stretches", conflict: "CONFLICT" };

function Matrix({ label, rows, mark, onPick, selected, name }) {
  return (
    <figure className="matrix">
      <figcaption>{label}</figcaption>
      <table>
        <tbody>
          {rows.map((r, j) => (
            <tr key={j}>
              {r.map((v, k) => {
                const hit = mark && mark[0] === j && mark[1] === k;
                const label = `${name}[${j}, ${k}] = ${v}${hit ? " (traced)" : ""}`;
                return (
                  <td key={k} className={hit ? "hit" : ""} aria-current={hit ? "true" : undefined}>
                    {onPick ? (
                      <button type="button" onClick={() => onPick(j, k)} aria-label={label} aria-pressed={!!hit}>{v}</button>
                    ) : (
                      <span aria-label={label}>{v}</span>
                    )}
                    {hit && <span className="mark" aria-hidden="true">◂</span>}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}

export function BroadcastInputs({ inputs, dispatch }) {
  const edit = (name, shape) => dispatch({ type: "setInput", name, value: shape });
  return (
    <fieldset className="inputs">
      <legend>Edit the shapes (each axis has 1 to {MAX_SIZE} cells)</legend>
      {["shapeA", "shapeB"].map((name) => {
        const shape = inputs[name];
        const letter = name === "shapeA" ? "A" : "B";
        return (
          <div key={name} className="shape-row">
            <label>Axes in {letter}
              <select value={shape.length} onChange={(e) => {
                const n = Number(e.target.value);
                edit(name, n <= shape.length ? shape.slice(shape.length - n) : [...Array(n - shape.length).fill(1), ...shape]);
              }}>
                {Array.from({ length: MAX_DIMS }, (_, n) => <option key={n} value={n + 1}>{n + 1}</option>)}
              </select>
            </label>
            {shape.map((size, d) => (
              <label key={d}>{letter} axis {d}
                <select value={size} onChange={(e) => edit(name, shape.map((s, x) => (x === d ? Number(e.target.value) : s)))}>
                  {Array.from({ length: MAX_SIZE }, (_, n) => <option key={n} value={n + 1}>{n + 1}</option>)}
                </select>
              </label>
            ))}
            <span className="shape-text">shape {fmt(shape)}</span>
          </div>
        );
      })}
    </fieldset>
  );
}

function CodeView({ lines }) {
  return (
    <pre className="codeview" aria-label="Code for this step">
      {CODE.map((line, n) => (
        <code key={n} className={lines.includes(n) ? "on" : ""}>{lines.includes(n) ? "▶ " : "  "}{line}{"\n"}</code>
      ))}
    </pre>
  );
}

export function AxesTable({ axes }) {
  return (
    <table className="plain axes">
      <caption>Axes are compared from the right</caption>
      <thead><tr><th scope="col">Axis from right</th><th scope="col">Size in A</th><th scope="col">Size in B</th><th scope="col">Result</th></tr></thead>
      <tbody>
        {[...axes].reverse().map((x) => (
          <tr key={x.axis} className={x.status === "conflict" ? "conflict" : ""}>
            <th scope="row">{x.fromRight}</th><td>{x.a}</td><td>{x.b}</td>
            <td>{x.status === "conflict" ? "CONFLICT: sizes differ and neither is 1" : `${STATUS_TEXT[x.status]} → ${x.size}`}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function BroadcastView({ frame, state, dispatch }) {
  const v = frame.view;
  return (
    <div className="bview">
      {v.kind === "align" && (
        <table className="plain">
          <caption>Shapes padded with leading 1s</caption>
          <thead><tr><th scope="col" /><th scope="col">Axis 0</th><th scope="col">Axis 1</th><th scope="col">Axis 2</th></tr></thead>
          <tbody>
            {[["A", v.padA, v.shapeA], ["B", v.padB, v.shapeB]].map(([n, p, orig]) => (
              <tr key={n}>
                <th scope="row">{n} {fmt(orig)}</th>
                {p.map((s, d) => <td key={d}>{s}{d < MAX_DIMS - orig.length && <span className="sub inline"> (added)</span>}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {v.kind === "axes" && <AxesTable axes={v.axes} />}
      {v.kind === "result" && (
        <>
          <AxesTable axes={v.axes} />
          <p className="big">Result shape: <strong>{fmt(v.shape)}</strong></p>
        </>
      )}
      {v.kind === "trace" && (
        <>
          {v.shape[0] > 1 && (
            <div role="group" aria-label="Choose output slice i" className="slicer">
              Output slice i:
              {Array.from({ length: v.shape[0] }, (_, i) => (
                <button key={i} type="button" aria-pressed={v.sel.i === i}
                  onClick={() => dispatch({ type: "select", selection: { ...v.sel, i } })}>{i}</button>
              ))}
            </div>
          )}
          <div className="matrices">
            <Matrix name="A" label={`A at axis-0 index ${v.iaSlice}`} rows={v.sliceA} mark={[v.ia[1], v.ia[2]]} />
            <Matrix name="B" label={`B at axis-0 index ${v.ibSlice}`} rows={v.sliceB} mark={[v.ib[1], v.ib[2]]} />
            <Matrix name="C" label={`C = A + B at i = ${v.sel.i} (select a cell)`} rows={v.sliceC} mark={[v.sel.j, v.sel.k]}
              onPick={(j, k) => dispatch({ type: "select", selection: { i: v.sel.i, j, k } })} />
          </div>
          <div className="eq" aria-label={`C[${v.sel.i},${v.sel.j},${v.sel.k}] equals ${v.a} plus ${v.b} equals ${v.out}`}>
            <Tex display tex={`C_{${v.sel.i},${v.sel.j},${v.sel.k}} = A_{${v.ia}} + B_{${v.ib}} = ${v.a} + ${v.b} = ${v.out}`} />
          </div>
        </>
      )}
      <CodeView lines={frame.codeLines} />
    </div>
  );
}

// Compact view for the "compare two cases" block.
export function BroadcastCompareView({ inputs, computed }) {
  const axes = computed.frames[1].view.axes;
  return (
    <>
      <p className="big">{computed.summary.text}</p>
      <AxesTable axes={axes} />
    </>
  );
}

export { resultShape };
