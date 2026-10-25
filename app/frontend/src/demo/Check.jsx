import { useState } from "react";
import { Inline } from "../Inline.jsx";
import { judge } from "./check.js";

const KIND = { prediction: "Predict before you look", "new-case": "Try a new case", "explain-back": "Explain it back" };

export function CheckBlock({ id, check }) {
  const [guess, setGuess] = useState("");
  const [result, setResult] = useState(null);   // { correct, message } | { empty } | { revealed }
  const [revealed, setRevealed] = useState(false);
  if (!check) return <p role="alert" className="error">Check “{id}” is not available.</p>;
  const free = check.kind === "explain-back";
  const reset = () => { setGuess(""); setResult(null); setRevealed(false); };

  return (
    <form className="check" onSubmit={(e) => { e.preventDefault(); setResult(judge(check, guess)); }} aria-labelledby={`${id}-q`}>
      <h3>{KIND[check.kind]}</h3>
      <p id={`${id}-q`}><Inline text={check.prompt} /></p>
      {check.case && (
        <p className="case">{Object.entries(check.case).map(([k, v]) => <span key={k}>{k}: <code>{JSON.stringify(v)}</code> </span>)}</p>
      )}
      {check.answer_type === "choice" && !free && (
        <fieldset><legend className="sr-only">Choose one</legend>
          {check.options.map((o) => (
            <label key={o} className="opt"><input type="radio" name={id} value={o} checked={guess === o} onChange={() => setGuess(o)} /> {o}</label>
          ))}
        </fieldset>
      )}
      {(check.answer_type === "shape" || check.answer_type === "number") && (
        <label>Your answer
          <input value={guess} onChange={(e) => setGuess(e.target.value)} inputMode={check.answer_type === "number" ? "decimal" : "text"}
            placeholder={check.answer_type === "shape" ? "(2, 4, 3) or error" : "number"} />
        </label>
      )}
      {check.answer_type === "text" && (
        <label>Your explanation <textarea rows={3} value={guess} onChange={(e) => setGuess(e.target.value)} /></label>
      )}
      <div className="actions">
        {!free && <button type="submit">Check my answer</button>}
        <button type="button" onClick={() => setRevealed(true)}>{free ? "Show a model explanation" : "Show the answer"}</button>
        {(result || revealed) && <button type="button" onClick={reset}>Try again</button>}
      </div>
      <div role="status" aria-live="polite" className="feedback">
        {result?.empty && <p>Enter an answer first, or choose “Show the answer”.</p>}
        {result && !result.empty && (
          <p className={result.correct ? "right" : "wrong"}>
            <strong>{result.correct ? "Correct. " : "Not quite. "}</strong><Inline text={result.message} />
          </p>
        )}
        {revealed && (
          <p className="answer">
            {!free && <><strong>Answer: </strong><code>{String(check.answer.value)}</code>. </>}
            <Inline text={check.explanation.correct} />
          </p>
        )}
      </div>
    </form>
  );
}
