import "./conceptStories.css";

const TOKENS = {
  mha: [
    { word:"The", idea:"points to a specific thing", role:"context" },
    { word:"cat", idea:"the animal being discussed", role:"subject" },
    { word:"sat", idea:"the action", role:"event" },
  ],
  source: [
    { word:"A", idea:"introduces one thing" }, { word:"red", idea:"describes colour" },
    { word:"fox", idea:"the source subject" }, { word:"jumps", idea:"the source action" },
  ],
  target: [{ word:"Le", idea:"target context" }, { word:"renard", idea:"asks for source evidence" }],
};

function Token({ item, tone="teal", active=false, small=false }) {
  return <div className={`cs-token ${tone} ${active?"active":""} ${small?"small":""}`}><b>{item.word}</b><span>{item.idea}</span>{item.role&&<i>{item.role}</i>}</div>;
}

function MhaStory({ step }) {
  const projected=step>=0, split=step>=1, linked=step>=2, weighted=step>=3, gathered=step>=4, joined=step>=5;
  return <div className="cs-story cs-mha" aria-label="Concept-first multi-head attention story">
    <div className="cs-note">Illustrative roles, not claims about learned embedding dimensions.</div>
    <div className="cs-row cs-words">{TOKENS.mha.map((t,i)=><Token key={t.word} item={t} active={linked&&i===2}/>)}</div>
    <div className={`cs-projections ${projected?"show":""}`}>
      <div className="cs-lane query"><strong>Q · asks</strong><span>“What information would help me?”</span></div>
      <div className="cs-lane key"><strong>K · offers</strong><span>“Here is what I can be matched by.”</span></div>
      <div className="cs-lane value"><strong>V · carries</strong><span>“Here is the information I can pass on.”</span></div>
    </div>
    <div className={`cs-heads ${split?"show":""}`}><div><b>Head A</b><span>tracks who relates to the action</span></div><div><b>Head B</b><span>tracks another relationship</span></div></div>
    {linked&&<svg className="cs-links" viewBox="0 0 800 110" aria-hidden="true"><path className={weighted?"strong":""} d="M645 92 C560 12 350 12 265 92"/><path d="M645 92 C610 44 570 44 530 92"/><circle cx="645" cy="92" r="6"/></svg>}
    <div className={`cs-result ${gathered?"show":""}`}><span>{weighted?"weighted ideas":"candidate ideas"}</span><b>“sat” gathers useful context from “cat”</b>{joined&&<em>Heads join into one contextual token representation.</em>}</div>
  </div>;
}

function CrossStory({ step }) {
  const project=step>=1, linked=step>=2, weighted=step>=3, output=step>=4;
  return <div className="cs-story cs-cross" aria-label="Concept-first cross-attention story">
    <div className="cs-note">Two streams stay separate: queries come from the target; keys and values come from the source.</div>
    <section><header>ENCODER MEMORY · source text</header><div className="cs-row">{TOKENS.source.map((t)=><Token key={t.word} item={t} tone="amber" small/>)}</div>{project&&<div className="cs-rolebar">K: how each source token can be found · V: what it can contribute</div>}</section>
    <div className="cs-cross-arrow">{linked?"target query searches source memory":"different sequence lengths are allowed"}</div>
    <section><header>DECODER · text being produced</header><div className="cs-row">{TOKENS.target.map((t,i)=><Token key={t.word} item={t} tone="violet" active={linked&&i===1} small/>)}</div>{project&&<div className="cs-rolebar">Q: what the current target token needs from the source</div>}</section>
    {linked&&<div className={`cs-evidence ${weighted?"weighted":""}`}><i/><i/><i className="best"/><i/><span>{weighted?"Softmax turns compatibility into a distribution over source tokens.":"Every target query compares with every allowed source key."}</span></div>}
    {output&&<div className="cs-result show"><b>Source evidence is blended into the target token</b><em>The decoder keeps its own identity; cross-attention adds relevant encoder information.</em></div>}
  </div>;
}

function RopeStory({ step }) {
  const angle=step===0?0:step===1?18:step===2?48:step===3?48:78;
  return <div className="cs-story cs-rope" aria-label="Concept-first rotary position embedding story">
    <div className="cs-note">The token idea stays; position changes the orientation of coordinate pairs.</div>
    <div className="cs-rope-words"><Token item={{word:"cat",idea:"token content before position"}}/><Token item={{word:"cat",idea:"same content at a later position"}} tone="violet"/></div>
    <div className="cs-compasses">
      <div><strong>position 1</strong><span className="cs-compass"><i style={{transform:`rotate(${angle}deg)`}}/></span><small>content pair rotated by θ</small></div>
      <div><strong>position 4</strong><span className="cs-compass violet"><i style={{transform:`rotate(${angle+72}deg)`}}/></span><small>same pair rotated by 4θ</small></div>
    </div>
    <div className="cs-rope-rule"><b>{step<2?"Pair coordinates, then compute the angle.":step===2?"Rotate each pair without stretching it.":step===3?"Length is preserved; direction carries position.":"Their dot product depends on the relative angle, 4θ − θ."}</b><span>RoPE changes phase, not the token’s vector norm.</span></div>
  </div>;
}

function CacheStory({ step }) {
  const projected=step>=1, appended=step>=2, reused=step>=3, attended=step>=4;
  const old=["The","cat","sat","on"], fresh="the";
  return <div className="cs-story cs-cache" aria-label="Concept-first KV cache story">
    <div className="cs-note">Generation example: “The cat sat on” → predict the next token. Cached cards are reused, not recomputed.</div>
    <div className="cs-prompt">{old.map((w)=><span key={w}>{w}</span>)}<span className="new">{fresh}</span><i>current position</i></div>
    <div className="cs-cache-shelf"><header>GPU memory · KV cache</header><div>{old.map((w,i)=><div key={w} className={reused?"reused":""}><b>{w}</b><span>K{i}</span><span>V{i}</span></div>)}{appended&&<div className="fresh"><b>{fresh}</b><span>K4</span><span>V4</span></div>}</div></div>
    {projected&&<div className="cs-current"><b>Only the new token is projected</b><span>new key locates it · new value carries its information</span></div>}
    {attended&&<div className="cs-query"><strong>current query</strong><i/><i/><i/><i/><i/><span>reads every cached key, then blends the corresponding values</span></div>}
    <div className={`cs-result ${attended?"show":""}`}><b>Work per step grows with cache length—not with re-projecting the whole prefix.</b></div>
  </div>;
}

const STORY = {
  "mha-steps": { ids:["project","split","scores","weights","output","concat"], View:MhaStory },
  "xattn-steps": { ids:["sequences","project","scores","weights","output"], View:CrossStory },
  "q331-mechanism": { ids:["concept-0","concept-1","concept-2","concept-3","concept-4"], View:RopeStory },
  "q391-mechanism": { ids:["concept-0","concept-1","concept-2","concept-3","concept-4"], View:CacheStory },
};

export function hasConceptStory(id) { return !!STORY[id]; }
export function ConceptStory({ demoId, frameId }) {
  const story=STORY[demoId];
  if(!story) return null;
  const step=Math.max(0,story.ids.indexOf(frameId));
  return <story.View step={step}/>;
}
