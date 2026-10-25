"""Mechanism-first visual lessons for the selected questions in chapters 13-36.

The hand-authored fields state each concept's central invariant.  The animation
stages and highlighted line ranges are derived from the supplied workbook source,
so a lesson cannot silently describe a different implementation.
"""
from __future__ import annotations

import ast
import html
import re

import kit
from kit import Article, code_view, edge, expl, frame, graph, node, text


# q316 and q328 already have detailed, numerically checked attention lessons in c16.py.
SELECTED = [
    248, 249, 267, 287, 294, 308, 331, 339, 354, 361, 374, 379,
    391, 397, 409, 413, 420, 429, 433, 439, 461, 462, 469, 476,
    485, 499, 503, 509, 515, 519, 525, 528, 542, 545, 549, 555,
    557, 562, 567, 573, 577, 581, 588, 592, 593, 598,
]

CHAPTER = {
    248: ("adaptive optimization", "First and second moments turn one gradient into a scale-aware update."),
    249: ("decoupled regularization", "AdamW shrinks parameters separately from the moment-normalized gradient update."),
    267: ("spatial scatter-add", "Each input pixel scatters a weighted kernel footprint into the larger output."),
    287: ("images as token sequences", "Patch projection changes a spatial grid into tokens; a class token carries the final summary."),
    294: ("learned subword construction", "BPE repeatedly counts adjacent pairs and merges the best pair everywhere."),
    308: ("packed-document isolation", "A valid attention edge must satisfy document equality and causal order at the same time."),
    331: ("position as rotation", "RoPE rotates adjacent coordinate pairs while preserving each pair's Euclidean norm."),
    339: ("causal decoder composition", "Token and position embeddings flow through masked pre-norm blocks into tied vocabulary logits."),
    354: ("low-rank adaptation", "LoRA adds a factored update BA to a frozen linear map, scaled by alpha divided by rank."),
    361: ("relative preference margins", "DPO compares policy preference with reference preference before applying a logistic loss."),
    374: ("geometric mean surprise", "Perplexity exponentiates average negative log probability over exactly the evaluated tokens."),
    379: ("sample-based success", "Pass@k measures the chance that a size-k subset contains a success; voting answers a different question."),
    391: ("autoregressive reuse", "A KV cache appends only new projected keys and values along the sequence axis."),
    397: ("speculative acceptance", "Draft tokens are accepted until the first target mismatch; one target bonus token is always available."),
    409: ("affine quantization", "Scale and zero point map a real interval onto a finite unsigned integer grid that contains real zero."),
    413: ("error-aware sequential quantization", "GPTQ-style compensation pushes each column's quantization error into later columns."),
    420: ("distributed gradient agreement", "DDP averages the same parameter gradient across workers before every worker updates."),
    429: ("roofline bottlenecks", "Compare compute time with weight-transfer time; the larger lower bound names the bottleneck."),
    433: ("shared-memory tiling", "A CUDA block reuses tiles of A and B while accumulating one output tile."),
    439: ("online softmax", "Flash-style attention updates a running maximum, denominator, and weighted value without an N-by-N matrix."),
    461: ("temporal credit with boundaries", "GAE carries advantage backward inside an episode and stops that recursion at every episode boundary."),
    462: ("bounded policy change", "PPO clips probability ratios only when an unclipped change would improve the sampled objective too far."),
    469: ("reconstruction versus latent regularity", "The negative ELBO adds reconstruction cost and beta-weighted KL for each example."),
    476: ("one reverse diffusion step", "A noise estimate produces a clean-sample estimate and then the previous noisy state."),
    485: ("term evidence saturation", "BM25 combines inverse document frequency with length-normalized, saturating term frequency."),
    499: ("time-respecting evaluation", "Purging separates training observations from test windows so overlapping labels cannot leak."),
    503: ("cross-view self-distillation", "Each teacher global view supervises different student views; aligned views are excluded."),
    509: ("teacher-student state updates", "A DINO step updates the student by gradient, the teacher by EMA, and the center from old teacher logits."),
    515: ("selective Q/V adaptation", "A fused QKV projection keeps K unchanged and adds separate low-rank paths only to Q and V."),
    519: ("shifted-window validity", "Tokens may interact only when they are real and share the same unwrapped shifted region."),
    525: ("late interaction", "ColBERT finds each query token's best document-token match and then sums over valid query tokens."),
    528: ("sparse vocabulary evidence", "SPLADE activates token logits, max-pools over tokens, and penalizes mass and corpus-wide activation."),
    542: ("metric-weighted ranking updates", "LambdaRank weights pair pressure by the NDCG change that swapping the pair would cause."),
    545: ("retrieval ceiling", "A reranker can reorder retrieved candidates, but it cannot recover a relevant item that retrieval omitted."),
    549: ("point-in-time correctness", "An as-of join accepts a version only when both event time and availability time precede the cutoff."),
    555: ("rolling-origin fairness", "Competing forecasts use identical origins; every forecast cell contributes to pooled WAPE."),
    557: ("adapter delta composition", "Merge complete scaled BA deltas; separately averaging factors creates cross-terms that were never trained."),
    562: ("token-weighted accumulation", "Sum all valid token losses and divide once by the global valid-token count."),
    567: ("clipped noisy aggregation", "Sum already clipped per-example gradients, add Gaussian noise, then apply the supplied normalization."),
    573: ("one DP-SGD numerical step", "Compute per-example gradients, clip jointly, aggregate with noise, and update once; accounting remains external."),
    577: ("two different bottlenecks", "Additive noise perturbs a representation; a variational bottleneck also pays KL toward a prior."),
    581: ("encoder-decoder with skips", "U-Net preserves fine spatial detail by concatenating aligned encoder features into the decoder."),
    588: ("durable workflow state", "A transaction claims one dependency-ready node; explicit recovery returns interrupted work to pending."),
    592: ("effect/checkpoint crash gaps", "A crash after an effect but before its checkpoint is safe only when replayed effects are idempotent."),
    593: ("collision-tolerant counting", "A Count-Min Sketch updates one counter per hash row and estimates by the minimum counter."),
    598: ("streaming commitment boundary", "Failover is safe before nonempty output; after output begins, switching providers would splice responses."),
}

EQUATION = {
    248: r"m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2",
    249: r"\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)",
    267: r"Y_{n,o,is-p+a,js-p+b}\mathrel{+}=X_{n,c,i,j}W_{c,o,a,b}",
    287: r"N_{tokens}=(H/P)^2+1,\quad z_{out}=\operatorname{LN}(z_L[:,0])",
    294: r"(a,b)^*=\arg\max_{(a,b)}\operatorname{count}(a,b)",
    308: r"M_{ij}=[d_i=d_j]\land[i\ge j]",
    331: r"(x',y')=(x\cos\theta-y\sin\theta,\ x\sin\theta+y\cos\theta)",
    339: r"\operatorname{logits}=\operatorname{LN}(h_L)E^\top",
    354: r"y=xW^\top+\frac{\alpha}{r}(xA^\top)B^\top",
    361: r"\ell=-\log\sigma\{\beta[(\log\pi_c-\log\pi_r)-(\log\pi^0_c-\log\pi^0_r)]\}",
    374: r"\operatorname{PPL}=\exp\left(-\frac1N\sum_{t=1}^N\log p_t\right)",
    379: r"\operatorname{pass@k}=1-\binom{n-c}{k}/\binom nk",
    391: r"K_{cache}\leftarrow\operatorname{concat}(K_{cache},K_{new};\,axis=T)",
    397: r"output=target[:j+1]\quad\text{at the first }draft_j\ne target_j",
    409: r"q=\operatorname{clip}(\operatorname{round}(x/s)+z,0,2^b-1),\quad \hat x=s(q-z)",
    413: r"e_i=(w_i-\hat w_i)/U_{ii},\quad W_{i:}\leftarrow W_{i:}-e_iU_{i,i:}",
    420: r"\bar g_p=\frac1R\sum_{r=1}^{R}g_{r,p}",
    429: r"t\ge\max(\mathrm{FLOPs}/F_{peak},\ \mathrm{bytes}/B_{mem})",
    433: r"C_{ij}=\sum_t\sum_{k\in tile_t}A_{ik}B_{kj}",
    439: r"m'=\max(m,s),\ l'=e^{m-m'}l+e^{s-m'},\ a'=e^{m-m'}a+e^{s-m'}v",
    461: r"\delta_t=r_t+\gamma(1-terminal_t)V_{t+1}-V_t,\quad A_t=\delta_t+\gamma\lambda(1-end_t)A_{t+1}",
    462: r"L=-\mathbb E[\min(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)]",
    469: r"-ELBO=BCE(x,\hat x)+\frac\beta2\sum_j(\mu_j^2+e^{\log\sigma_j^2}-1-\log\sigma_j^2)",
    476: r"\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta)/\sqrt{\bar\alpha_t}",
    485: r"BM25=\sum_q IDF(q)\frac{tf(k_1+1)}{tf+k_1(1-b+b|D|/avgdl)}",
    499: r"train\_end < test\_start-gap",
    503: r"L=\frac1{|P|}\sum_{(t,v):t\ne v}H(p^{teacher}_t,p^{student}_v)",
    509: r"\theta_T\leftarrow m\theta_T+(1-m)\theta_S^{new}",
    515: r"Q'=Q+sB_QA_Qx,\quad K'=K,\quad V'=V+sB_VA_Vx",
    519: r"M_{ab}=valid_a\land valid_b\land region_a=region_b",
    525: r"score(q,d)=\sum_{i:mask_q}\max_{j:mask_d}\langle\hat q_i,\hat d_j\rangle",
    528: r"w_v=\max_t\log(1+\operatorname{ReLU}(logit_{t,v}))",
    542: r"\lambda_{ij}\propto |\Delta NDCG_{ij}|\,\sigma(1-\sigma(s_i-s_j))",
    545: r"NDCG\le f(\text{relevant documents present in candidates})",
    549: r"event\_time\le cutoff\ \land\ available\_time\le cutoff",
    555: r"WAPE=\sum|y-\hat y|/\sum|y|",
    557: r"W'=W+\sum_i c_i(\alpha_i/r_i)B_iA_i",
    562: r"L=\frac{\sum_b\sum_{t:mask_{bt}}\ell_{bt}}{\sum_b\sum_t mask_{bt}}",
    567: r"\tilde g=(\sum_i\bar g_i+\mathcal N(0,\sigma^2C^2I))/D",
    573: r"\bar g_i=g_i\min(1,C/\|g_i\|_2)",
    577: r"z=\mu+\exp(\tfrac12\log\sigma^2)\epsilon,\quad KL=\tfrac12\sum(\mu^2+\sigma^2-1-\log\sigma^2)",
    581: r"decoder_l=block([upsample(decoder_{l+1});\ encoder_l])",
    588: r"pending\rightarrow running\rightarrow\{done,failed\}",
    592: r"effect\rightarrow checkpoint;\quad crash\ between\ them\Rightarrow replay",
    593: r"\hat f(x)=\min_r table[r,h_r(x)]\ge f(x)",
    598: r"emitted=0\Rightarrow retry;\quad emitted>0\Rightarrow terminal\ partial",
}

# These labels describe the conceptual mechanism, not merely the number of
# statements in the reference function. They keep short implementations from
# collapsing a deep idea into "enter -> return".
FLOW = {
    248: ["Read gradient", "Update first moment", "Update second moment", "Correct startup bias", "Scale parameter step"],
    249: ["Read prior state", "Update moments", "Correct bias", "Shrink parameter", "Apply adaptive step"],
    267: ["Choose input pixel", "Place kernel footprint", "Multiply channels", "Scatter-add overlaps", "Crop padding boundary"],
    287: ["Cut image patches", "Project patch tokens", "Add class and positions", "Mix tokens in encoder", "Normalize class token"],
    294: ["Split into characters", "Count adjacent pairs", "Choose frequent pair", "Merge every occurrence", "Repeat with new symbols"],
    308: ["Label document IDs", "Compare document identity", "Compare token order", "Combine both permissions", "Return causal block mask"],
    331: ["Pair adjacent coordinates", "Compute position angles", "Rotate every pair", "Preserve pair norm", "Expose relative phase"],
    339: ["Embed token and position", "Apply causal mask", "Run pre-norm blocks", "Normalize final states", "Project with tied weights"],
    354: ["Run frozen base path", "Project into rank r", "Project back to output", "Scale low-rank delta", "Add base and delta"],
    361: ["Measure policy preference", "Measure reference preference", "Subtract reference margin", "Scale by beta", "Apply stable softplus"],
    374: ["Select evaluated tokens", "Read token log probabilities", "Average log probability", "Negate mean surprise", "Exponentiate perplexity"],
    379: ["Count samples and successes", "Count all size-k subsets", "Count all-failure subsets", "Subtract failure probability", "Vote answers separately"],
    391: ["Process prompt tokens", "Project new keys and values", "Append along token axis", "Reuse all old cache entries", "Attend with current query"],
    397: ["Draft several tokens", "Run target verification", "Accept matching prefix", "Correct first mismatch", "Use bonus if all match"],
    409: ["Find real range with zero", "Choose integer grid", "Compute scale and zero-point", "Round and clip integers", "Dequantize for comparison"],
    413: ["Build damped Hessian", "Factor inverse curvature", "Quantize one column", "Measure scaled error", "Compensate later columns"],
    420: ["Compute worker gradients", "Align same parameters", "Stack worker values", "Average elementwise", "Share identical mean"],
    429: ["Count operation work", "Count weight bytes", "Compute ideal compute time", "Compute ideal memory time", "Choose larger lower bound"],
    433: ["Map thread to output", "Load A and B tiles", "Synchronize the block", "Accumulate tile products", "Store valid output cell"],
    439: ["Load one query", "Stream a key/value tile", "Update running maximum", "Rescale old accumulator", "Normalize final weighted value"],
    461: ["Compute one-step TD residuals", "Start at rollout end", "Carry advantage backward", "Stop at episode boundary", "Add values for returns"],
    462: ["Compute log-ratio", "Exponentiate probability ratio", "Form unclipped objective", "Form clipped objective", "Choose pessimistic term"],
    469: ["Score reconstruction pixels", "Sum per example", "Compute latent KL", "Scale KL by beta", "Average negative ELBO"],
    476: ["Read noisy state and schedule", "Predict added noise", "Estimate clean sample", "Compute previous mean", "Add scheduled noise"],
    485: ["Tokenize query and documents", "Count term frequency", "Compute inverse document frequency", "Normalize document length", "Sum saturated term scores"],
    499: ["Choose rolling test origin", "Reserve purge gap", "Keep earlier training rows", "Move origin forward", "Evaluate without overlap leakage"],
    503: ["Create teacher global targets", "Create student view logits", "Skip aligned view pair", "Score cross-view entropy", "Average valid pairs"],
    509: ["Create or load teacher", "Compute old teacher targets", "Update student by loss", "EMA-update teacher", "Update center from old logits"],
    515: ["Freeze fused QKV base", "Split Q, K, and V", "Compute low-rank Q delta", "Leave K unchanged", "Compute low-rank V delta"],
    519: ["Pad spatial grid", "Apply cyclic shift", "Recover original regions", "Partition shifted windows", "Allow same real region only"],
    525: ["Normalize token vectors", "Compute every token similarity", "Mask padded document tokens", "Take max per query token", "Sum valid query maxima"],
    528: ["Activate vocabulary logits", "Mask padded tokens", "Max-pool across tokens", "Measure document L1 mass", "Measure corpus FLOPS penalty"],
    542: ["Freeze current ranks", "Compute gains and discounts", "Measure swap NDCG change", "Scale pair pressure", "Accumulate item gradients"],
    545: ["Retrieve candidate IDs", "Deduplicate candidates", "Score candidates", "Stable-sort reranked list", "Measure recall and NDCG"],
    549: ["Read entity and cutoff", "Match entity versions", "Reject late event times", "Reject late availability", "Choose greatest eligible version"],
    555: ["Choose common forecast origins", "Require enough history", "Generate both forecasts", "Record overlapping cells", "Pool errors into WAPE"],
    557: ["Copy base weight", "Form each B times A", "Scale by alpha over rank", "Weight each complete delta", "Sum deltas into base"],
    562: ["Count all valid tokens", "Select valid losses", "Sum each micro-batch", "Divide by global count", "Backpropagate each contribution"],
    567: ["Receive clipped example gradients", "Sum each parameter tensor", "Sample Gaussian noise", "Scale noise by sigma C", "Divide by denominator"],
    573: ["Select example subset", "Compute per-example gradients", "Clip joint norms", "Add Gaussian aggregation", "Apply one optimizer step"],
    577: ["Choose bottleneck mode", "Sample caller noise", "Perturb or reparameterize", "Compute KL when variational", "Use mean at evaluation"],
    581: ["Encode high-resolution features", "Downsample into context", "Process bottleneck", "Upsample and concatenate skips", "Predict one logit per pixel"],
    588: ["Validate dependency DAG", "Persist matching configuration", "Claim one ready node", "Finish terminal state", "Recover interrupted claims"],
    592: ["Start one logical operation", "Cross effect boundary", "Crash at chosen point", "Restart worker", "Replay idempotently or duplicate"],
    593: ["Hash key once per row", "Increment selected counters", "Accumulate collisions", "Read same positions", "Return minimum estimate"],
    598: ["Choose first provider", "Read streamed chunks", "Classify failure timing", "Fail over before output only", "Return complete or partial status"],
}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def target_function(tree: ast.Module, n: int):
    preferred = f"q{n:03d}"
    funcs = [x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return next((x for x in funcs if x.name == preferred), funcs[-1])


def stage_label(stmt: ast.stmt) -> str:
    if isinstance(stmt, ast.If): return "Validate the contract"
    if isinstance(stmt, (ast.For, ast.While)): return "Repeat the core update"
    if isinstance(stmt, ast.Return): return "Return the promised result"
    if isinstance(stmt, ast.ClassDef): return "Build the module structure"
    if isinstance(stmt, ast.FunctionDef): return f"Define helper {stmt.name}"
    if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        raw = ast.unparse(stmt).split("=", 1)[0].strip()
        return f"Compute {raw[:44]}"
    if isinstance(stmt, ast.Expr): return "Apply the operation"
    return type(stmt).__name__.replace("_", " ")


def source_stages(n: int):
    source = kit.solution_source(n)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        lines = source.splitlines()
        significant = [(i + 1, line.strip()) for i, line in enumerate(lines)
                       if line.strip() and not line.lstrip().startswith(('#', '//', '#include'))]
        anchors = []
        for i, line in significant:
            if any(word in line for word in ('__global__', 'for(', 'for (', '__syncthreads', 'if(', 'if (', '<<<')):
                anchors.append((i, line))
        anchors = anchors[:6] or significant[:6]
        stages = [("contract", "Read inputs and output", anchors[0][0], anchors[0][0], "Enter the CUDA function with the dimensions and memory contract in the question.")]
        for j, (line_no, snippet) in enumerate(anchors):
            label = ("Map threads to output" if j == 0 else
                     "Load or update one tile" if 'for' in snippet or '__syncthreads' in snippet else
                     "Guard the boundary" if 'if' in snippet else
                     "Launch the kernel")
            stages.append((f"s{j}", label, line_no, line_no, snippet[:180]))
        stages.append(("result", "Return the promised result", len(lines), len(lines), "The kernel writes the output buffer defined by the contract."))
        return source, type("CudaFunction", (), {"name": "solve", "lineno": anchors[0][0], "end_lineno": len(lines), "args": type("Args", (), {"args": []})()})(), stages[:8]
    fn = target_function(tree, n)
    statements = list(fn.body)
    chosen = []
    # Keep the important endpoints and spread intermediate stages over the body.
    for stmt in statements:
        if isinstance(stmt, (ast.If, ast.For, ast.While, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.ClassDef, ast.FunctionDef, ast.Expr, ast.Return)):
            chosen.append(stmt)
    if len(chosen) > 6:
        idx = sorted(set([0, 1, len(chosen)//3, 2*len(chosen)//3, len(chosen)-2, len(chosen)-1]))
        chosen = [chosen[i] for i in idx]
    stages = [("contract", "Read inputs and output", fn.lineno, fn.lineno, f"Enter `{fn.name}` with the shapes and conditions in the question.")]
    for i, stmt in enumerate(chosen):
        snippet = ast.unparse(stmt).replace("\n", " ")
        stages.append((f"s{i}", stage_label(stmt), stmt.lineno, getattr(stmt, "end_lineno", stmt.lineno), snippet[:180]))
    if stages[-1][1] != "Return the promised result":
        stages.append(("result", "Return the promised result", fn.end_lineno, fn.end_lineno, "Return the output defined by the exercise contract."))
    if len(stages) < 3:
        stages.insert(1, ("math", "Apply the central equation", fn.lineno, fn.end_lineno,
                          "Evaluate the expression while preserving the documented axis and scaling roles."))
    return source, fn, stages[:8]


def boundary_sentence(prompt: str) -> str:
    candidates = re.split(r"(?<=[.!?])\s+", prompt.replace("\n", " "))
    for c in candidates:
        if re.search(r"\b(require|invalid|empty|mask|padding|boundary|must|only|without|do not|caller|tie|odd|zero)\b", c, re.I):
            return c.strip()
    return "Change one input at the edge of the stated contract and trace where the behavior changes."


def roles(fn) -> str:
    names = [a.arg for a in fn.args.args]
    if not names: return "The function has no explicit input arguments."
    return "The inputs are " + ", ".join(f"`{x}`" for x in names) + ". Keep their roles separate while you trace the solution."


def pipeline_svg(title, labels):
    labels = labels[:6]
    w, h, boxw = 980, 220, 132
    gap = (w - 80 - boxw * len(labels)) / max(1, len(labels) - 1)
    colors = ["#d9efeb", "#ffedcf", "#ece2ff", "#dfe8f4", "#e8f0dc", "#f7e0e5"]
    strokes = ["#08776e", "#9a5d00", "#7139ba", "#385f8f", "#557a31", "#9a3e54"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img">',
             '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10z" fill="#738197"/></marker></defs>',
             f'<rect width="{w}" height="{h}" rx="14" fill="#f5f8fa"/><text x="40" y="38" font-family="Georgia,serif" font-size="20" font-weight="700" fill="#14223e">{html.escape(title)}</text>']
    xs = []
    for i, lab in enumerate(labels):
        x = 40 + i * (boxw + gap); xs.append(x)
        if i: parts.append(f'<line x1="{xs[i-1]+boxw}" y1="130" x2="{x-12}" y2="130" stroke="#738197" stroke-width="2" marker-end="url(#a)"/>')
        words = lab.split(); lines = [" ".join(words[:3]), " ".join(words[3:6]), " ".join(words[6:])]
        parts.append(f'<rect x="{x}" y="88" width="{boxw}" height="84" rx="10" fill="{colors[i%6]}" stroke="{strokes[i%6]}"/>')
        for j, line in enumerate([q for q in lines if q]):
            parts.append(f'<text x="{x+boxw/2}" y="{116+j*18}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#24344d">{html.escape(line[:22])}</text>')
    parts.append('</svg>')
    return "".join(parts)


def conceptual_stages(n, source_stages_):
    flow = FLOW[n]
    out = []
    for i, label in enumerate(flow):
        # Map conceptual progress monotonically onto the parsed source ranges.
        src = source_stages_[round(i * (len(source_stages_) - 1) / max(1, len(flow) - 1))]
        out.append((f"concept-{i}", label, src[2], src[3], src[4]))
    return out


def make_frames(n, source, stages, mode="baseline"):
    title = kit.title_of(n)
    nodes = [node(sid, 85 + i * (760 / max(1, len(stages)-1)), 105, label[:22], sub=("current" if i == 0 else None), sel=sid, cls=("active" if i == 0 else "")) for i, (sid, label, *_rest) in enumerate(stages)]
    edges = [edge(stages[i][0], stages[i+1][0], "data") for i in range(len(stages)-1)]
    boundary = boundary_sentence(kit.record(n)["question"])
    result = []
    for active, (sid, label, lo, hi, snippet) in enumerate(stages):
        ns = []
        for i, nd in enumerate(nodes):
            q = dict(nd); q["sub"] = "now" if i == active else ("done" if i < active else "next"); q["cls"] = "src" if i == active else ("dim" if i > active else ""); ns.append(q)
        changed = f"{label}. The active source operation is `{snippet}`."
        if mode == "boundary": changed += f" Boundary focus: {boundary}"
        result.append(frame(sid, label, changed,
            CHAPTER[n][1],
            "The exercise contract and all earlier computed values remain fixed.",
            graph(ns, edges, f"Execution map for {title}; the active stage is {label}.", width=860, height=210, node_w=108, node_h=56),
            code_view(source, [[lo, hi]], language="C++" if kit.record(n)["language"] != "Python" else "Python", label=f"Supplied solution, lines {lo}-{hi}"),
            text(("Baseline trace." if mode == "baseline" else "Boundary trace.") + " Select a node to keep its identity visible across the diagram.")))
    return result


def article(n: int):
    record = kit.record(n); title = record["title"]; prompt = " ".join(record["question"].split())
    source, fn, source_trace = source_stages(n)
    stages = conceptual_stages(n, source_trace)
    aid = f"lesson-q{n:03d}-{slug(title)}"
    a = Article(aid, title, exercises=[(n, "deep-lesson")])
    a.h2("The question in plain language")
    a.p(CHAPTER[n][1] + " The task is: " + prompt)
    a.p(roles(fn) + " The animation follows the supplied implementation, so every highlighted operation connects to code you can inspect.")
    a.callout("definition", CHAPTER[n][1], title=f"Core idea: {CHAPTER[n][0].capitalize()}")
    a.figure("mechanism", pipeline_svg(title, FLOW[n]),
             f"A left-to-right mechanism map for {title}. The stages are " + ", ".join(x[1] for x in stages) + ".",
             "The static map shows the same stages as the animation. It remains useful when motion is disabled.")
    a.h2("The mathematical contract")
    a.eq(EQUATION[n], "This equation names the central relationship. The prose below connects its symbols to the function inputs.")
    a.p("Read each symbol as a value with a specific axis, time, rank, token, or state role. The code must preserve those roles. A compact formula does not remove the validation, masking, boundary, or state rules in the prompt.")
    a.p("Before following the implementation, identify the output's shape and meaning. Then identify which input values can affect one selected output. This gives you a testable mental model before the code adds implementation detail.")
    a.h2("Watch the mechanism")
    a.demo(f"q{n:03d}-mechanism", f"Trace question {n:03d} from contract to returned value.",
           lambda mode: make_frames(n, source, stages, mode),
           params=[("mode", "Trace", [("baseline", "Baseline"), ("boundary", "Boundary case")])],
           help=[{"step_id": stages[min(1,len(stages)-1)][0], "help_type": "code-connection", "content": "The highlighted lines implement the active stage. Follow the data-flow arrow into the next stage, then inspect the returned value's meaning."},
                 {"step_id": stages[-1][0], "help_type": "common-mistake", "content": boundary_sentence(prompt)}],
           facts=[("Stage order and highlighted line ranges come from the parsed supplied solution.", "workbook source AST"),
                  ("The equation and core invariant were reviewed against the exercise contract.", "question prompt and supplied solution")])
    a.h2("What changes in the boundary case")
    a.p(boundary_sentence(prompt) + " Use the Trace control to switch between the normal path and this boundary focus. The algorithm does not get permission to invent an unstated fallback.")
    a.p("A useful test changes one condition while holding the remaining inputs fixed. Compare the output shape, state transition, valid edges, selected tokens, or returned status. This isolates the reason for the change.")
    a.callout("mistake", "A diagram can show data moving and still hide a wrong contract. Check the mask direction, reduction axis, update order, and ownership of persistent state against the prompt.", title="Do not learn the motion without the rule")
    a.h2("Connect the explanation to the supplied solution")
    notes = []
    used_ranges = set()
    for _sid, label, lo, hi, snippet in stages:
        if (lo, hi) in used_ranges:
            continue
        used_ranges.add((lo, hi))
        match = source.splitlines()[lo-1].strip()[:28]
        notes.append(((lo, hi), match, f"{label}. This is the code stage shown in the mechanism animation."))
        if len(notes) == 4:
            break
    a.walkthrough(n, notes, "Walk through the original solution")
    a.h2("Check your model")
    labels = [x[1] for x in stages]
    correct = labels[-1]
    wrong = labels[0] if len(labels) > 1 else "Skip the contract"
    a.check(f"q{n:03d}-predict", "prediction", "Which stage must agree exactly with the exercise's promised output?", "choice", correct,
            expl("Yes. The returned object is where shape, value, and state promises become observable.", "Trace the data to the final returned object."), options=[correct, wrong, "Ignore the return value"])
    a.check(f"q{n:03d}-new", "new-case", "You change only the boundary condition described above. What is the best next action?", "choice", "Trace the first stage whose condition changes",
            expl("Correct. Holding other inputs fixed isolates cause and effect.", "Change one condition and locate the first divergent stage."),
            options=["Trace the first stage whose condition changes", "Rewrite the mathematical contract", "Assume the output is unchanged"])
    a.check(f"q{n:03d}-explain", "explain-back", f"Explain {CHAPTER[n][0]} in your own words. Name one input, one intermediate state, and the output meaning.", "text", None,
            expl("A strong answer preserves the input roles, the central invariant, and the output contract.", "Use the animation stages as a scaffold."))
    a.h2("A practical verification strategy")
    a.ol(["Choose the smallest input that still exercises the mechanism.", "Compute one output or transition by hand.", "Run the supplied solution on the same input.", "Change one mask, boundary, scale, or state condition.", "Check the first stage where the result diverges."])
    a.p("This lesson explains the educational core in the workbook. It does not turn a small reference function into a complete production system, benchmark, privacy guarantee, distributed runtime, or trained model.")
    return a


def build():
    assert set(SELECTED) == set(CHAPTER) == set(EQUATION) == set(FLOW)
    out = []
    for n in SELECTED:
        a = article(n)
        # Structural assurance before the shared importer performs its full validation.
        assert a.exercises == [{"id": n, "purpose": "deep-lesson", "order": 1}]
        assert len(a.demos[0]["data"]["cases"]) == 2
        assert len(a.demos[0]["steps"]) >= 3
        out.append(a)
    return out
