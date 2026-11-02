"""Batch C16: attention. Questions 316 (multi-head attention) and 328 (cross-attention).
Embeddings and weights are small invented numbers. They show the arithmetic. They are not learned values."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kit
from kit import Article, cell, expl, frame, grid, matrix, num, plot, series, text, equation, code_view, tokens, vec

ATTN = ("Attention Is All You Need (Vaswani et al., 2017)", "https://arxiv.org/abs/1706.03762")
INVENTED = "The numbers in this lesson are small invented values chosen so that you can follow the arithmetic. They are not learned from data and they do not show real language relationships."


def attention_svg(cross=False):
    left = ["decoder Q"] if cross else ["token Q", "token K", "token V"]
    source = ["encoder K", "encoder V"] if cross else []
    boxes = left + source + ["scores", "softmax", "weighted V", "join heads", "output"]
    width, height = 1080, 225
    gap = (width - 80) / len(boxes)
    colors = ["#d9efeb", "#ffedcf", "#ece2ff", "#dfe8f4"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
           '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10z" fill="#687991"/></marker></defs>',
           '<rect width="1080" height="225" rx="14" fill="#f5f8fa"/>']
    for i, label in enumerate(boxes):
        x = 40 + i * gap
        if i:
            out.append(f'<line x1="{x-gap+94}" y1="126" x2="{x-10}" y2="126" stroke="#687991" stroke-width="2" marker-end="url(#arrow)"/>')
        out.append(f'<rect x="{x}" y="88" width="94" height="76" rx="10" fill="{colors[i%4]}" stroke="#60738b"/>')
        words = label.split()
        for j, line in enumerate([" ".join(words[:1]), " ".join(words[1:])]):
            if line:
                out.append(f'<text x="{x+47}" y="{120+j*18}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="13" fill="#14223e">{line}</text>')
    heading = "Cross-attention: queries and source information follow different paths" if cross else "Multi-head attention: parallel score-and-value paths are joined"
    out.append(f'<text x="40" y="38" font-family="Georgia,serif" font-size="20" font-weight="700" fill="#14223e">{heading}</text></svg>')
    return "".join(out)


# =========================================================================== prerequisites
def matmul_article():
    a = Article("dot-products-and-matmul", "Dot products and matrix products", kind="prerequisite", prerequisites=["arrays-and-shapes"])
    a.p("A **dot product** multiplies two vectors entry by entry and adds the results. For $u, v \\in \\mathbb{R}^d$: $u \\cdot v = \\sum_{k=1}^{d} u_k v_k$. The result is one number. It is large when the vectors point in a similar direction.")

    def frames(row):
        U = np.array([[1, 2, 0], [0, 1, 1]]); V = np.array([[2, 0, 1], [1, 1, 1], [0, 3, 1]])
        i = int(row)
        P = U @ V.T
        rid = lambda i, j: f"c{i}_{j}"
        return [
            frame("vectors", "Take one row of each matrix", f"Row {i} of U is {vec(U[i])}. The rows of V are {vec(V[0])}, {vec(V[1])}, {vec(V[2])}.",
                  "A matrix product `U @ V.T` is a table of dot products of rows.", "The matrices stay the same.",
                  matrix(U, id_fn=lambda a, k: f"U{a}.{k}", cls_fn=lambda a, k: "src" if a == i else "dim", row_labels=["u0", "u1"], title="U  shape (2, 3)"),
                  matrix(V, id_fn=lambda a, k: f"V{a}.{k}", row_labels=["v0", "v1", "v2"], title="V  shape (3, 3)")),
            frame("dot", "Multiply and add", f"The dot product of u{i} and v0 is {' + '.join(str(int(U[i, k] * V[0, k])) for k in range(3))} = {int(P[i, 0])}.",
                  "Each entry multiplies its partner. The three products are added.", "Only row i of U is used.",
                  matrix(P, id_fn=rid, cls_fn=lambda a, j: "pos" if a == i else "dim", links_fn=lambda a, j: [f"U{a}.{k}" for k in range(3)] + [f"V{j}.{k}" for k in range(3)],
                         row_labels=["u0", "u1"], col_labels=["v0", "v1", "v2"], title="P = U @ V.T  shape (2, 3)"),
                  matrix(U, id_fn=lambda a, k: f"U{a}.{k}", row_labels=["u0", "u1"], title="U"), matrix(V, id_fn=lambda a, k: f"V{a}.{k}", row_labels=["v0", "v1", "v2"], title="V"),
                  text("Select a cell of P. Its row of U and its row of V are marked.",
                       by_sel={rid(a, j): f"P[{a}, {j}] = u{a} · v{j} = {' + '.join(str(int(U[a, k] * V[j, k])) for k in range(3))} = {int(P[a, j])}" for a in range(2) for j in range(3)})),
        ]

    assert (np.array([[1, 2, 0], [0, 1, 1]]) @ np.array([[2, 0, 1], [1, 1, 1], [0, 3, 1]]).T).tolist() == [[2, 3, 6], [1, 2, 4]]
    a.demo("dot-table", "A matrix product is a table of dot products", frames, params=[("row", "Row of U", [(0, "u0"), (1, "u1")])],
           facts=[("U @ V.T equals the table of row dot products", "numpy")],
           help=[{"step_id": "dot", "help_type": "smaller-example", "content": "Take $u = (1, 2)$ and $v = (3, 4)$. Then $u \\cdot v = 1 \\cdot 3 + 2 \\cdot 4 = 11$."}])
    a.h2("Shapes")
    a.p("If $A$ has shape $(n, d)$ and $B$ has shape $(d, m)$, then $AB$ has shape $(n, m)$. The inner sizes must match. Entry $(i, j)$ of $AB$ is the dot product of row $i$ of $A$ with column $j$ of $B$.")
    a.eq(r"(AB)_{ij} = \sum_{k=1}^{d} A_{ik} B_{kj}")
    a.callout("mistake", "In attention, scores are `Q @ K.T`. The transpose turns the rows of `K` into columns. Without it the inner sizes do not match.")
    return a


def softmax_article():
    a = Article("softmax", "Softmax", kind="prerequisite", prerequisites=["dot-products-and-matmul"])
    a.p("Softmax turns a list of numbers (scores) into positive weights that add up to 1. Larger scores get larger weights.")
    a.eq(r"\mathrm{softmax}(s)_j = \frac{e^{s_j}}{\sum_{k} e^{s_k}}")
    s0 = np.array([2.0, 1.0, 0.1])

    def sm(s):
        e = np.exp(s - s.max())
        return e / e.sum()

    def frames(temp):
        t = float(temp)
        s = s0 / t
        w = sm(s)
        assert abs(w.sum() - 1) < 1e-12
        return [
            frame("scores", "Start with scores", f"The scores are {vec(s0)}" + ("" if t == 1 else f", divided by the temperature {num(t)}: {vec(s)}") + ".",
                  "Scores can be any real numbers, even negative.", "The scores are not changed by softmax.",
                  plot((-0.5, 2.5), (min(0, s.min()) - 0.3, s.max() + 0.5), [series("bar", enumerate(s), "score", width=0.5)], alt=f"Bar chart of three scores {vec(s)}.", xlabel="position", ylabel="score", xticks=[0, 1, 2], width=420, height=240)),
            frame("exp", "Take the exponential", f"$e^{{s}}$ gives {vec(np.exp(s))}. All numbers are now positive.", "The exponential makes every number positive and makes bigger gaps bigger.", "The order of the numbers stays the same.",
                  plot((-0.5, 2.5), (0, np.exp(s).max() * 1.15), [series("bar", enumerate(np.exp(s)), "exp(score)", width=0.5)], alt=f"Bar chart of exponentials {vec(np.exp(s))}.", xlabel="position", ylabel="exp(score)", xticks=[0, 1, 2], width=420, height=240)),
            frame("normalize", "Divide by the total", f"The total is {num(np.exp(s).sum())}. The weights are {vec(w)} and they add up to 1.", "Dividing by the total makes the weights a distribution.", "The order stays the same.",
                  plot((-0.5, 2.5), (0, 1), [series("bar", enumerate(w), "weight", width=0.5)], alt=f"Bar chart of softmax weights {vec(w)}.", xlabel="position", ylabel="weight", xticks=[0, 1, 2], width=420, height=240),
                  text(f"Largest weight: {num(w.max())}. A low temperature makes the largest weight closer to 1. A high temperature makes the weights closer to 1/3.")),
        ]

    a.demo("softmax-steps", "Softmax, step by step", frames, params=[("temp", "Temperature", [(0.5, "0.5"), (1, "1"), (2, "2"), (5, "5")])],
           facts=[("softmax weights computed with a max-shifted exponential and verified to sum to 1", "numpy")])
    a.set_default("softmax-steps", temp=1)
    a.h2("Numerical stability")
    a.p("The exponential of a large score overflows. $e^{1000}$ is infinity in floating point. The fix is to subtract the largest score first. This does not change the result, because the same factor $e^{-\\max}$ appears in the numerator and the denominator.")
    a.eq(r"\mathrm{softmax}(s) = \mathrm{softmax}(s - \max_k s_k)")
    big = np.array([1000.0, 999.0, 998.0])
    with np.errstate(all="ignore"):
        naive = np.exp(big) / np.exp(big).sum()
    assert np.isnan(naive).all() and abs(sm(big).sum() - 1) < 1e-12
    a.p(f"With scores {vec(big)} the naive formula returns `nan` for every weight. The shifted formula returns {vec(sm(big))}. Question [[q:223|223]] asks for this stable version.")
    a.callout("boundary", "If every score is $-\\infty$ (all positions masked), the shifted formula divides $-\\infty$ by $-\\infty$ and returns `nan`. Question 328 handles this case by returning zeros. See [[article:attention-masks|Attention masks]].")
    return a


def sdpa_article():
    a = Article("scaled-dot-product-attention", "Scaled dot-product attention", kind="prerequisite",
                prerequisites=["dot-products-and-matmul", "softmax"], sources=[ATTN])
    a.p("Attention lets each position build its output as a weighted average of **value** vectors. The weights come from comparing a **query** with all the **keys**. " + INVENTED)
    a.eq(r"\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V")
    a.ul(["$Q$ has shape $(L, d_k)$: one query per output position.", "$K$ has shape $(S, d_k)$: one key per source position.", "$V$ has shape $(S, d_v)$: one value per source position.",
          "The scores $QK^\\top$ have shape $(L, S)$. Row $i$ says how much query $i$ matches each key.", "The weights have the same shape. Each row adds up to 1.", "The output has shape $(L, d_v)$."])
    Q = np.array([[1.0, 0.0], [0.0, 1.0]]); K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]); V = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ns = kit.supplied_ns(316)
    ref = ns["self_attention"](Q, K, V)
    S = Q @ K.T / np.sqrt(2); W = np.exp(S - S.max(1, keepdims=True)); W /= W.sum(1, keepdims=True)
    assert np.allclose(W @ V, ref)

    def frames():
        rid = lambda i, j: f"s{i}_{j}"
        return [
            frame("qkv", "Queries, keys, and values", "Two queries ask. Three keys answer. Each key has a value.", "The query and key must have the same length so that their dot product exists.", "The numbers are fixed in this example.",
                  matrix(Q, id_fn=lambda i, k: f"Q{i}.{k}", row_labels=["q0", "q1"], title="Q (2×2)"), matrix(K, id_fn=lambda i, k: f"K{i}.{k}", row_labels=["k0", "k1", "k2"], title="K (3×2)"), matrix(V, id_fn=lambda i, k: f"V{i}.{k}", row_labels=["k0", "k1", "k2"], title="V (3×2)")),
            frame("scores", "Compare queries with keys", f"Each score is a dot product divided by $\\sqrt{{d_k}} = {num(np.sqrt(2))}$.", "A large score means the query and key point the same way.", "Q, K, and V stay the same.",
                  matrix(S, id_fn=rid, links_fn=lambda i, j: [f"Q{i}.0", f"Q{i}.1", f"K{j}.0", f"K{j}.1"], row_labels=["q0", "q1"], col_labels=["k0", "k1", "k2"], title="scores  shape (2, 3)"),
                  matrix(Q, id_fn=lambda i, k: f"Q{i}.{k}", row_labels=["q0", "q1"], title="Q"), matrix(K, id_fn=lambda i, k: f"K{i}.{k}", row_labels=["k0", "k1", "k2"], title="K"),
                  text("Select a score to see which query and key made it.", by_sel={rid(i, j): f"score[{i}, {j}] = (q{i} · k{j}) / √2 = {num(Q[i] @ K[j])} / {num(np.sqrt(2))} = {num(S[i, j])}" for i in range(2) for j in range(3)})),
            frame("weights", "Softmax each row", "Each row of scores becomes weights that add up to 1.", "The weights say how much of each value to take.", "The scores are the input of softmax and are not changed.",
                  matrix(W, id_fn=lambda i, j: f"w{i}_{j}", h_fn=lambda i, j: float(W[i, j]), row_labels=["q0", "q1"], col_labels=["k0", "k1", "k2"], title="weights  (each row sums to 1)"),
                  text(f"Row sums: {vec(W.sum(1))}.")),
            frame("output", "Weighted sum of values", "Output row $i$ is $\\sum_j w_{ij} v_j$.", "This mixes the values. A key with a large weight contributes more.", "The weights stay the same.",
                  matrix(W, id_fn=lambda i, j: f"w{i}_{j}", h_fn=lambda i, j: float(W[i, j]), links_fn=lambda i, j: [f"o{i}"], row_labels=["q0", "q1"], col_labels=["k0", "k1", "k2"], title="weights"),
                  matrix(V, row_labels=["v0", "v1", "v2"], title="V"),
                  grid([[cell(num(ref[i, k]), id=f"o{i}", cls="pos") for k in range(2)] for i in range(2)], ["q0", "q1"], ["dim 0", "dim 1"], title="output (2×2)"),
                  text("Select an output row to see its weighted sum.", by_sel={f"o{i}": " + ".join(f"{num(W[i, j])}·{vec(V[j])}" for j in range(3)) + f" = {vec(ref[i])}" for i in range(2)})),
        ]

    a.demo("sdpa-steps", "One attention computation", lambda: frames(), facts=[("output equals the self_attention function in the supplied solution of question 316", "supplied solution q316")])
    a.h2("Why divide by $\\sqrt{d_k}$")
    a.p("A dot product of two random vectors with $d_k$ entries has a spread that grows like $\\sqrt{d_k}$. Large scores make softmax very sharp: one weight near 1 and the others near 0. A sharp softmax has tiny gradients. Dividing by $\\sqrt{d_k}$ keeps the spread near 1. The lesson on [[q:316|question 316]] shows this with numbers.")
    return a


def masks_article():
    a = Article("attention-masks", "Attention masks", kind="prerequisite", prerequisites=["softmax", "scaled-dot-product-attention"])
    a.p("A mask tells attention which positions it may use. Masked positions get a score of $-\\infty$ before softmax. Then $e^{-\\infty} = 0$, so they get weight 0, and the other weights still add up to 1.")
    S = np.array([[2.0, 1.0, 0.5]]); allowed = np.array([[True, True, False]])
    masked = np.where(allowed, S, -np.inf)
    e = np.exp(masked - masked.max()); w = e / e.sum()
    assert w[0, 2] == 0 and abs(w.sum() - 1) < 1e-12
    a.table(["Kind of mask", "Shape of the allowed array", "What it hides"],
            [["Padding (source) mask", "`(B, S)` then broadcast to `(B, 1, 1, S)`", "Padding tokens in the source"], ["Causal mask", "`(L, L)` lower triangle", "Future tokens"], ["Document mask", "`(L, L)` block diagonal", "Tokens of other documents"]])
    a.p(f"Example: scores {vec(S)} with the last position masked give weights {vec(w)}. The masked position has weight exactly 0.")
    a.callout("mistake", "Adding a large negative number (such as -1e9) works in most cases. Setting the score to $-\\infty$ and using a max-shifted softmax is cleaner. If **all** positions are masked, the result is `nan` unless you handle it. Question 328 returns zeros for such rows.")
    a.callout("boundary", "A mask applies on the **source** axis for cross-attention. For a source of length `S`, the mask has length `S`, not the query length `L`.")
    return a


# =========================================================================== lesson 316
def lesson_316():
    ns = kit.supplied_ns(316)
    compute_qkv, self_attention, mha = ns["compute_qkv"], ns["self_attention"], ns["multi_head_attention"]
    TOK = ["The", "cat", "sat"]
    X = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], float)
    Wq = np.array([[2, -1, 1, 2], [0, 2, 1, 1], [0, 2, 1, 0], [0, 0, 1, 2]], float)
    Wk = np.array([[0, -1, 1, 0], [-1, 0, 0, 2], [0, 2, -1, -1], [0, 0, 0, 1]], float)
    Wv = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]], float)
    H, T, D = 2, 3, 4
    dk = D // H
    Q, K, V = compute_qkv(X, Wq, Wk, Wv)
    assert np.allclose(Q, X @ Wq) and np.allclose(K, X @ Wk) and np.allclose(V, X @ Wv)
    out = mha(Q, K, V, H)
    heads = []
    for h in range(H):
        sl = slice(h * dk, (h + 1) * dk)
        S_ = Q[:, sl] @ K[:, sl].T / np.sqrt(dk)
        W_ = np.exp(S_ - S_.max(1, keepdims=True)); W_ /= W_.sum(1, keepdims=True)
        o_ = W_ @ V[:, sl]
        assert np.allclose(o_, self_attention(Q[:, sl], K[:, sl], V[:, sl]))
        heads.append((S_, W_, o_))
    assert np.allclose(np.concatenate([x[2] for x in heads], axis=-1), out)

    tok_id = lambda i: f"t{i}"
    def frames(head):
        h = int(head); sl = slice(h * dk, (h + 1) * dk); S_, W_, o_ = heads[h]
        cls_cols = lambda i, c: f"g{c // dk}"
        tk = tokens([{"t": t, "sub": f"x{i}", "id": tok_id(i)} for i, t in enumerate(TOK)], title="Tokens (invented embeddings)")
        proj = lambda M, name: matrix(M, id_fn=lambda i, c: f"{name}{i}.{c}", links_fn=lambda i, c: [tok_id(i)], cls_fn=cls_cols, row_labels=TOK, col_labels=[f"c{c}" for c in range(D)], title=f"{name}  shape ({T}, {D})")
        qh = lambda M, name: matrix(M[:, sl], id_fn=lambda i, c: f"{name}h{i}.{c}", links_fn=lambda i, c: [tok_id(i)], cls_fn=lambda i, c: f"g{h}", row_labels=TOK, col_labels=[f"c{h * dk + c}" for c in range(dk)], title=f"{name}, head {h}  shape ({T}, {dk})")
        sid = lambda i, j: f"s{i}_{j}"
        by_out = {f"o{i}": f"Output of **{TOK[i]}**, head {h}: " + " + ".join(f"{num(W_[i, j])}·{vec(V[j, sl])}" for j in range(T)) + f" = {vec(o_[i])}" for i in range(T)}
        return [
            frame("project", "Project the input three ways", f"Each token vector $x$ is multiplied by three weight matrices. This gives $Q$, $K$, and $V$, each with shape ({T}, {D}).",
                  "Queries ask, keys answer, and values carry the content. Different weights give each a different role.", "The tokens and the weight matrices stay the same.",
                  tk, matrix(X, id_fn=lambda i, c: f"X{i}.{c}", links_fn=lambda i, c: [tok_id(i)], row_labels=TOK, title="X  shape (3, 4)"), proj(Q, "Q"), proj(K, "K"), proj(V, "V"),
                  text("Select a token to mark its row in X, Q, K, and V. Colors show which columns belong to head 0 and head 1.")),
            frame("split", f"Give head {h} its own columns", f"Head {h} takes columns {h * dk} to {(h + 1) * dk - 1} of $Q$, $K$, and $V$. Each head works with {dk} columns.",
                  "`reshape(T, H, d_k).transpose(1, 0, 2)` cuts the columns into heads. Each head then sees all tokens but only part of each vector.", "$Q$, $K$, and $V$ are not changed. The heads are slices.",
                  tk, qh(Q, "Q"), qh(K, "K"), qh(V, "V")),
            frame("scores", f"Scores in head {h}", f"Scores are $Q_h K_h^\\top / \\sqrt{{{dk}}}$. The shape is ({T}, {T}).", "Row $i$ compares the query of token $i$ with the key of every token.", "The head slices stay the same.",
                  matrix(S_, id_fn=sid, links_fn=lambda i, j: [f"Qh{i}.0", f"Qh{i}.1", f"Kh{j}.0", f"Kh{j}.1"], row_labels=TOK, col_labels=TOK, title=f"scores, head {h}"),
                  matrix(Q[:, sl], id_fn=lambda i, c: f"Qh{i}.{c}", row_labels=TOK, title="Q_h"), matrix(K[:, sl], id_fn=lambda i, c: f"Kh{i}.{c}", row_labels=TOK, title="K_h"),
                  text("Select a score to see its query row and key row.", by_sel={sid(i, j): f"score[{TOK[i]}, {TOK[j]}] = (q · k)/√{dk} = {num(Q[i, sl] @ K[j, sl])}/{num(np.sqrt(dk))} = {num(S_[i, j])}" for i in range(T) for j in range(T)})),
            frame("weights", f"Softmax: weights in head {h}", "Each row of scores becomes weights that add up to 1.", "A token with a larger score gets a larger weight.", "The scores stay the same.",
                  matrix(W_, id_fn=lambda i, j: f"w{i}_{j}", h_fn=lambda i, j: float(W_[i, j]), row_labels=TOK, col_labels=TOK, title=f"weights, head {h}"),
                  text(f"Row sums: {vec(W_.sum(1))}. Head {1 - h} has different weights. Switch the head to compare.")),
            frame("output", f"Weighted sum of values in head {h}", "Output row $i$ is the weighted sum of the value rows.", "This is how information moves between tokens.", "The weights and values stay the same.",
                  matrix(W_, id_fn=lambda i, j: f"w{i}_{j}", h_fn=lambda i, j: float(W_[i, j]), links_fn=lambda i, j: [f"o{i}"], row_labels=TOK, col_labels=TOK, title="weights"),
                  matrix(V[:, sl], row_labels=TOK, title="V_h"),
                  grid([[cell(num(o_[i, c]), id=f"o{i}", cls=f"g{h}") for c in range(dk)] for i in range(T)], TOK, [f"c{h * dk + c}" for c in range(dk)], title=f"head {h} output  shape ({T}, {dk})"),
                  text("Select an output row. This is the 'Why this output?' trace.", by_sel=by_out)),
            frame("concat", "Concatenate the heads", f"The two head outputs ({T}, {dk}) are joined side by side to give ({T}, {D}). This is the result of `multi_head_attention`.",
                  "Joining keeps each head's result in its own columns. Nothing is averaged.", "The head outputs are unchanged. The supplied solution stops here.",
                  grid([[cell(num(heads[c // dk][2][i, c % dk]), cls=f"g{c // dk}") for c in range(D)] for i in range(T)], TOK, [f"h{c // dk}.c{c % dk}" for c in range(D)], title="concatenated output  shape (3, 4)"),
                  text("In a full transformer block a matrix $W_O$ (shape (4, 4)) multiplies this result to mix the heads. The supplied function for question 316 does **not** include $W_O$. Question 328 does.")),
        ]

    a = Article("lesson-q316-multi-head-attention", "Multi-head attention", kind="lesson",
                prerequisites=["scaled-dot-product-attention", "softmax", "dot-products-and-matmul", "arrays-and-shapes"], exercises=[(316, "deep-lesson"), (328, "related")],
                sources=[ATTN, ("Question 316 source", kit.record(316)["url"])])
    a.h2("The puzzle")
    a.p("One attention computation mixes the values of all tokens with a single set of weights. A token may need to look at one token for grammar and at another for meaning at the same time. **Multi-head attention** runs several attention computations side by side. Each one has its own part of every vector. Then it joins the results.")
    a.p(INVENTED)
    a.h2("What the question asks")
    a.p(kit.record(316)["question"].strip())
    a.callout("definition", "`multi_head_attention(Q, K, V, n_heads)` takes **already projected** matrices `Q`, `K`, `V` with shape `(T, d_model)`. It requires `d_model % n_heads == 0`. It returns shape `(T, d_model)`. `compute_qkv(X, W_q, W_k, W_v)` makes the projections. There is no mask and no output projection in this question.", title="Inputs and output")
    a.figure("multi-head-flow", attention_svg(False), "Multi-head attention flow from token projections through scores, softmax, weighted values, head concatenation, and output.", "The static diagram preserves the complete information path when animation is disabled.")
    a.h2("What you need first")
    a.ul(["[[article:scaled-dot-product-attention|Scaled dot-product attention]]: scores, softmax, and the weighted sum.", "[[article:softmax|Softmax]] and [[article:dot-products-and-matmul|dot products and matrix products]].", "[[article:arrays-and-shapes|Array shapes]]: `reshape` and `transpose` for the head split."])
    a.h2("Follow one computation")
    a.p("We use three tokens, a model width of 4, and 2 heads, so each head has width 2. Switch the head to see that the heads compare different things. Select a token or a number to trace it.")
    a.demo("mha-steps", "Multi-head attention on 'The cat sat'", frames, params=[("head", "Head", [(0, "Head 0"), (1, "Head 1")])],
           help=[{"step_id": "project", "help_type": "smaller-example", "content": "For one token $x = (1, 0, 1, 0)$ and one column $w = (1, 0, 1, 0)$ of $W_q$, the query entry is $1 \\cdot 1 + 0 \\cdot 0 + 1 \\cdot 1 + 0 \\cdot 0 = 2$."},
                 {"step_id": "project", "help_type": "prerequisite", "content": "Matrix products are explained here.", "prerequisite_id": "dot-products-and-matmul"},
                 {"step_id": "split", "help_type": "common-mistake", "content": "Use `reshape(T, H, d_k)` and then `transpose(1, 0, 2)`. Using `reshape(H, T, d_k)` directly mixes tokens and features. It runs without an error."},
                 {"step_id": "scores", "help_type": "expanded-equation", "content": "For head 0, token 'cat': $q \\cdot k$ for key 'The' is computed on 2 numbers only, then divided by $\\sqrt{2} = 1.414$."},
                 {"step_id": "weights", "help_type": "prerequisite", "content": "Softmax is explained here.", "prerequisite_id": "softmax"},
                 {"step_id": "output", "help_type": "code-connection", "content": "`np.matmul(attention_weights, V)` in `self_attention` is this weighted sum."},
                 {"step_id": "concat", "help_type": "code-connection", "content": "`np.concatenate(attentions, axis=-1)` joins the head outputs along the feature axis."}],
           facts=[("Q, K, V equal X @ W for all three weight matrices (compute_qkv of the supplied solution)", "supplied solution q316"),
                  ("each head's output equals self_attention on its column slice, and their concatenation equals multi_head_attention", "supplied solution q316")])
    a.set_default("mha-steps", head=0)
    a.h2("The mathematics")
    a.eq(r"Q = XW_Q,\quad K = XW_K,\quad V = XW_V")
    a.eq(r"\mathrm{head}_h = \mathrm{softmax}\!\left(\frac{Q_h K_h^{\top}}{\sqrt{d_k}}\right) V_h, \qquad d_k = d_{\text{model}} / H")
    a.eq(r"\mathrm{MHA}(Q, K, V) = \mathrm{concat}(\mathrm{head}_1, \dots, \mathrm{head}_H)")
    a.ul(["$Q_h$ is the slice of $Q$ with columns $h d_k$ to $(h+1) d_k - 1$.", "Every head has $T \\times T$ scores. For $H$ heads the scores have shape $(H, T, T)$.", "The total work is about the same as one head of width $d_{\\text{model}}$, because each head is smaller."])
    a.h2("Why divide by $\\sqrt{d_k}$?")
    a.p("Choose the head width and switch the scaling on or off. The vectors are fixed random numbers with mean 0 and variance 1. Without scaling, the score spread grows with the width. The softmax becomes sharp: one weight near 1.")
    rng = np.random.default_rng(7)
    BASE_Q = rng.standard_normal(128); BASE_K = rng.standard_normal((4, 128))

    def scale_frames(width, scale):
        w_ = int(width); sc = scale == "on"
        q = BASE_Q[:w_]; k = BASE_K[:, :w_]
        s = k @ q / (np.sqrt(w_) if sc else 1.0)
        wgt = np.exp(s - s.max()); wgt /= wgt.sum()
        lim = max(3.0, float(np.abs(s).max()) * 1.2)
        return [
            frame("raw", "Scores for four keys", f"The head width is {w_}. The scores are {vec(s)}" + (f" after dividing by $\\sqrt{{{w_}}}$." if sc else " without scaling."),
                  "Scores come from dot products. Their spread depends on the width unless you scale.", "The vectors are fixed random numbers.",
                  plot((-0.5, 3.5), (-lim, lim), [series("bar", enumerate(s), "score", width=0.5)], alt=f"Bar chart of four scores. The largest absolute score is {num(np.abs(s).max())}.", xlabel="key", ylabel="score", xticks=[0, 1, 2, 3], width=440, height=250)),
            frame("soft", "Softmax of the scores", f"The largest weight is {num(wgt.max())}.", "With big scores softmax puts almost all weight on one key.", "The scores stay the same.",
                  plot((-0.5, 3.5), (0, 1), [series("bar", enumerate(wgt), "weight", width=0.5)], alt=f"Bar chart of four attention weights. The largest is {num(wgt.max())}.", xlabel="key", ylabel="weight", xticks=[0, 1, 2, 3], width=440, height=250),
                  text("A weight near 1 for one key and near 0 for the others gives almost no gradient to the other keys." if wgt.max() > 0.95 else "The weights are spread out. Every key can still receive a gradient.")),
        ]
    for w_ in (2, 8, 32, 128):
        s_no = BASE_K[:, :w_] @ BASE_Q[:w_]; s_sc = s_no / np.sqrt(w_)
        assert np.abs(s_no).max() >= np.abs(s_sc).max()
    big_no = BASE_K @ BASE_Q; ex = np.exp(big_no - big_no.max()); ex /= ex.sum()
    a.demo("scale-effect", "Scaled and unscaled scores", scale_frames,
           params=[("width", "Head width", [(2, "2"), (8, "8"), (32, "32"), (128, "128")]), ("scale", "Divide by √d_k", [("on", "on"), ("off", "off")])],
           facts=[("scores are Q·K with and without division by sqrt(width), softmax computed in NumPy", "numpy")])
    a.set_default("scale-effect", width=128, scale="off")
    a.compare("scale-effect", "dividing the scores by $\\sqrt{d_k}$", [("Width 128, no scaling", {"width": 128, "scale": "off"}), ("Width 128, scaled", {"width": 128, "scale": "on"})],
              "Only the scaling changed. With scaling, the largest weight is smaller and the other keys still get weight.", frame="soft")

    a.h2("Break it, then repair it")
    a.p("The head split is easy to get wrong because the wrong version runs without an error. Choose a version and compare its first head with the correct one.")
    right = Q.reshape(T, H, dk).transpose(1, 0, 2)
    wrongsplit = Q.reshape(H, T, dk)
    assert right.shape == wrongsplit.shape == (H, T, dk) and not np.allclose(right, wrongsplit)

    def split_frames(version):
        arr = right if version == "correct" else wrongsplit
        expr = "Q.reshape(T, H, d_k).transpose(1, 0, 2)" if version == "correct" else "Q.reshape(H, T, d_k)"
        return [
            frame("expr", "The split expression", f"The expression is `{expr}`.", "Both versions return shape (H, T, d_k) = (2, 3, 2).", "Q is the same matrix.", code_view(f"heads = {expr}", [1], label="Head split")),
            frame("head0", "Head 0 after the split", f"Head 0 holds {vec(arr[0][0])}, {vec(arr[0][1])}, {vec(arr[0][2])} for the three tokens.",
                  "Head 0 must hold columns 0 and 1 of every token. In the wrong version it holds the first six numbers of Q in memory order.", "The correct head 0 is Q[:, :2].",
                  matrix(arr[0], row_labels=TOK, col_labels=["c0", "c1"], title=f"head 0 ({version})", cls_fn=lambda i, c: "pos" if version == "correct" else "neg"),
                  matrix(Q[:, :2], row_labels=TOK, col_labels=["c0", "c1"], title="correct head 0 = Q[:, :2]"),
                  text("The values match the correct head." if version == "correct" else "The values differ from the correct head. The shape is the same, so no error appears.")),
        ]
    a.demo("split-repair", "Which head split is right?", split_frames, params=[("version", "Version", [("wrong", "reshape(H, T, d_k)"), ("correct", "reshape(T, H, d_k).transpose(1, 0, 2)")])],
           facts=[("the wrong split has the same shape (2, 3, 2) but different values; the correct split equals Q[:, :2] for head 0", "numpy")])
    a.set_default("split-repair", version="wrong")
    a.callout("mistake", "A wrong reshape keeps the shape and changes the meaning. Check one head against a plain slice `Q[:, :d_k]`.", title="A silent bug")
    a.callout("boundary", "If `d_model` is not divisible by `n_heads`, the supplied function raises ValueError. A model width of 6 with 4 heads cannot be split.")

    a.h2("Read the supplied solution")
    a.walkthrough(316, [
        (kit.line_of(316, 'return X @ W_q'), 'return X @ W_q', 'Project the same input into queries, keys, and values with separate weights.'),
        (kit.line_of(316, 'scores = queries @'), 'scores = queries @', 'Scale dot products by the square root of the per-head width.'),
        (kit.line_of(316, 'weights = np.exp'), 'weights = np.exp', 'Subtract each row maximum before exponentiation to keep softmax stable.'),
        (kit.line_of(316, 'return weights @ values'), 'return weights @ values', 'Weight the value vectors by the attention probabilities.'),
        (kit.line_of(316, 'or model_dim % n_heads'), 'or model_dim % n_heads', 'Require a positive head count that divides the model width.'),
        (kit.line_of(316, 'return array.reshape'), 'return array.reshape', 'Reshape into heads and move the head axis to the front.'),
        (kit.line_of(316, 'context = self_attention'), 'context = self_attention', 'A batched matrix product evaluates independent attention for all heads.'),
        (kit.line_of(316, 'return context.transpose'), 'return context.transpose', 'Move the head axis back and combine features into the model width.'),

    ], title="Question 316: three small functions")

    a.h2("Test yourself")
    a.check("predict-shape", "prediction", "`Q` has shape (3, 4) and `n_heads` is 2. What is the shape of `Q.reshape(3, 2, 2).transpose(1, 0, 2)`?", "shape", "(2, 3, 2)",
            expl("The reshape gives (3, 2, 2) = (T, H, d_k). The transpose moves the head axis first: (2, 3, 2) = (H, T, d_k).", "`transpose(1, 0, 2)` swaps the first two axes of (T, H, d_k).",
                 **{"(3, 2, 2)": "That is the shape before the transpose.", "(3, 4)": "That is the shape of Q. The split adds a head axis."}))
    A_ = np.array([[1.0], [0.0]]); K2 = np.array([[0.0], [np.log(3)]]); V2 = np.array([[10.0], [20.0]])
    w2 = float(self_attention(A_[:1], K2, V2)[0, 0])
    assert abs(w2 - (0.25 * 10 + 0.75 * 20)) < 1e-9
    a.check("new-weight", "new-case", "A single head has width 1. The query is [1]. The keys are [0] and [1.0986]. The values are [10] and [20]. What is the attention output? (Round to one decimal.)", "number", round(w2, 1),
            expl("The scores are 0 and 1.0986, with $\\sqrt{d_k} = 1$. Softmax gives weights 0.25 and 0.75 because $e^{1.0986} = 3$. The output is $0.25 \\cdot 10 + 0.75 \\cdot 20 = 17.5$.", "Compute the two scores, take softmax to get the weights, then mix the values.",
                 **{"15": "That is the plain average. The second key has the larger score, so it gets the larger weight.", "20": "That would need weight 1 on the second value. The weights are 0.25 and 0.75."}),
            tolerance=0.05, case={"Q": [[1]], "K": [[0], [1.0986]], "V": [[10], [20]]})
    a.check("count-heads", "prediction", "`d_model` is 512 and `n_heads` is 8. What is the width of each head?", "number", 64, expl("512 / 8 = 64.", "Divide the model width by the number of heads."), tolerance=0)
    a.check("explain-heads", "explain-back", "In your own words: what does each head see, and why do the heads give different attention weights?", "text", None,
            {"correct": "Each head sees all tokens but only its own slice of every query, key, and value vector. The slices come from different columns of the projection matrices, so each head compares tokens in a different way and produces different weights. Concatenation puts the head results side by side."})
    a.h2("Write it yourself")
    cases = []
    rng2 = np.random.default_rng(3)
    for (t_, d_, h_) in ((3, 4, 2), (2, 6, 3)):
        q_, k_, v_ = (np.round(rng2.standard_normal((t_, d_)), 2) for _ in range(3))
        cases.append((q_, k_, v_, h_, mha(q_, k_, v_, h_)))
    checks = [("output shape is (T, d_model)", "r = multi_head_attention(np.ones((3, 4)), np.ones((3, 4)), np.ones((3, 4)), 2)\nassert np.asarray(r).shape == (3, 4), np.asarray(r).shape")]
    for n_, (q_, k_, v_, h_, o_) in enumerate(cases):
        checks.append((f"matches the supplied solution, case {n_ + 1}", f"Q = np.array({q_.tolist()}); K = np.array({k_.tolist()}); V = np.array({v_.tolist()})\nexpected = np.array({np.round(o_, 6).tolist()})\nr = multi_head_attention(Q, K, V, {h_})\nassert np.allclose(r, expected, atol=1e-4), np.round(r, 4)"))
    checks.append(("one head equals plain attention", "Q = np.array([[1., 0.], [0., 1.]]); K = Q.copy(); V = np.array([[1., 2.], [3., 4.]])\ns = Q @ K.T / np.sqrt(2); w = np.exp(s - s.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)\nassert np.allclose(multi_head_attention(Q, K, V, 1), w @ V)"))
    checks.append(("rejects a width that does not split", "try:\n    multi_head_attention(np.ones((2, 6)), np.ones((2, 6)), np.ones((2, 6)), 4)\n    raise SystemError('no error')\nexcept (AssertionError, ValueError):\n    pass"))
    a.playground(316, '''
        import numpy as np


        def multi_head_attention(Q, K, V, n_heads):
            # Q, K, V have shape (T, d_model). Return shape (T, d_model).
            # Split d_model into n_heads parts, run scaled dot-product attention in each part,
            # and concatenate the results. Raise an error when d_model % n_heads != 0.
            raise NotImplementedError
        ''', checks=checks, note="Write `multi_head_attention(Q, K, V, n_heads)` for question 316. The expected outputs come from the supplied solution.", max_input_size=100)
    return a


# =========================================================================== lesson 328
def lesson_328():
    q328 = kit.supplied(328)
    ns = kit.supplied_ns(328)
    q309 = ns["q309"]
    DEC = ["Le", "chat"]; ENC = ["The", "cat", "sat"]
    L, S, D, H = 2, 3, 4, 2
    dk = D // H
    dec = np.array([[[1, 0, 1, 0], [0, 1, 0, 1]]], float)
    enc = np.array([[[1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 1, 1]]], float)
    wq = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1]], float)
    wk = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1]], float)
    wv = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]], float)
    wo = np.eye(D) + np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]) * 0.5
    mask = np.array([[True, True, False]])
    ref_masked = q328(dec, enc, wq, wk, wv, wo, H, mask)
    ref_nomask = q328(dec, enc, wq, wk, wv, wo, H, None)
    assert ref_masked.shape == (1, L, D) and not np.allclose(ref_masked, ref_nomask)

    def parts(use_mask):
        q = (dec @ wq).reshape(1, L, H, dk).transpose(0, 2, 1, 3)
        k = (enc @ wk).reshape(1, S, H, dk).transpose(0, 2, 1, 3)
        v = (enc @ wv).reshape(1, S, H, dk).transpose(0, 2, 1, 3)
        allowed = None if not use_mask else mask[:, None, None, :]
        scores = q @ k.swapaxes(-1, -2) / np.sqrt(dk)
        sc = np.where(np.ones_like(scores, bool) if allowed is None else np.broadcast_to(allowed, scores.shape), scores, -np.inf)
        e = np.exp(sc - sc.max(-1, keepdims=True)); w = e / e.sum(-1, keepdims=True)
        o = w @ v
        cat = o.transpose(0, 2, 1, 3).reshape(1, L, D)
        return q, k, v, scores, w, o, cat, cat @ wo
    for um, rf in ((True, ref_masked), (False, ref_nomask)):
        assert np.allclose(parts(um)[-1], rf)
    SRC = [{"t": t, "sub": f"source {j}", "id": f"e{j}"} for j, t in enumerate(ENC)]

    def frames(head, mask_on):
        h = int(head); um = mask_on == "on"
        q, k, v, scores, w, o, cat, fin = parts(um)
        sl = slice(h * dk, (h + 1) * dk)
        qi = lambda i: f"d{i}"
        dt = tokens([{"t": t, "sub": f"decoder {i} → query", "id": qi(i), "cls": "src"} for i, t in enumerate(DEC)], title=f"Decoder sequence, length L = {L}: supplies the queries")
        et = tokens([{**s, "cls": "add" if um or True else None, "sub": s["sub"] + (" (padding)" if j == 2 and um else " → key, value"), **({"cls": "mask"} if j == 2 and um else {"cls": "add"})} for j, s in enumerate(SRC)],
                    title=f"Encoder sequence, length S = {S}: supplies the keys and values")
        sid = lambda i, j: f"a{i}_{j}"
        sc_cell = lambda i, j: cell("−∞" if (um and not mask[0, j]) else num(scores[0, h, i, j]), id=sid(i, j), cls="mask" if (um and not mask[0, j]) else None, links=[qi(i), f"e{j}"])
        w_cell = lambda i, j: cell(num(w[0, h, i, j]), id=sid(i, j), h=float(w[0, h, i, j]), cls="mask" if (um and not mask[0, j]) else None, links=[qi(i), f"e{j}"])
        by_out = {qi(i): f"Query **{DEC[i]}**, head {h}: " + " + ".join(f"{num(w[0, h, i, j])}·{vec(v[0, h, j])}" for j in range(S)) + f" = {vec(o[0, h, i])}" for i in range(L)}
        by_att = {sid(i, j): f"Weight of source token **{ENC[j]}** for query **{DEC[i]}**, head {h}: {num(w[0, h, i, j])}" for i in range(L) for j in range(S)}
        return [
            frame("sequences", "Two sequences of different length", f"The decoder has {L} tokens. The encoder has {S} tokens" + (", and the third one is padding." if um else "."),
                  "Queries come from the decoder. Keys and values come from the encoder. $L$ and $S$ may differ.", "No causal mask is used. Every query may look at every valid source token.",
                  dt, et, text(f"Shapes: decoder ({1}, {L}, {D}), encoder ({1}, {S}, {D}). The mask has shape ({1}, {S}): `True` marks a valid source token." + (" The third source token is masked." if um else " This run has no mask.")) ),
            frame("project", "Project each sequence", f"$Q$ is {L}×{D} from the decoder. $K$ and $V$ are {S}×{D} from the encoder.", "The projection weights are separate for $Q$, $K$, and $V$. The decoder is multiplied by $W_Q$. The encoder is multiplied by $W_K$ and $W_V$.", "The inputs and weights stay the same.",
                  matrix(q[0].transpose(1, 0, 2).reshape(L, D), id_fn=lambda i, c: f"Q{i}.{c}", links_fn=lambda i, c: [qi(i)], cls_fn=lambda i, c: f"g{c // dk}", row_labels=DEC, title=f"Q from the decoder ({L}, {D})"),
                  matrix(k[0].transpose(1, 0, 2).reshape(S, D), id_fn=lambda j, c: f"K{j}.{c}", links_fn=lambda j, c: [f"e{j}"], cls_fn=lambda j, c: f"g{c // dk}", row_labels=ENC, title=f"K from the encoder ({S}, {D})"),
                  matrix(v[0].transpose(1, 0, 2).reshape(S, D), id_fn=lambda j, c: f"V{j}.{c}", links_fn=lambda j, c: [f"e{j}"], cls_fn=lambda j, c: f"g{c // dk}", row_labels=ENC, title=f"V from the encoder ({S}, {D})"),
                  dt, et, text("Select a token to mark its row in Q, K, or V. Colors show the columns of head 0 and head 1.")),
            frame("scores", f"Scores in head {h}: decoder rows, encoder columns", f"The scores have shape ({L}, {S}), not square. Row $i$ is a decoder query. Column $j$ is an encoder key.", "$L \\ne S$, so the matrix is rectangular." + (" Masked source columns get $-\\infty$." if um else ""),
                  "Q and K stay the same.", grid([[sc_cell(i, j) for j in range(S)] for i in range(L)], DEC, ENC, title=f"scores, head {h}  shape ({L}, {S})"), dt, et,
                  text("Select a score. The decoder token and encoder token behind it are marked.", by_sel={sid(i, j): ("Masked: the source token is padding, so the score is −∞." if um and not mask[0, j] else f"score[{DEC[i]}, {ENC[j]}] = {num(scores[0, h, i, j])}") for i in range(L) for j in range(S)})),
            frame("weights", f"Softmax over the source axis, head {h}", "Each row is normalized over the $S$ source positions.", "Softmax runs along the source axis. Masked columns get weight exactly 0. Each row still adds up to 1.", "The scores stay the same.",
                  grid([[w_cell(i, j) for j in range(S)] for i in range(L)], DEC, ENC, title=f"weights, head {h}"), dt, et,
                  text(f"Row sums: {vec(w[0, h].sum(-1))}. Weight on the padding token: {num(w[0, h, :, 2].max())}.", by_sel=by_att)),
            frame("output", "Weighted sum, join heads, project", f"Each query gets a weighted sum of the source values ({L}, {dk}) per head. The heads are joined to ({L}, {D}). Then $W_O$ gives the final ({L}, {D}).",
                  "The output has the length of the **decoder**, not of the encoder.", "The weights stay the same.",
                  grid([[cell(num(o[0, h, i, c]), id=qi(i), cls=f"g{h}") for c in range(dk)] for i in range(L)], DEC, [f"c{h * dk + c}" for c in range(dk)], title=f"head {h} output ({L}, {dk})"),
                  grid([[cell(num(fin[0, i, c]), id=qi(i), cls="pos") for c in range(D)] for i in range(L)], DEC, [f"c{c}" for c in range(D)], title=f"final output after W_O ({L}, {D})"),
                  dt, text("Select a decoder token. This is the 'Why this output?' trace for head " + str(h) + ".", by_sel=by_out)),
        ]

    a = Article("lesson-q328-cross-attention", "Cross-attention with different sequence lengths", kind="lesson",
                prerequisites=["lesson-q316-multi-head-attention", "attention-masks", "scaled-dot-product-attention", "softmax"], exercises=[(328, "deep-lesson"), (316, "related")],
                sources=[ATTN, ("Question 328 source", kit.record(328)["url"])])
    a.h2("The puzzle")
    a.p("In a translation model the decoder writes one language and the encoder holds the other. The decoder needs to look at the encoder. **Cross-attention** does that. The queries come from the decoder. The keys and values come from the encoder. The two sequences usually have different lengths. " + INVENTED)
    a.h2("What the question asks")
    a.p(kit.record(328)["question"].strip())
    a.callout("definition", "`decoder` has shape `(B, L, D)`. `encoder` has shape `(B, S, D)`. `source_mask` has shape `(B, S)` and is `True` for valid source tokens. `wq`, `wk`, `wv`, `wo` have shape `(D, D)`. `heads` must divide `D`. The output has shape `(B, L, D)`. There is **no** causal mask.", title="Inputs and output")
    a.figure("cross-attention-flow", attention_svg(True), "Cross-attention flow. Queries originate in the decoder while keys and values originate in the encoder before scores and weighted values are computed.", "The split paths make the source of Q, K, and V explicit without motion.")
    a.h2("What you need first")
    a.ul(["[[article:lesson-q316-multi-head-attention|Multi-head attention]]: the head split and concatenation.", "[[article:attention-masks|Attention masks]]: what a source mask hides, and its shape.", "[[article:scaled-dot-product-attention|Scaled dot-product attention]] and [[article:softmax|softmax]]."])
    a.h2("Follow one cross-attention")
    a.p("There are two decoder tokens and three encoder tokens. The third encoder token is padding. Turn the mask off to see what changes. Select a decoder token for the full weighted sum.")
    a.demo("xattn-steps", "Cross-attention, step by step", frames,
           params=[("head", "Head", [(0, "Head 0"), (1, "Head 1")]), ("mask_on", "Source mask", [("on", "on (3rd token is padding)"), ("off", "off")])],
           help=[{"step_id": "sequences", "help_type": "smaller-example", "content": "One decoder token and two source tokens give a 1×2 score matrix. The single query is compared with two keys."},
                 {"step_id": "project", "help_type": "common-mistake", "content": "Do not project the decoder with $W_K$ or $W_V$. In cross-attention only $Q$ comes from the decoder."},
                 {"step_id": "scores", "help_type": "prerequisite", "content": "Masks are explained here.", "prerequisite_id": "attention-masks"},
                 {"step_id": "weights", "help_type": "expanded-equation", "content": "For query 'Le' in head 0 with the mask on, the third column is $-\\infty$, so $e^{-\\infty} = 0$ and the other two weights add up to 1."},
                 {"step_id": "output", "help_type": "code-connection", "content": "In the supplied solution `q309(q, k, v, allowed).transpose(0, 2, 1, 3).reshape(b, l, d) @ wo` does the weighted sum, joins the heads, and applies $W_O$."}],
           facts=[("the final output equals the supplied q328, with and without the source mask", "supplied solution q328"), ("the weight of a masked source token is exactly 0 and every row sums to 1", "numpy")])
    a.set_default("xattn-steps", head=0, mask_on="on")
    a.h2("The mathematics")
    a.eq(r"Q = X_{\mathrm{dec}} W_Q,\qquad K = X_{\mathrm{enc}} W_K,\qquad V = X_{\mathrm{enc}} W_V")
    a.eq(r"A_h = \mathrm{softmax}_{\text{source}}\!\left(\frac{Q_h K_h^{\top}}{\sqrt{d_k}} + M\right) \in \mathbb{R}^{L \times S}", caption="$M_{ij} = 0$ for a valid source token $j$ and $-\\infty$ for padding.")
    a.eq(r"\mathrm{out} = \mathrm{concat}_h(A_h V_h)\, W_O \in \mathbb{R}^{L \times D}")
    a.p("Compare with self-attention. There $L = S$ and all three come from one sequence. The score matrix is square, and a **causal** mask may be added. Here the score matrix has $L$ rows and $S$ columns, and the only mask hides padding.")

    # --- mask repair demo
    a.h2("Repair the mask")
    a.p("Choose how the mask is applied. Look at the weight on the padding token.")

    def mask_frames(version):
        q, k, v, scores, w, o, cat, fin = parts(False)
        outcome = {}
        if version == "none":
            ww = w[0, 0]; expr = "allowed = None"
            outcome = ("The padding token receives a weight.", "Without a mask the model reads a token that is only padding.", ww)
            msg = None
        elif version == "wrongaxis":
            expr = "allowed = np.asarray(source_mask)[:, None, :, None]"
            sc4 = np.zeros((1, H, L, S), float)
            try:
                np.broadcast_to(mask[:, None, :, None], sc4.shape)
                msg = None
            except ValueError as e:
                msg = str(e)
            assert msg is not None
            outcome = ("NumPy raises an error.", "The mask was given the source length on the **query** axis. The mask must go on the last axis.", None)
            ww = None
        else:
            expr = "allowed = np.asarray(source_mask)[:, None, None, :]"
            ww = parts(True)[4][0, 0]
            assert np.isclose(ww[:, 2].max(), 0) and np.allclose(ww.sum(-1), 1)
            outcome = ("The padding token has weight 0 and the rows still sum to 1.", "The mask shape `(B, 1, 1, S)` broadcasts over heads and queries and lines up with the source axis.", ww)
            msg = None
        views = [grid([[cell(num(ww[i, j]), cls="neg" if (version == "none" and j == 2) else ("mask" if (version == "correct" and j == 2) else None), h=float(ww[i, j])) for j in range(S)] for i in range(L)], DEC, ENC, title="weights, head 0")] if ww is not None else []
        return [
            frame("expr", "The mask expression", f"The line is `{expr}`.", "The mask must end up with shape (B, 1, 1, S) so it lines up with the scores (B, H, L, S).", "The source mask is `[[True, True, False]]`.",
                  code_view(expr, [1], label="Mask for the scores")),
            frame("effect", "What the mask does", outcome[0], outcome[1], "The scores before masking are unchanged.",
                  *(views or [text(f"Error from NumPy: `{msg}`")])),
        ]
    a.demo("mask-repair", "Which mask is right?", mask_frames, params=[("version", "Mask", [("none", "no mask"), ("wrongaxis", "[:, None, :, None]"), ("correct", "[:, None, None, :]")])],
           facts=[("no mask gives the padding token a positive weight; the wrong-axis mask fails to broadcast; the correct mask gives weight 0", "numpy")])
    a.set_default("mask-repair", version="none")
    a.callout("mistake", "A mask on the wrong axis may not raise an error when $L = S$. It then hides the wrong positions. Test with $L \\ne S$.", title="Why use different lengths")
    a.callout("boundary", "If every source token of one example is masked, a plain softmax returns `nan`. The supplied helper `q309` returns zeros for that row instead.")

    # --- controlled edits of projected keys and values
    a.h2("Change a key, change a value")
    a.p("Change the projected key or the projected value of source token 1 by +1 on both of its entries for head 0. All other projected vectors stay fixed. This is different from changing the input embedding, because an embedding change also changes the key **and** the value of that token.")
    q, k, v, scores, w, o, cat, fin = parts(True)
    q0 = q[:, 0]; k0 = k[:, 0].copy(); v0 = v[:, 0].copy()
    base_o = q309(q0, k0, v0, mask[:, None, :])
    k_edit = k0.copy(); k_edit[0, 1] += 1.0
    v_edit = v0.copy(); v_edit[0, 1] += 1.0
    cases = {"none": (k0, v0), "key": (k_edit, v0), "value": (k0, v_edit)}

    def edit_frames(edit):
        kk, vv = cases[edit]
        sc = q0 @ kk.swapaxes(-1, -2) / np.sqrt(dk); sc = np.where(mask[:, None, :], sc, -np.inf)
        ww = np.exp(sc - sc.max(-1, keepdims=True)); ww /= ww.sum(-1, keepdims=True)
        oo = q309(q0, kk, vv, mask[:, None, :])
        assert np.allclose(oo, ww @ vv)
        if edit == "key":
            assert not np.allclose(ww, w[:, 0])
        if edit == "value":
            assert np.allclose(ww, w[:, 0]) and not np.allclose(oo, base_o)
        return [
            frame("edit", "The edit", {"none": "Nothing is edited.", "key": "Key of source token 'cat' + (0, 1)... both entries +1.", "value": "Value of source token 'cat' +1 on both entries."}[edit].replace("+ (0, 1)... ", ""),
                  "We hold all other projected vectors fixed.", "Queries, other keys, other values, and the mask stay the same.",
                  matrix(kk[0], row_labels=ENC, col_labels=["c0", "c1"], title="K, head 0", cls_fn=lambda j, c: "add" if edit == "key" and j == 1 else None),
                  matrix(vv[0], row_labels=ENC, col_labels=["c0", "c1"], title="V, head 0", cls_fn=lambda j, c: "add" if edit == "value" and j == 1 else None)),
            frame("effect", "Weights and outputs after the edit", ("The weights changed." if edit == "key" else "The weights did not change.") if edit != "none" else "This is the baseline.",
                  "A key enters the **scores**, so it changes the weights. A value enters only the **weighted sum**, so it changes the output but not the weights.", "The queries stay the same.",
                  grid([[cell(num(ww[0, i, j]), h=float(ww[0, i, j])) for j in range(S)] for i in range(L)], DEC, ENC, title="weights, head 0"),
                  grid([[cell(num(oo[0, i, c]), cls="pos") for c in range(dk)] for i in range(L)], DEC, ["c0", "c1"], title="head 0 output")),
        ]
    a.demo("kv-edit", "Edit the projected key or value", edit_frames, params=[("edit", "Edit", [("none", "none"), ("key", "key of 'cat'"), ("value", "value of 'cat'")])],
           facts=[("editing a key changes the weights; editing a value changes the output but not the weights (computed with the supplied q309)", "supplied solution q328 helper q309")])
    a.set_default("kv-edit", edit="key")

    a.h2("Read the supplied solution")
    a.walkthrough(328, [
        (kit.line_of(328, "def q328"), "def q328", "The function gets the two sequences, the four weight matrices, the number of heads, and an optional source mask."),
        (kit.line_of(328, "if d % heads:"), "if d % heads:", "The width must divide evenly into heads."),
        ((kit.line_of(328, "(decoder @ wq)") - 1, kit.line_of(328, "(decoder @ wq)") + 3), "(decoder @ wq)", "Queries come from the **decoder**. Shape `(B, L, D)` becomes `(B, H, L, D/H)`."),
        ((kit.line_of(328, "(encoder @ wk)") - 1, kit.line_of(328, "(encoder @ wk)") + 3), "(encoder @ wk)", "Keys come from the **encoder**. The sequence length is `S`, not `L`."),
        (kit.line_of(328, "np.asarray(source_mask)[:, None, None, :]"), "[:, None, None, :]", "The source mask `(B, S)` becomes `(B, 1, 1, S)`. It lines up with the last axis of the scores."),
        (kit.line_of(328, "q309(q, k, v, allowed)"), "q309(q, k, v, allowed)", "Scaled dot-product attention with a mask. This is the helper that returns zeros for fully masked rows."),
        (kit.line_of(328, "scores = np.where(valid, scores, -np.inf)"), "-np.inf", "Masked scores become $-\\infty$ so that they get weight 0."),
        (kit.line_of(328, ".reshape(b, l, d) @ wo"), ".reshape(b, l, d) @ wo", "The heads are joined back to width `D` and mixed with $W_O$."),
    ], title="Question 328: the helper and the main function")
    a.h2("Test yourself")
    a.check("predict-shape", "prediction", "The decoder has L = 5 tokens and the encoder has S = 7 tokens. What is the shape of the score matrix for one head?", "shape", "(5, 7)",
            expl("Rows are queries from the decoder (5). Columns are keys from the encoder (7). The matrix is (5, 7), not square.", "Rows come from the sequence that supplies the queries. Columns come from the sequence that supplies the keys.",
                 **{"(7, 5)": "That would put the encoder on the query axis.", "(5, 5)": "That is self-attention on the decoder.", "(7, 7)": "That is self-attention on the encoder."}))
    mask2 = np.array([[True, True, True, False]])
    g = np.array([[2.0, 1.0, 0.0, 5.0]])
    e_ = np.exp(np.where(mask2, g, -np.inf) - 2.0); w4 = e_ / e_.sum()
    a.check("new-weight", "new-case", "One query has scores [2, 1, 0, 5] over 4 source tokens. The last source token is padding (masked). What is the weight of the first token? (Three decimals.)", "number", round(float(w4[0, 0]), 3),
            expl(f"The masked score is dropped. Softmax of [2, 1, 0] gives $e^2 / (e^2 + e^1 + e^0) = {w4[0, 0]:.3f}$. The padding weight is 0 even though its score was the largest.", "Remove the masked token first, then take softmax over the rest.",
                 **{"0.867": "That would use the padding score 5 in the sum. The masked token must get weight 0 and be left out of the total."}), tolerance=0.002, case={"scores": [2, 1, 0, 5], "valid": [True, True, True, False]})
    a.check("which-edit", "prediction", "You edit the **value** vector of one source token and keep all keys and queries fixed. What changes?", "choice", "The outputs, not the weights",
            expl("A value enters only the weighted sum. The scores and weights depend on queries and keys, so they stay the same.", "Think about where each of Q, K, V enters the formula.",
                 **{"The weights, not the outputs": "That is the effect of changing a key.", "Nothing": "The output is a weighted sum of values, so it changes.", "Both the weights and the outputs": "The weights depend only on queries and keys."}),
            options=["The weights, not the outputs", "The outputs, not the weights", "Both the weights and the outputs", "Nothing"])
    a.check("explain-xattn", "explain-back", "In your own words: which sequence supplies the queries, which supplies the keys and values, and why can the two lengths differ?", "text", None,
            {"correct": "The decoder supplies the queries. The encoder supplies the keys and values. The score matrix has one row per query and one column per key, so it is $L \\times S$. Each output row is a weighted sum of $S$ values, so the output keeps the decoder length $L$."})
    a.h2("Write it yourself")
    rng3 = np.random.default_rng(11)
    d_, e_in = np.round(rng3.standard_normal((1, 2, 4)), 2), np.round(rng3.standard_normal((1, 3, 4)), 2)
    Ws = [np.round(rng3.standard_normal((4, 4)) * 0.5, 2) for _ in range(4)]
    m_ = np.array([[True, True, False]])
    exp_m = q328(d_, e_in, *Ws, 2, m_)
    exp_n = q328(d_, e_in, *Ws, 2, None)
    lit = lambda x: np.array2string(np.asarray(x), separator=", ", threshold=10**6).replace("\n", "")
    setup = f"D_ = np.array({d_.tolist()}); E_ = np.array({e_in.tolist()}); W = [np.array(w) for w in {[w.tolist() for w in Ws]}]; M = np.array([[True, True, False]])"
    a.playground(328, '''
        import numpy as np


        def cross_attention(decoder, encoder, wq, wk, wv, wo, heads, source_mask=None):
            # decoder: (B, L, D). encoder: (B, S, D). source_mask: (B, S), True for valid tokens.
            # Queries come from the decoder. Keys and values come from the encoder. No causal mask.
            raise NotImplementedError
        ''', checks=[
        ("output has the decoder length", f"{setup}\nr = cross_attention(D_, E_, *W, 2, M)\nassert np.asarray(r).shape == (1, 2, 4), np.asarray(r).shape"),
        ("matches the supplied solution with a mask", f"{setup}\nr = cross_attention(D_, E_, *W, 2, M)\nassert np.allclose(r, np.array({np.round(exp_m, 6).tolist()}), atol=1e-4), np.round(r, 4)"),
        ("matches the supplied solution without a mask", f"{setup}\nr = cross_attention(D_, E_, *W, 2)\nassert np.allclose(r, np.array({np.round(exp_n, 6).tolist()}), atol=1e-4), np.round(r, 4)"),
        ("masked source tokens do not change the output", f"{setup}\nE2 = E_.copy(); E2[0, 2] = 99.0\na1 = cross_attention(D_, E_, *W, 2, M); a2 = cross_attention(D_, E2, *W, 2, M)\nassert np.allclose(a1, a2, atol=1e-6)"),
    ], note="Write `cross_attention(...)` for question 328. Use different lengths for the decoder (2) and the encoder (3).", max_input_size=100)
    return a


def build():
    return [matmul_article(), softmax_article(), sdpa_article(), masks_article(), lesson_316(), lesson_328()]
