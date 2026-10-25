"""Batch C02: NumPy arrays, broadcasting, and vectorization. Questions 027 and 030."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kit
from kit import Article, cell, expl, frame, grid, matrix, num, plot, series, text, equation, code_view, tokens, vec
from scenes_c02 import pairwise_scene, mismatch_scene, broadcast_concept_scene

sys.path.insert(0, str(kit.ROOT / "tools"))
from gen_broadcast_expected import case as broadcast_case  # noqa: E402  (NumPy reference values)

NUMPY_DOC = ("NumPy documentation: Broadcasting", "https://numpy.org/doc/stable/user/basics.broadcasting.html")
NUMPY_COPIES = ("NumPy documentation: Copies and views", "https://numpy.org/doc/stable/user/basics.copies.html")


# =========================================================================== prerequisite: arrays and shapes
def arrays_and_shapes():
    a = Article("arrays-and-shapes", "Array shapes and indexing", kind="prerequisite", sources=[NUMPY_DOC])
    a.p("An array is a grid of numbers of one type. Its **shape** lists how many numbers lie along each **axis**.")
    a.h2("Read a shape")
    a.p("A shape is a tuple. The array `np.zeros((2, 3))` has shape `(2, 3)`. Axis 0 has 2 rows. Axis 1 has 3 columns.")
    a.p("Use the demonstration to change the shape. Select a cell to see its index. The index has one number for each axis.")

    def frames(shape):
        r, c = (int(s) for s in shape.split("x"))
        arr = np.arange(1, r * c + 1).reshape(r, c) * 10
        ids = lambda i, j: f"a{i}_{j}"
        return [
            frame("shape", "Read the shape", f"The array has shape ({r}, {c}). It has {r} rows and {c} columns, so {r * c} numbers.",
                  "The first number of a shape is the length of axis 0. The second number is the length of axis 1.",
                  "The values stay the same.",
                  matrix(arr, id_fn=ids, row_labels=[f"row {i}" for i in range(r)], col_labels=[f"col {j}" for j in range(c)],
                         title=f"shape ({r}, {c})"),
                  text(f"Select a cell to read its index.", by_sel={ids(i, j): f"`a[{i}, {j}]` = {num(arr[i, j])}. The first index picks the row. The second index picks the column."
                                                                  for i in range(r) for j in range(c)})),
            frame("index", "Index with a slice", f"`a[:, 0]` takes column 0 of every row and gives shape ({r},).",
                  "A colon keeps a whole axis. A number picks one position and removes the axis.",
                  "The array does not change.",
                  matrix(arr, id_fn=ids, cls_fn=lambda i, j: "pos" if j == 0 else "dim", row_labels=[f"row {i}" for i in range(r)],
                         col_labels=[f"col {j}" for j in range(c)], title="a[:, 0] is the highlighted column"),
                  text(f"`a[:, 0]` = {vec(arr[:, 0])}. Its shape is `({r},)`: one axis, because the column index removed axis 1.")),
            frame("flat", "Rows come first in memory", f"NumPy stores the numbers row by row: {vec(arr.ravel())}.",
                  "This row-major order is why reshaping keeps the order of the numbers.", "The values stay the same.",
                  tokens([{"t": num(v), "sub": f"flat {n}"} for n, v in enumerate(arr.ravel())], title="Flat order in memory")),
        ]

    a.demo("shape-explorer", "Shapes and indices", frames,
           params=[("shape", "Shape", [("2x3", "2 × 3"), ("3x2", "3 × 2"), ("3x3", "3 × 3"), ("1x4", "1 × 4")])],
           facts=[("a.shape and a[i, j] values computed with NumPy for each shape option", "numpy")],
           help=[{"step_id": "shape", "help_type": "common-mistake", "content": "A shape `(3,)` has one axis. A shape `(3, 1)` has two axes. NumPy treats them differently."},
                 {"step_id": "index", "help_type": "smaller-example", "content": "Take `a = [[10, 20], [30, 40]]`. Then `a[:, 0]` is `[10, 30]` and `a[0, :]` is `[10, 20]`."}])
    a.h2("Rules to remember")
    a.ul(["The shape `(n, d)` means `n` rows and `d` columns. Many ML arrays use `n` for the number of samples and `d` for the number of features.",
          "`a[i, j]` reads one number. The index has one entry per axis.",
          "`a[:, j]` keeps all rows and picks column `j`. The result has one axis fewer.",
          "`a.reshape(...)` changes the shape but not the order of the numbers."])
    a.callout("mistake", "Do not confuse `(3,)` and `(3, 1)`. The first is a flat vector. The second is a column with one number per row. They behave differently when you add them to a matrix.")
    a.h2("Insert an axis")
    a.p("Write `None` (or `np.newaxis`) inside an index to insert a new axis of size 1. For example, a vector `v` with shape `(3,)` gives `v[:, None]` with shape `(3, 1)` and `v[None, :]` with shape `(1, 3)`. This is the tool you need for [[article:broadcasting|broadcasting]].")
    return [a]


# =========================================================================== prerequisite: broadcasting
def broadcasting():
    a = Article("broadcasting", "Broadcasting", kind="prerequisite", prerequisites=["arrays-and-shapes"], sources=[NUMPY_DOC])
    a.p("Broadcasting lets NumPy combine arrays of different shapes without loops. It compares the two shapes **from the right**. Two sizes fit together when they are equal or when one of them is 1.")
    a.scene_demo("broadcast-concept", "Broadcasting A (2, 1, 3) + B (1, 4, 1)", lambda: broadcast_concept_scene(), help=[
        {"step_id": "inputs", "help_type": "prerequisite", "content": "Shapes and axes are explained here.", "prerequisite_id": "arrays-and-shapes"},
        {"step_id": "align", "help_type": "smaller-example", "content": "Take shapes `(3,)` and `(2, 3)`. Write `(3,)` as `(1, 3)`. Now both have two axes. Axis 1 compares 3 with 3. Axis 0 compares 1 with 2."},
        {"step_id": "align", "help_type": "common-mistake", "content": "Do not start at the left. `(3,)` and `(4, 3)` match because both end in 3."},
        {"step_id": "expand", "help_type": "common-mistake", "content": "The pale blocks are not new memory. NumPy reads the stored numbers again. Only the result is a new array."},
        {"step_id": "add", "help_type": "expanded-equation", "content": "For the result cell $(1, 2, 2)$: $A[1,0,2] + B[0,2,0] = 6 + 30 = 36$. Index 0 is used on every axis where the input has size 1."},
        {"step_id": "add", "help_type": "code-connection", "content": "`C = A + B` does all 24 additions. The loop over positions happens inside NumPy."}],
        facts=[("C = A + B has shape (2, 4, 3) and C[i, j, k] = A[i, 0, k] + B[0, j, 0] for all 24 cells, computed by NumPy", "numpy"), ("C[1, 2, 2] = 6 + 30 = 36", "numpy")])
    a.hero("One rule. Every pair.", "Understand NumPy broadcasting by following the numbers.")
    a.h2("The four steps as still images")
    a.p("These images show the end of each animation step. They work without 3D and are easy to compare side by side.")
    a.gallery([
        ("step1-inputs", "Teal block A with shape 2 by 1 by 3 holding 1 to 6, and amber block B with shape 1 by 4 by 1 holding 10, 20, 30, 40, next to an empty frame for the result.", "Step 1: the inputs", "**1. Inputs.** `A` has shape (2, 1, 3). `B` has shape (1, 4, 1)."),
        ("step2-align", "The same blocks with a shape table: A is 2, 1, 3. B is 1, 4, 1. Reading from the right, 3 with 1 gives 3, 1 with 4 gives 4, and 2 with 1 gives 2.", "Step 2: align the shapes from the right", "**2. Align.** Read the shapes from the right. Every pair is equal or contains a 1."),
        ("step3-expand", "Block A and block B each grown with pale copies until both have shape 2 by 4 by 3, labelled same values, more positions.", "Step 3: expand the size-1 axes", "**3. Expand.** The pale blocks are the same stored numbers, read again."),
        ("step4-add", "The violet result block with shape 2 by 4 by 3. The first layer reads 11, 21, 31, 41 across, then 12, 22, 32, 42, and so on up to 46.", "Step 4: add matching cells", "**4. Add.** Cell [i, j, k] is A[i, 0, k] + B[0, j, 0]. The result has 24 numbers."),
    ])
    a.h2("Read shapes from the right")
    a.p("NumPy compares shapes starting from the rightmost axis. Sizes are compatible when they are equal or when one of them is 1. Missing axes on the left count as size 1.")
    a.table(["Axis (from the right)", "A", "B", "Result"],
            [["2", "3", "1", "3"], ["1", "1", "4", "4"], ["0", "2", "1", "2"]], caption="Shape alignment for this example: (2, 1, 3) with (1, 4, 1) gives (2, 4, 3)")
    a.callout("definition", "A size of 1 \"stretches\" in a conceptual way. NumPy does not copy the data. It reads the same numbers again for each position.", title="No copy is made")
    a.h2("The rule in three steps")
    a.ol(["Write the two shapes and align them on the right. Add leading 1s to the shorter shape.",
          "For each pair of sizes: equal sizes match. A size of 1 stretches to the other size. Any other pair is an error.",
          "The result shape takes the larger size of each pair."])
    a.h2("Try your own shapes")
    a.p("Change the sizes of A and B in this 2D version. Step through the rule. Select an output cell in the last step to see which cell of A and which cell of B it reads.")
    pairs = [((2, 1, 3), (1, 4, 1)), ((3,), (4,)), ((2, 3), (3,)), ((3, 1), (1, 4)), ((2, 3), (4,)), ((4, 4, 4), (4, 4, 4))]
    demo = {"id": "broadcast-explorer", "component": "broadcast-trace", "seed": None,
            "input_schema": {"defaults": {"shapeA": [2, 1, 3], "shapeB": [1, 4, 1]}, "max_dims": 3, "max_size": 4},
            "steps": [{"id": "align", "title": "Write both shapes with 3 axes"}, {"id": "check-axes", "title": "Compare each pair of axes"},
                      {"id": "result-shape", "title": "Read the result shape"}, {"id": "trace-cell", "title": "Trace one output cell"}],
            "expected": [broadcast_case(*s) for s in pairs],
            "selection_links": ["output cell C[i,j,k] -> A[i',j',k'], B[i'',j'',k''], equation, code line 3"],
            "help": [
                {"step_id": "align", "help_type": "smaller-example", "content": "Take shapes `(3,)` and `(2, 3)`. Write `(3,)` as `(1, 3)`. Now both have 2 axes."},
                {"step_id": "check-axes", "help_type": "common-mistake", "content": "Do not start at the left. `(3,)` and `(4, 3)` match because both end in 3."},
                {"step_id": "check-axes", "help_type": "prerequisite", "content": "Shapes and axes are explained here.", "prerequisite_id": "arrays-and-shapes"},
                {"step_id": "trace-cell", "help_type": "expanded-equation", "content": "For output cell $(1,2,1)$ with the default shapes: A is read at $(1,0,1)$ and B at $(0,2,0)$. The result is $5 + 30 = 35$."},
                {"step_id": "trace-cell", "help_type": "code-connection", "content": "The highlighted last code line is one cell of `C = A + B`."}]}
    a.demo_raw(demo, "Broadcasting two arrays of your own shapes")
    a.h2("Examples")
    a.table(["A shape", "B shape", "Result", "Why"],
            [["(2, 1, 3)", "(1, 4, 1)", "(2, 4, 3)", "Every pair has a 1 or equal sizes"],
             ["(3,)", "(4,)", "error", "3 and 4 differ and neither is 1"],
             ["(2, 3)", "(3,)", "(2, 3)", "(3,) is read as (1, 3)"],
             ["(3, 1)", "(1, 4)", "(3, 4)", "A column plus a row gives a grid"]])
    a.callout("boundary", "If both shapes are `(n, d)` with the same `n` and `d`, the operation is element by element. No axis is stretched. This is the case where a missing axis can hide a mistake.")
    a.h2("Why it matters for ML")
    a.ul(["Adding a bias vector `(d,)` to a batch `(n, d)`.", "Scaling each column by a vector.",
          "All pairs of rows: `x[:, None, :] - y[None, :, :]` (question [[q:30|030]]).",
          "Gradients of broadcast values must be summed back to the original shape (question [[q:108|108]])."])
    return [a]


# =========================================================================== prerequisite: views and copies
def views_and_copies():
    a = Article("views-and-copies", "Views and copies", kind="prerequisite", prerequisites=["arrays-and-shapes"], sources=[NUMPY_COPIES])
    a.p("A NumPy array has two parts. One part is the block of memory that holds the numbers. The other part describes how to read that block: the shape and the steps. A **view** is a new description of the same memory. A **copy** is a new block of memory.")
    a.h2("Which operations make which")
    a.table(["Operation", "Result", "Shares memory?"],
            [["`a[::2]` (basic slice)", "view", "yes"], ["`a[2:5]`", "view", "yes"], ["`a.reshape(...)` of a contiguous array", "view", "yes"],
             ["`a[[0, 2, 4]]` (index list, fancy indexing)", "copy", "no"], ["`a[a > 0]` (boolean mask)", "copy", "no"], ["`a.copy()`", "copy", "no"]],
            caption="Checked with np.shares_memory")
    # verify the table with NumPy
    base = np.arange(8)
    assert np.shares_memory(base, base[::2]) and np.shares_memory(base, base[2:5]) and np.shares_memory(base, base.reshape(2, 4))
    assert not np.shares_memory(base, base[[0, 2, 4]]) and not np.shares_memory(base, base[base > 0]) and not np.shares_memory(base, base.copy())
    a.p("The rule of thumb: slices with a start, stop, and step make views. Index lists and masks make copies.")
    a.callout("mistake", "Writing into a view changes the original array. Writing into a copy does not. This is the usual source of surprising changes in ML code, for example when you \"clean\" a slice of a dataset in place.")
    a.h2("How a view works")
    a.p("A view stores a start position and a **stride**. The stride is how many bytes to jump to reach the next element. The slice `a[::2]` uses a stride of two elements. It owns no numbers of its own.")
    return [a]


# =========================================================================== lesson 030
def move_section(a, heading, before):
    """Move the section that starts at `heading` so that it sits just before the section `before`."""
    idx = lambda h: next(i for i, b in enumerate(a.blocks) if b.get("type") == "heading" and b["text"] == h)
    start = idx(heading)
    end = next(i for i in range(start + 1, len(a.blocks)) if a.blocks[i].get("type") == "heading" and a.blocks[i].get("level") == 2)
    chunk = a.blocks[start:end]
    del a.blocks[start:end]
    at = idx(before)
    a.blocks[at:at] = chunk


def lesson_030():
    q30 = kit.supplied(30)
    X = np.array([[1, 2], [3, 4]])
    Y_ALL = np.array([[0, 1], [1, 1], [2, 2], [4, 0]])
    n, d = X.shape

    # ---- reference: explicit loops, asserted equal to the supplied solution
    def by_loops(x, y):
        out = np.zeros((len(x), len(y), x.shape[1]), dtype=x.dtype)
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(x.shape[1]):
                    out[i, j, k] = x[i, k] - y[j, k]
        return out

    for m in range(1, 5):
        ref = by_loops(X, Y_ALL[:m])
        assert np.array_equal(ref, q30(X, Y_ALL[:m])), "lesson loop differs from the supplied solution"
        assert ref.shape == (n, m, d)

    a = Article("lesson-q030-pairwise-differences", "Pairwise differences with broadcasting", kind="lesson",
                prerequisites=["arrays-and-shapes", "broadcasting"], exercises=[(30, "deep-lesson"), (27, "related")],
                sources=[NUMPY_DOC, ("Question 030 source", kit.record(30)["url"])])

    # ---- figure: the geometry
    fig, ax = plt.subplots(figsize=(5.2, 4), dpi=100)
    Y3 = Y_ALL[:3]
    ax.scatter(*X.T, s=90, color="#0072b2", label="x points", zorder=3)
    ax.scatter(*Y3.T, s=90, color="#d55e00", marker="s", label="y points", zorder=3)
    for i, p in enumerate(X):
        ax.annotate(f"x{i}", p, textcoords="offset points", xytext=(7, 6), color="#0072b2")
    for j, p in enumerate(Y3):
        ax.annotate(f"y{j}", p, textcoords="offset points", xytext=(7, -13), color="#d55e00")
    for i, xp in enumerate(X):
        for j, yp in enumerate(Y3):
            ax.annotate("", xy=xp, xytext=yp, arrowprops=dict(arrowstyle="->", color="#666", lw=1, shrinkA=6, shrinkB=6, alpha=.7))
    ax.set_xlim(-0.8, 5); ax.set_ylim(-0.8, 5); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_title("Each arrow is one difference x[i] − y[j]")
    ax.legend(loc="upper left")

    a.h2("The puzzle")
    a.p("You have `n` points in a list `x` and `m` points in a list `y`. Each point has `d` numbers. You want the difference `x[i] - y[j]` for **every** pair `(i, j)`. That is `n × m` differences.")
    a.p("A loop is easy to write. NumPy can do all pairs in one line, with no loop. This lesson shows why that line works.")
    a.figure("pairs", fig, alt="Plot of two x points and three y points on a plane, with an arrow from each y point to each x point. Each arrow is a difference.",
             description="Geometry of pairwise differences: 2 x points, 3 y points, 6 difference arrows",
             caption="Two x points and three y points give six differences. The result has one vector per arrow.")
    a.h2("What the question asks")
    a.p(kit.record(30)["question"].strip())
    a.callout("definition", "Input `x` has shape `(n, d)`. Input `y` has shape `(m, d)`. The output has shape `(n, m, d)`, and `out[i, j] = x[i] - y[j]`. Both inputs may be lists. The solution converts them with `np.asarray`.", title="Inputs and output")
    a.h2("What you need first")
    a.ul(["[[article:arrays-and-shapes|Array shapes and indexing]]: shapes, axes, and the index `None` that inserts an axis.",
          "[[article:broadcasting|Broadcasting]]: how two shapes combine, and why a size of 1 stretches."])

    # ---- demo 1: build the answer
    def build(m):
        Y = Y_ALL[:m]
        out = by_loops(X, Y)
        pid = lambda i, j: f"p{i}_{j}"
        xid = lambda i, k: f"X{i}.{k}"
        yid = lambda j, k: f"Y{j}.{k}"
        pair = lambda v: f"({num(v[0])}, {num(v[1])})"
        by_pair = {pid(i, j): f"Pair ({i}, {j}): `x[{i}] - y[{j}]` = {pair(X[i])} − {pair(Y[j])} = {pair(out[i, j])}." for i in range(n) for j in range(m)}
        links = lambda i, j: [xid(i, 0), xid(i, 1), yid(j, 0), yid(j, 1)]
        xg = lambda title="x  shape (n, d)": matrix(X, id_fn=lambda i, k: xid(i, k), row_labels=[f"x{i}" for i in range(n)], col_labels=["k=0", "k=1"], title=title)
        yg = lambda title="y  shape (m, d)": matrix(Y, id_fn=lambda j, k: yid(j, k), row_labels=[f"y{j}" for j in range(m)], col_labels=["k=0", "k=1"], title=title)
        pair_grid = lambda title, getter, cls: grid([[cell(pair(getter(i, j)), id=pid(i, j), links=links(i, j), cls=cls) for j in range(m)] for i in range(n)],
                                                    [f"i={i}" for i in range(n)], [f"j={j}" for j in range(m)], title=title)
        geometry = plot((-0.8, 5), (-0.8, 5), [series("scatter", X, "x points"), series("scatter", Y, "y points")],
                        alt=f"Plot of {n} x points and {m} y points.", xlabel="feature 0", ylabel="feature 1", width=420, height=330)
        arrows = [{"from": [float(Y[j][0]), float(Y[j][1])], "to": [float(X[i][0]), float(X[i][1])]} for i in range(n) for j in range(m)]
        geometry_arrows = dict(geometry, arrows=arrows,
                               alt=f"Plot of {n} x points and {m} y points with {n * m} arrows. Each arrow runs from a y point to an x point and is one difference.")
        return [
            frame("inputs", "Start with two matrices", f"`x` has shape ({n}, {d}) and `y` has shape ({m}, {d}).",
                  "Each row is one point. The question needs one difference for each pair of rows.", "Nothing is computed yet.",
                  xg(), yg(), geometry),
            frame("insert-axes", "Insert one axis in each input", f"`x[:, None, :]` has shape ({n}, 1, {d}). `y[None, :, :]` has shape (1, {m}, {d}).",
                  "The new axes of size 1 will stretch in the next step. The old axis 0 of `y` moves to axis 1.", "The numbers stay the same. `None` only adds an axis.",
                  xg("x"), yg("y"),
                  grid([[cell(pair(X[i]), id=f"x{i}", links=[xid(i, 0), xid(i, 1)])] for i in range(n)], [f"i={i}" for i in range(n)], ["j (size 1)"], title=f"x[:, None, :]  shape ({n}, 1, {d})"),
                  grid([[cell(pair(Y[j]), id=f"y{j}", links=[yid(j, 0), yid(j, 1)]) for j in range(m)]], ["i (size 1)"], [f"j={j}" for j in range(m)], title=f"y[None, :, :]  shape (1, {m}, {d})"),
                  text("Each cell holds one point (a pair of numbers). Axis 2 is inside the cell.")),
            frame("stretch", "Let the size-1 axes stretch", f"Both shapes become ({n}, {m}, {d}) in the comparison. Each cell of `x` is read {m} times. Each cell of `y` is read {n} times.",
                  "A size of 1 matches any size. NumPy reads the same numbers again. It does not copy them.", "The stored arrays do not change.",
                  xg("x"), yg("y"), pair_grid("x as read for each pair", lambda i, j: X[i], "src"), pair_grid("y as read for each pair", lambda i, j: Y[j], "add"),
                  text("Select a cell. The same pair is marked in both grids, with the source rows.", by_sel=by_pair)),
            frame("subtract", "Subtract cell by cell", f"Each pair gives `x[i] - y[j]`. The result has shape ({n}, {m}, {d}) and {n * m} cells.",
                  "Subtraction works on matching cells. Because every cell now has a partner, no loop is needed.", "`x` and `y` stay as they were.",
                  xg("x"), yg("y"), pair_grid("out = x[:, None, :] - y[None, :, :]", lambda i, j: out[i, j], "pos"),
                  equation(r"\mathrm{out}_{i,j,k} = x_{i,k} - y_{j,k}",
                           by_sel={pid(i, j): rf"\mathrm{{out}}_{{{i},{j}}} = x_{{{i}}} - y_{{{j}}} = {pair(X[i])} - {pair(Y[j])} = {pair(out[i, j])}" for i in range(n) for j in range(m)},
                           caption="Select an output cell to see its equation with numbers."),
                  geometry_arrows),
            frame("check", "Compare with the supplied solution", f"The supplied solution returns the same array. Its shape is ({n}, {m}, {d}).",
                  "Line 5 of the solution is the whole computation.", "The numbers do not change.",
                  code_view(kit.solution_source(30), [5], label="Supplied solution for question 030"),
                  text(f"Check: `np.array_equal(out, q030(x, y))` is true for {m} point(s) in `y`.")),
        ]

    a.h2("Watch it happen in 3D")
    a.p("Press **Play**. `x` stands on the left wall and `y` hangs from the ceiling. Copies slide into a box, and every pair fuses into one result cube. Drag the picture to rotate it. Scroll to zoom. Click a cube to see which inputs made it. Use the chapter buttons to replay one part.")
    a.scene_demo("pairwise-3d", "Pairwise differences in 3D", lambda m: pairwise_scene(m),
                 params=[("m", "Number of y points", [(3, "3"), (2, "2"), (4, "4")])], help=[
                     {"step_id": "inputs", "help_type": "prerequisite", "content": "Shapes and axes are explained here.", "prerequisite_id": "arrays-and-shapes"},
                     {"step_id": "insert-axes", "help_type": "smaller-example", "content": "Take `x = [10, 20]` (shape `(2,)`) and `y = [1, 2, 3]`. Then `x[:, None] - y[None, :]` has shape `(2, 3)`. Row `i`, column `j` holds `x[i] - y[j]`."},
                     {"step_id": "stretch", "help_type": "prerequisite", "content": "The stretching rule is explained here.", "prerequisite_id": "broadcasting"},
                     {"step_id": "stretch", "help_type": "common-mistake", "content": "Stretching is conceptual. NumPy does not build an `(n, m, d)` copy of `x` before it subtracts."},
                     {"step_id": "subtract", "help_type": "expanded-equation", "content": "For the pair $(1, 2)$: $x_1 - y_2$ in feature 0 is $4 - 2 = 2$ when `y` has at least 3 points."},
                     {"step_id": "result", "help_type": "code-connection", "content": "`np.asarray(x)[:, None, :] - np.asarray(y)[None, :, :]` in the supplied solution builds exactly this box."}],
                 facts=[("the box holds out[i, j, k] = x[i, k] - y[j, k], equal to the supplied q030 and an explicit triple loop, for m = 2, 3, 4", "supplied solution q030"),
                        ("out has shape (n, m, d) = (2, m, 3)", "numpy")])
    a.p("If you prefer to read, open **Text version** under the animation. It lists every chapter and lets you select numbers without the 3D view.")

    # ---- the mathematics
    a.h2("The mathematics")
    a.p("Write the two matrices as $X \\in \\mathbb{R}^{n \\times d}$ and $Y \\in \\mathbb{R}^{m \\times d}$. The result is a three-axis array $D$ with one entry for each triple $(i, j, k)$:")
    a.eq(r"D_{i,j,k} = X_{i,k} - Y_{j,k}", caption="Index $i$ picks a row of $X$. Index $j$ picks a row of $Y$. Index $k$ picks a feature.")
    a.ul(["$i$ runs over the $n$ rows of $X$.", "$j$ runs over the $m$ rows of $Y$.", "$k$ runs over the $d$ features. It is shared by both inputs.",
          "The output has $n \\cdot m \\cdot d$ numbers. For the demo it has $2 \\cdot 3 \\cdot 2 = 12$."])
    a.p("In shapes, the operation is:")
    a.eq(r"(n,\,1,\,d) - (1,\,m,\,d) \;\to\; (n,\,m,\,d)")
    a.p("The squared distance between two rows follows from the result. Sum the squares over $k$: $\\lVert x_i - y_j \\rVert^2 = \\sum_k D_{i,j,k}^2$. This is how many nearest-neighbour and clustering methods compute all distances at once.")

    # ---- failure case in 3D
    a.h2("When the shapes do not fit")
    sc_err, err_text = mismatch_scene()
    a.p("Now `x` has 3 features and `y` has only 2. Watch where the animation stops.")
    a.scene_demo("pairwise-mismatch", "Different numbers of features", lambda: mismatch_scene()[0], help=[
        {"step_id": "conflict", "help_type": "common-mistake", "content": "Axes `i` and `j` stretch because each contains a 1. The feature axis has no 1, so sizes 3 and 2 cannot match."}],
        facts=[(f"x[:, None, :] - y[None, :, :] with feature sizes 3 and 2 raises ValueError: {err_text}", "numpy")])
    a.callout("mistake", f"NumPy reports: `{err_text}`. The three numbers per shape are `(n, 1, d)` and `(1, m, d)`. Both inputs must have the same `d`.", title="The error message")

    # ---- failure case: break and repair
    a.h2("Break it, then repair it")
    a.p("Choose a wrong version of the expression. Read the shape table. Then repair the expression. The shapes here are `x` with `(2, 2)` and `y` with `(3, 2)`.")
    xs, ys = X, Y_ALL[:3]
    attempts = {
        "missing": ("x - y", lambda: xs - ys),
        "swapped": ("x[None, :, :] - y[:, None, :]", lambda: xs[None, :, :] - ys[:, None, :]),
        "sign": ("y[None, :, :] - x[:, None, :]", lambda: ys[None, :, :] - xs[:, None, :]),
        "correct": ("x[:, None, :] - y[None, :, :]", lambda: xs[:, None, :] - ys[None, :, :]),
    }
    good = by_loops(xs, ys)

    def fix(attempt):
        expr, run = attempts[attempt]
        try:
            res = run()
            err = None
        except ValueError as e:
            res, err = None, str(e)
        sh = {"missing": ("(2, 2)", "(3, 2)"), "swapped": ("(1, 2, 2)", "(3, 1, 2)"), "sign": ("(1, 3, 2)", "(2, 1, 2)"), "correct": ("(2, 1, 2)", "(1, 3, 2)")}[attempt]
        shape_grid = grid([[cell(sh[0]), cell(sh[1]), cell(str(res.shape) if res is not None else "error")]], None,
                          ["left operand", "right operand", "result"], title="Shapes after the expression")
        if attempt == "missing":
            outcome, why = "NumPy raises an error.", "Axis 0 has sizes 2 and 3. Neither is 1."
            assert err and "shape" in err
        elif attempt == "swapped":
            outcome, why = f"The result has shape {res.shape}, not (2, 3, 2).", "The axes of the pair are in the wrong order. Entry `[j, i]` holds `x[i] - y[j]`."
            assert res.shape == (3, 2, 2) and np.array_equal(res.transpose(1, 0, 2), good)
        elif attempt == "sign":
            outcome, why = "The shape is right but every number has the wrong sign.", "The operands are in the wrong order. `y[j] - x[i]` is the negative of `x[i] - y[j]`."
            assert np.array_equal(res, -good)
        else:
            outcome, why = "The shape is (2, 3, 2) and the values match the loop and the supplied solution.", "Both new axes are in the right place."
            assert np.array_equal(res, good) and np.array_equal(res, q30(xs, ys))
        return [
            frame("expression", "Write the expression", f"The expression is `{expr}`.", "Each version inserts axes in a different place.", "`x` and `y` are the same arrays.",
                  code_view(f"out = {expr}", [1], label="Expression to test")),
            frame("shapes", "Check the shapes", f"The operand shapes are {sh[0]} and {sh[1]}.", "NumPy aligns the shapes from the right.", "The values are unchanged.", shape_grid),
            frame("outcome", "Read the outcome", outcome, why, "The inputs are not modified.",
                  text(f"Error from NumPy: `{err}`" if err else f"First pair `[0, 0]` = {vec(res[0, 0])}. Correct `[0, 0]` = {vec(good[0, 0])}.")),
        ]

    a.demo("pairwise-fix", "Which expression is right?", fix,
           params=[("attempt", "Expression", [("missing", "x - y"), ("swapped", "x[None] - y[:, None]"), ("sign", "y[None] - x[:, None]"), ("correct", "x[:, None] - y[None]")])],
           help=[{"step_id": "shapes", "help_type": "smaller-example", "content": "Try shapes `(2, 2)` and `(2, 2)`. `x - y` works there. It subtracts row by row, not pair by pair. A missing axis can hide a mistake."},
                 {"step_id": "outcome", "help_type": "common-mistake", "content": "Check the shape first. Then check one cell by hand: `out[0, 1]` must equal `x[0] - y[1]`."}],
           facts=[("x - y raises ValueError for shapes (2,2) and (3,2); swapped gives shape (3,2,2) = transpose of the answer; sign is the negative; correct equals the supplied q030", "numpy + supplied solution q030")])
    a.set_default("pairwise-fix", attempt="missing")
    a.compare("pairwise-fix", "the order of the inserted axes",
              [("Axes inserted as in the solution", {"attempt": "correct"}), ("Axes inserted in the other order", {"attempt": "swapped"})],
              "Both versions run without an error. Only one has the shape `(n, m, d)`. Always check the output shape and one cell.", frame="shapes")
    a.callout("mistake", "`x - y` can run without an error when `x` and `y` have the same shape. It then subtracts matching rows only and returns shape `(n, d)`. NumPy cannot tell that you wanted all pairs.", title="A silent mistake")
    a.callout("boundary", "If `x` or `y` has no rows (`n = 0` or `m = 0`), the output has shape `(0, m, d)` or `(n, 0, d)`. The expression still works. It returns an empty array.")

    # ---- memory
    a.h2("Does this copy the data?")
    a.p("The two inputs are not copied. Broadcasting reads the same numbers again for each pair. The **result** is a new array with `n × m × d` numbers, so the memory for the result grows with `n × m`. Inserting axes with `None` makes views. Learn about views in [[article:views-and-copies|Views and copies]] and test your understanding with question [[q:27|027]].")
    a.p("For large inputs the result can be too big. With `n = m = 10000` and `d = 128`, the result has about 12.8 billion numbers. If you only need distances, use the identity $\\lVert x - y \\rVert^2 = \\lVert x \\rVert^2 - 2 x \\cdot y + \\lVert y \\rVert^2$ and a matrix product instead.")

    # ---- the solution
    a.h2("Read the supplied solution")
    a.walkthrough(30, [
        (4, "def q030", "The function takes the two collections of points."),
        (5, "np.asarray(x)", "`np.asarray` accepts lists or arrays and gives an array without copying when it can."),
        (5, "[:, None, :]", "This inserts an axis of size 1 between the rows and the features. `x` becomes `(n, 1, d)`."),
        (5, "- np.asarray(y)[None, :, :]", "This inserts the size-1 axis in front. `y` becomes `(1, m, d)`. The subtraction broadcasts the two shapes to `(n, m, d)`."),
    ], title="Question 030: the whole solution is one line")

    # ---- practice
    a.h2("Test yourself")
    a.check("predict-shape", "prediction", "Predict the shape of `x[:, None, :] - y[None, :, :]` when `x` has shape (4, 3) and `y` has shape (5, 3). Do this before you use the demo.",
            "shape", "(4, 5, 3)", expl("`x[:, None, :]` has shape (4, 1, 3). `y[None, :, :]` has shape (1, 5, 3). The pairs are (4, 1), (1, 5), (3, 3), so the result is (4, 5, 3).",
                                       "Insert the axes, then compare from the right: (4, 1, 3) with (1, 5, 3). Each axis takes the larger size of its pair.",
                                       **{"(5, 4, 3)": "That is the shape of the swapped version. `x` supplies axis 0, so its size 4 comes first.", "(4, 3)": "That is the shape of `x`. The result has a new axis for the `y` rows.", "error": "These shapes are compatible. Each pair of sizes is equal or contains a 1."}),
            case={"x": "(4, 3)", "y": "(5, 3)"})
    xn, yn = np.array([[2, 5, 1], [7, 3, 4]]), np.array([[1, 1, 1], [4, 0, 2]])
    ans = int(q30(xn, yn)[1, 0, 1])
    assert ans == 7 - 1 if False else ans == xn[1, 1] - yn[0, 1]
    a.check("new-cell", "new-case", "Use the supplied solution on new data. `x = [[2, 5, 1], [7, 3, 4]]` and `y = [[1, 1, 1], [4, 0, 2]]`. What is `out[1, 0, 1]`?",
            "number", ans, expl(f"`out[1, 0, 1] = x[1][1] - y[0][1] = {xn[1, 1]} - {yn[0, 1]} = {ans}`. The first index picks a row of `x`. The second index picks a row of `y`. The third index picks the feature.",
                                 f"Read the three indices as (row of x, row of y, feature). Here that is `x[1][1] - y[0][1]`.",
                                 **{str(-ans): "The sign is reversed. The result is `x` minus `y`, not `y` minus `x`.", str(int(xn[0, 1] - yn[1, 1])): "You read `x[0]` and `y[1]`. The first index is the row of `x` and the second is the row of `y`."}),
            tolerance=0, case={"x": xn.tolist(), "y": yn.tolist()})
    a.check("pick-expression", "prediction", "Which expression gives an array of shape (m, n, d), with `out[j, i] = x[i] - y[j]`?", "choice",
            "x[None, :, :] - y[:, None, :]",
            expl("`x[None, :, :]` has shape (1, n, d) and `y[:, None, :]` has shape (m, 1, d). The result has shape (m, n, d).", "Put the `y` axis first by inserting the new axis for `x` in front.",
                 **{"x[:, None, :] - y[None, :, :]": "This gives shape (n, m, d), the shape of question 030.", "x - y": "This does not insert axes. It fails or subtracts matching rows.", "x.T - y.T": "The transposes have shapes (d, n) and (d, m). They do not match."}),
            options=["x[:, None, :] - y[None, :, :]", "x[None, :, :] - y[:, None, :]", "x - y", "x.T - y.T"])
    a.check("explain-copy", "explain-back", "In your own words: does the expression copy `y` once for every row of `x`? What does NumPy do instead?", "text", None,
            {"correct": "No. NumPy does not build a stretched copy of `y`. It reads the same stored numbers again for each position, because a size-1 axis has a stride of zero. Only the result array is new memory."})

    # ---- code practice
    a.h2("Write it yourself")
    a.playground(30, '''
        import numpy as np


        def pairwise_differences(x, y):
            # Return an array of shape (n, m, d) with out[i, j] = x[i] - y[j].
            # Use axis insertion. Do not use a loop.
            raise NotImplementedError
        ''', checks=[
        ("output shape is (n, m, d)", "r = pairwise_differences(np.zeros((2, 3)), np.zeros((4, 3)))\nassert r.shape == (2, 4, 3), f'shape {r.shape}'"),
        ("values: out[i, j] = x[i] - y[j]", "x = np.array([[1., 2.], [3., 4.]]); y = np.array([[0., 1.], [1., 1.], [2., 2.]])\nr = pairwise_differences(x, y)\nfor i in range(2):\n    for j in range(3):\n        assert np.array_equal(r[i, j], x[i] - y[j]), f'cell {i},{j}'"),
        ("accepts lists", "r = pairwise_differences([[1, 2]], [[0, 1], [2, 2]])\nassert np.asarray(r).tolist() == [[[1, 1], [-1, 0]]], np.asarray(r).tolist()"),
        ("does not change the inputs", "x = np.array([[1., 2.]]); y = np.array([[3., 4.]])\npairwise_differences(x, y)\nassert x.tolist() == [[1., 2.]] and y.tolist() == [[3., 4.]]"),
    ], note="Write `pairwise_differences(x, y)` for question 030. Insert axes so that the shapes broadcast. The checks run after your code.", max_input_size=200)
    a.p("When you finish, compare with the walkthrough above. Then read how [[q:30|question 030]] connects to gradients in [[q:108|question 108]], where a broadcast value must be summed back to its original shape.")
    move_section(a, "Watch it happen in 3D", before="What the question asks")
    return a


# =========================================================================== lesson 027
def lesson_027():
    q27 = kit.supplied(27)
    base = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    a_out, b_out = q27(base)
    assert a_out.tolist() == [-1, 20, -1, 40, -1, 60, -1, 80] and b_out.tolist() == base.tolist()

    OPS = {
        "slice": ("view = a[::2]", "A slice with a step", lambda a: a[::2], True),
        "fancy": ("copied = a[[0, 2, 4, 6]]", "An index list (fancy indexing)", lambda a: a[np.array([0, 2, 4, 6])], False),
        "mask": ("copied = a[a % 20 == 10]", "A boolean mask", lambda a: a[a % 20 == 10], False),
        "reshape": ("view = a.reshape(2, 4)", "reshape of a contiguous array", lambda a: a.reshape(2, 4), True),
    }
    n = len(base)

    def frames(op):
        code, label, make, shares = OPS[op]
        a = base.copy()
        r = make(a)
        assert bool(np.shares_memory(a, r)) == shares, f"{op}: shares_memory differs from the lesson"
        flat = r.ravel()
        src_idx = {"slice": [0, 2, 4, 6], "fancy": [0, 2, 4, 6], "mask": [0, 2, 4, 6], "reshape": list(range(8))}[op]
        rid = lambda k: f"r{k}"
        aid = lambda k: f"a{k}"
        # the result cell k is the value at base position src_idx[k]
        def a_strip(vals, mark=()):
            return tokens([{"t": num(v), "sub": f"a[{k}]", "id": aid(k), "links": [], "cls": "add" if k in mark else None} for k, v in enumerate(vals)], title="a  (the original array)")
        def r_strip(vals, linked, title="result"):
            return tokens([{"t": num(v), "sub": f"[{k}]", "id": rid(k), "links": [aid(src_idx[k])] if linked else [], "cls": "src"} for k, v in enumerate(vals)], title=title)
        # write -1 into the result
        w = make(a.copy())
        a2 = a.copy()
        r2 = make(a2)
        r2[...] = -1
        changed = [k for k in range(n) if a2[k] != a[k]]
        assert (len(changed) > 0) == shares
        return [
            frame("start", "Start with an array", f"`a` holds {n} numbers: {vec(a)}.", "Every NumPy array has a block of memory. The numbers sit in it in order.", "Nothing has been copied yet.",
                  a_strip(a)),
            frame("make", "Make the second array", f"`{code}` gives {vec(flat)}. This is a **{'view' if shares else 'copy'}**.",
                  "A view reads the same memory as `a`. A copy has its own memory." + (" Select a result cell to see which cell of `a` it reads." if shares else " The result cells have no link to `a`."),
                  f"`np.shares_memory(a, result)` is {'true' if shares else 'false'}.", a_strip(a), r_strip(flat, shares, label)),
            frame("write", "Write -1 into the result", f"`result[...] = -1` sets every number of the result to -1.",
                  "A write goes to the memory the array reads. For a view that memory belongs to `a`.", "The code did not touch `a` directly.",
                  a_strip(a), tokens([{"t": "-1", "sub": f"[{k}]", "id": rid(k), "links": [aid(src_idx[k])] if shares else [], "cls": "neg"} for k in range(len(flat))], title="result after the write")),
            frame("read", "Look at the original again", (f"`a` changed at positions {changed}: {vec(a2)}." if shares else f"`a` did not change: {vec(a2)}."),
                  "A view writes through to its owner. A copy does not." , f"We never wrote to `a` by name.",
                  tokens([{"t": num(v), "sub": f"a[{k}]", "id": aid(k), "cls": "neg" if k in changed else None} for k, v in enumerate(a2)], title="a  after the write"),
                  text("Changed cells are marked." if shares else "No cell changed.")),
        ]

    a = Article("lesson-q027-array-mutation", "Views, copies, and surprising writes", kind="lesson",
                prerequisites=["arrays-and-shapes", "views-and-copies"], exercises=[(27, "deep-lesson"), (30, "related")],
                sources=[NUMPY_COPIES, ("Question 027 source", kit.record(27)["url"])])
    a.h2("The puzzle")
    a.p("You copy an array into `a` and `b`. You set every second number to -1, once through a slice and once through an index list. Which of `a` and `b` changes?")
    a.code("""
        a = np.array([10, 20, 30, 40, 50, 60, 70, 80])
        b = np.array([10, 20, 30, 40, 50, 60, 70, 80])
        a[::2][:] = -1                      # slice, then write
        b[np.arange(0, 8, 2)][:] = -1       # index list, then write
        """, caption="One of these writes works. The other one is lost.")
    # verify the claim in the snippet
    A = base.copy(); B = base.copy(); A[::2][:] = -1; B[np.arange(0, 8, 2)][:] = -1
    assert A.tolist() == [-1, 20, -1, 40, -1, 60, -1, 80] and B.tolist() == base.tolist()
    a.p(f"After these lines `a` is `{A.tolist()}` and `b` is still `{B.tolist()}`. The slice write reached `a`. The index-list write changed a temporary copy and was lost. This lesson shows why.")
    a.h2("What the question asks")
    a.p(kit.record(27)["question"].strip())
    a.callout("definition", "The input is one array. The output is a pair `(a, b)`. Both start as copies of the input. In `a`, every second element is replaced by -1 through a **slice**. In `b`, the same positions are replaced through **fancy indexing**. The expected result is: `a` has -1 at positions 0, 2, 4, 6 and `b` equals the input.", title="Inputs and output")
    a.h2("What you need first")
    a.ul(["[[article:arrays-and-shapes|Array shapes and indexing]]", "[[article:views-and-copies|Views and copies]]: which indexing makes a view and which makes a copy."])
    a.h2("Watch a write travel")
    a.p("Choose how the second array is made. Step through. In the first two steps, select a result cell. In the view cases it marks the cell of `a` that it reads.")
    a.demo("mutation-trace", "Does the write reach the original?", frames,
           params=[("op", "How the second array is made", [(k, v[1]) for k, v in OPS.items()])],
           help=[{"step_id": "make", "help_type": "smaller-example", "content": "Take `a = [1, 2, 3, 4]`. The slice `a[::2]` is `[1, 3]` and shares memory. The index list `a[[0, 2]]` is `[1, 3]` too but owns new memory."},
                 {"step_id": "make", "help_type": "prerequisite", "content": "The table of views and copies is here.", "prerequisite_id": "views-and-copies"},
                 {"step_id": "write", "help_type": "common-mistake", "content": "`b[idx][:] = -1` does two things. First `b[idx]` makes a copy. Then the write goes into that copy. The write never reaches `b`."},
                 {"step_id": "read", "help_type": "code-connection", "content": "In the supplied solution, `view[:] = -1` changes `a`. `copied[:] = -1` leaves `b` unchanged."}],
           facts=[("np.shares_memory is true for slice and reshape, false for index list and mask; writes reach the original only for views", "numpy"),
                  ("supplied q027 returns a with -1 at even positions and b equal to the input", "supplied solution q027")])
    a.set_default("mutation-trace", op="slice")
    a.h2("The rule in one picture")
    a.p("An array object is a small description: where its memory starts, its shape, and its steps. Slicing makes a new description of the **same** memory. Fancy indexing must gather numbers from scattered places. Because the places are not evenly spaced, NumPy cannot describe them with a start and a step. So it **copies** them into new memory.")
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=100)
    for k, v in enumerate(base):
        ax.add_patch(plt.Rectangle((k, 1), 0.92, 0.9, fc="#e6f0fa", ec="#0072b2"))
        ax.text(k + .46, 1.45, str(v), ha="center", va="center")
        if k % 2 == 0:
            ax.annotate("", xy=(k + .46, 1), xytext=(k / 2 * 1.0 + .46 + 0, 0.5 + 0.0), arrowprops=dict(arrowstyle="->", color="#009e73"))
    for k in range(4):
        ax.add_patch(plt.Rectangle((k, -0.3), 0.92, 0.8, fc="#fff3c4", ec="#b08900", ls="--"))
        ax.text(k + .46, 0.1, str(base[2 * k]), ha="center", va="center")
    ax.text(5.2, 0.1, "new memory (a copy)", va="center", color="#8a6a00")
    ax.text(5.2, 1.45, "original memory", va="center", color="#0072b2")
    ax.set_xlim(-.2, 8.2); ax.set_ylim(-.6, 2.2); ax.axis("off")
    ax.set_title("Index list: the numbers are gathered into new memory", fontsize=10)
    a.figure("gather", fig, alt="Eight numbers in the original memory. Four of them, at positions 0, 2, 4 and 6, are copied into a separate row of new memory below.",
             description="Fancy indexing copies the selected numbers into new memory", caption="Fancy indexing gathers numbers into a new block. A write to the new block cannot reach the original.")
    a.h2("The mathematics of a view")
    a.p("A one-axis view is described by three numbers: a start $s$, a step $t$, and a length $L$. Element $k$ of the view is element $s + t k$ of the original:")
    a.eq(r"\mathrm{view}[k] = \mathrm{base}[\,s + t\,k\,], \qquad k = 0, \dots, L-1")
    a.p("For `a[::2]` on 8 numbers: $s = 0$, $t = 2$, $L = 4$. So the view reads positions 0, 2, 4, 6. A writing to `view[k]` writes to `base[s + t k]`. An index list like `[0, 2, 4, 6]` could have been described this way, but NumPy does not check. A general list such as `[5, 1, 6]` has no such rule, so NumPy treats every index list the same way and copies.")
    a.callout("boundary", "A reshape returns a view only when the memory layout allows it. After a transpose, `a.T.reshape(-1)` must copy. Use `np.shares_memory(a, b)` to check, and never rely on it in code that you cannot test.")
    a.callout("mistake", "In pandas and NumPy code, chained expressions such as `x[mask][:] = 0` look like an in-place update. They update a copy. Use one assignment: `x[mask] = 0`.", title="The classic bug")
    a.h2("Read the supplied solution")
    a.walkthrough(27, [
        (5, "np.array(array, copy=True)", "Two independent copies of the input. Changes to the input later do not reach `a` or `b`."),
        (6, "view = a[::2]", "A basic slice. `view` shares memory with `a`."),
        (7, "view[:] = -1", "The write goes through `view` into the memory of `a`. This is why `a` changes."),
        (8, "np.arange(0, len(b), 2)", "An index array. Indexing with an array is fancy indexing, which copies."),
        (9, "copied[:] = -1", "This changes only the copy. `b` stays as it was."),
        (10, "return (a, b)", "The pair shows both outcomes side by side."),
    ], title="Question 027: the same positions, two different outcomes")
    a.h2("Test yourself")
    a.check("predict-slice", "prediction", "`a = np.arange(6)`. You run `v = a[1:4]`, then `v[0] = 99`. What is `a[1]` afterwards? Predict before you try it.", "number", 99,
            expl("`a[1:4]` is a slice, so `v` is a view. `v[0]` is `a[1]`, so `a[1]` becomes 99.", "A slice with a start and a stop makes a view. A write through a view changes the original.",
                 **{"1": "That is the value before the write. A slice shares memory, so the write is visible in `a`."}), tolerance=0)
    arr = np.arange(6); idx = arr[[1, 3]]; idx[0] = 99
    assert arr[1] == 1
    a.check("predict-fancy", "new-case", "`a = np.arange(6)`. You run `c = a[[1, 3]]`, then `c[0] = 99`. What is `a[1]` afterwards?", "number", int(arr[1]),
            expl("`a[[1, 3]]` is fancy indexing, so `c` is a copy. The write changes `c` only. `a[1]` stays 1.", "An index list makes a copy. Writes to a copy never reach the original.",
                 **{"99": "That would be true for a view. An index list always makes a copy."}), tolerance=0)
    a.check("which-copy", "prediction", "Which of these makes a view of `a`?", "choice", "a[2:5]",
            expl("A basic slice with start and stop makes a view.", "Slices make views. Index lists, masks, and `.copy()` make copies.",
                 **{"a[[2, 3, 4]]": "An index list makes a copy.", "a[a > 2]": "A boolean mask makes a copy.", "a.copy()": "By definition this is a copy."}),
            options=["a[[2, 3, 4]]", "a[2:5]", "a[a > 2]", "a.copy()"])
    a.check("explain-view", "explain-back", "In your own words: why does `view[:] = -1` change `a`, but `copied[:] = -1` does not?", "text", None,
            {"correct": "`view` is only a new description of the memory that `a` owns, so a write through `view` writes into that memory. `copied` owns separate memory that NumPy filled by gathering numbers, so a write to it cannot reach `a`."})
    a.h2("Write it yourself")
    a.playground(27, '''
        import numpy as np


        def mutation_demo(array):
            # Return (a, b). Both start as copies of `array`.
            # In a, set every second element (positions 0, 2, 4, ...) to -1 through a slice.
            # In b, set the same positions to -1 through fancy indexing, and show that b is unchanged
            # when you write into the gathered copy (a[idx][:] = -1 style).
            raise NotImplementedError
        ''', checks=[
        ("a has -1 at even positions", "a, b = mutation_demo([10, 20, 30, 40, 50])\nassert list(a) == [-1, 20, -1, 40, -1], list(a)"),
        ("b equals the input", "a, b = mutation_demo([10, 20, 30, 40, 50])\nassert list(b) == [10, 20, 30, 40, 50], list(b)"),
        ("the input is not changed", "x = np.array([1, 2, 3, 4])\nmutation_demo(x)\nassert x.tolist() == [1, 2, 3, 4]"),
    ], note="Write `mutation_demo(array)` for question 027. The checks run after your code.", max_input_size=100)
    return a


def build():
    return [*arrays_and_shapes(), *broadcasting(), *views_and_copies(), lesson_030(), lesson_027()]
