"""3D timeline scenes for batch C02."""
import numpy as np

import kit
from scene import K, Scene, orbit, vis

BLUE, ORANGE, GREEN, PINK, GREY, RED = "#0072b2", "#d55e00", "#009e73", "#cc79a7", "#8a949e", "#d62728"
S = 1.5            # spacing between cells
WALL_X = -3.4      # x lives on a wall at the left
CEIL_Y = 3.3       # y lives on a ceiling above
GAP = 0.22         # a blue and an orange cube share one cell, side by side
SZ = 0.8           # size of a main cube
GSZ = 0.44         # size of a ghost cube


def pairwise_scene(m: int) -> Scene:
    q30 = kit.supplied(30)
    X = np.array([[1, 2, 3], [4, 5, 6]])
    Y = np.array([[0, 1, 2], [1, 1, 1], [2, 0, 1], [3, 2, 0]])[:m]
    n, d = X.shape
    out = q30(X, Y)
    assert out.shape == (n, m, d) and np.array_equal(out, X[:, None, :] - Y[None, :, :])
    cx, cy, cz = (m - 1) * S / 2, -(n - 1) * S / 2, (d - 1) * S / 2
    P = lambda i, j, k: (j * S, -i * S, k * S)

    sc = Scene(33, f"A 3D box of {n * m * d} numbers. Matrix x with {n} rows and {d} features sits on the left wall. Matrix y with {m} rows and {d} features hangs from the ceiling. Copies of x slide right and copies of y slide down until every pair meets. Each pair fuses into one cube holding x minus y.", start_t=4.8)
    sc.chapter("inputs", "Two matrices", 0, 5,
               f"`x` has shape ({n}, {d}) and stands on the left wall. `y` has shape ({m}, {d}) and hangs from the ceiling. The axes are `i` (rows of x, down), `j` (rows of y, right), and `k` (features, towards you).",
               "The question needs one difference for every pair of rows: one row of `x` and one row of `y`. The two inputs share the feature axis `k`.", "Nothing is computed yet.")
    sc.chapter("insert-axes", "Insert one axis in each input", 5, 11,
               f"`x[:, None, :]` has shape ({n}, 1, {d}): a slab one cube thick along `j`. `y[None, :, :]` has shape (1, {m}, {d}): a slab one cube thick along `i`.",
               "The new axes have size 1. A size of 1 can stretch to any size in the next step.", "The numbers stay the same. `None` only adds an axis.")
    sc.chapter("stretch", "Stretch the size-1 axes", 11, 19,
               f"The slab of `x` is read again for every `j` until `j` has size {m}. The slab of `y` is read again for every `i` until `i` has size {n}. Both now fill the box of shape ({n}, {m}, {d}).",
               "A size of 1 matches any other size. NumPy does not copy the numbers. It reads the same stored numbers again at every position, so the extra cubes only picture the re-reading.",
               "The stored arrays do not change.")
    sc.chapter("subtract", "Subtract cell by cell", 19, 27,
               f"At every position a blue `x` cube and an orange `y` cube meet. They fuse into one cube that holds `x - y`. The {n * m * d} cubes appear in a wave.",
               "Each position now has a partner, so subtraction works element by element. No loop is needed.", "`x` and `y` stay on their walls.")
    sc.chapter("result", "Read the result", 27, 33,
               f"The result has shape ({n}, {m}, {d}). Click any result cube to see which `x` cube and which `y` cube it reads.",
               "Result cube `[i, j, k]` equals `x[i, k] - y[j, k]`. The feature `k` is shared. `i` comes from `x` and `j` comes from `y`.",
               "The inputs and the result are not changed. Drag to look at the box from every side.")

    # ---- axis legend (a text line, in its own band under the box)
    band = lambda r: -n * S - 0.7 - 0.75 * r
    sc.text("lbl_axes", (cx, band(0), cz), "i = row of x  (down)      j = row of y  (right)      k = feature  (towards you)", "fg", 0.42, tr={"opacity": vis((3.0, 26.0))})

    # ---- inputs: x on the wall, y on the ceiling
    for i in range(n):
        for k in range(d):
            t0 = 0.3 + 0.09 * (i * d + k)
            sc.cube(f"x_{i}_{k}", (WALL_X, -i * S, k * S), X[i, k], BLUE, size=SZ, title=f"x[{i}, {k}] = {X[i, k]}",
                    tr={"opacity": [(0, 0), (t0, 0), (t0 + 0.6, 1), (27, 1), (28.5, 0.7)]})
    for j in range(m):
        for k in range(d):
            t0 = 1.5 + 0.09 * (j * d + k)
            sc.cube(f"y_{j}_{k}", (j * S, CEIL_Y, k * S), Y[j, k], ORANGE, size=SZ, title=f"y[{j}, {k}] = {Y[j, k]}",
                    tr={"opacity": [(0, 0), (t0, 0), (t0 + 0.6, 1), (27, 1), (28.5, 0.7)]})
    sc.text("lbl_x", (WALL_X, 1.6, cz), "x", BLUE, 0.55, tr={"text": [(0, "x   shape (2, 3)"), (7.6, "x[:, None, :]   shape (2, 1, 3)")], "opacity": vis((0.5, None))})
    sc.text("lbl_y", (cx, CEIL_Y + 1.3, cz), "y", ORANGE, 0.55, tr={"text": [(0, f"y   shape ({m}, 3)"), (9.9, f"y[None, :, :]   shape (1, {m}, 3)")], "opacity": vis((1.6, None))})
    sc.text("lbl_legend", (cx, band(0), cz), "result colors:  green  x − y > 0     pink  x − y < 0     grey  x − y = 0", "fg", 0.42,
            tr={"opacity": vis((27.5, None))})

    # ---- ghosts and outputs
    for i in range(n):
        for j in range(m):
            for k in range(d):
                a, b, c = int(X[i, k]), int(Y[j, k]), int(out[i, j, k])
                assert a - b == c
                tf = 19.2 + 0.6 * (i + j + k)          # when this position fuses
                cell = np.array(P(i, j, k))
                rest_x, rest_y = cell + [-GAP, 0, 0], cell + [GAP, 0, 0]
                xr0 = np.array(P(i, 0, k)) + [-GAP, 0, 0]
                yr0 = np.array(P(0, j, k)) + [GAP, 0, 0]
                wall = np.array([WALL_X, -i * S, k * S]); ceil = np.array([j * S, CEIL_Y, k * S])
                # x ghost: slides from the wall into slab j=0, then j>0 copies slide out of it
                if j == 0:
                    pos = [(0, wall), (5.4, wall), (7.4, rest_x)]
                    op = [(0, 0), (5.0, 0), (5.4, 1), (tf, 1), (tf + 0.9, 0)]
                else:
                    ts = 11.3 + 0.8 * (j - 1)
                    pos = [(0, xr0), (ts + 0.3, xr0), (ts + 1.3, rest_x)]
                    op = [(0, 0), (ts, 0), (ts + 0.3, 1), (tf, 1), (tf + 0.9, 0)]
                pos = pos + [(tf, rest_x), (tf + 0.9, cell)]
                sc.cube(f"gx_{i}_{j}_{k}", rest_x, a, BLUE, size=GSZ, ghost=j > 0, pickable=False,
                        tr={"pos": [(t, list(map(float, p))) for t, p in pos], "opacity": op, "size": [(0, GSZ), (tf, GSZ), (tf + 0.9, 0.2)]})
                # y ghost
                if i == 0:
                    pos = [(0, ceil), (7.9, ceil), (9.9, rest_y)]
                    op = [(0, 0), (7.6, 0), (7.9, 1), (tf, 1), (tf + 0.9, 0)]
                else:
                    ts = 14.6 + 0.8 * (i - 1)
                    pos = [(0, yr0), (ts + 0.3, yr0), (ts + 1.2, rest_y)]
                    op = [(0, 0), (ts, 0), (ts + 0.3, 1), (tf, 1), (tf + 0.9, 0)]
                pos = pos + [(tf, rest_y), (tf + 0.9, cell)]
                sc.cube(f"gy_{i}_{j}_{k}", rest_y, b, ORANGE, size=GSZ, ghost=i > 0, pickable=False,
                        tr={"pos": [(t, list(map(float, p))) for t, p in pos], "opacity": op, "size": [(0, GSZ), (tf, GSZ), (tf + 0.9, 0.2)]})
                color = GREEN if c > 0 else PINK if c < 0 else GREY
                sc.cube(f"out_{i}_{j}_{k}", cell, c, color, size=SZ,
                        title=f"out[{i}, {j}, {k}] = x[{i}, {k}] − y[{j}, {k}] = {a} − {b} = {c}", links=[f"x_{i}_{k}", f"y_{j}_{k}"],
                        tr={"opacity": [(0, 0), (tf + 0.45, 0), (tf + 1.1, 1)], "size": [(0, 0.3), (tf + 0.45, 0.3), (tf + 1.1, SZ)]})
                sc.by_sel[f"out_{i}_{j}_{k}"] = f"`out[{i}, {j}, {k}]` = `x[{i}, {k}]` − `y[{j}, {k}]` = {a} − ({b}) = **{c}**. The blue cube `x[{i}, {k}]` and the orange cube `y[{j}, {k}]` are marked."
    for i in range(n):
        for k in range(d):
            sc.by_sel[f"x_{i}_{k}"] = f"`x[{i}, {k}]` = {X[i, k]}. The result reads it once for every `j`, so {m} times. The result cubes that use it are marked."
    for j in range(m):
        for k in range(d):
            sc.by_sel[f"y_{j}_{k}"] = f"`y[{j}, {k}]` = {Y[j, k]}. The result reads it once for every `i`, so {n} times. The result cubes that use it are marked."

    # ---- guides
    sc.box("out_box", (cx, cy, cz), (m * S, n * S, d * S), GREY, label=f"out   shape ({n}, {m}, {d})", tr={"opacity": [(0, 0), (16.2, 0), (17.4, 0.9)]})
    sc.text("note_x", (cx, band(1), cz), "x is read again for every j  →", BLUE, 0.45, tr={"opacity": vis((11.4, 18.6), fade_s=0.5)})
    sc.text("note_y", (m * S + 1.7, cy + 0.3, cz), "y is read again\nfor every i  ↓", ORANGE, 0.45, tr={"opacity": vis((14.8, 18.6), fade_s=0.5)})
    sc.text("note_nocopy", (cx, band(2), cz), "no copy in memory: the same stored numbers are read again", "fg", 0.45, tr={"opacity": vis((17.2, 19.4), fade_s=0.5)})
    sc.text("note_click", (cx, band(1), cz), "click a cube to trace it", "fg", 0.5, tr={"opacity": vis((28.0, None))})

    tg = (cx - 0.4, cy + 0.3, cz)
    tg2 = (cx - 0.9, cy - 0.2, cz)
    sc.camera_keys((0, orbit(tg, 38, 30, 14.5), tg), (5, orbit(tg, 36, 28, 14.5), tg), (11, orbit(tg, 46, 32, 15), tg), (19, orbit(tg, 38, 28, 14.5), tg),
                   (27, orbit(tg2, 30, 24, 17), tg2), (33, orbit(tg2, -22, 20, 17), tg2))
    sc.view("front", "Front view (i down, j right)", (cx, cy + 0.5, 32), (cx, cy + 0.5, cz))
    sc.view("top", "Top view (j right, k toward you)", (cx, 32, cz + 0.01), (cx, cy, cz))
    sc.view("side", "Side view (i down, k)", (-32, cy + 0.5, cz), (cx, cy + 0.5, cz))
    return sc


def mismatch_scene() -> Scene:
    """x has 3 features and y has 2: the feature axis k cannot broadcast."""
    X = np.array([[1, 2, 3], [4, 5, 6]]); Y = np.array([[0, 1], [1, 1]])
    n, dx = X.shape; m, dy = Y.shape
    try:
        np.asarray(X)[:, None, :] - np.asarray(Y)[None, :, :]
        raise AssertionError("expected a broadcasting error")
    except ValueError as e:
        err = str(e)
    assert "(2,1,3)" in err and "(1,2,2)" in err
    cx, cy, cz = (m - 1) * S / 2, -(n - 1) * S / 2, (dx - 1) * S / 2
    P = lambda i, j, k: (j * S, -i * S, k * S)
    sc = Scene(18, f"A 3D scene where x has {dx} features and y has {dy}. The feature axis cannot match, so the third layer of x has no partner and NumPy reports an error.", start_t=4.8)
    sc.chapter("inputs", "Two matrices with different feature counts", 0, 5, f"`x` has shape ({n}, {dx}). `y` has shape ({m}, {dy}). The feature axis `k` has size {dx} in `x` and {dy} in `y`.",
               "Both inputs need the same number of features. A difference is only defined for matching features.", "Nothing is computed yet.")
    sc.chapter("align", "Insert axes and stretch", 5, 12, f"The slabs are built exactly as before: `x[:, None, :]` has shape ({n}, 1, {dx}) and `y[None, :, :]` has shape (1, {m}, {dy}). Along `i` and `j` they stretch as usual.",
               "Axes `i` and `j` contain a 1, so they stretch. The feature axis `k` has no 1.", "The numbers stay the same.")
    sc.chapter("conflict", "The feature axis does not fit", 12, 18, f"At `k = 2` the blue `x` cubes have no orange partner. Sizes {dx} and {dy} are different and neither is 1.",
               f"NumPy compares sizes from the right. The last axis has sizes {dx} and {dy}, so the operation stops with an error.", "Axes `i` and `j` were fine.")
    for i in range(n):
        for k in range(dx):
            sc.cube(f"x_{i}_{k}", (WALL_X, -i * S, k * S), X[i, k], BLUE, title=f"x[{i}, {k}] = {X[i, k]}", tr={"opacity": [(0, 0), (0.3 + 0.09 * (i * dx + k), 0), (0.9 + 0.09 * (i * dx + k), 1)]})
    for j in range(m):
        for k in range(dy):
            sc.cube(f"y_{j}_{k}", (j * S, CEIL_Y, k * S), Y[j, k], ORANGE, title=f"y[{j}, {k}] = {Y[j, k]}", tr={"opacity": [(0, 0), (1.5 + 0.09 * (j * dy + k), 0), (2.1 + 0.09 * (j * dy + k), 1)]})
    for i in range(n):
        for j in range(m):
            for k in range(dx):
                cell = np.array(P(i, j, k)); rx = cell + [-GAP, 0, 0]; ry = cell + [GAP, 0, 0]
                wall = np.array([WALL_X, -i * S, k * S])
                ts = 5.4 + 0.5 * j
                x0 = np.array(P(i, 0, k)) + [-GAP, 0, 0]
                posx = [(0, wall), (5.4, wall), (6.6, x0)] + ([(ts + 1.3, x0), (ts + 2.4, rx)] if j else [])
                sc.cube(f"gx_{i}_{j}_{k}", rx, X[i, k], BLUE, size=0.5, ghost=j > 0, pickable=False,
                        tr={"pos": [(t, list(map(float, p))) for t, p in posx], "opacity": [(0, 0), (5.0 if j == 0 else ts + 1.3, 0), ((5.4 if j == 0 else ts + 1.6), 1)]})
                if k < dy:
                    ceil = np.array([j * S, CEIL_Y, k * S]); y0 = np.array(P(0, j, k)) + [GAP, 0, 0]
                    posy = [(0, ceil), (8.0, ceil), (9.2, y0)] + ([(10.0, y0), (10.9, ry)] if i else [])
                    sc.cube(f"gy_{i}_{j}_{k}", ry, Y[j, k], ORANGE, size=0.5, ghost=i > 0, pickable=False,
                            tr={"pos": [(t, list(map(float, p))) for t, p in posy], "opacity": [(0, 0), (7.6 if i == 0 else 10.0, 0), (8.0 if i == 0 else 10.3, 1)]})
                else:
                    sc.cube(f"missing_{i}_{j}_{k}", ry, "?", RED, size=0.5, pickable=False,
                            tr={"opacity": [(0, 0), (12.0 + 0.15 * (i + j), 0), (12.6 + 0.15 * (i + j), 1)], "size": [(12.0, 0.2), (12.9, 0.62), (13.5, 0.5), (14.2, 0.62)]})
    sc.text("lbl_x", (WALL_X, 1.0, cz), "x   shape (2, 3)", BLUE, 0.5, tr={"opacity": vis((0.5, None))})
    sc.text("lbl_y", (cx, CEIL_Y + 1.0, cz / 1.5), "y   shape (2, 2)", ORANGE, 0.5, tr={"opacity": vis((1.6, None))})
    sc.text("err", (cx, -n * S - 0.9, cz), "ValueError: operands could not be\nbroadcast together with shapes (2,1,3) (1,2,2)", RED, 0.46, tr={"opacity": vis((13.6, None))})
    sc.text("axis_k", (cx, 1.1, dx * S + 0.6), "k  features:  x has 3,  y has 2", "fg", 0.5, tr={"opacity": vis((12.0, None))})
    sc.camera_keys((0, (12, 6.5, 14), (cx - 0.7, cy + 0.2, cz)), (12, (11, 5.5, 14), (cx - 0.2, cy, cz)), (18, (8, 4.5, 15), (cx, cy, cz)))
    sc.view("front", "Front view (i down, j right)", (cx, cy, 26), (cx, cy, cz))
    sc.view("side", "Side view (i down, k)", (-26, cy, cz), (cx, cy, cz))
    return sc, err


# =========================================================================== concept style: A (2,1,3) + B (1,4,1)
TEAL, TEAL_EDGE = "#35b9ac", "#08776e"
AMBER, AMBER_EDGE = "#f4b94f", "#9a5d00"
VIOLET, VIOLET_EDGE = "#a47ae8", "#7139ba"
NAVY = "#14223e"
CS = 1.12          # cell spacing in the concept scene
ZS = 1.8           # spacing along the depth axis (axis 2): larger so that numbers on cubes behind each other do not collide
CSZ = 0.98         # cube size


def broadcast_concept_scene() -> Scene:
    A = np.array([[[1, 2, 3]], [[4, 5, 6]]])           # shape (2, 1, 3)
    B = np.array([[[10], [20], [30], [40]]])           # shape (1, 4, 1)
    C = A + B                                           # numpy does the broadcasting: the reference for every number below
    assert A.shape == (2, 1, 3) and B.shape == (1, 4, 1) and C.shape == (2, 4, 3)
    N, M, D = C.shape
    for i in range(N):
        for j in range(M):
            for k in range(D):
                assert C[i, j, k] == A[i, 0, k] + B[0, j, 0]
    assert C[1, 2, 2] == 36 and A[1, 0, 2] == 6 and B[0, 2, 0] == 30        # the worked cell from the design

    A0, B0, C0 = -10.2, -2.9, 4.6
    pa = lambda i, j, k: (A0 + j * CS, -i * CS, k * ZS)
    pb = lambda i, j, k: (B0 + j * CS, -i * CS, k * ZS)
    pc = lambda i, j, k: (C0 + j * CS, -i * CS, k * ZS)

    sc = Scene(26, "Three 3D blocks of numbers. A teal block A with shape 2 by 1 by 3 and an amber block B with shape 1 by 4 by 1. Their size-1 axes stretch until both have shape 2 by 4 by 3. Then matching cells are added to give the violet result block.",
               start_t=4.8, style="concept", ortho_span=13)
    sc.chapter("inputs", "Start with the inputs", 0, 5,
               "`A` has shape (2, 1, 3): two layers of three numbers. `B` has shape (1, 4, 1): four numbers on its middle axis. Their shapes say where each number lives.",
               "Every array has one size for each axis. Broadcasting works by comparing those sizes.", "Nothing has been copied or added yet.", short="Inputs")
    sc.chapter("align", "Align the shapes from the right", 5, 10,
               "Write the shapes in a column and read them from the right. Axis 2 compares 3 with 1. Axis 1 compares 1 with 4. Axis 0 compares 2 with 1.",
               "Two sizes are compatible when they are equal or when one of them is 1. Every pair here has a 1, so the shapes fit. The result takes the larger size of each pair: (2, 4, 3).", "The numbers and the shapes of `A` and `B` are unchanged.", short="Align")
    sc.chapter("expand", "Same values, more positions", 10, 18,
               "The single layer of `A` is read again at four positions along axis 1. The four numbers of `B` are read again for both layers and all three columns. Both now cover the shape (2, 4, 3).",
               "A size-1 axis matches any size. NumPy does not copy the data. The pale blocks show the same stored numbers being read again.", "The stored values of `A` and `B` do not change.", short="Expand")
    sc.chapter("add", "Add the corresponding values", 18, 26,
               "Each position now has one value from `A` and one from `B`. They add into one violet cell. The result has 24 numbers. Click a violet cell to see which two inputs made it.",
               "Addition works on matching positions. The result cell `[i, j, k]` equals `A[i, 0, k] + B[0, j, 0]`.", "`A` and `B` stay as they were. The result is new memory.", short="Add")

    # ---- originals
    t_in = lambda n: 0.35 + 0.12 * n
    for i in range(N):
        for k in range(D):
            sc.cube(f"A_{i}_{k}", pa(i, 0, k), A[i, 0, k], TEAL, size=CSZ, title=f"A[{i}, 0, {k}] = {A[i, 0, k]}",
                    tr={"opacity": [(0, 0), (t_in(i * D + k), 0), (t_in(i * D + k) + 0.6, 1), (18.5, 1), (22.5, 0.4)], "size": [(0, 0.5), (t_in(i * D + k), 0.5), (t_in(i * D + k) + 0.7, CSZ)]}).update({"edge": TEAL_EDGE})
    for j in range(M):
        sc.cube(f"B_{j}", pb(0, j, 0), B[0, j, 0], AMBER, size=CSZ, title=f"B[0, {j}, 0] = {B[0, j, 0]}",
                tr={"opacity": [(0, 0), (t_in(6 + j), 0), (t_in(6 + j) + 0.6, 1), (18.5, 1), (22.5, 0.4)], "size": [(0, 0.5), (t_in(6 + j), 0.5), (t_in(6 + j) + 0.7, CSZ)]}).update({"edge": AMBER_EDGE})

    # ---- ghost readers and results
    for i in range(N):
        for j in range(M):
            for k in range(D):
                tf = 18.3 + 0.34 * (i + j + k)           # when this position fuses into the result
                a_from, b_from = np.array(pa(i, 0, k)), np.array(pb(0, j, 0))
                a_rest, b_rest = np.array(pa(i, j, k)), np.array(pb(i, j, k))
                c_pos = np.array(pc(i, j, k))
                # A reader
                ta = 10.3 + 0.55 * j if j else 17.6
                apos = [(0, a_from), (ta + 0.2, a_from), (ta + 1.4, a_rest), (tf, a_rest), (tf + 1.0, c_pos)] if j else [(0, a_from), (17.6, a_from), (tf, a_from), (tf + 1.0, c_pos)]
                sc.cube(f"gA_{i}_{j}_{k}", a_rest, A[i, 0, k], TEAL, size=CSZ, ghost=j > 0, pickable=False,
                        tr={"pos": [(t, list(map(float, p))) for t, p in apos],
                            "opacity": [(0, 0), (ta, 0), (ta + 0.4, 1 if j == 0 else 0.55), (tf, 1 if j == 0 else 0.55), (tf + 0.9, 0)],
                            "size": [(0, CSZ), (tf, CSZ), (tf + 1.0, 0.3)]}).update({"edge": TEAL_EDGE})
                # B reader
                tb = 10.3 + 0.45 * (i + k) if (i or k) else 17.6
                bpos = [(0, b_from), (tb + 0.2, b_from), (tb + 1.4, b_rest), (tf, b_rest), (tf + 1.0, c_pos)] if (i or k) else [(0, b_from), (17.6, b_from), (tf, b_from), (tf + 1.0, c_pos)]
                sc.cube(f"gB_{i}_{j}_{k}", b_rest, B[0, j, 0], AMBER, size=CSZ, ghost=bool(i or k), pickable=False,
                        tr={"pos": [(t, list(map(float, p))) for t, p in bpos],
                            "opacity": [(0, 0), (tb, 0), (tb + 0.4, 1 if not (i or k) else 0.55), (tf, 1 if not (i or k) else 0.55), (tf + 0.9, 0)],
                            "size": [(0, CSZ), (tf, CSZ), (tf + 1.0, 0.3)]}).update({"edge": AMBER_EDGE})
                a, b, c = int(A[i, 0, k]), int(B[0, j, 0]), int(C[i, j, k])
                sc.cube(f"C_{i}_{j}_{k}", c_pos, c, VIOLET, size=CSZ,
                        title=f"C[{i}, {j}, {k}] = A[{i}, 0, {k}] + B[0, {j}, 0] = {a} + {b} = {c}", links=[f"A_{i}_{k}", f"B_{j}"],
                        tr={"opacity": [(0, 0), (tf + 0.7, 0), (tf + 1.3, 1)], "size": [(0, 0.3), (tf + 0.7, 0.3), (tf + 1.3, CSZ)]}).update({"edge": VIOLET_EDGE})
                sc.by_sel[f"C_{i}_{j}_{k}"] = f"`C[{i}, {j}, {k}]` = `A[{i}, 0, {k}]` + `B[0, {j}, 0]` = {a} + {b} = **{c}**. Index 0 is used on the axes where `A` and `B` have size 1."
                sc.set_calc(f"C_{i}_{j}_{k}", [(f"A[{i}, 0, {k}]", a, "a"), (f"B[0, {j}, 0]", b, "b"), (f"C[{i}, {j}, {k}]", c, "c")], ["+", "="])
    for i in range(N):
        for k in range(D):
            sc.by_sel[f"A_{i}_{k}"] = f"`A[{i}, 0, {k}]` = {A[i, 0, k]}. It is read again at all {M} positions along axis 1, so it appears in {M} result cells."
    for j in range(M):
        sc.by_sel[f"B_{j}"] = f"`B[0, {j}, 0]` = {B[0, j, 0]}. It is read again for both layers and all {D} columns, so it appears in {N * D} result cells."
    sc.calc_default = "C_1_2_2"

    # ---- panel titles and axis brackets
    mono = lambda t, c="fg": {"t": t, "c": c, "f": "mono", "w": 600}
    head = lambda name, color, shape: [{"t": name, "c": color, "w": 800}, {"t": "   ", "w": 400}, mono(shape)]
    sc.text("title_A", (A0 + 0.2, 1.9, 0), head("A", TEAL_EDGE, "(2, 1, 3)"), "fg", 0.62, tr={"opacity": [(0, 0), (0.4, 1)]})
    sc.text("title_B", (B0 + 1.7, 1.9, 0), head("B", AMBER_EDGE, "(1, 4, 1)"), "fg", 0.62, tr={"opacity": [(0, 0), (0.8, 1)]})
    sc.text("title_C", (C0 + 1.7, 1.9, 0), head("A + B", VIOLET_EDGE, "(2, 4, 3)"), "fg", 0.62, tr={"opacity": [(0, 0), (0.4, 1), (17.9, 1)]})
    # the result panel is an empty outline until it fills
    sc.box("C_frame", (C0 + (M - 1) * CS / 2, -(N - 1) * CS / 2, (D - 1) * ZS / 2), (M * CS, N * CS, (D - 1) * ZS + CSZ + 0.1), "#b9a4e8", tr={"opacity": [(0, 0), (1.5, 0.9), (22, 0.5)]})

    def brackets(name, x0, ncols, nrows, ndep, vis_to, color="#5b6776", t_on=3.4):
        """Three brackets on the front-left, front-bottom and right-bottom edges of a block."""
        xl, xr = x0 - CSZ / 2, x0 + (ncols - 1) * CS + CSZ / 2
        yt, yb = CSZ / 2, -(nrows - 1) * CS - CSZ / 2
        zf, zb = (ndep - 1) * ZS + CSZ / 2, -CSZ / 2
        op = [(0, 0), (t_on, 0), (t_on + 0.8, 1)]
        if name in ("dA", "dB"):      # the brackets of the inputs step aside while the pale copies fill the space
            op = op + [(10.0, 1), (10.6, 0.15), (18.0, 0.15), (18.8, 1)]
        sc.dim(f"{name}_ax0", (xl - 0.55, yt, zf), (xl - 0.55, yb, zf), f"axis 0  ({nrows})", tick=(0.22, 0, 0), color=color, label_offset=(-1.0, 0, 0), tr={"opacity": op})
        sc.dim(f"{name}_ax1", (xl, yb - 0.55, zf), (xr, yb - 0.55, zf), f"axis 1  ({ncols})", tick=(0, 0.22, 0), color=color, label_offset=(0, -0.5, 0), tr={"opacity": op})
        sc.dim(f"{name}_ax2", (xr + 0.55, yb - 0.55, zb), (xr + 0.55, yb - 0.55, zf), f"axis 2  ({ndep})", tick=(0, 0.22, 0), color=color, label_offset=(0.75, -0.2, 0), tr={"opacity": op})
    brackets("dA", A0, 1, N, D, 0)
    brackets("dB", B0, M, 1, 1, 0)
    brackets("dC", C0, M, N, D, 0)

    # ---- chapter 2: shape alignment, drawn in the scene
    row = lambda r: -N * CS - 2.1 - 0.8 * r
    cx_mid = (A0 + C0 + M * CS) / 2 - 1.0
    sc.text("al_A", (cx_mid, row(0), CS), head("A", TEAL_EDGE, "  2    1    3"), "fg", 0.5, tr={"opacity": [(0, 0), (5.2, 0), (5.8, 1), (9.6, 1), (10.4, 0)]})
    sc.text("al_B", (cx_mid, row(1), CS), head("B", AMBER_EDGE, "  1    4    1"), "fg", 0.5, tr={"opacity": [(0, 0), (5.4, 0), (6.0, 1), (9.6, 1), (10.4, 0)]})
    sc.text("al_C", (cx_mid, row(2), CS), head("A + B", VIOLET_EDGE, "2    4    3"), "fg", 0.5, tr={"opacity": [(0, 0), (8.6, 0), (9.2, 1), (9.8, 1), (10.4, 0)]})
    sc.text("al_note", (cx_mid, row(3) + 0.1, CS), "read from the right:   3 with 1  →  3      1 with 4  →  4      2 with 1  →  2", "fg", 0.44, tr={"opacity": [(0, 0), (6.4, 0), (7.0, 1), (9.6, 1), (10.4, 0)]})
    sc.text("ex_note", (cx_mid, row(0), CS), "same values, more positions", "fg", 0.56, tr={"opacity": [(0, 0), (11.0, 0), (11.6, 1), (17.4, 1), (18.0, 0)]})
    sc.text("add_note", (cx_mid, row(0), CS), "click a violet cell to trace it", "fg", 0.52, tr={"opacity": [(0, 0), (24.0, 0), (24.8, 1)]})
    # pulse the bracket pairs while each pair of sizes is read (axis 2, then 1, then 0)
    pulse = {"ax2": (5.8, 7.0), "ax1": (7.0, 8.2), "ax0": (8.2, 9.4)}
    for name in ("dA", "dB", "dC"):
        for ax, (t0, t1) in pulse.items():
            o = next(x for x in sc.objects if x["id"] == f"{name}_{ax}")
            o["tracks"]["color"] = [[0, "#5b6776"], [t0, "#5b6776"], [t0 + 0.4, VIOLET_EDGE], [t1, VIOLET_EDGE], [t1 + 0.4, "#5b6776"]]

    tg = (-1.9, -1.0, 2.0)
    az, el = 36, 24
    sc.camera_keys((0, orbit(tg, az, el, 40), tg, None, 1.0), (10, orbit(tg, az, el, 40), tg, None, 1.0), (18, orbit(tg, az - 3, el, 40), tg, None, 1.0),
                   (26, orbit(tg, az - 6, el + 1, 40), (tg[0] + 0.6, tg[1], tg[2]), None, 1.06))
    sc.view("front", "Front view", orbit(tg, 0, 0, 40), tg)
    sc.view("top", "Top view", (tg[0], 40, tg[2] + 0.01), tg)
    sc.view("iso", "3D view", orbit(tg, az, el, 40), tg)
    return sc
