"""Builders for 3D timeline scenes (component `scene`). A scene is plain data: objects with keyframed properties.

Time is in seconds. A key is [t, value] or [t, value, ease]. Eases: linear, in, out, inout (default), step.
"""
from kit import num

CUBE = 0.9


def K(*keys):
    """K((0, v), (2, w, 'out')) -> track list."""
    return [list(k) for k in keys]


def fade(t0, t1, lo=0.0, hi=1.0):
    return K((t0, lo), (t1, hi))


def clean(tracks):
    """Sort keys by time, drop exact duplicates, and nudge equal times so that every key time is strictly increasing."""
    out = {}
    for prop, keys in tracks.items():
        ks = sorted((list(k) for k in keys), key=lambda k: k[0])
        res = []
        for k in ks:
            k = [round(float(k[0]), 4)] + k[1:]
            if res and k[0] <= res[-1][0]:
                k[0] = round(res[-1][0] + 0.001, 4)
            res.append(k)
        out[prop] = res
    return out


def vis(*spans, hi=1.0, lo=0.0, fade_s=0.4):
    """Opacity track that is `hi` inside each (t_on, t_off) span and `lo` outside. A t_off of None keeps it on."""
    keys = [(0.0, lo)]
    for on, off in spans:
        keys += [(on, lo), (on + fade_s, hi)]
        if off is not None:
            keys += [(off, hi), (off + fade_s, lo)]
    return keys


class Scene:
    def __init__(self, duration, alt, start_t=None, style="glass", ortho_span=None):
        self.duration, self.alt, self.start_t = float(duration), alt, start_t
        self.style, self.ortho_span = style, ortho_span
        self.calc, self.calc_default = {}, None
        self.chapters, self.objects, self.by_sel, self.camera, self.views = [], [], {}, [], {}
        self._ids = set()

    def chapter(self, id, title, t0, t1, changed, why, fixed, short=None):
        c = {"id": id, "title": title, "t0": float(t0), "t1": float(t1), "changed": changed, "why": why, "fixed": fixed}
        if short:
            c["short"] = short
        self.chapters.append(c)
        return self

    def add(self, obj):
        assert obj["id"] not in self._ids, f"duplicate scene object {obj['id']}"
        self._ids.add(obj["id"])
        self.objects.append(obj)
        return obj

    def cube(self, id, pos, label, color, size=CUBE, title=None, links=None, ghost=False, pickable=True, tr=None):
        o = {"id": id, "type": "cube", "pos": list(map(float, pos)), "label": str(label), "color": color, "size": size}
        if title:
            o["title"] = title
        if links:
            o["links"] = list(links)
        if ghost:
            o["ghost"] = True
        if not pickable:
            o["pickable"] = False
        if tr:
            o["tracks"] = clean(tr)
        return self.add(o)

    def text(self, id, pos, text, color="fg", height=0.6, weight=600, tr=None):
        o = {"id": id, "type": "text", "pos": list(map(float, pos)), "text": text, "color": color, "height": height, "weight": weight}
        if tr:
            o["tracks"] = clean(tr)
        return self.add(o)

    def arrow(self, id, a, b, color="#888888", label=None, label_offset=None, tr=None):
        o = {"id": id, "type": "arrow", "from": list(map(float, a)), "to": list(map(float, b)), "color": color}
        if label:
            o["label"] = label
        if label_offset:
            o["label_offset"] = label_offset
        if tr:
            o["tracks"] = clean(tr)
        return self.add(o)

    def dim(self, id, a, b, label, tick=(0, -0.22, 0), color="#6b7680", label_offset=None, tr=None):
        """A dimension bracket along an edge: a line with end ticks and a label, like 'axis 1 (4)'."""
        o = {"id": id, "type": "dim", "from": list(map(float, a)), "to": list(map(float, b)), "tick": list(map(float, tick)), "label": label, "color": color}
        if label_offset:
            o["label_offset"] = list(map(float, label_offset))
        if tr:
            o["tracks"] = clean(tr)
        return self.add(o)

    def set_calc(self, id, terms, ops):
        """Worked-computation card for a selected object. terms: [(label, value, tone)], tone a|b|c|n. ops: symbols between terms."""
        self.calc[id] = {"terms": [{"label": l, "value": str(v), "tone": t} for l, v, t in terms], "ops": list(ops)}
        return self

    def box(self, id, center, size, color="#888888", label=None, tr=None):
        o = {"id": id, "type": "box", "pos": list(map(float, center)), "size": list(map(float, size)), "color": color}
        if label:
            o["label"] = label
        if tr:
            o["tracks"] = clean(tr)
        return self.add(o)

    def camera_keys(self, *keys):
        """keys: (t, pos, target[, ease[, zoom]])"""
        self.camera = [{"t": float(k[0]), "pos": list(map(float, k[1])), "target": list(map(float, k[2])),
                        **({"ease": k[3]} if len(k) > 3 and k[3] else {}), **({"zoom": float(k[4])} if len(k) > 4 else {})} for k in keys]
        return self

    def view(self, name, label, pos, target):
        self.views[name] = {"label": label, "pos": list(map(float, pos)), "target": list(map(float, target))}
        return self

    def to_dict(self):
        d = {"duration": self.duration, "alt": self.alt, "chapters": self.chapters, "objects": self.objects,
             "camera": {"keys": self.camera, **({"mode": "ortho", "span": self.ortho_span} if self.ortho_span else {})}, "views": self.views}
        if self.style != "glass":
            d["style"] = self.style
        if self.calc:
            d["calc"] = self.calc
            d["calc_default"] = self.calc_default or next(iter(self.calc))
        if self.by_sel:
            d["by_sel"] = self.by_sel
        if self.start_t is not None:
            d["start_t"] = float(self.start_t)
        return d


def orbit(target, az, el, dist):
    """Camera position for a target, an azimuth (degrees from +z towards +x), an elevation, and a distance."""
    import math
    a, e = math.radians(az), math.radians(el)
    return (target[0] + dist * math.sin(a) * math.cos(e), target[1] + dist * math.sin(e), target[2] + dist * math.cos(a) * math.cos(e))
