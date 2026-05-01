"""
Smart 3D generator — two modes:

1. Rule-based (no training needed):
   - Parses multiple shape keywords and composes them with smooth union/subtraction
   - Extracts adjectives ("tall", "wide", "flat") to modify SDF parameters
   - Supports "with a hole", "on top of", "inside", "hollow" etc.

2. Neural (after training):
   - Loads weights from models/weights/<name>.npz (numpy) or .pth (PyTorch)
   - OccupancyNetwork is a genuine trained occupancy/SDF decoder
   - Falls back to rule-based if no weights found
"""

import re, os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple

from .mesh import Mesh
from .sdf import (
    sdf_sphere, sdf_box, sdf_cylinder, sdf_cone, sdf_torus,
    sdf_terrain, sdf_capsule, sdf_ellipsoid,
    smooth_union, smooth_sub,
)
from .primitives import make_sphere


# ─────────────────────────────────────────────────────────────────────────────
# Text understanding
# ─────────────────────────────────────────────────────────────────────────────

# Shape tokens → SDF factory (returns a callable(xyz)->values)
_SHAPE_FACTORIES = {
    "sphere":    lambda p: (lambda xyz: sdf_sphere(xyz, p.get("r", 0.42))),
    "ball":      lambda p: (lambda xyz: sdf_sphere(xyz, p.get("r", 0.42))),
    "globe":     lambda p: (lambda xyz: sdf_sphere(xyz, p.get("r", 0.42))),
    "orb":       lambda p: (lambda xyz: sdf_sphere(xyz, p.get("r", 0.42))),
    "cube":      lambda p: (lambda xyz: sdf_box(xyz, p.get("hx",0.38), p.get("hy",0.38), p.get("hz",0.38))),
    "box":       lambda p: (lambda xyz: sdf_box(xyz, p.get("hx",0.38), p.get("hy",0.38), p.get("hz",0.38))),
    "block":     lambda p: (lambda xyz: sdf_box(xyz, p.get("hx",0.38), p.get("hy",0.38), p.get("hz",0.38))),
    "brick":     lambda p: (lambda xyz: sdf_box(xyz, p.get("hx",0.35), p.get("hy",0.20), p.get("hz",0.18))),
    "cylinder":  lambda p: (lambda xyz: sdf_cylinder(xyz, p.get("r",0.32), p.get("h",0.65))),
    "tube":      lambda p: (lambda xyz: sdf_cylinder(xyz, p.get("r",0.32), p.get("h",0.65))),
    "pipe":      lambda p: (lambda xyz: sdf_cylinder(xyz, p.get("r",0.18), p.get("h",0.70))),
    "barrel":    lambda p: (lambda xyz: sdf_cylinder(xyz, p.get("r",0.38), p.get("h",0.45))),
    "pillar":    lambda p: (lambda xyz: sdf_cylinder(xyz, p.get("r",0.15), p.get("h",0.80))),
    "cone":      lambda p: (lambda xyz: sdf_cone(xyz, p.get("angle",0.5), p.get("h",0.70))),
    "pyramid":   lambda p: (lambda xyz: sdf_cone(xyz, p.get("angle",0.45), p.get("h",0.70))),
    "spike":     lambda p: (lambda xyz: sdf_cone(xyz, p.get("angle",0.25), p.get("h",0.80))),
    "torus":     lambda p: (lambda xyz: sdf_torus(xyz, p.get("R",0.36), p.get("r",0.14))),
    "donut":     lambda p: (lambda xyz: sdf_torus(xyz, p.get("R",0.36), p.get("r",0.14))),
    "ring":      lambda p: (lambda xyz: sdf_torus(xyz, p.get("R",0.38), p.get("r",0.08))),
    "terrain":   lambda p: (lambda xyz: sdf_terrain(xyz)),
    "mountain":  lambda p: (lambda xyz: sdf_terrain(xyz)),
    "landscape": lambda p: (lambda xyz: sdf_terrain(xyz)),
    "hill":      lambda p: (lambda xyz: sdf_terrain(xyz)),
    "capsule":   lambda p: (lambda xyz: sdf_capsule(xyz, p.get("r",0.24), p.get("h",0.50))),
    "pill":      lambda p: (lambda xyz: sdf_capsule(xyz, p.get("r",0.20), p.get("h",0.45))),
    "ellipsoid": lambda p: (lambda xyz: sdf_ellipsoid(xyz, p.get("rx",0.45), p.get("ry",0.28), p.get("rz",0.35))),
    "egg":       lambda p: (lambda xyz: sdf_ellipsoid(xyz, p.get("rx",0.30), p.get("ry",0.40), p.get("rz",0.28))),
    "disc":      lambda p: (lambda xyz: sdf_cylinder(xyz, p.get("r",0.45), p.get("h",0.08))),
    "disk":      lambda p: (lambda xyz: sdf_cylinder(xyz, p.get("r",0.45), p.get("h",0.08))),
    "plate":     lambda p: (lambda xyz: sdf_box(xyz, p.get("hx",0.45), p.get("hy",0.06), p.get("hz",0.35))),
    "plane":     lambda p: (lambda xyz: xyz[:, 1] + 0.02),
}

# Adjective → parameter modifiers
_ADJECTIVE_MODS = {
    "tall":    {"h": 1.6,  "hy": 1.6},
    "short":   {"h": 0.5,  "hy": 0.5},
    "wide":    {"hx": 1.5, "hz": 1.5, "r": 1.4,  "R": 1.4},
    "narrow":  {"hx": 0.5, "hz": 0.5, "r": 0.6},
    "thin":    {"hx": 0.4, "hz": 0.4, "r": 0.5,  "h": 1.4},
    "thick":   {"hx": 1.3, "hz": 1.3, "r": 1.3},
    "flat":    {"hy": 0.35,"h": 0.3},
    "large":   {"r": 1.5,  "hx":1.5,  "hy":1.5,  "hz":1.5, "R":1.5},
    "big":     {"r": 1.4,  "hx":1.4,  "hy":1.4,  "hz":1.4},
    "small":   {"r": 0.6,  "hx":0.6,  "hy":0.6,  "hz":0.6},
    "tiny":    {"r": 0.4,  "hx":0.4,  "hy":0.4,  "hz":0.4},
    "round":   {"angle": 0.6, "r": 0.22},
    "sharp":   {"angle": 0.2},
    "stretched":{"hy":1.8},
    "squashed":{"hy":0.4},
    "fat":     {"r": 1.4, "R": 1.2},
    "slim":    {"r": 0.5},
}

# Boolean operation keywords
_OP_PATTERNS = [
    (r"\bwith (?:a |an )?\bhole\b",      "hollow"),
    (r"\bhollow\b",                        "hollow"),
    (r"\bwith (?:a |an )?(\w+) on top\b", "stack"),
    (r"\bon top of\b",                     "stack"),
    (r"\bwith (?:a |an )?(\w+) cut\b",    "subtract"),
    (r"\bminus\b",                         "subtract"),
    (r"\band\b",                           "union"),
    (r"\bmerged? with\b",                  "union"),
    (r"\bcombined with\b",                 "union"),
]


def _default_params() -> dict:
    return {"r":0.38,"R":0.36,"hx":0.38,"hy":0.38,"hz":0.38,
            "h":0.65,"angle":0.5,"rx":0.45,"ry":0.28,"rz":0.35}


def _apply_adjectives(params: dict, prompt: str) -> dict:
    p = prompt.lower()
    for adj, mods in _ADJECTIVE_MODS.items():
        if adj in p:
            for k, mult in mods.items():
                if k in params:
                    params[k] *= mult
    # clamp to sane range
    for k in params:
        params[k] = float(np.clip(params[k], 0.02, 2.0))
    return params


def _find_shapes(prompt: str) -> List[str]:
    p = prompt.lower()
    found = []
    for kw in _SHAPE_FACTORIES:
        if re.search(rf"\b{kw}\b", p):
            found.append(kw)
    return found if found else ["sphere"]  # default


def parse_prompt(prompt: str):
    """
    Returns a callable sdf(xyz) -> values by composing shapes found in the prompt.
    """
    p = prompt.lower()
    params = _apply_adjectives(_default_params(), p)
    shapes = _find_shapes(p)

    # Build individual SDFs
    sdfs = [_SHAPE_FACTORIES[s](params) for s in shapes[:4]]  # max 4 shapes

    # Start with primary shape
    def composed(xyz):
        base = sdfs[0](xyz)
        # Additional shapes → smooth union
        for extra in sdfs[1:]:
            base = smooth_union(base, extra(xyz), k=0.18)
        return base

    # Boolean modifications
    if re.search(r"\bhollow\b|\bwith (?:a |an )?hole\b", p):
        # Carve out a smaller version of the primary shape
        inner_params = {k: v * 0.65 for k, v in params.items()}
        inner_fn = _SHAPE_FACTORIES[shapes[0]](inner_params)
        outer = composed

        def composed(xyz, _o=outer, _i=inner_fn):
            return smooth_sub(_o(xyz), _i(xyz), k=0.05)

    if re.search(r"\btwisted?\b|\bspiral\b|\bhelix\b", p):
        base_fn = composed

        def composed(xyz, _b=base_fn):
            angle = xyz[:, 1] * 2.5
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rx = xyz[:, 0] * cos_a - xyz[:, 2] * sin_a
            rz = xyz[:, 0] * sin_a + xyz[:, 2] * cos_a
            xyz2 = np.stack([rx, xyz[:, 1], rz], axis=1)
            return _b(xyz2)

    if re.search(r"\bstacked?\b|\bon top\b|\bon a\b", p) and len(shapes) >= 2:
        # Stack second shape above first
        fn1 = _SHAPE_FACTORIES[shapes[0]](params)
        fn2 = _SHAPE_FACTORIES[shapes[1] if len(shapes) > 1 else shapes[0]](params)
        hy = params.get("hy", 0.38)

        def composed(xyz, _a=fn1, _b=fn2, _hy=hy):
            shifted = xyz - np.array([0, _hy * 1.6, 0], np.float32)
            return smooth_union(_a(xyz), _b(shifted), k=0.12)

    if re.search(r"\bbumpy\b|\bspiky\b|\bnodules?\b", p):
        base_fn = composed

        def composed(xyz, _b=base_fn):
            freq = 8.0
            noise = 0.06 * np.sin(freq * xyz[:, 0]) * np.sin(freq * xyz[:, 1]) * np.sin(freq * xyz[:, 2])
            return _b(xyz) + noise

    if re.search(r"\bsmooth\b|\bpolished\b", p):
        base_fn = composed

        def composed(xyz, _b=base_fn):  # smooth = slight offset (rounded edges)
            return _b(xyz) - 0.02

    return composed


# ─────────────────────────────────────────────────────────────────────────────
# Network — numpy inference, PyTorch training
# ─────────────────────────────────────────────────────────────────────────────

_EMBED_DIM = 64
_RNG0 = np.random.default_rng(42)

_KEYWORDS = list(_SHAPE_FACTORIES.keys()) + list(_ADJECTIVE_MODS.keys()) + [
    "smooth","rough","organic","sharp","abstract","complex","simple",
    "detailed","hollow","solid","twisted","spiral","bumpy","spiky",
]
_EMBED_TABLE = {kw: _RNG0.standard_normal(_EMBED_DIM).astype(np.float32) for kw in _KEYWORDS}


def encode_text(prompt: str) -> np.ndarray:
    tokens = re.findall(r"[a-z]+", prompt.lower())
    vecs = []
    for t in tokens:
        if t in _EMBED_TABLE:
            vecs.append(_EMBED_TABLE[t])
        else:
            h = abs(hash(t)) % len(_KEYWORDS)
            vecs.append(_EMBED_TABLE[_KEYWORDS[h]] * 0.2)
    return np.mean(vecs, axis=0).astype(np.float32) if vecs else np.zeros(_EMBED_DIM, np.float32)


def _relu(x): return np.maximum(0.0, x)


class _Linear:
    def __init__(self, din, dout, rng=None, W=None, b=None):
        if W is not None:
            self.W, self.b = W, b
        else:
            s = np.sqrt(2.0 / din)
            self.W = (rng.standard_normal((din, dout)) * s).astype(np.float32)
            self.b = np.zeros(dout, np.float32)

    def __call__(self, x): return x @ self.W + self.b


class OccupancyNetwork:
    """
    5-layer MLP, float32 standard weights.
    Input : (N, 3+64)   Output: (N,) occupancy / SDF
    Skip connection at layer 3 (like DeepSDF).
    Weights can be loaded from a .npz file produced by training.
    """
    H = 256

    def __init__(self, seed: int = 0, weights_path: Optional[str] = None):
        rng = np.random.default_rng(seed)
        d0, h = 3 + _EMBED_DIM, self.H
        self.layers = [
            _Linear(d0,    h,  rng),
            _Linear(h,     h,  rng),
            _Linear(h,     h,  rng),
            _Linear(h+d0,  h,  rng),
            _Linear(h,     h,  rng),
            _Linear(h,     1,  rng),
        ]
        if weights_path and os.path.exists(weights_path):
            self.load(weights_path)

    def load(self, path: str):
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            layer.W = data[f"W{i}"].astype(np.float32)
            layer.b = data[f"b{i}"].astype(np.float32)

    def save(self, path: str):
        kv = {}
        for i, layer in enumerate(self.layers):
            kv[f"W{i}"] = layer.W
            kv[f"b{i}"] = layer.b
        np.savez(path, **kv)

    def condition(self, emb: np.ndarray, noise: float = 0.08):
        key = int(abs(emb.sum() * 1e6)) % (2**31)
        rng = np.random.default_rng(key)
        for layer in self.layers:
            layer.W += (rng.standard_normal(layer.W.shape) * noise).astype(np.float32)
        self.layers[0].b[:_EMBED_DIM] += emb * 0.4

    def __call__(self, xyz: np.ndarray, emb: np.ndarray) -> np.ndarray:
        e  = np.broadcast_to(emb[None], (len(xyz), _EMBED_DIM))
        x0 = np.concatenate([xyz, e], axis=-1).astype(np.float32)
        x  = x0
        for i, layer in enumerate(self.layers[:-1]):
            x = _relu(layer(x))
            if i == 2: x = np.concatenate([x, x0], axis=-1)
        return self.layers[-1](x)[:, 0]


# ─────────────────────────────────────────────────────────────────────────────
# Params + generator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationParams:
    resolution:   int   = 48
    iso_value:    float = 0.0
    neural_blend: float = 0.0    # 0 = pure rule-based (recommended without training)
    smooth_iters: int   = 1
    scale:        float = 1.0
    seed:         int   = 0
    weights_path: Optional[str] = None  # path to .npz trained weights


_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")


def _find_weights(prompt: str) -> Optional[str]:
    """Look for a .npz weights file matching the dominant shape keyword."""
    if not os.path.isdir(_WEIGHTS_DIR):
        return None
    shapes = _find_shapes(prompt)
    for shape in shapes:
        p = os.path.join(_WEIGHTS_DIR, f"{shape}.npz")
        if os.path.exists(p): return p
    general = os.path.join(_WEIGHTS_DIR, "model.npz")
    return general if os.path.exists(general) else None


class ModelGenerator:
    BATCH = 8192

    def __init__(self):
        self._cb = None

    def set_progress_callback(self, cb): self._cb = cb

    def _p(self, f, msg):
        if self._cb: self._cb(f, msg)

    def generate(self, prompt: str, params: Optional[GenerationParams] = None) -> Mesh:
        if params is None: params = GenerationParams()

        self._p(0.05, "Parsing prompt…")
        rule_sdf = parse_prompt(prompt)

        # Check for trained weights
        wpath = params.weights_path or _find_weights(prompt)
        use_neural = (params.neural_blend > 0) and (wpath is not None)

        if use_neural:
            self._p(0.12, f"Loading weights: {os.path.basename(wpath)}")
            emb = encode_text(prompt)
            net = OccupancyNetwork(seed=params.seed, weights_path=wpath)
        else:
            if params.neural_blend > 0:
                self._p(0.12, "No trained weights found — using rule-based SDF only")
            params = GenerationParams(**{**params.__dict__, "neural_blend": 0.0})

        self._p(0.18, "Building voxel grid…")
        res = params.resolution
        lin = np.linspace(-0.58, 0.58, res, dtype=np.float32)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        xyz = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

        self._p(0.28, "Evaluating SDF…")
        rule_vals = rule_sdf(xyz)

        if use_neural:
            self._p(0.40, "Neural forward pass…")
            neural_vals = np.zeros(len(xyz), np.float32)
            for s in range(0, len(xyz), self.BATCH):
                e = min(s + self.BATCH, len(xyz))
                neural_vals[s:e] = net(xyz[s:e], emb)
                self._p(0.40 + 0.22 * e / len(xyz), "Neural forward pass…")
            α = params.neural_blend
            sdf_grid = ((1 - α) * rule_vals + α * neural_vals).reshape(res, res, res)
        else:
            sdf_grid = rule_vals.reshape(res, res, res)

        self._p(0.65, "Marching cubes…")
        mesh = self._mc(sdf_grid, params.iso_value, 1.16 / res)

        if mesh is None or mesh.vertex_count < 4:
            self._p(0.90, "Fallback mesh…")
            mesh = make_sphere(32, 16)
        else:
            mesh.normalize().scale(params.scale)
            for _ in range(params.smooth_iters):
                mesh.compute_vertex_normals()

        mesh.name = prompt[:30].strip()
        self._p(1.0, "Done.")
        return mesh

    @staticmethod
    def _mc(field, iso, spacing) -> Optional[Mesh]:
        try:
            from skimage.measure import marching_cubes as ski_mc
            v, f, n, _ = ski_mc(field, level=iso, spacing=(spacing,) * 3)
            return Mesh(v.astype(np.float32), f.astype(np.int32), normals=n.astype(np.float32))
        except Exception:
            pass
        return _mc_fallback(field, iso, spacing)


def _mc_fallback(field, iso, sp) -> Optional[Mesh]:
    nx, ny, nz = field.shape
    verts, faces = [], []
    ep = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    co = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                   [0,0,1],[1,0,1],[1,1,1],[0,1,1]], int)
    for ix in range(nx-1):
        for iy in range(ny-1):
            for iz in range(nz-1):
                c = co + [ix,iy,iz]
                vals = np.array([field[c[j,0],c[j,1],c[j,2]] for j in range(8)], np.float32)
                if (vals<iso).all() or (vals>=iso).all(): continue
                pts = c * sp
                ev = {}
                for e,(i1,i2) in enumerate(ep):
                    if (vals[i1]<iso) != (vals[i2]<iso):
                        t = (iso-vals[i1])/(vals[i2]-vals[i1]+1e-9)
                        ev[e] = pts[i1] + t*(pts[i2]-pts[i1])
                evl = list(ev.values())
                if len(evl) < 3: continue
                b = len(verts); verts.extend(evl)
                for i in range(1, len(evl)-1):
                    faces.append([b, b+i, b+i+1])
    if not verts: return None
    return Mesh(np.array(verts, np.float32), np.array(faces, np.int32))
