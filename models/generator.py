"""
3D Model Generator — standard-weight occupancy network + analytic SDF hybrid.

Architecture
------------
1.  Text → 64-d embedding via keyword lookup table
2.  Analytic SDF selected from keyword map (sphere, torus, terrain …)
3.  OccupancyNetwork MLP (3 + 64 → 256 × 5 layers → 1) produces neural SDF
4.  Blend: sdf = (1-α)·analytic + α·neural
5.  Marching cubes → Mesh
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable

from .mesh import Mesh
from .sdf import parse_sdf, sdf_sphere, smooth_union
from .primitives import make_sphere


# ─── text encoder ────────────────────────────────────────────────────────────

_EMBED_DIM = 64
_RNG = np.random.default_rng(42)

_KEYWORDS = [
    "sphere","ball","cube","box","cylinder","tube","cone","pyramid",
    "torus","donut","ring","terrain","mountain","landscape","capsule",
    "smooth","rough","organic","sharp","twisted","tall","wide","flat",
    "small","large","hollow","solid","simple","complex","abstract",
    "round","angular","curved","thick","thin","detailed",
]
_EMBED_TABLE = {kw: _RNG.standard_normal(_EMBED_DIM).astype(np.float32) for kw in _KEYWORDS}


def encode_text(prompt: str) -> np.ndarray:
    tokens = re.findall(r"[a-z]+", prompt.lower())
    vecs = []
    for t in tokens:
        if t in _EMBED_TABLE:
            vecs.append(_EMBED_TABLE[t])
        else:
            # deterministic unknown-token embedding via hash
            h = abs(hash(t)) % len(_KEYWORDS)
            vecs.append(_EMBED_TABLE[_KEYWORDS[h]] * 0.25)
    return np.mean(vecs, axis=0).astype(np.float32) if vecs else np.zeros(_EMBED_DIM, np.float32)


# ─── network layers ───────────────────────────────────────────────────────────

def _relu(x):  return np.maximum(0.0, x)


class _Linear:
    def __init__(self, din, dout, rng):
        s = np.sqrt(2.0 / din)
        self.W = (rng.standard_normal((din, dout)) * s).astype(np.float32)
        self.b = np.zeros(dout, np.float32)
    def __call__(self, x): return x @ self.W + self.b


class OccupancyNetwork:
    """
    Standard-weight (float32) 5-layer MLP with skip connection at layer 3.
    Conditioned on a text embedding via bias modulation in layer 0.
    """
    H = 256

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        d0 = 3 + _EMBED_DIM
        h  = self.H
        self.layers = [
            _Linear(d0,    h,  rng),
            _Linear(h,     h,  rng),
            _Linear(h,     h,  rng),
            _Linear(h+d0,  h,  rng),   # skip
            _Linear(h,     h,  rng),
            _Linear(h,     1,  rng),
        ]

    def condition(self, text_emb: np.ndarray, noise: float = 0.12):
        """Modulate weights so each prompt yields a different shape."""
        key = int(abs(text_emb.sum() * 1e6)) % (2**31)
        rng = np.random.default_rng(key)
        for layer in self.layers:
            layer.W += (rng.standard_normal(layer.W.shape) * noise).astype(np.float32)
        self.layers[0].b[:_EMBED_DIM] += text_emb * 0.5

    def __call__(self, xyz: np.ndarray, emb: np.ndarray) -> np.ndarray:
        """xyz: (N,3), emb: (64,)  →  (N,) SDF values."""
        e  = np.broadcast_to(emb[None], (len(xyz), _EMBED_DIM))
        x0 = np.concatenate([xyz, e], axis=-1).astype(np.float32)
        x  = x0
        for i, layer in enumerate(self.layers[:-1]):
            x = _relu(layer(x))
            if i == 2:
                x = np.concatenate([x, x0], axis=-1)
        return self.layers[-1](x)[:, 0]


# ─── params / generator ──────────────────────────────────────────────────────

@dataclass
class GenerationParams:
    resolution:    int   = 48     # voxel grid N³
    iso_value:     float = 0.0    # marching-cubes threshold
    neural_blend:  float = 0.4    # 0 = pure analytic, 1 = pure neural
    smooth_iters:  int   = 1      # normal-smoothing passes
    scale:         float = 1.0    # final mesh scale
    seed:          int   = 0


class ModelGenerator:
    """Text-to-3D mesh generator."""

    BATCH = 8192   # points evaluated per forward pass (CPU friendly)

    def __init__(self):
        self._cb: Optional[Callable[[float, str], None]] = None

    def set_progress_callback(self, cb: Callable[[float, str], None]):
        self._cb = cb

    def _progress(self, f: float, msg: str):
        if self._cb: self._cb(f, msg)

    def generate(self, prompt: str, params: Optional[GenerationParams] = None) -> Mesh:
        if params is None: params = GenerationParams()

        self._progress(0.05, "Encoding prompt…")
        emb   = encode_text(prompt)
        net   = OccupancyNetwork(seed=params.seed)
        net.condition(emb)

        self._progress(0.15, "Building voxel grid…")
        res = params.resolution
        lin = np.linspace(-0.55, 0.55, res, dtype=np.float32)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        xyz = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

        self._progress(0.25, "Analytic SDF…")
        analytic_fn = parse_sdf(prompt) or sdf_sphere
        analytic    = analytic_fn(xyz)

        self._progress(0.40, "Neural forward pass…")
        neural = np.zeros(len(xyz), np.float32)
        for start in range(0, len(xyz), self.BATCH):
            end = min(start + self.BATCH, len(xyz))
            neural[start:end] = net(xyz[start:end], emb)
            self._progress(0.40 + 0.25 * end / len(xyz), "Neural forward pass…")

        self._progress(0.67, "Blending SDFs…")
        α   = params.neural_blend
        sdf = ((1 - α) * analytic + α * neural).reshape(res, res, res)

        self._progress(0.75, "Marching cubes…")
        mesh = self._mc(sdf, params.iso_value, 1.1 / res)

        if mesh is None or mesh.vertex_count < 4:
            self._progress(0.90, "Fallback sphere…")
            mesh = make_sphere(32, 16)
        else:
            mesh.normalize().scale(params.scale)
            for _ in range(params.smooth_iters):
                mesh.compute_vertex_normals()

        mesh.name = (prompt[:30] if prompt else "model").strip()
        self._progress(1.0, "Done.")
        return mesh

    @staticmethod
    def _mc(field: np.ndarray, iso: float, spacing: float) -> Optional[Mesh]:
        try:
            from skimage.measure import marching_cubes as ski_mc
            v, f, n, _ = ski_mc(field, level=iso, spacing=(spacing,)*3)
            return Mesh(v.astype(np.float32), f.astype(np.int32),
                        normals=n.astype(np.float32))
        except ImportError:
            pass
        except Exception:
            pass
        # Pure-numpy fallback — coarser but always works
        return _mc_numpy(field, iso, spacing)


# ─── minimal marching-cubes fallback ─────────────────────────────────────────

def _interp(p1, p2, v1, v2, iso):
    if abs(v1 - v2) < 1e-9: return p1
    t = (iso - v1) / (v2 - v1)
    return p1 + t*(p2 - p1)


_EDGE_PAIRS = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def _mc_numpy(field, iso, sp) -> Optional[Mesh]:
    """Simplified marching cubes — only extracts surface cells."""
    nx, ny, nz = field.shape
    verts, faces = [], []

    corners_offsets = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                                 [0,0,1],[1,0,1],[1,1,1],[0,1,1]], int)
    for ix in range(nx-1):
        for iy in range(ny-1):
            for iz in range(nz-1):
                co = corners_offsets + np.array([ix,iy,iz])
                vals = np.array([field[c[0],c[1],c[2]] for c in co], np.float32)
                if (vals < iso).all() or (vals >= iso).all():
                    continue
                pts = co * sp  # world positions
                edge_verts = {}
                for e,(i1,i2) in enumerate(_EDGE_PAIRS):
                    if (vals[i1] < iso) != (vals[i2] < iso):
                        edge_verts[e] = _interp(pts[i1], pts[i2], vals[i1], vals[i2], iso)
                ev = list(edge_verts.values())
                if len(ev) < 3: continue
                base = len(verts)
                verts.extend(ev)
                for i in range(1, len(ev)-1):
                    faces.append([base, base+i, base+i+1])

    if not verts: return None
    return Mesh(np.array(verts, np.float32), np.array(faces, np.int32))
