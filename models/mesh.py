import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class Material:
    name: str = "default"
    diffuse: Tuple[float, float, float] = (0.7, 0.7, 0.9)
    specular: Tuple[float, float, float] = (0.3, 0.3, 0.3)
    shininess: float = 32.0
    opacity: float = 1.0


class Mesh:
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: Optional[np.ndarray] = None,
        uvs: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        material: Optional[Material] = None,
        name: str = "mesh",
    ):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.faces    = np.asarray(faces,    dtype=np.int32)
        self.uvs      = np.asarray(uvs,      dtype=np.float32) if uvs     is not None else None
        self.colors   = np.asarray(colors,   dtype=np.float32) if colors  is not None else None
        self.material = material or Material()
        self.name     = name
        if normals is not None:
            self.normals = np.asarray(normals, dtype=np.float32)
        else:
            self.normals = None
            self.compute_vertex_normals()

    # ── geometry ─────────────────────────────────────────────
    def compute_vertex_normals(self) -> np.ndarray:
        nv = len(self.vertices)
        n  = np.zeros((nv, 3), dtype=np.float64)
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        for i in range(len(self.faces)):
            n[self.faces[i, 0]] += fn[i]
            n[self.faces[i, 1]] += fn[i]
            n[self.faces[i, 2]] += fn[i]
        l = np.linalg.norm(n, axis=1, keepdims=True)
        self.normals = (n / np.where(l == 0, 1.0, l)).astype(np.float32)
        return self.normals

    @property
    def vertex_count(self) -> int: return len(self.vertices)
    @property
    def face_count(self)   -> int: return len(self.faces)

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(0), self.vertices.max(0)

    def extents(self) -> np.ndarray:
        mn, mx = self.bounds(); return mx - mn

    def center(self) -> np.ndarray:
        mn, mx = self.bounds(); return (mn + mx) / 2

    def normalize(self) -> "Mesh":
        mn, mx = self.bounds()
        c = (mn + mx) / 2
        s = (mx - mn).max()
        if s > 0:
            self.vertices = (self.vertices - c) / s
        return self

    def translate(self, t) -> "Mesh":
        self.vertices += np.asarray(t, dtype=np.float32); return self

    def scale(self, s) -> "Mesh":
        self.vertices *= np.asarray(s, dtype=np.float32); return self

    def apply_matrix(self, m: np.ndarray) -> "Mesh":
        v = np.hstack([self.vertices, np.ones((len(self.vertices), 1), np.float32)])
        self.vertices = (v @ m.T)[:, :3]
        if self.normals is not None:
            rot = np.linalg.inv(m[:3, :3]).T
            n   = self.normals @ rot.T
            l   = np.linalg.norm(n, axis=1, keepdims=True)
            self.normals = (n / np.where(l == 0, 1.0, l)).astype(np.float32)
        return self

    def merge(self, other: "Mesh") -> "Mesh":
        off  = len(self.vertices)
        v    = np.vstack([self.vertices, other.vertices])
        f    = np.vstack([self.faces,    other.faces + off])
        n    = np.vstack([self.normals,  other.normals]) \
               if self.normals is not None and other.normals is not None else None
        return Mesh(v, f, normals=n, name=self.name)

    def volume(self) -> float:
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        return float(abs(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2)))) / 6.0)

    def surface_area(self) -> float:
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        return float(np.sum(np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)) / 2.0)


def merge_meshes(meshes: List[Mesh]) -> Mesh:
    if not meshes: raise ValueError("empty list")
    result = meshes[0]
    for m in meshes[1:]:
        result = result.merge(m)
    return result
