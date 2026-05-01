"""
Signed Distance Field (SDF) functions used by the generator.
Each function accepts xyz (N,3) float32 and returns (N,) SDF values.
Negative = inside the surface, positive = outside.
"""
import numpy as np


def sdf_sphere(xyz, r=0.42):
    return np.linalg.norm(xyz, axis=1) - r


def sdf_box(xyz, hx=0.38, hy=0.38, hz=0.38):
    q = np.abs(xyz) - np.array([hx, hy, hz], np.float32)
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0)


def sdf_cylinder(xyz, r=0.35, h=0.65):
    xz = xyz[:, [0, 2]]
    return np.maximum(np.linalg.norm(xz, axis=1) - r, np.abs(xyz[:, 1]) - h/2)


def sdf_cone(xyz, angle=0.5, h=0.7):
    q = np.stack([np.linalg.norm(xyz[:, [0, 2]], axis=1), -xyz[:, 1]+h/2], axis=1)
    tip = np.array([np.sin(angle), np.cos(angle)])
    d = q @ tip - h*np.cos(angle)
    return np.maximum(d, -q[:, 1]-h)


def sdf_torus(xyz, R=0.38, r=0.16):
    q = np.stack([np.linalg.norm(xyz[:, [0, 2]], axis=1) - R, xyz[:, 1]], axis=1)
    return np.linalg.norm(q, axis=1) - r


def sdf_terrain(xyz):
    x, z = xyz[:, 0], xyz[:, 2]
    h = (np.sin(x*5)*0.08 + np.sin(z*4.3)*0.07
         + np.sin(x*2+z*3.7)*0.1 + np.sin(x*8-z*6)*0.04)
    return xyz[:, 1] - (h - 0.18)


def sdf_capsule(xyz, r=0.25, h=0.5):
    y = np.clip(xyz[:, 1], -h/2, h/2)
    q = np.stack([np.linalg.norm(xyz[:, [0, 2]], axis=1), xyz[:, 1]-y], axis=1)
    return np.linalg.norm(q, axis=1) - r


def sdf_ellipsoid(xyz, rx=0.5, ry=0.3, rz=0.4):
    k0 = np.linalg.norm(xyz / np.array([rx, ry, rz], np.float32), axis=1)
    k1 = np.linalg.norm(xyz / np.array([rx**2, ry**2, rz**2], np.float32), axis=1)
    return k0*(k0 - 1.0) / k1


def smooth_union(a, b, k=0.25):
    h = np.clip(0.5 + 0.5*(b-a)/k, 0, 1)
    return b + h*(a-b) - k*h*(1-h)


def smooth_sub(a, b, k=0.2):
    h = np.clip(0.5 - 0.5*(b+a)/k, 0, 1)
    return a + h*(-b-a) + k*h*(1-h)


KEYWORD_MAP = {
    "sphere": sdf_sphere, "ball": sdf_sphere, "globe": sdf_sphere, "orb": sdf_sphere,
    "cube":   sdf_box,    "box":  sdf_box,    "block": sdf_box,
    "cylinder": sdf_cylinder, "tube": sdf_cylinder, "pipe": sdf_cylinder,
    "cone":   sdf_cone,   "pyramid": sdf_cone, "spike": sdf_cone,
    "torus":  sdf_torus,  "donut": sdf_torus, "ring": sdf_torus,
    "terrain": sdf_terrain, "mountain": sdf_terrain, "landscape": sdf_terrain,
    "capsule": sdf_capsule, "pill": sdf_capsule,
    "ellipsoid": sdf_ellipsoid, "oval": sdf_ellipsoid,
}


def parse_sdf(prompt: str):
    """Return the best-matching SDF function for a text prompt."""
    p = prompt.lower()
    for kw, fn in KEYWORD_MAP.items():
        if kw in p:
            return fn
    return None
