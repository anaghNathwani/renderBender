"""Analytic mesh primitives — no neural net required."""
import numpy as np
from .mesh import Mesh


def make_sphere(radius: float = 0.5, segs: int = 32, rings: int = 16) -> Mesh:
    verts, faces = [], []
    for i in range(rings + 1):
        phi = np.pi * i / rings
        for j in range(segs):
            theta = 2 * np.pi * j / segs
            verts.append([radius * np.sin(phi) * np.cos(theta),
                          radius * np.cos(phi),
                          radius * np.sin(phi) * np.sin(theta)])
    for i in range(rings):
        for j in range(segs):
            a = i * segs + j
            b = i * segs + (j + 1) % segs
            c = (i + 1) * segs + (j + 1) % segs
            d = (i + 1) * segs + j
            faces += [[a, b, c], [a, c, d]]
    return Mesh(np.array(verts, np.float32), np.array(faces, np.int32), name="sphere")


def make_box(wx=1.0, wy=1.0, wz=1.0) -> Mesh:
    x, y, z = wx / 2, wy / 2, wz / 2
    v = np.array([
        [-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],
        [-x,-y, z],[x,-y, z],[x,y, z],[-x,y, z],
    ], np.float32)
    f = np.array([
        [0,1,2],[0,2,3],[4,6,5],[4,7,6],
        [0,5,1],[0,4,5],[2,6,7],[2,7,3],
        [0,3,7],[0,7,4],[1,5,6],[1,6,2],
    ], np.int32)
    return Mesh(v, f, name="cube")


def make_cylinder(radius=0.5, height=1.0, segs=32) -> Mesh:
    verts = [[0, -height/2, 0], [0, height/2, 0]]
    for cap in [-height/2, height/2]:
        for i in range(segs):
            a = 2*np.pi*i/segs
            verts.append([radius*np.cos(a), cap, radius*np.sin(a)])
    faces = []
    bs, ts = 2, 2 + segs
    for i in range(segs):
        j = (i+1) % segs
        faces += [[0, bs+j, bs+i], [1, ts+i, ts+j],
                  [bs+i, ts+j, ts+i], [bs+i, bs+j, ts+j]]
    return Mesh(np.array(verts, np.float32), np.array(faces, np.int32), name="cylinder")


def make_torus(R=0.5, r=0.2, maj=32, mn=16) -> Mesh:
    verts, faces = [], []
    for i in range(maj):
        phi = 2*np.pi*i/maj
        for j in range(mn):
            th = 2*np.pi*j/mn
            x = (R + r*np.cos(th)) * np.cos(phi)
            y = r*np.sin(th)
            z = (R + r*np.cos(th)) * np.sin(phi)
            verts.append([x, y, z])
    for i in range(maj):
        ni = (i+1) % maj
        for j in range(mn):
            nj = (j+1) % mn
            a,b,c,d = i*mn+j, ni*mn+j, ni*mn+nj, i*mn+nj
            faces += [[a,b,c],[a,c,d]]
    return Mesh(np.array(verts, np.float32), np.array(faces, np.int32), name="torus")


def make_cone(radius=0.5, height=1.0, segs=32) -> Mesh:
    verts = [[0, height/2, 0], [0, -height/2, 0]]
    for i in range(segs):
        a = 2*np.pi*i/segs
        verts.append([radius*np.cos(a), -height/2, radius*np.sin(a)])
    faces = []
    base = 2
    for i in range(segs):
        j = (i+1) % segs
        faces += [[0, base+i, base+j], [1, base+j, base+i]]
    return Mesh(np.array(verts, np.float32), np.array(faces, np.int32), name="cone")


def make_plane(w=1.0, h=1.0, nx=10, ny=10) -> Mesh:
    verts, faces = [], []
    for iy in range(ny+1):
        for ix in range(nx+1):
            verts.append([(ix/nx - 0.5)*w, 0.0, (iy/ny - 0.5)*h])
    for iy in range(ny):
        for ix in range(nx):
            a = iy*(nx+1)+ix; b=a+1; c=a+(nx+1); d=c+1
            faces += [[a,b,d],[a,d,c]]
    return Mesh(np.array(verts, np.float32), np.array(faces, np.int32), name="plane")


def make_capsule(r=0.25, h=0.5, segs=24, rings=8) -> Mesh:
    """Cylinder body capped with hemispheres."""
    verts, faces = [], []
    # Top hemisphere
    for i in range(rings+1):
        phi = np.pi/2 * i/rings
        for j in range(segs):
            th = 2*np.pi*j/segs
            verts.append([r*np.cos(phi)*np.cos(th), h/2+r*np.sin(phi), r*np.cos(phi)*np.sin(th)])
    # Bottom hemisphere
    for i in range(rings+1):
        phi = np.pi/2 * i/rings
        for j in range(segs):
            th = 2*np.pi*j/segs
            verts.append([r*np.cos(phi)*np.cos(th), -(h/2+r*np.sin(phi)), r*np.cos(phi)*np.sin(th)])
    for half in range(2):
        off = half*(rings+1)*segs
        for i in range(rings):
            for j in range(segs):
                a = off+i*segs+j; b=off+i*segs+(j+1)%segs
                c=off+(i+1)*segs+(j+1)%segs; d=off+(i+1)*segs+j
                if half == 0:
                    faces += [[a,b,c],[a,c,d]]
                else:
                    faces += [[a,c,b],[a,d,c]]
    return Mesh(np.array(verts, np.float32), np.array(faces, np.int32), name="capsule")
