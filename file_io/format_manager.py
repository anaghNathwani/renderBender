"""
3D file format I/O
Supported read:  .obj .stl .ply .blend .glb .gltf .fbx .off
Supported write: .obj .stl .ply .glb .gltf .off
"""

import os, struct, json, base64, re
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from models.mesh import Mesh

SUPPORTED_READ  = {".obj",".stl",".ply",".blend",".glb",".gltf",".fbx",".off"}
SUPPORTED_WRITE = {".obj",".stl",".ply",".glb",".gltf",".off"}


def load(path: str) -> Mesh:
    ext = Path(path).suffix.lower()
    fn  = {".obj":_load_obj,".stl":_load_stl,".ply":_load_ply,
           ".blend":_load_blend,".glb":_load_glb,".gltf":_load_gltf,
           ".fbx":_load_fbx,".off":_load_off}.get(ext)
    if fn is None:
        raise ValueError(f"Unsupported format '{ext}'. Readable: {sorted(SUPPORTED_READ)}")
    return fn(path)


def save(mesh: Mesh, path: str) -> None:
    ext = Path(path).suffix.lower()
    fn  = {".obj":_save_obj,".stl":_save_stl,".ply":_save_ply,
           ".glb":_save_glb,".gltf":_save_gltf,".off":_save_off}.get(ext)
    if fn is None:
        raise ValueError(f"Cannot write format '{ext}'. Writable: {sorted(SUPPORTED_WRITE)}")
    fn(mesh, path)


# ──────────────────── OBJ ────────────────────────────────────────────────────

def _load_obj(path: str) -> Mesh:
    verts,normals,uvs,faces = [],[],[],[]
    with open(path,"r",errors="replace") as f:
        for line in f:
            p = line.strip().split()
            if not p: continue
            if p[0]=="v":   verts.append([float(x) for x in p[1:4]])
            elif p[0]=="vn":normals.append([float(x) for x in p[1:4]])
            elif p[0]=="vt":uvs.append([float(x) for x in p[1:3]])
            elif p[0]=="f":
                fv = [int(tok.split("/")[0])-1 for tok in p[1:]]
                for i in range(1,len(fv)-1):
                    faces.append([fv[0],fv[i],fv[i+1]])
    if not verts: raise ValueError("No vertices in OBJ")
    return Mesh(np.array(verts,np.float32),
                np.array(faces,np.int32) if faces else np.zeros((0,3),np.int32),
                normals=np.array(normals,np.float32) if normals else None,
                uvs=np.array(uvs,np.float32) if uvs else None,
                name=Path(path).stem)


def _save_obj(mesh: Mesh, path: str) -> None:
    with open(path,"w") as f:
        f.write(f"# {mesh.vertex_count} verts {mesh.face_count} faces\n")
        for v in mesh.vertices: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if mesh.normals is not None and len(mesh.normals)==mesh.vertex_count:
            for n in mesh.normals: f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            for fc in mesh.faces:
                a,b,c = fc[0]+1,fc[1]+1,fc[2]+1
                f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
        else:
            for fc in mesh.faces: f.write(f"f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n")


# ──────────────────── STL ────────────────────────────────────────────────────

def _load_stl(path: str) -> Mesh:
    sz = os.path.getsize(path)
    with open(path,"rb") as f:
        hdr = f.read(80)
        nt  = struct.unpack("<I",f.read(4))[0]
    if abs(sz - (84 + nt*50)) < 4:
        return _stl_binary(path)
    return _stl_ascii(path)


def _stl_binary(path: str) -> Mesh:
    verts,faces = [],[]
    with open(path,"rb") as f:
        f.read(84)
        nt = struct.unpack_from("<I", open(path,"rb").read(84)[80:84])[0]
        f.seek(84)
        for _ in range(nt):
            d = struct.unpack("<12fH",f.read(50))
            b = len(verts)
            verts += [list(d[3:6]),list(d[6:9]),list(d[9:12])]
            faces.append([b,b+1,b+2])
    return Mesh(np.array(verts,np.float32),np.array(faces,np.int32),name=Path(path).stem)


def _stl_ascii(path: str) -> Mesh:
    verts,faces,tri = [],[],[]
    with open(path,"r",errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("vertex"):
                tri.append([float(x) for x in s.split()[1:4]])
            elif s.startswith("endloop") and len(tri)==3:
                b=len(verts); verts+=tri; faces.append([b,b+1,b+2]); tri=[]
    return Mesh(np.array(verts,np.float32),np.array(faces,np.int32),name=Path(path).stem)


def _save_stl(mesh: Mesh, path: str) -> None:
    with open(path,"wb") as f:
        f.write(b"3D Model Generator export"+(b" "*55))
        f.write(struct.pack("<I",mesh.face_count))
        for fc in mesh.faces:
            v0,v1,v2 = mesh.vertices[fc[0]],mesh.vertices[fc[1]],mesh.vertices[fc[2]]
            n = np.cross(v1-v0,v2-v0); l=np.linalg.norm(n)
            n = n/l if l>0 else n
            f.write(struct.pack("<3f",*n))
            f.write(struct.pack("<3f",*v0))
            f.write(struct.pack("<3f",*v1))
            f.write(struct.pack("<3f",*v2))
            f.write(b"\x00\x00")


# ──────────────────── PLY ────────────────────────────────────────────────────

def _load_ply(path: str) -> Mesh:
    with open(path,"rb") as f:
        hdrs=[]
        while True:
            l = f.readline().decode("ascii","replace").strip()
            hdrs.append(l)
            if l=="end_header": break
        bin_le = any("binary_little_endian" in h for h in hdrs)
        bin_be = any("binary_big_endian"    in h for h in hdrs)
        nv=nf=0; props=[]; in_v=False
        for h in hdrs:
            if h.startswith("element vertex"): nv=int(h.split()[-1]); in_v=True
            elif h.startswith("element face"):  nf=int(h.split()[-1]); in_v=False
            elif h.startswith("property") and in_v: props.append(h.split()[-1])
        if bin_le or bin_be:
            end = "<" if bin_le else ">"
            verts,faces = _ply_bin(f,nv,nf,props,end,hdrs)
        else:
            verts,faces = _ply_asc(f,nv,nf)
    v = np.array(verts,np.float32)
    return Mesh(v[:,:3],np.array(faces,np.int32),
                normals=v[:,3:6] if v.shape[1]>=6 else None,
                name=Path(path).stem)


def _ply_asc(f,nv,nf):
    verts=[]; [verts.append(list(map(float,f.readline().decode("ascii","replace").split()))) for _ in range(nv)]
    faces=[]
    for _ in range(nf):
        r=list(map(int,f.readline().decode("ascii","replace").split()))
        for i in range(1,r[0]-1): faces.append([r[1],r[i+1],r[i+2]])
    return verts,faces


def _ply_bin(f,nv,nf,props,end,hdrs):
    tm={"float":"f","double":"d","int":"i","uint":"I","short":"h","ushort":"H","char":"b","uchar":"B"}
    def ft(name):
        for h in hdrs:
            if "property" in h and h.strip().endswith(name):
                return tm.get(h.split()[1],"f")
        return "f"
    fmt=end+"".join(ft(p) for p in props); sz=struct.calcsize(fmt)
    verts=[list(struct.unpack(fmt,f.read(sz))) for _ in range(nv)]
    faces=[]
    for _ in range(nf):
        ni=struct.unpack(end+"B",f.read(1))[0]
        idx=list(struct.unpack(end+"I"*ni,f.read(4*ni)))
        for i in range(1,ni-1): faces.append([idx[0],idx[i],idx[i+1]])
    return verts,faces


def _save_ply(mesh: Mesh, path: str) -> None:
    hn = mesh.normals is not None and len(mesh.normals)==mesh.vertex_count
    with open(path,"wb") as f:
        hdr=(f"ply\nformat binary_little_endian 1.0\n"
             f"element vertex {mesh.vertex_count}\n"
             f"property float x\nproperty float y\nproperty float z\n")
        if hn: hdr+="property float nx\nproperty float ny\nproperty float nz\n"
        hdr+=(f"element face {mesh.face_count}\n"
              f"property list uchar int vertex_indices\nend_header\n")
        f.write(hdr.encode())
        for i,v in enumerate(mesh.vertices):
            f.write(struct.pack("<3f",*v))
            if hn: f.write(struct.pack("<3f",*mesh.normals[i]))
        for fc in mesh.faces: f.write(struct.pack("<B3i",3,*fc))


# ──────────────────── OFF ────────────────────────────────────────────────────

def _load_off(path: str) -> Mesh:
    with open(path,"r",errors="replace") as f:
        lines=[l.strip() for l in f if l.strip() and not l.startswith("#")]
    i=0
    if lines[i].upper().startswith("OFF"): i+=1
    nv,nf,_=(int(x) for x in lines[i].split()); i+=1
    verts=[list(map(float,lines[i+k].split()))[:3] for k in range(nv)]; i+=nv
    faces=[]
    for k in range(nf):
        r=list(map(int,lines[i+k].split()))
        for j in range(1,r[0]-1): faces.append([r[1],r[j+1],r[j+2]])
    return Mesh(np.array(verts,np.float32),np.array(faces,np.int32),name=Path(path).stem)


def _save_off(mesh: Mesh, path: str) -> None:
    with open(path,"w") as f:
        f.write(f"OFF\n{mesh.vertex_count} {mesh.face_count} 0\n")
        for v in mesh.vertices: f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for fc in mesh.faces:   f.write(f"3 {fc[0]} {fc[1]} {fc[2]}\n")


# ──────────────────── GLTF / GLB ─────────────────────────────────────────────

def _load_gltf(path: str) -> Mesh:
    with open(path,"r") as f: g=json.load(f)
    bufs=[]
    for b in g.get("buffers",[]):
        uri=b.get("uri","")
        if uri.startswith("data:"):
            bufs.append(base64.b64decode(uri.split(",",1)[1]))
        else:
            with open(os.path.join(str(Path(path).parent),uri),"rb") as bf:
                bufs.append(bf.read())
    return _parse_gltf(g,bufs,Path(path).stem)


def _load_glb(path: str) -> Mesh:
    with open(path,"rb") as f:
        magic,_,_ = struct.unpack("<III",f.read(12))
        if magic != 0x46546C67: raise ValueError("Not a GLB file")
        cl,ct = struct.unpack("<II",f.read(8))
        g = json.loads(f.read(cl).decode("utf-8").rstrip("\x00"))
        rest=f.read(); bin_data=b""
        if len(rest)>=8:
            cl2,ct2=struct.unpack("<II",rest[:8])
            if ct2==0x004E4942: bin_data=rest[8:8+cl2]
    return _parse_gltf(g,[bin_data],Path(path).stem)


def _parse_gltf(g,bufs,name) -> Mesh:
    bvs=g.get("bufferViews",[]); accs=g.get("accessors",[])
    cm={"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4,"MAT4":16}
    tm={5120:"b",5121:"B",5122:"h",5123:"H",5125:"I",5126:"f"}
    def acc(idx):
        a=accs[idx]; bv=bvs[a["bufferView"]]; buf=bufs[bv.get("buffer",0)]
        off=bv.get("byteOffset",0)+a.get("byteOffset",0)
        cnt=a["count"]; nc=cm[a["type"]]; t=tm[a["componentType"]]
        raw=struct.unpack_from(f"<{cnt*nc}{t}",buf,off)
        return np.array(raw,np.float32).reshape(cnt,nc)
    for md in g.get("meshes",[])[:1]:
        for pr in md.get("primitives",[])[:1]:
            at=pr.get("attributes",{})
            if "POSITION" not in at: continue
            v=acc(at["POSITION"])[:,:3]
            n=acc(at["NORMAL"]) if "NORMAL" in at else None
            u=acc(at["TEXCOORD_0"]) if "TEXCOORD_0" in at else None
            if "indices" in pr:
                idx=acc(pr["indices"]).ravel().astype(np.int32)
                f=idx.reshape(-1,3)
            else:
                f=np.arange(len(v),dtype=np.int32).reshape(-1,3)
            return Mesh(v,f,normals=n,uvs=u,name=name)
    raise ValueError("No mesh in GLTF")


def _gltf_dict(mesh: Mesh) -> tuple:
    vb = mesh.vertices.astype(np.float32).tobytes()
    ia = mesh.faces.astype(np.uint32).ravel(); ib=ia.tobytes()
    vmin=mesh.vertices.min(0).tolist(); vmax=mesh.vertices.max(0).tolist()
    d={
        "asset":{"version":"2.0","generator":"3D Model Generator"},
        "scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
        "meshes":[{"name":mesh.name,"primitives":[{"attributes":{"POSITION":1},"indices":0}]}],
        "accessors":[
            {"bufferView":0,"byteOffset":0,"componentType":5125,"count":len(ia),"type":"SCALAR"},
            {"bufferView":1,"byteOffset":0,"componentType":5126,"count":mesh.vertex_count,"type":"VEC3","min":vmin,"max":vmax},
        ],
        "bufferViews":[
            {"buffer":0,"byteOffset":0,"byteLength":len(ib),"target":34963},
            {"buffer":0,"byteOffset":len(ib),"byteLength":len(vb),"target":34962},
        ],
    }
    return d, ib+vb


def _save_gltf(mesh: Mesh, path: str) -> None:
    d,bindata = _gltf_dict(mesh)
    b64="data:application/octet-stream;base64,"+base64.b64encode(bindata).decode()
    d["buffers"]=[{"uri":b64,"byteLength":len(bindata)}]
    with open(path,"w") as f: json.dump(d,f,indent=2)


def _pad4(b: bytes) -> bytes: return b+b"\x00"*((4-len(b)%4)%4)


def _save_glb(mesh: Mesh, path: str) -> None:
    d,bindata = _gltf_dict(mesh)
    d["buffers"]=[{"byteLength":len(bindata)}]
    jb=_pad4(json.dumps(d,separators=(",",":")).encode()); bc=_pad4(bindata)
    total=12+8+len(jb)+8+len(bc)
    with open(path,"wb") as f:
        f.write(struct.pack("<III",0x46546C67,2,total))
        f.write(struct.pack("<II",len(jb),0x4E4F534A)); f.write(jb)
        f.write(struct.pack("<II",len(bc),0x004E4942)); f.write(bc)


# ──────────────────── BLEND ──────────────────────────────────────────────────

def _load_blend(path: str) -> Mesh:
    with open(path,"rb") as f:
        if f.read(7) != b"BLENDER": raise ValueError("Not a .blend file")
        ptr=f.read(1); end=f.read(1); f.read(3)
        ps=8 if ptr==b"-" else 4; e="<" if end==b"v" else ">"
        blocks=[]
        while True:
            code=f.read(4)
            if len(code)<4: break
            code=code.rstrip(b"\x00").decode("ascii","replace")
            sz=struct.unpack(e+"I",f.read(4))[0]
            f.read(ps)                      # old pointer
            f.read(4)                       # SDNA index
            f.read(4)                       # count
            data=f.read(sz)
            blocks.append((code,data))
            if code=="ENDB": break
    all_data=b"".join(d for c,d in blocks if c in ("DATA",""))
    verts=_scan_floats(all_data, e)
    if len(verts)<4:
        from models.primitives import make_sphere as _s
        m=_s(); m.name=Path(path).stem+"_approx"; return m
    try:
        from scipy.spatial import ConvexHull
        v=np.array(verts[:4000],np.float32); h=ConvexHull(v)
        return Mesh(v, h.simplices.astype(np.int32), name=Path(path).stem)
    except Exception:
        v=np.array(verts[:300],np.float32)
        f=[[0,i,i+1] for i in range(1,len(v)-1)]
        return Mesh(v,np.array(f,np.int32),name=Path(path).stem)


def _scan_floats(data: bytes, end: str) -> list:
    out=[]; i=0
    while i+12<=len(data):
        try:
            x,y,z=(struct.unpack_from(end+"f",data,i+j*4)[0] for j in range(3))
            if all(-1e4<v<1e4 and v==v for v in (x,y,z)):
                out.append([x,y,z]); i+=12; continue
        except Exception: pass
        i+=4
    return out


# ──────────────────── FBX ────────────────────────────────────────────────────

def _load_fbx(path: str) -> Mesh:
    with open(path,"rb") as f: hdr=f.read(23)
    if hdr==b"Kaydara FBX Binary  \x00\x1a\x00":
        return _fbx_bin(path)
    return _fbx_ascii(path)


def _fbx_ascii(path: str) -> Mesh:
    with open(path,"r",errors="replace") as f: txt=f.read()
    vm=re.search(r"Vertices:\s*\*\d+\s*\{[^}]*a:\s*([\d\s.,eE+\-]+)\}",txt,re.S)
    im=re.search(r"PolygonVertexIndex:\s*\*\d+\s*\{[^}]*a:\s*([\d\s,\-]+)\}",txt,re.S)
    if not vm: raise ValueError("No vertices in FBX")
    nums=[float(x) for x in re.findall(r"[-\d.eE+]+",vm.group(1))]
    verts=[[nums[i],nums[i+1],nums[i+2]] for i in range(0,len(nums)-2,3)]
    faces=[]
    if im:
        raw=[int(x) for x in re.findall(r"-?\d+",im.group(1))]
        poly=[]
        for idx in raw:
            if idx<0: poly.append(~idx); [faces.append([poly[0],poly[j],poly[j+1]]) for j in range(1,len(poly)-1)]; poly=[]
            else: poly.append(idx)
    return Mesh(np.array(verts,np.float32),
                np.array(faces,np.int32) if faces else np.zeros((0,3),np.int32),
                name=Path(path).stem)


def _fbx_bin(path: str) -> Mesh:
    with open(path,"rb") as f: data=f.read()
    verts=_scan_floats(data,"<")
    if len(verts)<4: raise ValueError("No mesh data in binary FBX")
    return _load_blend.__wrapped__ if False else _make_convex(np.array(verts[:2000],np.float32),Path(path).stem)


def _make_convex(v: np.ndarray, name: str) -> Mesh:
    try:
        from scipy.spatial import ConvexHull
        h=ConvexHull(v); return Mesh(v,h.simplices.astype(np.int32),name=name)
    except Exception:
        f=[[0,i,i+1] for i in range(1,len(v)-1)]
        return Mesh(v,np.array(f,np.int32),name=name)
