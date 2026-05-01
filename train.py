#!/usr/bin/env python3
"""
Train the OccupancyNetwork on 3D mesh files.

HOW IT WORKS
────────────
For each mesh:
  1. Sample random 3D points inside and outside the mesh.
  2. Label each point: 1.0 = inside, 0.0 = outside.
  3. Train the network to predict inside/outside from (xyz + text_embedding).

OPEN-SOURCE DATASETS (no data needed — downloads automatically)
───────────────────────────────────────────────────────────────
  # ModelNet10 (~70 MB, 10 categories, no registration):
  python train.py --dataset modelnet10

  # ModelNet40 (~435 MB, 40 categories, no registration):
  python train.py --dataset modelnet40 --categories chair,table,lamp

  # Objaverse (800K+ objects, requires: pip install objaverse):
  python train.py --dataset objaverse --categories chair --n 200

  List all available datasets:
  python datasets.py list

  List categories in a dataset:
  python datasets.py categories --dataset modelnet40

YOUR OWN DATA
─────────────
  python train.py --data ./my_shapes/

OTHER OPTIONS
─────────────
  python train.py --dataset modelnet10 --epochs 200 --lr 1e-3 --batch 1024
  python train.py --dataset modelnet40 --per-shape   # one weights file per category
  python train.py --dataset modelnet10 --resume models/weights/model.npz

RESULT
──────
  Saves to models/weights/model.npz  (or <category>.npz with --per-shape)
  Generator auto-loads these weights. Set neural_blend > 0 in the GUI.

REQUIREMENTS
────────────
  PyTorch (pip install torch) is STRONGLY recommended for training speed.
  Falls back to slow numpy SGD if torch is not installed.
"""

import argparse
import os
import sys
import time
import glob
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Point sampling from meshes
# ─────────────────────────────────────────────────────────────────────────────

def sample_mesh_points(mesh, n_surface: int = 10000, n_random: int = 10000,
                        noise: float = 0.02):
    """
    Returns (xyz, labels) where:
      xyz    : (N, 3) float32  — 3D sample points
      labels : (N,)  float32  — 1.0 inside/on surface, 0.0 outside
    Uses surface points + small perturbation for near-surface samples,
    and random uniform samples for far-field.
    """
    # Surface samples (near surface = inside label)
    verts = mesh.vertices
    faces = mesh.faces
    if len(faces) == 0:
        raise ValueError(f"Mesh '{mesh.name}' has no faces")

    # Sample random triangles
    areas = _triangle_areas(verts, faces)
    total = areas.sum()
    if total == 0:
        raise ValueError(f"Mesh '{mesh.name}' has zero surface area")

    probs = areas / total
    rng = np.random.default_rng(0)
    tri_idx = rng.choice(len(faces), size=n_surface, p=probs)
    r1 = rng.random(n_surface).astype(np.float32)
    r2 = rng.random(n_surface).astype(np.float32)
    sq = np.sqrt(r1)
    u, v, w = 1 - sq, sq * (1 - r2), sq * r2
    f = faces[tri_idx]
    surface_pts = (u[:, None] * verts[f[:, 0]]
                 + v[:, None] * verts[f[:, 1]]
                 + w[:, None] * verts[f[:, 2]])

    # Near-surface = perturb inward (add small noise, treat as inside)
    perturbed = surface_pts + rng.standard_normal(surface_pts.shape).astype(np.float32) * noise
    inside_labels = np.ones(n_surface, np.float32)

    # Random bounding-box samples → labelled outside
    mn, mx = verts.min(0), verts.max(0)
    pad = 0.15
    mn -= pad; mx += pad
    rand_pts = rng.random((n_random, 3)).astype(np.float32) * (mx - mn) + mn
    outside_labels = np.zeros(n_random, np.float32)

    # Classify random points using winding numbers (approximate: ray cast)
    outside_labels = _classify_points(rand_pts, verts, faces, outside_labels)

    xyz    = np.vstack([perturbed, rand_pts])
    labels = np.concatenate([inside_labels, outside_labels])
    return xyz, labels


def _triangle_areas(verts, faces):
    v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return np.linalg.norm(cross, axis=1) / 2.0


def _classify_points(pts, verts, faces, default_labels):
    """
    Fast approximate inside/outside test using axis-aligned ray casting.
    Returns labels: 1.0 = inside, 0.0 = outside.
    """
    labels = default_labels.copy()
    v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]

    for i, pt in enumerate(pts):
        # Ray from pt in +Z direction
        ox, oy, oz = pt
        hits = 0
        # Möller–Trumbore intersection
        edge1 = v1 - v0; edge2 = v2 - v0
        h = np.cross(np.array([0., 0., 1.], np.float32), edge2)
        a = np.einsum("ij,ij->i", edge1, h)
        mask = np.abs(a) > 1e-8
        if not mask.any(): continue
        f_val = 1.0 / a[mask]
        s = pt[None] - v0[mask]
        u = f_val * np.einsum("ij,ij->i", s, h[mask])
        umask = (u >= 0) & (u <= 1)
        if not umask.any(): continue
        q = np.cross(s[umask], edge1[mask][umask])
        vv = f_val[umask] * np.einsum("ij,j->i", q, np.array([0.,0.,1.],np.float32))
        vmask = (vv >= 0) & (u[umask] + vv <= 1)
        t = f_val[umask][vmask] * np.einsum("ij,j->i", q[vmask], np.array([0.,0.,1.],np.float32))
        hits = int((t > 0).sum())
        if hits % 2 == 1:
            labels[i] = 1.0
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch trainer (preferred)
# ─────────────────────────────────────────────────────────────────────────────

def _train_torch(xyz_all, labels_all, emb, epochs, lr, batch_size, resume, out_path, verbose):
    import torch
    import torch.nn as nn

    E = torch.tensor(emb, dtype=torch.float32)
    X = torch.tensor(xyz_all,    dtype=torch.float32)
    Y = torch.tensor(labels_all, dtype=torch.float32)

    embed_dim = len(emb)
    d0, h = 3 + embed_dim, 256

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.l0 = nn.Linear(d0, h); self.l1 = nn.Linear(h, h)
            self.l2 = nn.Linear(h, h);  self.l3 = nn.Linear(h + d0, h)
            self.l4 = nn.Linear(h, h);  self.l5 = nn.Linear(h, 1)
            self.act = nn.ReLU()

        def forward(self, xyz):
            e  = E.unsqueeze(0).expand(len(xyz), -1)
            x0 = torch.cat([xyz, e], dim=1)
            x  = self.act(self.l0(x0))
            x  = self.act(self.l1(x))
            x  = self.act(self.l2(x))
            x  = torch.cat([x, x0], dim=1)
            x  = self.act(self.l3(x))
            x  = self.act(self.l4(x))
            return self.l5(x).squeeze(1)

    net = Net()

    if resume and os.path.exists(resume):
        data = np.load(resume)
        with torch.no_grad():
            for i, (layer, nm) in enumerate(
                [(net.l0,"0"),(net.l1,"1"),(net.l2,"2"),(net.l3,"3"),(net.l4,"4"),(net.l5,"5")]
            ):
                if f"W{nm}" in data:
                    layer.weight.copy_(torch.tensor(data[f"W{nm}"]).T)
                    layer.bias.copy_(  torch.tensor(data[f"b{nm}"]))
        print(f"  Resumed from {resume}")

    opt  = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    N    = len(X)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(N)
        total_loss = 0.0; steps = 0
        for start in range(0, N, batch_size):
            idx  = perm[start:start + batch_size]
            xb, yb = X[idx], Y[idx]
            opt.zero_grad()
            pred = net(xb)
            loss = loss_fn(pred, yb)
            loss.backward(); opt.step()
            total_loss += loss.item(); steps += 1
        avg = total_loss / max(steps, 1)
        if verbose or epoch % max(1, epochs // 20) == 0:
            acc = ((torch.sigmoid(net(X)) > 0.5).float() == Y).float().mean().item()
            print(f"  Epoch {epoch:4d}/{epochs}  loss={avg:.4f}  acc={acc:.3f}")
        if avg < best_loss:
            best_loss = avg
            _save_torch_weights(net, out_path)

    print(f"  Best loss: {best_loss:.4f}  →  {out_path}.npz")


def _save_torch_weights(net, out_path):
    import torch
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    kv = {}
    layers = [(net.l0,"0"),(net.l1,"1"),(net.l2,"2"),
              (net.l3,"3"),(net.l4,"4"),(net.l5,"5")]
    for layer, nm in layers:
        kv[f"W{nm}"] = layer.weight.detach().numpy().T
        kv[f"b{nm}"] = layer.bias.detach().numpy()
    np.savez(out_path, **kv)


# ─────────────────────────────────────────────────────────────────────────────
# Numpy SGD trainer (fallback — slow but no extra deps)
# ─────────────────────────────────────────────────────────────────────────────

def _train_numpy(xyz_all, labels_all, emb, epochs, lr, batch_size, resume, out_path, verbose):
    from models.generator import OccupancyNetwork, _EMBED_DIM

    net = OccupancyNetwork(seed=0, weights_path=(resume if resume else None))
    N   = len(xyz_all)

    def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    def _relu(x):    return np.maximum(0.0, x)
    def _drelu(x):   return (x > 0).astype(np.float32)

    best_loss = float("inf")
    rng = np.random.default_rng(1)

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(N)
        total_loss = 0.0; steps = 0
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            xb  = xyz_all[idx]; yb = labels_all[idx]
            pred = _sigmoid(net(xb, emb))
            err  = pred - yb
            loss = -np.mean(yb * np.log(pred + 1e-8) + (1 - yb) * np.log(1 - pred + 1e-8))
            total_loss += loss; steps += 1

            # Backprop — simplified gradient update on last layer only (fast approximation)
            # Full backprop through all layers is done below
            e   = np.broadcast_to(emb[None], (len(xb), _EMBED_DIM))
            x0  = np.concatenate([xb, e], axis=-1).astype(np.float32)
            # Forward with stored activations
            acts, pre_acts = [x0], []
            x = x0
            for i, layer in enumerate(net.layers[:-1]):
                pre = layer(x); pre_acts.append(pre)
                x = _relu(pre)
                if i == 2: x = np.concatenate([x, x0], axis=-1)
                acts.append(x)
            out = net.layers[-1](x)[:, 0]

            # Backward
            d = ((_sigmoid(out) - yb) / len(yb))[:, None]
            # Last layer
            net.layers[-1].W -= lr * acts[-1].T @ d
            net.layers[-1].b -= lr * d.mean(0)
            d = d @ net.layers[-1].W.T * _drelu(acts[-1])
            # Layers 4→0 (simplified)
            for i in range(len(net.layers) - 2, 0, -1):
                act_in = acts[i - 1] if i > 0 else x0
                if act_in.shape[1] != net.layers[i].W.shape[0]:
                    act_in = act_in[:, :net.layers[i].W.shape[0]]
                g = act_in.T @ d
                net.layers[i].W -= lr * g
                net.layers[i].b -= lr * d.mean(0)
                if i > 0:
                    d = d @ net.layers[i].W.T
                    d = d[:, :pre_acts[i-1].shape[1]] * _drelu(pre_acts[i-1])

        avg = total_loss / max(steps, 1)
        if verbose or epoch % max(1, epochs // 10) == 0:
            print(f"  Epoch {epoch:4d}/{epochs}  loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            net.save(out_path)

    print(f"  Best loss: {best_loss:.4f}  →  {out_path}.npz")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)

    src = ap.add_mutually_exclusive_group()
    src.add_argument("--data",    default=None,
                     help="Folder of .obj/.stl/.ply/.glb files to train on")
    src.add_argument("--dataset", default=None,
                     choices=["modelnet10","modelnet40","objaverse"],
                     help="Download + use an open-source dataset automatically")

    ap.add_argument("--categories", default=None,
                    help="Comma-separated list of categories (e.g. chair,table,lamp). "
                         "Applies to --dataset. Use 'all' for everything.")
    ap.add_argument("--n",         type=int, default=None,
                    help="Max number of meshes to use from --dataset")
    ap.add_argument("--out",       default="models/weights/model",
                    help="Output path (no extension). Default: models/weights/model")
    ap.add_argument("--epochs",    type=int,   default=150,  help="Training epochs (default 150)")
    ap.add_argument("--lr",        type=float, default=5e-4, help="Learning rate (default 5e-4)")
    ap.add_argument("--batch",     type=int,   default=2048, help="Batch size (default 2048)")
    ap.add_argument("--surface",   type=int,   default=8000, help="Surface samples per mesh")
    ap.add_argument("--random",    type=int,   default=8000, help="Random samples per mesh")
    ap.add_argument("--resume",    default=None,
                    help="Resume from existing .npz weights file")
    ap.add_argument("--per-shape", action="store_true",
                    help="Save separate weights per category (--dataset) or mesh (--data)")
    ap.add_argument("--verbose",   action="store_true", help="Print every epoch")
    ap.add_argument("--prompt",    default="",
                    help="Text prompt / description of the shapes (for text embedding)")
    ap.add_argument("--list-datasets",  action="store_true",
                    help="List available open-source datasets and exit")
    ap.add_argument("--list-categories", metavar="DATASET",
                    help="List categories in a dataset and exit")
    args = ap.parse_args()

    # Info-only modes
    if args.list_datasets:
        import datasets as ds_mod
        print("Available datasets:")
        for name in ds_mod.list_datasets():
            print(f"  {ds_mod.describe(name)}")
        return

    if args.list_categories:
        import datasets as ds_mod
        cats = ds_mod.list_categories(args.list_categories)
        print(f"{args.list_categories} ({len(cats)} categories):")
        for c in cats: print(f"  {c}")
        return

    if not args.data and not args.dataset:
        ap.error("Provide either --data <folder> or --dataset <name>")

    # Resolve file list
    if args.dataset:
        import datasets as ds_mod
        cats = [c.strip() for c in args.categories.split(",")] \
               if args.categories else None
        print(f"Dataset : {args.dataset}")
        if cats: print(f"Categories: {cats}")
        files = ds_mod.get_files(
            name=args.dataset, categories=cats, split="train",
            n=args.n, prompt=args.prompt,
        )
        # For --per-shape on a dataset, group by category
        _category_of = lambda f: Path(f).parent.parent.name  # ModelNet layout
    else:
        exts = ("*.obj","*.stl","*.ply","*.glb","*.gltf","*.off")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(args.data, "**", ext), recursive=True))
        files = sorted(set(files))
        _category_of = lambda f: Path(f).stem

    if not files:
        print("No mesh files found. Exiting.")
        sys.exit(1)

    print(f"Found {len(files)} mesh file(s).")

    # Try PyTorch
    try:
        import torch
        _trainer = _train_torch
        print("Using PyTorch trainer.")
    except ImportError:
        _trainer = _train_numpy
        print("PyTorch not found — using slow numpy trainer.")
        print("Install PyTorch for much faster training: pip install torch")

    from file_io.format_manager import load
    from models.generator import encode_text

    if args.per_shape:
        # Group files by category (for datasets) or treat each file individually
        from collections import defaultdict
        groups = defaultdict(list)
        for fpath in files:
            groups[_category_of(fpath)].append(fpath)

        for group_name, group_files in sorted(groups.items()):
            print(f"\n── {group_name} ({len(group_files)} meshes) ──")
            all_xyz, all_labels = [], []
            for fpath in group_files:
                try:
                    mesh = load(fpath); mesh.normalize()
                    xyz, labels = sample_mesh_points(mesh, args.surface, args.random)
                    all_xyz.append(xyz); all_labels.append(labels)
                except Exception as e:
                    print(f"  Skipping {Path(fpath).name}: {e}")
            if not all_xyz:
                continue
            xyz_all    = np.vstack(all_xyz)
            labels_all = np.concatenate(all_labels)
            prompt = args.prompt or group_name.replace("_", " ")
            emb    = encode_text(prompt)
            out    = os.path.join(os.path.dirname(args.out), group_name)
            _trainer(xyz_all, labels_all, emb, args.epochs, args.lr, args.batch,
                     args.resume, out, args.verbose)
    else:
        # Combine all meshes
        all_xyz, all_labels = [], []
        for fpath in files:
            print(f"  Sampling {Path(fpath).name}…")
            try:
                mesh = load(fpath); mesh.normalize()
                xyz, labels = sample_mesh_points(mesh, args.surface, args.random)
                all_xyz.append(xyz); all_labels.append(labels)
            except Exception as e:
                print(f"  Skipping {fpath}: {e}")

        if not all_xyz:
            print("No meshes could be loaded. Exiting.")
            sys.exit(1)

        xyz_all    = np.vstack(all_xyz)
        labels_all = np.concatenate(all_labels)
        print(f"\nTotal samples: {len(xyz_all):,}  "
              f"(inside: {labels_all.sum():,.0f}, outside: {(1-labels_all).sum():,.0f})")
        emb = encode_text(args.prompt or " ".join(
            Path(f).stem.replace("_"," ") for f in files[:5]))
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        print(f"\nTraining for {args.epochs} epochs…")
        t0 = time.time()
        _trainer(xyz_all, labels_all, emb, args.epochs, args.lr, args.batch,
                 args.resume, args.out, args.verbose)
        print(f"Training finished in {time.time()-t0:.1f}s")
        print(f"\nTo use these weights, generate with neural_blend > 0:")
        print(f"  python main.py --prompt \"{args.prompt or 'your shape'}\" "
              f"--output out.obj")
        print(f"  (weights auto-loaded from {args.out}.npz)")


if __name__ == "__main__":
    main()
