#!/usr/bin/env python3
"""
3D Model Generator
==================
GUI mode (default):
    python main.py

CLI mode:
    python main.py --prompt "a twisted torus" --output model.obj --res 48
    python main.py --prompt "mountain terrain" --output terrain.stl --res 64

Supported output formats: .obj  .stl  .ply  .glb  .gltf  .off
"""

import argparse, sys


def _gui():
    from gui.app import ModelGeneratorApp
    ModelGeneratorApp().run()


def _cli(args):
    from models.generator import ModelGenerator, GenerationParams
    from file_io.format_manager import save

    print(f"Prompt : {args.prompt!r}")
    print(f"Res    : {args.res}  blend={args.blend}  seed={args.seed}")

    gen    = ModelGenerator()
    params = GenerationParams(resolution=args.res, neural_blend=args.blend, seed=args.seed)

    def _cb(f, msg):
        bar = "█" * int(f * 28) + "░" * (28 - int(f * 28))
        print(f"\r  [{bar}] {f*100:5.1f}%  {msg:<40}", end="", flush=True)

    gen.set_progress_callback(_cb)
    mesh = gen.generate(args.prompt, params)
    print()
    print(f"Mesh   : {mesh.vertex_count} vertices, {mesh.face_count} faces")

    out = args.output or f"{args.prompt[:20].replace(' ','_')}.obj"
    save(mesh, out)
    print(f"Saved  : {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt",  type=str,   default=None,  help="Text description (omit for GUI)")
    p.add_argument("--output",  type=str,   default=None,  help="Output file path")
    p.add_argument("--res",     type=int,   default=48,    help="Voxel resolution (default 48)")
    p.add_argument("--blend",   type=float, default=0.35,  help="Neural blend 0..1 (default 0.35)")
    p.add_argument("--seed",    type=int,   default=0,     help="Random seed (default 0)")
    args = p.parse_args()
    _cli(args) if args.prompt else _gui()


if __name__ == "__main__":
    main()
