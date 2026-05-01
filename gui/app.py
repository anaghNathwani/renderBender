"""
Main application window — 3D Model Generator
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, queue
from pathlib import Path
from typing import Optional

from models.mesh import Mesh
from models.generator import ModelGenerator, GenerationParams
from models.primitives import (make_sphere, make_box, make_cylinder,
                                make_torus, make_cone, make_plane, make_capsule)
from file_io.format_manager import load, save, SUPPORTED_READ, SUPPORTED_WRITE
from gui.viewer3d import Viewer3D

# ── theme ─────────────────────────────────────────────────────────────────────
BG     = "#16213e"; PBG  = "#0f3460"; ACC  = "#e94560"
FG     = "#eaeaea"; EBG  = "#1a1a3e"; BFG  = "#ffffff"
MONO   = ("Consolas", 9); SANS = ("Segoe UI", 9)


def _style():
    s = ttk.Style()
    s.theme_use("clam")
    s.configure(".",            background=BG,  foreground=FG,  font=SANS)
    s.configure("TLabel",       background=BG,  foreground=FG)
    s.configure("P.TFrame",     background=PBG)
    s.configure("TFrame",       background=BG)
    s.configure("TEntry",       fieldbackground=EBG, foreground=FG, insertcolor=FG)
    s.configure("TButton",      background=ACC, foreground=BFG, relief="flat", padding=4)
    s.map("TButton",            background=[("active","#c73652")])
    s.configure("TProgressbar", troughcolor=EBG, background=ACC)
    s.configure("TCombobox",    fieldbackground=EBG, foreground=FG)
    s.configure("TRadiobutton", background=PBG, foreground=FG)
    s.configure("TCheckbutton", background=PBG, foreground=FG)
    s.configure("TNotebook",    background=BG)
    s.configure("TNotebook.Tab",background=PBG, foreground=FG, padding=(8,4))
    s.map("TNotebook.Tab",      background=[("selected",ACC)])
    s.configure("H.TLabel",     background=PBG, foreground=ACC, font=("Segoe UI",10,"bold"))
    s.configure("Stat.TLabel",  background=PBG, foreground=FG,  font=MONO)
    s.configure("TScale",       background=BG,  troughcolor=EBG)


class ModelGeneratorApp:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("3D Model Generator")
        self.root.geometry("1300x820"); self.root.minsize(900,600)
        self.root.configure(bg=BG)
        _style()

        self._mesh: Optional[Mesh] = None
        self._gen   = ModelGenerator()
        self._q: queue.Queue = queue.Queue()
        self._history: list  = []

        self._build_menu()
        self._build_ui()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.after(100, self._poll)

    # ── menu ──────────────────────────────────────────────────────────────────

    def _menu(self, mb, label, entries):
        m = tk.Menu(mb, tearoff=0, bg=PBG, fg=FG, activebackground=ACC, activeforeground=BFG)
        for e in entries:
            if e is None: m.add_separator()
            else:
                lbl,cmd,*acc = e
                kw = {"accelerator":acc[0]} if acc else {}
                m.add_command(label=lbl, command=cmd, **kw)
        mb.add_cascade(label=label, menu=m)
        return m

    def _build_menu(self):
        mb = tk.Menu(self.root, bg=BG, fg=FG, activebackground=ACC, activeforeground=BFG,
                     relief="flat", bd=0)
        self.root.config(menu=mb)
        self._menu(mb, "File", [
            ("Open…",       self._open,       "Ctrl+O"),
            ("Save…",       self._save,       "Ctrl+S"),
            ("Export PNG…", self._export_png),
            None,
            ("Quit",        self._quit,       "Ctrl+Q"),
        ])
        self._menu(mb, "Generate", [
            ("From Prompt", self._do_generate, "Ctrl+G"),
            None,
            ("Sphere",   lambda: self._prim(make_sphere())),
            ("Cube",     lambda: self._prim(make_box())),
            ("Cylinder", lambda: self._prim(make_cylinder())),
            ("Torus",    lambda: self._prim(make_torus())),
            ("Cone",     lambda: self._prim(make_cone())),
            ("Plane",    lambda: self._prim(make_plane())),
            ("Capsule",  lambda: self._prim(make_capsule())),
        ])
        self._menu(mb, "View", [
            ("Solid",     lambda: self.viewer.set_display_mode("solid")),
            ("Wireframe", lambda: self.viewer.set_display_mode("wireframe")),
            ("Points",    lambda: self.viewer.set_display_mode("points")),
            None,
            ("Reset View", self.viewer.reset_view if hasattr(self,"viewer") else lambda:None, "R"),
        ])
        self._menu(mb, "Help", [("About", self._about)])

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = ttk.Frame(pw, style="P.TFrame", width=270); left.pack_propagate(False)
        pw.add(left, weight=0); self._build_left(left)

        center = ttk.Frame(pw); pw.add(center, weight=3)
        self.viewer = Viewer3D(center); self.viewer.pack(fill=tk.BOTH, expand=True)

        right = ttk.Frame(pw, style="P.TFrame", width=230); right.pack_propagate(False)
        pw.add(right, weight=0); self._build_right(right)

        sb = ttk.Frame(self.root); sb.pack(fill=tk.X, side=tk.BOTTOM)
        self._sv = tk.StringVar(value="Ready.")
        self._pv = tk.DoubleVar(value=0.0)
        ttk.Label(sb, textvariable=self._sv, font=MONO).pack(side=tk.LEFT, padx=8)
        ttk.Progressbar(sb, variable=self._pv, maximum=1.0, length=220).pack(
            side=tk.RIGHT, padx=8, pady=2)

    # ── left panel ────────────────────────────────────────────────────────────

    def _build_left(self, p):
        nb = ttk.Notebook(p); nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        g=ttk.Frame(nb,style="P.TFrame"); nb.add(g, text=" Generate ")
        pr=ttk.Frame(nb,style="P.TFrame"); nb.add(pr,text=" Primitives ")
        h=ttk.Frame(nb,style="P.TFrame"); nb.add(h, text=" History ")
        self._gen_tab(g); self._prim_tab(pr); self._hist_tab(h)

    def _lbl(self,p,t): ttk.Label(p,text=t,background=PBG,foreground=FG).pack(anchor=tk.W,padx=8,pady=(6,1))

    def _gen_tab(self, p):
        ttk.Label(p,text="3D Model Generator",style="H.TLabel").pack(anchor=tk.W,padx=8,pady=(10,4))
        ttk.Separator(p,orient=tk.HORIZONTAL).pack(fill=tk.X,padx=8,pady=2)

        self._lbl(p,"Text Prompt:")
        self._prompt = tk.Text(p,height=4,bg=EBG,fg=FG,insertbackground=FG,
                               font=SANS,relief="flat",bd=2,wrap=tk.WORD)
        self._prompt.pack(fill=tk.X,padx=8); self._prompt.insert("1.0","a smooth sphere")

        self._lbl(p,"Resolution:")
        rf=ttk.Frame(p,style="P.TFrame"); rf.pack(fill=tk.X,padx=8)
        self._res=tk.IntVar(value=48)
        for lbl,v in [("Low (32)",32),("Med (48)",48),("High (64)",64)]:
            ttk.Radiobutton(rf,text=lbl,variable=self._res,value=v).pack(side=tk.LEFT,padx=2)

        self._lbl(p,"Neural blend  (0 = analytic):")
        self._blend=tk.DoubleVar(value=0.35)
        ttk.Scale(p,variable=self._blend,from_=0,to=1,orient=tk.HORIZONTAL).pack(fill=tk.X,padx=8)

        self._lbl(p,"Seed:")
        self._seed=tk.IntVar(value=0)
        ttk.Entry(p,textvariable=self._seed,width=8).pack(anchor=tk.W,padx=8)

        self._smooth=tk.BooleanVar(value=True)
        ttk.Checkbutton(p,text="Smooth normals",variable=self._smooth).pack(anchor=tk.W,padx=8,pady=4)

        ttk.Separator(p,orient=tk.HORIZONTAL).pack(fill=tk.X,padx=8,pady=6)
        self._gbtn=ttk.Button(p,text="⚡  Generate",command=self._do_generate)
        self._gbtn.pack(fill=tk.X,padx=8,pady=4)
        ttk.Button(p,text="Clear",command=self._clear).pack(fill=tk.X,padx=8,pady=2)

    def _prim_tab(self, p):
        ttk.Label(p,text="Quick Primitives",style="H.TLabel").pack(anchor=tk.W,padx=8,pady=(10,4))
        ttk.Separator(p,orient=tk.HORIZONTAL).pack(fill=tk.X,padx=8,pady=2)
        prims=[
            ("Sphere",   make_sphere),("Cube",      make_box),
            ("Cylinder", make_cylinder),("Torus",   make_torus),
            ("Cone",     make_cone),  ("Plane",      make_plane),
            ("Capsule",  make_capsule),
        ]
        for lbl,fn in prims:
            ttk.Button(p,text=lbl,command=lambda f=fn:self._prim(f())).pack(fill=tk.X,padx=8,pady=2)

    def _hist_tab(self, p):
        ttk.Label(p,text="Generation History",style="H.TLabel").pack(anchor=tk.W,padx=8,pady=(10,4))
        ttk.Separator(p,orient=tk.HORIZONTAL).pack(fill=tk.X,padx=8,pady=2)
        f=ttk.Frame(p,style="P.TFrame"); f.pack(fill=tk.BOTH,expand=True,padx=4,pady=4)
        sb=ttk.Scrollbar(f); sb.pack(side=tk.RIGHT,fill=tk.Y)
        self._hist_lb=tk.Listbox(f,bg=EBG,fg=FG,selectbackground=ACC,
                                  font=MONO,relief="flat",yscrollcommand=sb.set)
        self._hist_lb.pack(fill=tk.BOTH,expand=True); sb.config(command=self._hist_lb.yview)
        self._hist_lb.bind("<Double-Button-1>",self._restore)
        ttk.Button(p,text="Restore selected",command=self._restore).pack(fill=tk.X,padx=8,pady=4)

    # ── right panel ───────────────────────────────────────────────────────────

    def _build_right(self, p):
        ttk.Label(p,text="Mesh Properties",style="H.TLabel").pack(anchor=tk.W,padx=8,pady=(10,4))
        ttk.Separator(p,orient=tk.HORIZONTAL).pack(fill=tk.X,padx=8,pady=2)
        self._stats: dict={}
        for key in ("Name","Vertices","Faces","Volume","Surface area","X extent","Y extent","Z extent"):
            row=ttk.Frame(p,style="P.TFrame"); row.pack(fill=tk.X,padx=8,pady=1)
            ttk.Label(row,text=key+":",width=13,background=PBG,
                      foreground="#aabbdd",font=MONO).pack(side=tk.LEFT)
            var=tk.StringVar(value="—")
            ttk.Label(row,textvariable=var,style="Stat.TLabel").pack(side=tk.LEFT)
            self._stats[key]=var

        ttk.Separator(p,orient=tk.HORIZONTAL).pack(fill=tk.X,padx=8,pady=8)
        ttk.Label(p,text="Export",style="H.TLabel").pack(anchor=tk.W,padx=8,pady=(4,2))
        self._fmt=tk.StringVar(value=".obj")
        ttk.Combobox(p,textvariable=self._fmt,values=sorted(SUPPORTED_WRITE),
                     state="readonly",width=10).pack(anchor=tk.W,padx=8,pady=2)
        ttk.Button(p,text="Save mesh…",command=self._save).pack(fill=tk.X,padx=8,pady=2)
        ttk.Button(p,text="Export PNG", command=self._export_png).pack(fill=tk.X,padx=8,pady=2)

        ttk.Separator(p,orient=tk.HORIZONTAL).pack(fill=tk.X,padx=8,pady=8)
        ttk.Label(p,text="Display",style="H.TLabel").pack(anchor=tk.W,padx=8,pady=(4,2))
        for lbl,mode in [("Solid","solid"),("Wireframe","wireframe"),("Points","points")]:
            ttk.Button(p,text=lbl,
                       command=lambda m=mode:self.viewer.set_display_mode(m)
                       ).pack(fill=tk.X,padx=8,pady=1)

    # ── generation ────────────────────────────────────────────────────────────

    def _do_generate(self):
        prompt=self._prompt.get("1.0",tk.END).strip()
        if not prompt: messagebox.showwarning("No prompt","Enter a text prompt."); return
        params=GenerationParams(resolution=self._res.get(),
                                neural_blend=self._blend.get(),
                                seed=self._seed.get(),
                                smooth_iters=1 if self._smooth.get() else 0)
        self._gbtn.config(state=tk.DISABLED)
        self._status(0.0,f"Generating: {prompt!r}…")

        def _cb(f,msg): self._q.put(("progress",f,msg))
        def _worker():
            try:
                self._gen.set_progress_callback(_cb)
                mesh=self._gen.generate(prompt,params)
                self._q.put(("done",mesh,prompt))
            except Exception as ex:
                self._q.put(("error",str(ex)))

        threading.Thread(target=_worker,daemon=True).start()

    def _prim(self, mesh: Mesh):
        self._set_mesh(mesh)
        self._history.append((mesh.name, mesh))
        self._hist_lb.insert(tk.END,f"{len(self._history):>3}. {mesh.name}")
        self._status(1.0,f"Primitive: {mesh.vertex_count} verts")

    def _poll(self):
        try:
            while True:
                item=self._q.get_nowait()
                if item[0]=="progress":
                    _,f,msg=item; self._status(f,msg)
                elif item[0]=="done":
                    _,mesh,lbl=item
                    self._set_mesh(mesh)
                    self._gbtn.config(state=tk.NORMAL)
                    self._history.append((lbl,mesh))
                    self._hist_lb.insert(tk.END,f"{len(self._history):>3}. {lbl[:30]}")
                    self._status(1.0,f"Done — {mesh.vertex_count} verts, {mesh.face_count} faces")
                elif item[0]=="error":
                    _,msg=item
                    messagebox.showerror("Generation error",msg)
                    self._gbtn.config(state=tk.NORMAL); self._status(0.0,"Error.")
        except queue.Empty:
            pass
        self.root.after(100,self._poll)

    def _clear(self):
        self._mesh=None; self.viewer.clear(); self._update_stats(None); self._status(0.0,"Cleared.")

    def _restore(self, _=None):
        sel=self._hist_lb.curselection()
        if not sel: return
        idx=sel[0]
        if 0<=idx<len(self._history):
            lbl,mesh=self._history[idx]; self._set_mesh(mesh); self._status(1.0,f"Restored: {lbl}")

    # ── file I/O ──────────────────────────────────────────────────────────────

    def _open(self):
        exts=" ".join(f"*{e}" for e in sorted(SUPPORTED_READ))
        path=filedialog.askopenfilename(title="Open 3D File",
                                         filetypes=[("3D files",exts),("All files","*.*")])
        if not path: return
        try:
            self._status(0.3,f"Loading {Path(path).name}…")
            mesh=load(path); self._set_mesh(mesh)
            self._status(1.0,f"Loaded: {Path(path).name} — {mesh.vertex_count} verts")
        except Exception as ex:
            messagebox.showerror("Load error",str(ex)); self._status(0.0,"Load failed.")

    def _save(self):
        if self._mesh is None: messagebox.showwarning("No mesh","Generate or load a mesh first."); return
        ext=self._fmt.get()
        exts=" ".join(f"*{e}" for e in sorted(SUPPORTED_WRITE))
        path=filedialog.asksaveasfilename(title="Save Mesh",defaultextension=ext,
                                           initialfile=f"{self._mesh.name}{ext}",
                                           filetypes=[("3D files",exts),("All files","*.*")])
        if not path: return
        try:
            save(self._mesh,path); self._status(1.0,f"Saved: {Path(path).name}")
        except Exception as ex:
            messagebox.showerror("Save error",str(ex))

    def _export_png(self):
        path=filedialog.asksaveasfilename(title="Export PNG",defaultextension=".png",
                                           filetypes=[("PNG image","*.png")])
        if not path: return
        with open(path,"wb") as f: f.write(self.viewer.get_screenshot())
        self._status(1.0,f"PNG exported: {Path(path).name}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _set_mesh(self, mesh: Mesh):
        self._mesh=mesh; self.viewer.set_mesh(mesh); self._update_stats(mesh)

    def _update_stats(self, mesh: Optional[Mesh]):
        if mesh is None:
            for v in self._stats.values(): v.set("—"); return
        ex=mesh.extents()
        self._stats["Name"].set(mesh.name[:18])
        self._stats["Vertices"].set(f"{mesh.vertex_count:,}")
        self._stats["Faces"].set(f"{mesh.face_count:,}")
        try:
            self._stats["Volume"].set(f"{mesh.volume():.4f}")
            self._stats["Surface area"].set(f"{mesh.surface_area():.4f}")
        except Exception:
            self._stats["Volume"].set("n/a"); self._stats["Surface area"].set("n/a")
        self._stats["X extent"].set(f"{ex[0]:.4f}")
        self._stats["Y extent"].set(f"{ex[1]:.4f}")
        self._stats["Z extent"].set(f"{ex[2]:.4f}")

    def _status(self, frac: float, msg: str):
        self._sv.set(msg); self._pv.set(frac); self.root.update_idletasks()

    def _bind_keys(self):
        self.root.bind("<Control-o>", lambda _: self._open())
        self.root.bind("<Control-s>", lambda _: self._save())
        self.root.bind("<Control-g>", lambda _: self._do_generate())
        self.root.bind("<Control-q>", lambda _: self._quit())
        self.root.bind("<r>",         lambda _: self.viewer.reset_view())

    def _about(self):
        messagebox.showinfo("About","3D Model Generator\n\n"
            "Hybrid OccupancyNetwork + analytic SDF pipeline.\n"
            f"Read:  {', '.join(sorted(SUPPORTED_READ))}\n"
            f"Write: {', '.join(sorted(SUPPORTED_WRITE))}\n\n"
            "Controls:\n  Left-drag → rotate\n  Right-drag / scroll → zoom\n  R → reset view")

    def _quit(self): self.root.quit(); self.root.destroy()

    def run(self): self.root.mainloop()
