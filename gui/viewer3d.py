"""
Interactive 3D viewport — matplotlib 3D axes embedded in tkinter.
Left-drag: rotate  |  Right-drag / scroll: zoom  |  R: reset
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import Optional
import io

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d  # noqa: F401

from models.mesh import Mesh

BG       = "#16213e"
GRID_CLR = "#223355"
TICK_CLR = "#8899bb"
MESH_CLR = (0.28, 0.58, 0.95, 0.88)
WIRE_CLR = (0.15, 0.35, 0.80, 0.55)
LIGHT    = np.array([1.0, 1.6, 2.0]); LIGHT /= np.linalg.norm(LIGHT)


class Viewer3D(ttk.Frame):

    SOLID     = "solid"
    WIREFRAME = "wireframe"
    POINTS    = "points"

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._mesh: Optional[Mesh] = None
        self._mode   = self.SOLID
        self._azim   = 30.0
        self._elev   = 20.0
        self._zoom   = 1.0
        self._bg     = BG
        self._drag   = None
        self._btn    = None
        self._build()

    # ── build ────────────────────────────────────────────────

    def _build(self):
        self.fig = plt.figure(figsize=(6, 5), facecolor=self._bg)
        self.ax  = self.fig.add_subplot(111, projection="3d", facecolor=self._bg)
        self._style()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._build_toolbar()
        self._bind()
        self._placeholder()

    def _style(self):
        ax = self.ax
        ax.set_facecolor(self._bg)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False; pane.set_edgecolor(GRID_CLR)
        ax.grid(True, color=GRID_CLR, linestyle="--", linewidth=0.3)
        ax.tick_params(colors=TICK_CLR, labelsize=7)
        for lbl, fn in [("X", ax.set_xlabel),("Y", ax.set_ylabel),("Z", ax.set_zlabel)]:
            fn(lbl, color=TICK_CLR, fontsize=8)

    def _build_toolbar(self):
        bar = ttk.Frame(self)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        self._mode_var = tk.StringVar(value=self.SOLID)
        for lbl, val in [("Solid",self.SOLID),("Wire",self.WIREFRAME),("Points",self.POINTS)]:
            ttk.Radiobutton(bar, text=lbl, variable=self._mode_var, value=val,
                            command=self._on_mode).pack(side=tk.LEFT, padx=4)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)
        ttk.Button(bar, text="Reset",  command=self.reset_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Fit",    command=self.fit_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="BG",     command=self._toggle_bg).pack(side=tk.RIGHT, padx=4)

    def _bind(self):
        w = self.canvas.get_tk_widget()
        for btn in (1,2,3):
            w.bind(f"<ButtonPress-{btn}>",   self._press)
            w.bind(f"<B{btn}-Motion>",        self._drag_move)
            w.bind(f"<ButtonRelease-{btn}>",  self._release)
        w.bind("<MouseWheel>", self._scroll)
        w.bind("<Button-4>",   self._scroll)
        w.bind("<Button-5>",   self._scroll)

    # ── public ───────────────────────────────────────────────

    def set_mesh(self, mesh: Optional[Mesh]):
        self._mesh = mesh; self.redraw()

    def clear(self):
        self._mesh = None; self._placeholder()

    def set_display_mode(self, mode: str):
        self._mode = mode; self._mode_var.set(mode); self.redraw()

    def reset_view(self):
        self._azim=30.0; self._elev=20.0; self._zoom=1.0; self.redraw()

    def fit_view(self):
        self._zoom=1.0; self.redraw()

    def get_screenshot(self) -> bytes:
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        return buf.getvalue()

    def redraw(self):
        self.ax.cla(); self._style()
        if self._mesh is None or self._mesh.vertex_count == 0:
            self._placeholder(); return
        {self.SOLID:    self._draw_solid,
         self.WIREFRAME:self._draw_wire,
         self.POINTS:   self._draw_points}[self._mode]()
        self._limits()
        self.ax.view_init(elev=self._elev, azim=self._azim)
        self.canvas.draw_idle()

    # ── drawing ──────────────────────────────────────────────

    def _placeholder(self):
        self.ax.cla(); self._style()
        self.ax.text(0,0,0,"No mesh\nGenerate or open a file",
                     ha="center",va="center",color="#445577",fontsize=10,fontstyle="italic")
        self.ax.set_xlim(-1,1); self.ax.set_ylim(-1,1); self.ax.set_zlim(-1,1)
        self.ax.view_init(elev=self._elev,azim=self._azim)
        self.canvas.draw_idle()

    def _face_colors(self):
        m = self._mesh
        v0=m.vertices[m.faces[:,0]]; v1=m.vertices[m.faces[:,1]]; v2=m.vertices[m.faces[:,2]]
        fn=np.cross(v1-v0,v2-v0); l=np.linalg.norm(fn,axis=1,keepdims=True)
        fn=fn/np.where(l==0,1,l)
        intensity=np.clip(fn @ LIGHT, 0.08, 1.0)
        base=np.array(MESH_CLR[:3]); amb=0.22
        colors=np.outer(intensity,base)*(1-amb)+base*amb
        return np.hstack([np.clip(colors,0,1), np.full((len(colors),1),MESH_CLR[3])])

    def _draw_solid(self):
        m=self._mesh; tris=m.vertices[m.faces]
        poly=Poly3DCollection(tris,facecolors=self._face_colors(),edgecolors="none",linewidths=0)
        self.ax.add_collection3d(poly)

    def _draw_wire(self):
        m=self._mesh; tris=m.vertices[m.faces]
        poly=Poly3DCollection(tris,facecolors=(0,0,0,0),edgecolors=WIRE_CLR,linewidths=0.4)
        self.ax.add_collection3d(poly)

    def _draw_points(self):
        v=self._mesh.vertices
        self.ax.scatter(v[:,0],v[:,1],v[:,2],s=1.5,c=[MESH_CLR[:3]],alpha=0.7,depthshade=True)

    def _limits(self):
        m=self._mesh; mn,mx=m.bounds(); c=(mn+mx)/2
        ext=max((mx-mn).max()/2*self._zoom, 1e-6)
        self.ax.set_xlim(c[0]-ext,c[0]+ext)
        self.ax.set_ylim(c[1]-ext,c[1]+ext)
        self.ax.set_zlim(c[2]-ext,c[2]+ext)

    # ── interaction ──────────────────────────────────────────

    def _press(self, e):   self._drag=(e.x,e.y); self._btn=e.num
    def _release(self, e): self._drag=None; self._btn=None

    def _drag_move(self, e):
        if self._drag is None: return
        dx=e.x-self._drag[0]; dy=e.y-self._drag[1]
        self._drag=(e.x,e.y)
        if self._btn==1:
            self._azim-=dx*0.5; self._elev=np.clip(self._elev+dy*0.5,-89,89)
        elif self._btn==3:
            self._zoom=np.clip(self._zoom*(1+dy*0.006),0.05,20)
        self.redraw()

    def _scroll(self, e):
        if e.num==4 or getattr(e,"delta",0)>0: self._zoom*=0.88
        else: self._zoom*=1.12
        self._zoom=np.clip(self._zoom,0.05,20); self.redraw()

    def _on_mode(self): self._mode=self._mode_var.get(); self.redraw()

    def _toggle_bg(self):
        self._bg = "#f5f5f5" if self._bg==BG else BG
        self.fig.set_facecolor(self._bg); self.redraw()
