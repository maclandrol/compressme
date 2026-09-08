"""Rebuild the technical report's vector diagrams and measured timing figure."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(exist_ok=True)
SOURCE = ROOT / "benchmarks" / "moljepa_runtime_macos.json"
data = json.loads(SOURCE.read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "svg.fonttype": "none", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})
INK, TEAL, BLUE, GOLD, GREY = "#192F3C", "#087F8C", "#5979AF", "#B17930", "#DCE3E8"


def save(fig, name):
    for ext in ("svg", "png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(12, 5.5))
ax.set(xlim=(-.02, 1.02), ylim=(0, 1)); ax.axis("off")
ax.text(0, .995, "Mol-JEPA: what the compiler changes", fontsize=19, color=INK, weight="bold", va="top")
ax.text(0, .875, "Selected rewrites under a SMILES-only input contract", fontsize=11, color=INK)
cards = [(.005, "1  SPECIALISE THE INPUT", "15.46M fewer parameters", GREY),
         (.343, "2  COMPOSE AFFINE MAPS", "5.50M fewer parameters", BLUE),
         (.681, "3  STORE THE INTERACTION", "4.68M fewer parameters", GOLD)]
for x, title, saving, colour in cards:
    ax.add_patch(FancyBboxPatch((x, .23), .309, .61, boxstyle="round,pad=0.012,rounding_size=0.016",
                               linewidth=.9, edgecolor="#D7E0E5", facecolor="#F7F9FA"))
    ax.text(x+.012, .786, title, fontsize=10.2, color=INK, weight="bold")
    ax.text(x+.012, .276, saving, fontsize=13.2, color=TEAL, weight="bold")

ax.text(.023, .68, "SMILES → graph encoder", fontsize=12, color=INK)
ax.text(.023, .59, "Extra modality inputs: absent", fontsize=11, color=INK)
ax.annotate("", xy=(.157,.46), xytext=(.157,.54), arrowprops={"arrowstyle":"->","color":TEAL,"lw":1.8})
ax.text(.023, .40, "Omit their inactive encoders", fontsize=11, color=INK)
ax.text(.023, .335, "Saves weights; no active work removed", fontsize=8.5, color="#53636C")

ax.text(.361, .68, r"$x\ \to\ Ax+a\ \to\ W(Ax+a)+b$", fontsize=13, color=INK)
ax.annotate("", xy=(.493,.53), xytext=(.493,.62), arrowprops={"arrowstyle":"->","color":TEAL,"lw":1.8})
ax.text(.361, .47, r"$x\ \to\ (WA)x+(Wa+b)$", fontsize=14, color=TEAL)
ax.text(.361, .388, "The consumer sees the narrow input.", fontsize=9.6, color=INK)
ax.text(.361, .335, "Retain the original projection for residuals.", fontsize=8.4, color="#53636C")

ax.text(.699, .68, r"$(Qx_i)^\top(Kx_j)$", fontsize=15, color=INK)
ax.annotate("", xy=(.831,.53), xytext=(.831,.62), arrowprops={"arrowstyle":"->","color":TEAL,"lw":1.8})
ax.text(.699, .47, r"$x_i^\top (Q^\top K)x_j$", fontsize=15, color=TEAL)
ax.text(.699, .388, "Store the bilinear map when smaller.", fontsize=9.6, color=INK)
ax.text(.699, .335, "Preserve biases, edge terms and each head.", fontsize=8.4, color="#53636C")

ax.add_patch(FancyBboxPatch((.005,.07),.985,.092,boxstyle="round,pad=0.009",facecolor="#E8F4F4",edgecolor="none"))
ax.text(.025,.118,"RETAINED OUTPUTS",fontsize=10,color=TEAL,weight="bold",va="center")
ax.text(.23,.118,"12 predicted embeddings  ·  CLS  ·  13 latent embeddings  ·  requested attentions",
        fontsize=10.4,color=INK,va="center")
ax.text(.005,.006,"Schematic of selected projections, not a full architecture diagram. Float32 reassociation can change final bits.",
        fontsize=9,color="#53636C")
save(fig, "moljepa-rewrites")

fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.15), gridspec_kw={"width_ratios":[1.1,1]})
fig.subplots_adjust(left=.065,right=.985,top=.79,bottom=.25,wspace=.27)
fig.text(.02,.965,"Smaller weights and faster execution are separate results",fontsize=18,weight="bold",color=INK)
fig.text(.02,.91,"SMILES-only Mol-JEPA · float32 · all output heads retained",fontsize=11,color=INK)
counts = np.array([45406721, 29944320, 24440320, 19763160, 19763160]) / 1e6
bottoms = [0, counts[1], counts[2], counts[3], 0]
heights = [counts[0], counts[0]-counts[1], counts[1]-counts[2], counts[2]-counts[3], counts[3]]
colours = [INK, GREY, BLUE, GOLD, TEAL]
left.bar(range(5), heights, bottom=bottoms, color=colours, width=.69)
for i in range(4):
    level = counts[0] if i == 0 else counts[i]
    left.plot([i+.35,i+.65],[level,level],color="#8998A0",lw=.8,ls="--")
for i in (0,4):
    left.text(i, heights[i]+1.1, f"{heights[i]:.2f}M",ha="center",fontsize=10,weight="bold",color=INK)
for i in (1,2,3):
    left.text(i,bottoms[i]+heights[i]/2,f"−{heights[i]:.2f}M",ha="center",va="center",fontsize=9.5,
              color=INK if i==1 else "white",weight="bold")
left.set_xticks(range(5),["Original","Unused\ninputs","Affine\ncomposition","Attention\ninteraction","Final"])
left.tick_params(axis="x",length=0,labelsize=9)
left.set_ylim(0,52); left.set_ylabel("Parameter values (millions)")
left.set_title("56.47% fewer parameters",loc="left",fontsize=12,pad=15)
left.yaxis.grid(True,color="#ECF0F2");left.set_axisbelow(True)

batches = ["1","4","32"]
records = data["devices"]["mps"]
for key, offset, colour, label in [("original",-.18,INK,"Original"),("production",.18,TEAL,"Compressed + runtime")]:
    values=np.array([records[b][key]["median_ms"] for b in batches])
    low=np.array([records[b][key]["p10_ms"] for b in batches])
    high=np.array([records[b][key]["p90_ms"] for b in batches])
    bars=right.bar(np.arange(3)+offset,values,width=.33,color=colour,label=label,
                   yerr=[values-low,high-values],error_kw={"elinewidth":1,"capsize":3,"ecolor":"#475E6B"})
    for bar,value,upper in zip(bars,values,high):
        if value < 13:
            right.text(bar.get_x()+bar.get_width()/2,upper+2.5,f"{value:.2f}",ha="center",va="bottom",color=colour,fontsize=8.7)
        else:
            right.text(bar.get_x()+bar.get_width()/2,value/2,f"{value:.2f}",ha="center",va="center",color="white",fontsize=9,rotation=90)
for i,b in enumerate(batches):
    ratio=records[b]["original"]["median_ms"]/records[b]["production"]["median_ms"]
    high=max(records[b][key]["p90_ms"] for key in ("original","production"))
    right.text(i,high+4,f"{ratio:.2f}×",ha="center",color=TEAL,weight="bold",fontsize=12)
right.set_xticks(range(3),["1 molecule","4 molecules","32 molecules"])
right.tick_params(axis="x",length=0,labelsize=9)
right.set_ylim(0,89);right.set_ylabel("Complete-call latency (ms)")
right.set_title("Measured Apple MPS speedup",loc="left",fontsize=12,pad=15)
right.yaxis.grid(True,color="#ECF0F2");right.set_axisbelow(True)
right.legend(loc="upper left",frameon=False,fontsize=9)
fig.text(.065,.095,"Inactive encoders save storage. Sparse graph operations and batched readouts provide additional execution gains.",fontsize=9.2,color=INK)
fig.text(.065,.048,"Latency: median and p10–p90 over 20 interleaved rounds after 10 warmups, including parsing and transfers; Apple M5, 16 GB.",fontsize=8.6,color="#53636C")
fig.text(.065,.009,"Source: benchmarks/moljepa_runtime_macos.json. Apple MPS only; NVIDIA CUDA unverified. No per-rewrite speed attribution is inferred.",fontsize=8.6,color="#53636C")
save(fig, "moljepa-results")

manifest={"runtime_source":"benchmarks/moljepa_runtime_macos.json",
          "runtime_source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          "parameter_waterfall":{"original":45406721,"inactive_inputs_removed":15462401,"affine_removed":5504000,
                                 "attention_removed":4677160,"final":19763160},
          "runtime_statistic":"medians; p10/p90 empirical run intervals, not confidence intervals",
          "hardware":"Apple M5 MPS, 16 GB unified memory; no NVIDIA validation",
          "figures":["moljepa-rewrites.svg","moljepa-results.svg"]}
assert 45406721-15462401-5504000-4677160==19763160
(OUT/"provenance.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(manifest,indent=2))
