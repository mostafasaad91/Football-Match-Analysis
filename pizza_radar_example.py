from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle

OUT = Path("output/poster_redesign/pizza_radar_example.png")
labels = ["Progression", "Creation", "Threat", "Finishing", "Pressing", "Ball security"]
player = np.array([82, 74, 68, 91, 63, 79])
role_avg = np.array([58, 55, 52, 61, 57, 64])
fig, ax = plt.subplots(figsize=(10, 10), facecolor="#080A0B")
ax.set_facecolor("#080A0B"); ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)
center=(0,0); outer=3.0; inner=.45; gap=2.2
for i,(label,value,avg) in enumerate(zip(labels,player,role_avg)):
    start=90-i*360/len(labels)-360/len(labels)+gap/2
    end=90-i*360/len(labels)-gap/2
    # One independent slice: distance from centre encodes the player value.
    ax.add_patch(Wedge(center,outer,start,end,facecolor="#1A2022",edgecolor="#485154",lw=1.2))
    r=inner+(outer-inner)*value/100
    ax.add_patch(Wedge(center,r,start,end,facecolor="#EF3340",alpha=.88,edgecolor="#080A0B",lw=2.5))
    # Thin grey role-average ring inside the same slice.
    ar=inner+(outer-inner)*avg/100
    ax.add_patch(Wedge(center,ar-.035,start,end,facecolor="none",edgecolor="#C4CBCC",lw=2.2))
    mid=np.deg2rad((start+end)/2)
    ax.text(3.35*np.cos(mid),3.35*np.sin(mid),label.upper(),color="#B8C0C2",ha="center",va="center",fontsize=10,weight="bold")
    ax.text(2.35*np.cos(mid),2.35*np.sin(mid),str(value),color="#FFFFFF",ha="center",va="center",fontsize=11,weight="bold")
ax.add_patch(Circle(center,inner,facecolor="#080A0B",edgecolor="#485154",lw=1.2))
ax.text(0, .12, "PLAYER", color="#62D8CC", ha="center", fontsize=10, weight="bold")
ax.text(0,-.18, "78", color="#F4F7F7", ha="center", fontsize=25, weight="bold")
ax.text(0, 4.1, "PLAYER IMPACT", color="#62D8CC", ha="center", fontsize=13, weight="bold")
ax.text(0, 3.78, "Bukayo Saka  ·  78 min  ·  right winger", color="#F4F7F7", ha="center", fontsize=18, weight="bold")
ax.text(0, -3.92, "RED SLICE = PLAYER VALUE   ·   GREY RING = ROLE AVERAGE   ·   SCALE 0–100", color="#8B9699", ha="center", fontsize=10)
fig.text(.08,.06,"MATCH STUDY / PLAYER PROFILE", color="#8B9699", fontsize=9, weight="bold")
fig.text(.92,.06,"LOCAL MODEL VALUES", color="#8B9699", fontsize=9, ha="right")
fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(OUT)
