"""Direct point labels with deterministic placement in the rendered axes.

Coordinates are never jittered. Where values coincide, a leader connects each
full player name to that exact point. Placement uses actual font extents.
"""
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox


def label_players(ax, frame, x, y, *, color, background, fontsize=9):
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = ax.get_window_extent(renderer).padded(-5)
    points = ax.transData.transform(frame[[x, y]].to_numpy(float))
    font = FontProperties(size=fontsize)
    occupied = []
    labels = []
    # Crowded points get first choice; a stable sort makes exports reproducible.
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    density = (distances < 70).sum(axis=1)
    for i in np.argsort(-density, kind='stable'):
        name = str(frame.iloc[i]['player'])
        w, h, _ = renderer.get_text_width_height_descent(name, font, False)
        w += 8; h = max(h + 8, fontsize * fig.dpi / 72 + 7)
        px, py = points[i]
        candidates = []
        # Nearby offsets first, then a full axes grid for dense zero clusters.
        for radius in [12, 26, 44, 66, 92, 125, 170, 225]:
            for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
                candidates.append((px + radius*np.cos(angle), py + radius*np.sin(angle)))
        candidates += [(cx, cy) for cy in np.arange(bounds.y0+h/2, bounds.y1-h/2, h+3)
                       for cx in np.arange(bounds.x0+w/2, bounds.x1-w/2, max(24,w/3))]
        best = None
        for cx, cy in candidates:
            cx = np.clip(cx, bounds.x0+w/2, bounds.x1-w/2)
            cy = np.clip(cy, bounds.y0+h/2, bounds.y1-h/2)
            box = Bbox.from_bounds(cx-w/2, cy-h/2, w, h)
            if any(box.overlaps(other) for other in occupied):
                continue
            covers = sum(box.padded(5).contains(*p) for p in points)
            cost = (cx-px)**2+(cy-py)**2 + covers*10000
            if best is None or cost < best[0]:best=(cost,cx,cy,box)
        if best is None:
            raise ValueError(f'Insufficient chart space for every player label: {name}')
        _,cx,cy,box=best;occupied.append(box)
        position=ax.transAxes.inverted().transform((cx,cy))
        text=ax.annotate(name, xy=tuple(frame.iloc[i][[x,y]]), xycoords='data',
            xytext=position, textcoords='axes fraction', ha='center', va='center',
            color=color, fontsize=fontsize, zorder=8,
            bbox={'facecolor':background,'edgecolor':'none','alpha':.88,'pad':1},
            arrowprops={'arrowstyle':'-','color':color,'lw':.55,'alpha':.55,
                        'shrinkA':2,'shrinkB':5},annotation_clip=False)
        labels.append(text)
    return labels
