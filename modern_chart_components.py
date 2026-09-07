"""Readable alternatives to repeated bars; values retain their true scale."""
import numpy as np


def paired_dots(ax, categories, first, second, colors, fg, muted, *, labels=('First','Second')):
    positions=np.arange(len(categories))
    for i,(a,b) in enumerate(zip(first,second)):
        if np.isfinite(a) and np.isfinite(b):ax.plot([a,b],[i,i],color=muted,lw=2,alpha=.4)
    for values,c,marker,label,dy in [(first,colors[0],'o',labels[0],-.14),(second,colors[1],'s',labels[1],.14)]:
        ax.scatter(values,positions,c=c,s=80,marker=marker,label=label,zorder=3)
        for i,v in enumerate(values):
            if np.isfinite(v):ax.annotate(f'{v:g}',(v,i),xytext=(5,-12 if dy>0 else 9),textcoords='offset points',color=fg,size=9)
    ax.set_yticks(positions,categories,color=fg);ax.set_ylim(len(categories)-.5,-.5)
    valid=[v for v in list(first)+list(second) if np.isfinite(v)]
    ax.set_xlim(-max(valid+[1])*.04,max(valid+[1])*1.25)


def stage_flow(ax, counts, color, fg, muted):
    """Aligned stage nodes: size encodes remaining possessions, not flow width."""
    y=np.arange(len(counts))
    ax.plot(counts,y,color=color,lw=2,alpha=.65)
    ax.scatter(counts,y,s=95,color=color,edgecolor=fg,lw=.6,zorder=3)
    ax.set_yticks(y,['Possessions','Final third','Then box','Then shot','On target'],color=fg)
    ax.set_ylim(4.55,-.55);ax.set_xlim(-max(counts[0],1)*.04,max(counts[0],1)*1.4)
    for i,n in enumerate(counts):ax.annotate(f'{n} / {counts[0]}  ({100*n/max(counts[0],1):.0f}%)',(n,i),xytext=(12,0),textcoords='offset points',color=fg,size=10,va='center')
    ax.set_xlabel('Possessions remaining',color=fg)
