"""One role-aware player page, replacing duplicated historical radar exports."""
import re
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Wedge, Circle
from matplotlib.lines import Line2D
from match_insights import flags, numeric
from metric_registry import label, format_value


# The pizza used to be painted in fixed slate and near-black: track #20272A,
# rim #536064, hub #090B0C, average ring #D5DADB, and the percentile number in
# flat white. Every one of those is a dark-theme value. On the light page the
# hub and track arrived as dark blobs on grey paper, the average ring vanished
# into it, and white digits on a gold wedge were unreadable at any size. The
# colours are derived from the active theme instead, and the digit is solved
# against the wedge it sits on rather than assumed.
_PIZZA = {
    True:  {"track": "#E7E7E7", "rim": "#B4B4B4", "avg": "#4A4A4A"},   # light
    False: {"track": "#16191C", "rim": "#3D454A", "avg": "#D5DADB"},   # dark
}

# Short forms for the ring only. The registry labels are written for table
# rows, where a line can run as long as it likes; on a wedge "Possession-
# adjusted defensive actions" overran its neighbours on both sides.
_WEDGE_LABEL = {
    "padj_defensive_actions": "pAdj def.\nactions",
    "progression_metres": "Progression\nmetres",
    "aerials_won": "Aerials\nwon",
    "defensive_height": "Defensive\nheight",
    "final_third_receptions": "Final-third\nreceptions",
    "line_breaking_passes": "Line-break\npasses",
    "progressive_pass_pct": "Progressive\npass share",
    "xT_per_100_touches": "xT / 100\ntouches",
    "box_entries": "Box\nentries",
    "takeons_won": "Take-ons\nwon",
    "completed_passes": "Completed\npasses",
    "tackles_won": "Tackles\nwon",
    "pass_pct": "Pass %",
    "xGChain": "xGChain",
    "xG": "xG",
    "xA": "xA",
    "saves": "Saves",
    "claims": "Claims",
    "sweeps": "Sweeps",
}


def wedge_label(key):
    """The ring's own name for a metric, wrapped to fit beside its neighbours."""
    import textwrap
    if key in _WEDGE_LABEL:
        return _WEDGE_LABEL[key]
    return "\n".join(textwrap.wrap(str(label(key)), 12)) or str(key)


def _role_pizza(fig, rect, labels, values, colour, *, role_average=None,
                printed=None):
    """Segmented, role-aware pizza profile; each slice is independently readable.

    The number is printed **outside** the ring, beside its own label, on the
    page background. Inside the wedge it sat on whatever the bar happened to
    reach -- a saturated kit colour, a dark track, or the boundary between them
    -- and no single text colour is readable on all three. Chris Wood's "100"
    was white on Forest red at about four to one. Moving the number off the
    coloured fill removes the whole problem rather than tuning around it: one
    background, one text colour, every card, both themes.

    ``values`` are percentiles and set the length of each wedge. ``printed``,
    when given, is what is written beside it.

    They used to be the same thing, and a percentile is the wrong thing to
    print. The pool is the players in one fixture who share a role -- eight
    midfielders, four forwards -- so the scale has as many steps as the pool
    has players: with eight, the only values it can take are 6, 19, 31, 44, 56,
    69, 81 and 94. Eight wedges on an eight-step scale must repeat, and one
    card in this project showed five wedges reading 94. Printing "94" also
    claims a hundred-point resolution the comparison cannot support, and a
    forward compared with three others can never print more than 88 however
    well he played. The wedge keeps the ranking, which is what it is good at;
    the figure beside it is what the player actually did.
    """
    from insight_visuals import FG, MUTED
    from visualization_components import BG_DARK, IS_LIGHT_THEME
    shades = _PIZZA[bool(IS_LIGHT_THEME)]
    ax = fig.add_axes(rect)
    ax.set_aspect('equal')
    ax.axis('off')
    # Wider than tall on purpose: the labels and their numbers sit out to the
    # left and right of the ring and nothing sits above or below it, so the
    # room they need is horizontal.
    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.30, 1.30)
    n = len(labels)
    gap = 2.5
    outer = 1.0
    inner = .16
    for i, (caption, value) in enumerate(zip(labels, values)):
        start = 90 - i * 360 / n - 360 / n + gap / 2
        end = 90 - i * 360 / n - gap / 2
        ax.add_patch(Wedge((0, 0), outer, start, end,
                           facecolor=shades["track"], edgecolor=shades["rim"], lw=.7))
        mid = np.deg2rad((start + end) / 2)
        if np.isfinite(value):
            radius = inner + (outer - inner) * float(np.clip(value, 0, 100)) / 100
            ax.add_patch(Wedge((0, 0), radius, start, end, facecolor=colour,
                               alpha=.9, edgecolor=BG_DARK, lw=1.2))
        if role_average is not None and np.isfinite(role_average[i]):
            avg = inner + (outer - inner) * float(np.clip(role_average[i], 0, 100)) / 100
            ax.add_patch(Wedge((0, 0), avg, start, end, facecolor='none',
                               edgecolor=shades["avg"], lw=1.1))

        # Anchor the block on the side it sits on, so a long label grows away
        # from the ring instead of across the wedge beside it. The number is
        # the anchor and the label stacks above it, which keeps the numbers on
        # a predictable ring however many lines a label runs to.
        across, up = np.cos(mid), np.sin(mid)
        if across > .2:
            align, reach = 'left', 1.07
        elif across < -.2:
            align, reach = 'right', 1.07
        else:
            align, reach = 'center', 1.14
        x, y = reach * across, reach * up
        if printed is not None and i < len(printed) and printed[i] is not None:
            shown = str(printed[i])
        else:
            shown = f'{value:.0f}' if np.isfinite(value) else 'n/a'
        readable = np.isfinite(value) or (printed is not None and shown != 'n/a')
        ax.text(x, y - .015, shown,
                color=FG if readable else MUTED, ha=align, va='top',
                fontsize=10.5 if readable else 7,
                weight='bold' if readable else 'normal',
                style='normal' if readable else 'italic')
        ax.text(x, y + .045, str(caption).upper(), color=MUTED, ha=align,
                va='bottom', fontsize=6.2, weight='bold', linespacing=1.25)
    ax.add_patch(Circle((0, 0), inner, facecolor=BG_DARK,
                        edgecolor=shades["rim"], lw=.7))
    return ax


# Three columns of rows the role actually generates, instead of one grid of
# twenty-six columns-worth for everybody. Nothing here is already a wedge on
# that role's ring: a reader who has seen a measure as a percentile does not
# need it again as a raw number two inches below.
_TABLE = {
    'Defender': [
        ('ON THE BALL', [('Touches', 'touches'), ('Passes', '_passes_pct'),
                         ('Progressive passes', 'progressive_passes'),
                         ('Line-breaking passes', 'line_breaking_passes'),
                         ('Dispossessed', 'dispossessed')]),
        ('DEFENDING', [('Recoveries', 'recoveries'),
                       ('Interceptions', 'interceptions'),
                       ('Tackles won', 'tackles_won'),
                       ('Dribbled past', 'dribbled_past'),
                       ('Aerials won', '_aerials')]),
        ('THREAT', [('Shots', 'shots'), ('xG', 'xG'), ('Inferred xA', 'xA'),
                    ('Positive xT', 'positive_xT'),
                    ('xGBuildup', 'xGBuildup')]),
    ],
    'Midfielder': [
        ('ON THE BALL', [('Touches', 'touches'), ('Passes', '_passes_pct'),
                         ('Progressive passes', 'progressive_passes'),
                         ('Take-ons won', '_takeons'),
                         ('Dispossessed', 'dispossessed')]),
        ('CREATION', [('Shots', 'shots'), ('xG', 'xG'),
                      ('Box entries', 'box_entries'),
                      ('Final-third receptions', 'final_third_receptions'),
                      ('Positive xT', 'positive_xT')]),
        ('WITHOUT THE BALL', [('Recoveries', 'recoveries'),
                              ('Interceptions', 'interceptions'),
                              ('Tackles won', 'tackles_won'),
                              ('Dribbled past', 'dribbled_past'),
                              ('Aerials won', '_aerials')]),
    ],
    'Forward': [
        ('FINISHING', [('Goals', 'goals'), ('Shots', 'shots'),
                       ('xG / shot', 'xG_per_shot'), ('xG + xA', 'xG_xA'),
                       ('Positive xT', 'positive_xT')]),
        ('INVOLVEMENT', [('Touches', 'touches'),
                         ('Completed passes', 'completed_passes'),
                         ('Progressive passes', 'progressive_passes'),
                         ('Take-ons', 'takeons'),
                         ('Take-ons won', 'takeons_won')]),
        ('THE WORK', [('Aerials won', 'aerials_won'),
                      ('Dispossessed', 'dispossessed'),
                      ('Recoveries', 'recoveries'),
                      ('pAdj def. actions', 'padj_defensive_actions'),
                      ('Progression metres', 'progression_metres')]),
    ],
    'Goalkeeper': [
        ('THE GOAL', [('Saves', 'saves'), ('Claims', 'claims'),
                      ('Sweeps', 'sweeps'), ('Recoveries', 'recoveries'),
                      ('Clearances', 'clearances')]),
        ('DISTRIBUTION', [('Passes', 'passes'),
                          ('Completed', 'completed_passes'),
                          ('Completion %', 'pass_pct'),
                          ('Progressive passes', 'progressive_passes'),
                          ('Line-breaking passes', 'line_breaking_passes')]),
        ('WHAT IT WAS WORTH', [('Positive xT', 'positive_xT'),
                               ('xT / 100 touches', 'xT_per_100_touches'),
                               ('Touches', 'touches'),
                               ('xGBuildup', 'xGBuildup'),
                               ('Progression metres', 'progression_metres')]),
    ],
}
# A player the feed never gave a position gets the midfielder's set: it is the
# only one of the four that asks about every phase.
_TABLE['Unknown'] = _TABLE['Midfielder']


def _verdict(labels, values, minutes):
    """The three things this player did best, for the strip under his name.

    A card of forty-odd numbers at one size says nothing about which of them
    mattered. These three are the top of his own ring, so the strip is a
    summary of the card rather than a fifth opinion on it.
    """
    import numpy as np

    ranked = sorted(
        ((v, l) for l, v in zip(labels, values) if np.isfinite(v)),
        reverse=True)
    return ranked[:3]




# Eight wedges: four that belong to the role, four every outfielder is asked
# the same question on. Module level because the caption writer needs the same
# sets the ring is drawn from, and a copy in two places is a copy that drifts.
ROLE_WEDGES = {
    'Defender': ['padj_defensive_actions', 'aerials_won', 'defensive_height',
                 'progression_metres'],
    # xA ranks the volume of chance creation, and the table under the ring
    # already carries the passing volume it comes from. Per hundred passes
    # ranks how good the passing was, which is a different question: this
    # player is first of eight for xA because he passes more than anyone, and
    # second for xA per hundred.
    'Midfielder': ['padj_defensive_actions', 'progression_metres',
                   'line_breaking_passes', 'xA_per_100_passes'],
    'Forward': ['xG', 'xA', 'box_entries', 'final_third_receptions'],
    'Goalkeeper': ['saves', 'claims', 'sweeps', 'completed_passes'],
    'Unknown': ['xG', 'xA', 'completed_passes', 'padj_defensive_actions'],
}
# Six wedges, not eight. Two of the four shared ones were answered elsewhere on
# the same card -- pass_pct is now printed beside the pass count in the table,
# and xT_per_100_touches ranks the same movement the table's Positive xT
# counts -- and on a pool of eight the extra wedges could only repeat values
# the others had already taken.
SHARED_WEDGES = ['xGChain', 'progressive_pass_pct']

# What is written beside each wedge. The wedge length is the ranking; this is
# the figure the player actually produced, in the unit it is measured in.
WEDGE_FORMAT = {
    'padj_defensive_actions': lambda v: f'{v:.2f}',
    'progression_metres': lambda v: f'{v:.0f}',
    'line_breaking_passes': lambda v: f'{v:.0f}',
    'xA_per_100_passes': lambda v: f'{v:.2f}',
    'xA': lambda v: f'{v:.2f}',
    'xG': lambda v: f'{v:.2f}',
    'xGChain': lambda v: f'{v:.2f}',
    'progressive_pass_pct': lambda v: f'{v:.0f}%',
    'pass_pct': lambda v: f'{v:.0f}%',
    'xT_per_100_touches': lambda v: f'{v:.1f}',
    'aerials_won': lambda v: f'{v:.0f}',
    'defensive_height': lambda v: f'{v:.0f}',
    'box_entries': lambda v: f'{v:.0f}',
    'final_third_receptions': lambda v: f'{v:.0f}',
    'completed_passes': lambda v: f'{v:.0f}',
    'saves': lambda v: f'{v:.0f}',
    'claims': lambda v: f'{v:.0f}',
    'sweeps': lambda v: f'{v:.0f}',
}


def _paired(row, key):
    """A count printed with the denominator it came out of, or None.

    "Aerials won 0" is not a reading. Nought from one is a duel he lost and
    nought from none is a duel he was never in, and the card printed both as
    the same figure. The same goes for take-ons, and for a pass count sitting
    in one row with its completion in the row below it -- two rows for one
    fact, on a card where every row costs a reader something.
    """
    def count(name):
        try:
            value = pd.to_numeric(pd.Series([row.get(name, np.nan)]),
                                  errors='coerce').iloc[0]
        except Exception:
            return None
        return None if pd.isna(value) else int(round(float(value)))

    if key == '_passes_pct':
        attempted, completed = count('passes'), count('completed_passes')
        if attempted is None:
            return 'N/A'
        if not attempted or completed is None:
            return f'{attempted}'
        return f'{attempted} ({100 * completed / attempted:.0f}%)'
    if key == '_takeons':
        won, tried = count('takeons_won'), count('takeons')
        if won is None:
            return 'N/A'
        return f'{won}' if not tried else f'{won} of {tried}'
    if key == '_aerials':
        won, tried = count('aerials_won'), count('aerials')
        if won is None:
            return 'N/A'
        return f'{won}' if not tried else f'{won} of {tried}'
    return None


def wedge_figure(key, value):
    """The raw figure for one wedge, or None when there is nothing to print."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return WEDGE_FORMAT.get(key, lambda v: f'{v:.1f}')(number)


ROLE_POOL_MINIMUM = 4


_PEOPLE_CACHE = {}


def _people_for(events, players, info):
    """The enriched player frame for one fixture, computed once.

    profile_caption is called for every card in the package, and rebuilding the
    insights and the advanced metrics inside each call turned a forty-second
    document build into one that ran past ten minutes. The frame depends only
    on the fixture, so it is built once and kept.
    """
    from insight_visuals import role_group
    from match_insights import build_insights
    from player_advanced import enrich

    key = (str(info.get("match_id") or info.get("url") or ""),
           len(events), len(players))
    if key not in _PEOPLE_CACHE:
        people = build_insights(events, players, info)["players"].copy()
        people["role_group"] = people.role.map(role_group)
        _PEOPLE_CACHE[key] = enrich(people, events, players)
    return _PEOPLE_CACHE[key]


def profile_caption(events, players, info, player_name):
    """The paragraph for one player's card, computed without drawing it.

    The card writes this into the package manifest when it is rendered; the
    report reads it back when it is built. Recomputing it here means a wording
    change reaches every existing package for the cost of rebuilding two
    documents, instead of re-rendering thirty-one cards per package first.
    """
    import numpy as np
    import pandas as pd

    people = _people_for(events, players, info)
    match = people[people.player.astype(str) == str(player_name)]
    if match.empty:
        return ""
    row = match.iloc[0]
    keys, scores, pool, basis, compare = profile_percentiles(people, row)
    ranked = sorted(((v, wedge_label(k).replace(chr(10), " "))
                     for k, v in zip(keys, scores) if np.isfinite(v)), reverse=True)

    own = events[events.player.astype(str) == str(player_name)]
    x = pd.to_numeric(own.get("x"), errors="coerce").dropna()
    thirds = " / ".join(str(int(((x >= lo) & (x < hi)).sum()))
                        for lo, hi in ((0, 35), (35, 70), (70, 105))) if len(x) else ""
    defensive = int(own["type"].isin(
        {"Tackle", "Interception", "BallRecovery", "Clearance", "BlockedShot",
         "Aerial", "Challenge", "Foul"}).sum()) if "type" in own else 0
    final_third = int((x >= 66.7).sum()) if len(x) else 0
    team = str(info.get("home_name") if row.team_id == info.get("home_id")
               else info.get("away_name"))
    return _profile_reading(row, team, ranked[:3], keys, scores, pool, compare,
                            basis, defensive, final_third, thirds)


def profile_percentiles(people, row):
    """(radar keys, percentiles, pool, comparison basis, whether it is by role).

    The same maths the ring is drawn from, without drawing anything, so the
    paragraph that describes a card can be written without rendering it. The
    caption lives in the package manifest and is read when the report is built,
    which is minutes of work rather than the hours re-rendering every card in
    every package would cost.
    """
    import numpy as np
    import pandas as pd

    pool = people[(people.role_group == row.role_group) & (people.minutes >= 30)]
    compare = (row.minutes >= 30 and len(pool) >= ROLE_POOL_MINIMUM
               and row.role_group not in ('Goalkeeper', 'Unknown'))
    basis = pool if compare else people[people.minutes >= 30]
    keys = list(dict.fromkeys(
        ROLE_WEDGES.get(row.role_group, ROLE_WEDGES['Unknown']) + SHARED_WEDGES))
    scores = []
    for key in keys:
        value = pd.to_numeric(pd.Series([row.get(key, np.nan)]), errors='coerce').iloc[0]
        eligible = pd.to_numeric(basis.get(key, pd.Series(dtype=float)),
                                 errors='coerce').dropna()
        if len(eligible) < 2:
            eligible = pd.to_numeric(people.get(key, pd.Series(dtype=float)),
                                     errors='coerce').dropna()
        if len(eligible) >= 2 and pd.notna(value):
            scores.append(float(np.clip(
                100 * ((eligible < value).sum() + .5 * (eligible == value).sum())
                / len(eligible), 0, 100)))
        else:
            scores.append(0.0 if pd.notna(value) and len(eligible) >= 2 else np.nan)
    return keys, scores, pool, basis, compare


def _profile_reading(p, team, strip, radar_keys, vals, pool, compare, basis,
                     defensive_actions, final_touch, thirds):
    """What this card says about this player.

    It used to list five numbers already printed on the card -- "95.2 minutes;
    shots 2; progressive passes 7; xGChain 0.75; xGBuildup 0.29" -- and close
    on a sentence identical under every player in the package. A reader looking
    at the card has the numbers. What the card cannot say on its own is which
    of them is unusual for the job he was doing, and that is the whole point of
    ranking him inside his own line.
    """
    import numpy as np

    role = str(getattr(p, 'role_group', '') or 'player').lower()
    against = (f"the {len(pool)} other {role}s who played 30+ minutes" if compare
               else f"all {len(basis)} players over 30 minutes, too few {role}s to compare within")
    parts = [f"{p.player} played {p.minutes:.0f} minutes for {team}, ranked against {against}."]

    if strip:
        # wedge_label puts a newline in the long labels and a couple carry a
        # trailing space, so joining them left a double space mid-sentence.
        best = ", ".join(" ".join(name.lower().split()) + f" {score:.0f}"
                         for score, name in strip)
        parts.append(f"His strongest measures were {best} out of 100.")
        weakest = sorted(((v, wedge_label(k).replace(chr(10), ' '))
                          for k, v in zip(radar_keys, vals) if np.isfinite(v)))
        if weakest and weakest[0][0] <= 30 and weakest[0][0] < strip[-1][0] - 25:
            parts.append(
                f"The ring's short wedge is {' '.join(weakest[0][1].lower().split())} "
                f"at {weakest[0][0]:.0f}, "
                f"which is where this performance differed most from the rest of his line.")

    zones = str(thirds).split(' / ')
    if len(zones) == 3 and any(z.strip().isdigit() for z in zones):
        defensive, middle, attacking = (int(z) for z in zones)
        total = defensive + middle + attacking
        if total >= 15:
            if attacking >= total * 0.4:
                where = "spent most of his touches in the final third"
            elif defensive >= total * 0.5:
                where = "took most of his touches in his own third"
            else:
                where = "worked mainly between the two boxes"
            parts.append(
                f"The map shows where that happened: he {where}, with "
                f"{final_touch} touches in the final third and {defensive_actions} "
                f"defensive actions.")
    # "1 touches", "1 minutes". The same normaliser the board writer uses, so a
    # count of one reads as one wherever the report prints it.
    try:
        from report_narrative import one_reads_singular

        return one_reads_singular(" ".join(parts))
    except Exception:
        return " ".join(parts)


def compact_profiles(charts, players, events):
    from insight_visuals import role_group, FG, BG, MUTED
    people=players.copy();people['role_group']=people.role.map(role_group)
    # Possession-adjusted defending, progression distance, aerials, defensive
    # height and final-third receptions. Computed for the article's ranking
    # already; the ring was still drawing raw counts beside them.
    from player_advanced import enrich as _enrich_advanced
    people=_enrich_advanced(people,events,players)
    # Two metrics answer to this one name. enrich() has just written
    # player_advanced.line_breaking -- completed forward passes of 20 m or more
    # -- and the column is replaced here by the match_metrics measure, which
    # estimates the opponent's line over five-minute windows and counts the
    # passes that started behind it and finished past it. That is the one the
    # team-level board uses, so the card agrees with the rest of the package;
    # the footnote under the card describes it rather than the rule that is
    # overwritten. They are far apart -- twenty-three of twenty-six players in
    # one fixture, and 18 against 3 for a centre back who played out a lot --
    # so which one the label names is not a detail.
    from match_metrics import line_breaking_passes
    people['line_breaking_passes']=0
    for tid in people.team_id.dropna().unique():
        opponents=[x for x in people.team_id.dropna().unique() if x != tid]
        if opponents:
            br=line_breaking_passes(events,tid,opponents[0])
            if not br.empty and 'player' in br:
                counts=br.groupby('player').size()
                people.loc[people.team_id.eq(tid),'line_breaking_passes']=people.loc[people.team_id.eq(tid),'player'].map(counts).fillna(0)
    # Eight wedges: four that belong to the role, four every outfielder is
    # asked the same question on.
    #
    # The defensive four used to be raw interceptions, tackles, clearances and
    # recoveries. A defensive action can only happen while the other side has
    # the ball, so those four rewarded a player for his team being pinned:
    # Brighton recorded 103 of them to Chelsea's 125 and were still defending
    # twice as hard per opponent touch. They are one possession-adjusted wedge
    # now, with the questions the counts could not answer -- how high he won it,
    # how much ground he actually gained, whether he won the ball in the air --
    # taking the other three.
    roles=ROLE_WEDGES
    # xGBuildup is xGChain minus the shooter and the key-pass provider, and on
    # a player who was neither it is the same number twice. Vuskovic's ring
    # read 93 and 93. It stays in the table, where the pair can be compared;
    # two wedges moving together said nothing the first one had not.
    advanced=SHARED_WEDGES
    # Below this many players in a role the percentile is a ranking against
    # two or three people, which is a position in a queue rather than a rate.
    # Same floor the article's own ranking uses.
    for _,p in people[people.minutes>0].sort_values(['team_id','player']).iterrows():
        pool=people[(people.role_group==p.role_group)&(people.minutes>=30)]
        compare=(p.minutes>=30 and len(pool)>=ROLE_POOL_MINIMUM
                 and p.role_group not in ('Goalkeeper','Unknown'))
        # The card used to print "percentiles require 8 same-role players",
        # then draw them anyway off a pool of seven through a fallback that
        # only checked for fewer than two. One basis is chosen here and both
        # the ring and the caption below it are told which one it was.
        basis=pool if compare else people[people.minutes>=30]
        fig,axes=charts.figure(p.player,f'{charts.names[p.team_id]} · {p.role_group} · Touches {p.touches:.0f} · Shots {p.shots:.0f}',columns=2,height=12)
        fig.subplots_adjust(left=.07,right=.95,top=.74,bottom=.29,wspace=.20)
        # A compact, role-aware radar restores the visual player identity while
        # keeping the raw values and the action map on the same card.
        old=axes[0];fig.delaxes(old)
        ax=fig.add_axes([.12,.51,.29,.27])
        radar_keys=(roles[p.role_group]+advanced)
        radar_keys=list(dict.fromkeys(radar_keys))
        vals=[]
        role_avg=[]
        for k in radar_keys:
            value=pd.to_numeric(pd.Series([p.get(k,np.nan)]),errors='coerce').iloc[0]
            eligible=pd.to_numeric(basis.get(k,pd.Series(dtype=float)),errors='coerce').dropna()
            if len(eligible)<2:
                # Last resort so a profile is never an empty ring: everyone who
                # played, at whatever minutes.
                eligible=pd.to_numeric(people.get(k,pd.Series(dtype=float)),errors='coerce').dropna()
            if len(eligible)>=2 and pd.notna(value):
                vals.append(float(np.clip(
                    100*((eligible<value).sum()+.5*(eligible==value).sum())/len(eligible),0,100)))
            else:
                # No comparison and no value are different states, and only one
                # of them is a zero.
                vals.append(0.0 if pd.notna(value) and len(eligible)>=2 else np.nan)
            role_avg.append(
                float(100*((eligible<eligible.mean()).sum())/len(eligible))
                if len(eligible)>=2 else np.nan)
        fig.delaxes(ax)
        printed=[wedge_figure(k,p.get(k,np.nan)) for k in radar_keys]
        _role_pizza(fig,[.065,.435,.35,.30],[wedge_label(k) for k in radar_keys],vals,charts.colors[p.team_id],role_average=role_avg,printed=printed)
        # Three numbers at 26pt under the name, so the card leads with a
        # judgement instead of opening on forty numbers at one size and leaving
        # the reader to find the ones that mattered.
        strip=_verdict([wedge_label(k).replace(chr(10),' ') for k in radar_keys],vals,p.minutes)

        def _tile(x,number,name,under,colour):
            # The label sits under the number, not beside it.
            #
            # Beside it, the tile's width was the number's width, and the gap
            # was computed from the digit count -- which works while the label
            # is short. The tiles print the measure's own figure now, so the
            # labels are the long ones: "PROGRESSIVE PASS SHARE" beside "32%"
            # ran through the "18" of the tile after it. Stacked, every tile is
            # the same narrow column and nothing can reach its neighbour.
            fig.text(x,.806,number,color=colour,size=25,weight='bold',
                     ha='left',va='baseline')
            fig.text(x,.786,name,color=FG,size=7.2,weight='bold',ha='left',va='baseline')
            fig.text(x,.773,under,color=MUTED,size=6.6,ha='left',va='baseline')

        # The tiles print the figure and say the place under it, rather than
        # printing the percentile. Three tiles reading "94 percentile" side by
        # side was this card's worst instance of the pool problem: they were the
        # top three measures on an eight-step scale, so they were bound to be
        # the same number, and the card opened on it three times.
        label_by_wedge={wedge_label(k).replace(chr(10),' '):k for k in radar_keys}
        for column,(score,name) in enumerate(strip):
            key=label_by_wedge.get(name)
            figure=wedge_figure(key,p.get(key,np.nan)) if key else None
            if figure is None:
                figure=f'{score:.0f}'
                under='percentile'
            else:
                ahead=int(round((100-score)/100*max(len(basis),1)))
                under=f'{max(ahead,0)+1} of {len(basis)} in his role' if compare else f'{max(ahead,0)+1} of {len(basis)}'
            _tile(.068+column*.145,figure,name.upper(),under,
                  charts.colors[p.team_id])
        if strip:
            _tile(.505,f'{p.minutes:.0f}','MINUTES',
                  'played' if p.minutes>=89 else 'of the match',MUTED)
            fig.text(.068,.752,'His three strongest measures on the ring below.',
                     color=MUTED,size=7.5,ha='left',va='baseline')
        caption=(f'Ranked against {len(pool)} players in the same role (30+ min)'
                 if compare else
                 f'Only {len(pool)} in this role · ranked against all {len(basis)} players over 30 min')
        fig.text(.24,.408,caption+' · grey ring = pool average',ha='center',color=MUTED,size=8)
        ax=axes[1];ax.set_position([.545,.455,.33,.29]);ax.set_xlim(0,105);ax.set_ylim(0,68);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
        ax.add_patch(Rectangle((0,0),105,68,fill=False,ec=MUTED));ax.axvline(52.5,color=MUTED,lw=.6)
        ax.add_patch(Rectangle((88.5,13.84),16.5,40.32,fill=False,ec=MUTED))
        g=events[events.player.eq(p.player)&events.team_id.eq(p.team_id)]
        g=g[numeric(g,'x',np.nan).notna()&numeric(g,'y',np.nan).notna()]
        from match_metrics import touch_mask, progressive_pass_mask, FINAL_THIRD_X
        # The list the footnote below this card describes, and the one the
        # pAdj wedge on the ring is built from. The snapshot used to keep its
        # own: it counted fouls and aerials, which are not defensive actions
        # under that definition, and looked for "BlockedShot", which this feed
        # never emits -- the type is "BlockedPass" -- so a real block was
        # dropped while a foul was added. Nineteen of the twenty-seven players
        # in one fixture were given a figure the footnote did not describe.
        from player_advanced import _DEFENSIVE_TYPES
        touches=g[touch_mask(g)]
        # Tactical snapshot keeps the map interpretable without reading every
        # row below it.
        defensive_actions=int(g.type.isin(_DEFENSIVE_TYPES).sum()) if 'type' in g.columns else 0
        avg_x=float(touches.x.mean()) if len(touches) else float('nan')
        avg_y=float(touches.y.mean()) if len(touches) else float('nan')
        # x is the provider's 0-100 axis -- the heat map below converts it with
        # *1.05 to draw on a 105 m pitch. The bins were written in metres,
        # (0,35)/(35,70)/(70,105), and applied to it, so a touch at x=68 was
        # filed in midfield: the attacking third lost every touch between 66.7
        # and 70. Rice's card read "29" beside a final-third count of 45.
        #
        # Both figures now come from FINAL_THIRD_X, which is what the rest of
        # the project means by the final third, so the zones cannot disagree
        # with the count of the last one -- and the count is no longer printed
        # twice on the same line under two names.
        defensive_third_x=100.0-FINAL_THIRD_X
        bounds=[(0.0,defensive_third_x),(defensive_third_x,FINAL_THIRD_X),(FINAL_THIRD_X,100.01)]
        counts=[int((touches.x.between(lo,hi,inclusive="left")).sum()) if len(touches) else 0
                for lo,hi in bounds]
        thirds=' / '.join(str(c) for c in counts)
        # The prose under the card names the same figure. It reads the last bin
        # rather than recounting, so the sentence and the line above it cannot
        # come apart.
        final_touch=counts[-1]
        # One line starting at x=.56 could not hold five readings: it ran past
        # the right edge of the page and lost everything after "Line-breaking".
        # Two lines fit inside the map's own column.
        # "Avg touch 60, 36" was a coordinate pair, and the second half of it
        # read backwards. This feed numbers the width from the RIGHT touchline
        # -- a right back's touches average y=19 and a left back's y=81 -- so a
        # reader taking 36 for a position on a left-to-right axis put the
        # player on the wrong side of the pitch. Neither number is wrong; the
        # pair was unreadable without knowing the convention.
        #
        # Both are named now, and the width is given as a distance from the
        # centre with the side it lies on, so nothing depends on knowing which
        # touchline is zero.
        if np.isfinite(avg_x) and np.isfinite(avg_y):
            offset=abs(avg_y-50.0)
            if offset < 2.0:
                across='central'
            else:
                across=f'{offset:.0f} {"left" if avg_y > 50.0 else "right"} of centre'
            average=f'{avg_x:.0f} upfield · {across}'
        else:
            average='n/a'
        # Two figures left this line rather than being repeated on it. The
        # defensive-action count is the ring's pAdj wedge without its
        # denominator, and line-breaking passes is a wedge on the midfield ring
        # and a row in the defender's table. The comment above _TABLE says
        # nothing on the card should be a wedge printed twice; this line was
        # the exception to its own rule.
        fig.text(.545,.427,
                 f'TACTICAL SNAPSHOT   Touches by third · defensive / middle / final {thirds}\n'
                 f'Average touch {average}  ·  {defensive_actions} defensive '
                 f'{"action" if defensive_actions == 1 else "actions"}, '
                 f'{"smoothed on the map" if defensive_actions >= 6 else "too few to smooth"}',
                 color=MUTED,size=7.5,ha='left',va='top',linespacing=1.6)
        if len(touches):
            # Smooth team-colour density, matching the defensive-action maps
            # instead of using blocky hexagons.
            from matplotlib.colors import LinearSegmentedColormap
            from scipy.stats import gaussian_kde
            # Purple-to-gold density is deliberately independent of the
            # team's red/blue identity and the defensive marker colours.
            from matplotlib.colors import to_rgb
            is_dark=sum(to_rgb(BG)) < 1.5
            cmap=LinearSegmentedColormap.from_list('player_heat',
                ([(0,'#123a52'),(.45,'#0ea5e9'),(1,'#7dd3fc')]
                 if is_dark else
                 [(0,'#c7dcfb'),(.45,'#3b82f6'),(1,'#1d4ed8')]))
            tx,ty=touches.x.to_numpy()*1.05,touches.y.to_numpy()*.68
            if len(touches)>=6:
                gx,gy=np.mgrid[0:105:45j,0:68:30j]
                z=gaussian_kde(np.vstack([tx,ty]),bw_method=.20)(np.vstack([gx.ravel(),gy.ravel()])).reshape(gx.shape)
                peak=float(np.nanmax(z))
                if peak>0:
                    # Explicit lower cut-off removes invisible noise while
                    # keeping compact high-density zones visibly filled.
                    levels=np.linspace(peak*.08,peak,12)
                    ax.contourf(gx,gy,z,levels=levels,cmap=cmap,alpha=.88,zorder=1,antialiased=True)
        # Where he defended, in its own colour, over where he had the ball.
        #
        # A density needs points to smooth. Twelve of the thirty-one players in
        # the fixture this was built on produced six defensive actions or more
        # and the rest produced between nought and five, and smoothing four
        # points draws a hill that belongs to the kernel rather than the match.
        # So the surface is drawn only where the sample carries one. The height
        # line below is drawn either way: three actions establish where a
        # player was defending, and that is the reading a defensive map is for.
        from player_advanced import _DEFENSIVE_TYPES
        from scipy.stats import gaussian_kde as _kde
        from matplotlib.colors import to_rgb as _to_rgb
        # Every colour on this map is settled here, before anything is drawn.
        # The theme flag used to be recomputed twice in this block and the
        # defensive layer read it before either assignment.
        is_dark=sum(_to_rgb(BG)) < 1.5
        point_color='#f8fafc' if is_dark else '#1f2937'
        edge_color='#111827' if is_dark else '#ffffff'
        marker_edge='#0b0f14' if is_dark else '#ffffff'
        # Four things carry colour on this map and each has to hold against the
        # page it is on. The first attempt failed both ways: on the light page
        # the touch ramp began at white, so its lower half was white on white,
        # and on the dark page it began at the near-black background and the
        # same half disappeared into that. Both ramps start clear of the page
        # now and end bright on it.
        defend_colour='#f472b6' if is_dark else '#be185d'
        heat_peak='#7dd3fc' if is_dark else '#1d4ed8'
        # The arrows take the package's own two event colours rather than a
        # pair chosen for this map.
        #
        # Every other visual in the package already says "this one is the
        # finding" in EVENT_HIGHLIGHT and "this one is context" in
        # EVENT_NEUTRAL, and both follow the theme. Picking a grey and a white
        # here made the card the one page in fifty-odd speaking its own
        # dialect, and the light half of that pair was wrong on its own terms:
        # the package highlights in petrol on paper, not in near-black.
        from visualization_components import EVENT_HIGHLIGHT, EVENT_NEUTRAL
        FOCUS=EVENT_HIGHLIGHT
        quiet_arrow=EVENT_NEUTRAL
        defensive_events=g[g.type.isin(_DEFENSIVE_TYPES)].dropna(subset=['x','y']) if 'type' in g.columns else g.iloc[0:0]
        if len(defensive_events)>=6:
            from matplotlib.colors import LinearSegmentedColormap as _Ramp
            defend_cmap=_Ramp.from_list('player_defend',
                ([(0,'#6b2447'),(1,defend_colour)] if is_dark else
                 [(0,'#f9c4dc'),(1,defend_colour)]))
            dx,dy=defensive_events.x.to_numpy()*1.05,defensive_events.y.to_numpy()*.68
            gx,gy=np.mgrid[0:105:45j,0:68:30j]
            dz=_kde(np.vstack([dx,dy]),bw_method=.26)(np.vstack([gx.ravel(),gy.ravel()])).reshape(gx.shape)
            dpeak=float(np.nanmax(dz))
            if dpeak>0:
                # Lighter and in fewer steps than the touch surface. Eight
                # defensive actions against a hundred and seventeen touches is
                # the ratio the page should carry, and at equal weight the
                # smaller number was the louder mark.
                ax.contourf(gx,gy,dz,levels=np.linspace(dpeak*.15,dpeak,5),
                            cmap=defend_cmap,alpha=.36,zorder=2,antialiased=True)
        if len(defensive_events):
            heights=pd.to_numeric(defensive_events.x,errors='coerce').dropna()*1.05
            if len(heights):
                middle=float(heights.mean())
                spread=float(heights.std(ddof=0) or 0)
                # A band at the mean plus or minus a standard deviation was
                # the largest object on the map, and it stood for one figure.
                # The line is that figure; the band was decoration around it.
                ax.axvline(middle,color=defend_colour,lw=1.5,ls=(0,(5,3)),alpha=.95,zorder=4)
        # Hollow, not filled. Every touch was drawn again as an opaque dot over
        # the density that was made from those same touches, so the heat was
        # visible everywhere except where there was enough of it to matter. A
        # ring locates the touch and lets the surface through.
        ax.scatter(touches.x*1.05,touches.y*.68,s=22,facecolor='none',
                   edgecolors=point_color,linewidths=.6,alpha=.65,zorder=3)
        # One mark for defending, in the colour of the defensive density it
        # sits on, rather than eight markers in eight colours.
        #
        # The eight were a colour each for tackles, interceptions, recoveries,
        # clearances, blocks, aerials, challenges and fouls, and they cost more
        # than they returned: a key of eight entries beside a small pitch, a
        # sixth colour language on a card that already has the kit, the touch
        # ramp, the accent and the page, and -- because the split was by event
        # name -- one of the eight named a type this feed does not emit while
        # the type it does emit was missing. What a reader wants from the map
        # is where he defended, which one mark answers; which kind of action it
        # was is in the table two inches below.
        defensive_events=g[g.type.isin(_DEFENSIVE_TYPES)].dropna(subset=['x','y']) if 'type' in g.columns else g.iloc[0:0]
        if len(defensive_events):
            ax.scatter(defensive_events.x*1.05,defensive_events.y*.68,s=46,marker='D',
                       color=defend_colour,edgecolors=marker_edge,linewidths=.7,zorder=6)

        # Two weights, not one. Thirty progressive passes drawn in one colour
        # at full strength overlap into a single shape and none of them can be
        # followed. The ones that finished inside the penalty area are the
        # passes a reader is looking for, so they are drawn over the rest in
        # the accent; the others are thin and faint, and are still there to be
        # read as a body of work.
        team_colour=charts.colors[p.team_id]
        progressive_actions=g[progressive_pass_mask(g)].dropna(subset=['end_x','end_y'])
        into_box=(progressive_actions.end_x>=83.5)&progressive_actions.end_y.between(21.0,79.0)
        for action,boxed in zip(progressive_actions.itertuples(),into_box):
            ax.annotate('',(action.end_x*1.05,action.end_y*.68),(action.x*1.05,action.y*.68),
                        arrowprops={'arrowstyle':'-|>',
                                    'color':FOCUS if boxed else quiet_arrow,
                                    'alpha':.95 if boxed else .55,
                                    'lw':1.6 if boxed else .9,
                                    'mutation_scale':11 if boxed else 7},
                        annotation_clip=True,zorder=7 if boxed else 5)
        shots=g[flags(g,'is_shot')];ax.scatter(shots.x*1.05,shots.y*.68,s=88,marker='*',color=FOCUS,edgecolors=marker_edge,linewidths=.6,zorder=8)
        ax.set_title('Where he had the ball, and where he defended',color=FG,size=9.5,pad=12)
        # Under the pitch in one row, not stacked in a box beside it. Boxed on
        # the right it took width the pitch needed and set the reader scanning
        # sideways between a mark and its name; under it, the eye travels the
        # short way and the pitch keeps the column.
        handles=[Line2D([0],[0],marker='s',color='none',markerfacecolor=heat_peak,
                        markeredgecolor='none',label='touch density',markersize=6),
                 Line2D([0],[0],marker='o',color='none',markerfacecolor='none',
                        markeredgecolor=point_color,label='a touch',markersize=5)]
        if len(defensive_events):
            handles.append(Line2D([0],[0],marker='D',color='none',markerfacecolor=defend_colour,
                                  markeredgecolor='none',label='defending',markersize=5))
            handles.append(Line2D([0],[0],color=defend_colour,lw=1.2,ls=(0,(4,3)),
                                  label='defensive height'))
        if len(progressive_actions):
            handles.append(Line2D([0],[0],color=quiet_arrow,lw=1.4,alpha=.75,
                                  label='progressive pass'))
        if int(into_box.sum()):
            handles.append(Line2D([0],[0],color=FOCUS,lw=1.8,label='into the box'))
        if len(shots):
            handles.append(Line2D([0],[0],marker='*',color='none',markerfacecolor=FOCUS,
                                  markeredgecolor='none',label='shot',markersize=8))
        legend=ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.015),
                         fontsize=6.4,frameon=False,ncol=len(handles),
                         handletextpad=.4,columnspacing=1.2)
        for entry in legend.get_texts():
            entry.set_color(MUTED)
        # Four fixed columns held the same twenty-six rows whatever the player
        # did for a living. Measured across a match, an average card spent 9.9
        # of those rows saying nought -- a centre-back was given three lines of
        # take-on statistics and a goalkeeper thirteen empty rows out of
        # twenty-six. Three columns of role-appropriate rows say more in less
        # space, and anything already drawn on the ring is left off so a reader
        # does not meet the same measure twice in two forms.
        sections=_TABLE[p.role_group if p.role_group in _TABLE else 'Unknown']
        legacy=[
            ('ATTACK', [('Goals','goals'),('Shots','shots'),('xG','xG'),('xG / shot','xG_per_shot'),('Inferred xA','xA'),('xG + xA','xG_xA')]),
            ('PASSING', [('Attempts','passes'),('Completed','completed_passes'),('Completion %','pass_pct'),('Progressive passes','progressive_passes'),('Line-breaking passes','line_breaking_passes'),('Progressive share %','progressive_pass_pct'),('xA / 100 passes','xA_per_100_passes')]),
            ('PROGRESSION', [('Touches','touches'),('Progressive carries','progressive_carries'),('Box entries','box_entries'),('Positive xT','positive_xT'),('xT / 100 touches','xT_per_100_touches'),('xGChain','xGChain'),('xGBuildup','xGBuildup')]),
            ('DEFENDING / DUELS', [('Recoveries','recoveries'),('Interceptions','interceptions'),('Tackles won','tackles_won'),('Clearances','clearances'),('Take-ons attempted','takeons'),('Take-ons won','takeons_won'),('Take-on success %','takeon_success_pct')])]
        for i,(heading,metrics) in enumerate(sections):
            panel=fig.add_axes([.065+i*.30,.185,.255,.20],facecolor=BG)
            panel.set_xlim(0,1);panel.set_ylim(0,1);panel.axis('off')
            panel.text(0,1,heading,color=FG,size=9,weight='bold',va='top')
            for j,(caption,key) in enumerate(metrics):
                y=.82-j*.155
                shown=_paired(p,key)
                if shown is None:
                    value=p.get(key,np.nan)
                    shown=('N/A' if pd.isna(value) else format_value(value,digits=2 if key in {'xG','xG_per_shot','xA','xG_xA','positive_xT','xT_per_100_touches','xGChain','xGBuildup','xA_per_100_passes','pass_pct','progressive_pass_pct','takeon_success_pct'} else 0))
                panel.text(0,y,caption,color=MUTED,size=8,va='center')
                panel.text(1,y,shown,color=FG,size=9,weight='bold',ha='right',va='center')
                panel.plot([0,1],[y-.052,y-.052],color=MUTED,lw=.35,alpha=.25)
        definitions=('Raw match totals; unavailable values shown as N/A. The ring compares a player with others in his role when four or more played 30+ minutes, and with every 30+ minute player otherwise; the caption above says which.\n'
                     'pAdj defensive actions: tackles, interceptions, recoveries, clearances, challenges and blocked passes per 100 opponent touches. Defensive height: mean distance upfield of those actions, blank below three of them.\n'
                     'Progression metres: ground gained toward goal by completed passes and carries. Line-breaking passes: passes that started behind the opponent line, estimated over five-minute windows, and finished beyond it. Positive xT: threat added by successful movements.\n'
                     'xGChain: possession chance credit; xGBuildup excludes shooter and key-pass provider, and the credit overlaps across teammates. Progressive share uses completed passes; xT / 100 uses touches.')
        # The method footer is drawn at y=.055 and grows upward, so this block
        # has to finish above it. At .132 and five lines it did not, and the
        # two ran through each other.
        fig.text(.055,.152,definitions,color=MUTED,size=6.9,linespacing=1.45,va='top')
        method='Role and minutes govern comparisons; these are match observations, not a season ability rating. Nominal pitch 105 × 68 m.'
        filename='player_profiles/'+re.sub(r'[^\w.-]+','_',charts.names[p.team_id])+'/'+re.sub(r'[^\w.-]+','_',p.player)+'.png'
        reading=_profile_reading(p,charts.names[p.team_id],strip,radar_keys,vals,
                                 pool,compare,basis,defensive_actions,final_touch,thirds)
        charts.save(fig,filename,p.player+' — advanced role profile',reading+' '+method,method,len(pool))
