"""Publishing names, units and limitations shared by every output."""
from dataclasses import dataclass, asdict
import math

METRIC_VERSION = "2.2.0"

@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    unit: str
    definition: str
    model: str = "event-derived"
    lower_is_better: bool = False

SPECS = [
    MetricSpec('xGChain','xGChain','goals','Non-penalty possession xG credited to every participant; overlapping credits are not additive.'),
    MetricSpec('xGBuildup','xGBuildup','goals','Non-penalty possession xG credit excluding the shooter and key-pass provider.'),
    MetricSpec('xT_per_100_touches','xT / 100 touches','model units','Positive successful-movement xT / recorded touches × 100.'),
    MetricSpec('progressive_pass_pct','Progressive pass share','%','Progressive passes / completed passes × 100.'),
    MetricSpec('xA','Inferred xA','goals','Shot xG credited to a successful key pass in the same possession, within 15 seconds.'),
    MetricSpec('xA_per_100_passes','xA / 100 passes','goals','Inferred xA / attempted passes × 100.'),
    MetricSpec("xG", "Expected goals", "goals", "Chance quality before the shot; provider or versioned local model."),
    MetricSpec("xGoT", "Local post-shot estimate", "goals", "Placement-weighted pre-shot xG; uncalibrated, excludes shot velocity and actual keeper position.", "heuristic"),
    MetricSpec("field_tilt", "Field tilt", "%", "Share of both teams' completed passes ending in the final third."),
    MetricSpec("pitch_control", "Time-weighted touch influence", "%", "Duration-weighted short touch-position windows, split at substitutions/cards; includes contested space. Not tracking-based control.", "heuristic-v2"),
    MetricSpec("possession_share", "Possession share", "%", "Controlled-possession share under the local event model."),
    MetricSpec("ppda", "PPDA", "passes/action", "Opponent passes per defensive action in the defined pressing zone.", lower_is_better=True),
    MetricSpec("sequence_xT", "Positive sequence xT", "model units", "Sum of positive movement values; not net expected goals or additive player credit.", "local zone model"),
    MetricSpec("rest_defence_vulnerability", "Danger after advanced losses", "%", "Advanced open-play losses followed within 12 seconds by a dangerous counter.", lower_is_better=True),
    MetricSpec("shots", "Shots", "count", "Shot events excluding shootouts."),
    MetricSpec("big_chances", "Big chances", "count", "Source-tagged big chances."),
    MetricSpec("final_third_entries", "Final-third entries", "count", "Successful controlled movements crossing into the final third."),
    MetricSpec("box_entries", "Box entries", "count", "Successful controlled movements into the box from outside."),
]
METRICS = {s.key: s for s in SPECS}

def label(key):
    return METRICS[key].label if key in METRICS else key.replace("_", " ").capitalize()

def format_value(value, unit="", digits=1):
    try:
        n = float(value)
        if not math.isfinite(n):
            return "Unavailable"
    except (TypeError, ValueError):
        return "Unavailable"
    return f"{n:.{digits}f}{unit}"

def definitions():
    return [asdict(s) for s in SPECS]
