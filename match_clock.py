"""One convention for displayed event timestamps: elapsed mm:ss, never +1."""
import math

def event_label(row):
    def number(key):
        try:
            n=float(row.get(key,0));return n if math.isfinite(n) else 0
        except (TypeError,ValueError):return 0
    return f'{int(number("minute")):02d}:{int(number("second")):02d}'
