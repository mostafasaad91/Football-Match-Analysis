"""
Match_Analysis_Complete.py
═════════════════════════════════════════════════════════════════════════════
Single entry-point script — bundles everything:

    • All original Match_Analysis_Light analyses (the 39 visuals + PDFs)
    • the four new upgrades:
        1. PPDA for both teams
        2. Goals log with assist providers and classification (Open Play / Set Piece)
        3. Full per-player stats + a table per team
        4. Unified PDF report + every visual as a standalone PNG

Run this file directly:
    python Match_Analysis_Complete.py

Output:
  ── Original outputs (from Match_Analysis_Light) ──
    SAVE_DIR/<figs 1..39>.png        ← the 39 original visuals
    SAVE_DIR/<original PDFs>         ← original tactical reports
    SAVE_DIR/events_<ts>.csv
    SAVE_DIR/players_<ts>.csv
    SAVE_DIR/xg_<ts>.csv

  ── new outputs (extension pipeline) ──
    output/match_report_<ts>.pdf     ← the new unified report
    output/goals_log_<ts>.csv
    output/visuals/ppda_gauge.png
    output/visuals/player_stats_home.png
    output/visuals/player_stats_away.png

Note: Match_Analysis_Light.py is unchanged — it is already wired
to match_extensions at the end of main() from a previous change, so this call
runs the full pipeline end-to-end.
"""

from __future__ import annotations

import os
import sys
import traceback

# Make sure the project dir is on sys.path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console

console = Console()


def main() -> None:
    """Runs Match_Analysis_Light.main() which produces every output."""
    console.print(
        "[bold cyan]── Match_Analysis_Complete ──[/bold cyan]\n"
        "[dim]Running the full Light pipeline (39 visuals + original PDFs)\n"
        " then the extended analysis (PPDA, goals log, player stats PDF).[/dim]\n"
    )

    try:
        # Import and run the original Light.main().
        # Light itself calls _run_extended_analysis at the end of main(),
        # so the new upgrades run automatically after the 39 visuals.
        from Match_Analysis_Light import main as _light_main
        _light_main()

        console.print(
            "\n[bold green]✓ All outputs generated successfully:[/bold green]\n"
            "  [green]• 39 visuals + original PDFs → SAVE_DIR[/green]\n"
            "  [green]• the new unified report → output/[/green]\n"
            "  [green]• the new PNGs → output/visuals/[/green]"
        )

    except Exception as e:
        console.print(f"\n[bold red]✗ Run failed: {e}[/bold red]")
        console.print("[dim]Traceback:[/dim]")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
