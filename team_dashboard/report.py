"""PDF report generation for team dashboard visuals."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def build_pdf_report(visual_paths: list[str], output_path: str = "output/team_report.pdf") -> str:
    """Embed all generated visuals into a single PDF report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for visual in visual_paths:
            img = plt.imread(visual)
            fig, ax = plt.subplots(figsize=(11.69, 8.27), facecolor="white")
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight", facecolor="white")
            plt.close(fig)
    return str(path)

