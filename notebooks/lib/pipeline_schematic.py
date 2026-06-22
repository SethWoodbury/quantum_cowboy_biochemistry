"""Auto-generate a pipeline schematic figure from a Jupyter notebook.

Parses STEP headers in markdown cells + `### INPUTS ###`/`### OUTPUTS ###`
blocks in the following code cell, then renders a vertical flowchart with
matplotlib: one rounded box per step (number, short title, key I/O vars),
arrows between steps, optional steps drawn dashed/grey. No system deps.

Usage:
    from pipeline_schematic import render_pipeline
    render_pipeline("notebooks/foo.ipynb", out_png="schematic.png", show=True)
"""
from __future__ import annotations
import json
import re
from pathlib import Path

# Header line that names a pipeline step (markdown). e.g. "# **STEP 3: Minimize ...**"
# or an "(optional)" sub-step with no STEP number: "## *(optional)* 2-D Relaxed Scan ..."
_STEP_RE = re.compile(r"^#{1,3}\s+(.*?STEP\s*\d+.*|.*\boptional\b.*)$", re.I)
_STEPNUM_RE = re.compile(r"STEP\s*(\d+)", re.I)
_OPTIONAL_RE = re.compile(r"optional", re.I)
# A var assignment at the start of a (non-comment) line: "relaxed_pdb = ..."
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=")
_IO_BLOCK_RE = re.compile(r"###\s*(INPUTS|OUTPUTS)\s*###(.*?)(?=^\s*###|\Z)", re.S | re.M)


def _clean_title(header: str) -> tuple[str, bool]:
    """Strip markdown noise from a header line; return (short_title, is_optional)."""
    optional = bool(_OPTIONAL_RE.search(header))
    t = header.lstrip("#").strip()
    t = re.sub(r"`[^`]*`", "", t)                       # drop `code` spans (CLI names)
    t = re.sub(r"\*+", "", t)                           # drop bold/italic markers
    t = re.sub(r"\(\s*optional[^)]*\)", "", t, flags=re.I)  # drop "(optional ...)"
    t = re.sub(r"\bEXPERIMENTAL\b|\bOPTIONAL\b", "", t, flags=re.I)
    t = re.sub(r"^STEP\s*\d+\s*[:\-]\s*", "", t, flags=re.I)  # drop "STEP N:" prefix
    t = t.split("—")[0].split("(")[0].split("→")[0]    # keep the lead clause
    t = re.sub(r"\s+", " ", t).strip(" :-,*")
    return (t or header.strip("# ").strip()), optional


def _io_vars(code: str, kind: str) -> list[str]:
    """Return assigned variable names inside the INPUTS or OUTPUTS block of `code`."""
    for m in _IO_BLOCK_RE.finditer(code):
        if m.group(1).upper() == kind:
            names: list[str] = []
            for line in m.group(2).splitlines():
                if line.lstrip().startswith("#"):
                    continue
                am = _ASSIGN_RE.match(line)
                if am and am.group(1) not in names:
                    names.append(am.group(1))
            return names
    return []


def parse_pipeline(nb_path: str | Path) -> list[dict]:
    """Parse a notebook into an ordered list of step dicts.

    Each dict: {num, title, optional, inputs, outputs}. `num` may be None for
    optional sub-steps that carry no STEP number.
    """
    cells = json.loads(Path(nb_path).read_text())["cells"]
    steps: list[dict] = []
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "markdown":
            continue
        first = "".join(cell["source"]).split("\n")[0]
        if not _STEP_RE.match(first):
            continue
        title, optional = _clean_title(first)
        num_m = _STEPNUM_RE.search(first)
        # the step's code is the next code cell before any other step header
        code = ""
        for j in range(i + 1, len(cells)):
            c = cells[j]
            if c["cell_type"] == "code":
                code = "".join(c["source"])
                break
            if c["cell_type"] == "markdown" and _STEP_RE.match(
                "".join(c["source"]).split("\n")[0]
            ):
                break
        steps.append({
            "num": int(num_m.group(1)) if num_m else None,
            "title": title,
            "optional": optional,
            "inputs": _io_vars(code, "INPUTS"),
            "outputs": _io_vars(code, "OUTPUTS"),
        })
    return steps


def _box_text(step: dict) -> str:
    tag = f"STEP {step['num']}" if step["num"] is not None else "OPTIONAL"
    title = step["title"]
    if len(title) > 42:
        title = title[:41].rstrip() + "…"
    lines = [f"{tag}  —  {title}"]
    io = []
    if step["inputs"]:
        io.append("in: " + ", ".join(step["inputs"][:2]))
    if step["outputs"]:
        io.append("out: " + ", ".join(step["outputs"][:2]))
    if io:
        lines.append("   ".join(io))
    return "\n".join(lines)


def render_pipeline(nb_path, out_png=None, show=False, title=None):
    """Parse `nb_path` and render a vertical pipeline flowchart.

    Returns the matplotlib Figure. Saves a PNG if `out_png` is given;
    displays inline if `show=True` (e.g. inside a notebook).
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    steps = parse_pipeline(nb_path)
    if not steps:
        raise ValueError(f"No pipeline steps detected in {nb_path}")

    n = len(steps)
    box_w, box_h, gap = 7.4, 0.92, 0.55
    pitch = box_h + gap
    fig_h = max(4.0, n * pitch + 1.4)
    fig, ax = plt.subplots(figsize=(9.2, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    core_fc, core_ec = "#dbe9f6", "#2c6fb5"     # solid blue: core steps
    opt_fc, opt_ec = "#f0f0f0", "#9a9a9a"       # grey: optional steps
    cx = 5.0
    top = fig_h - 0.7
    centers = []
    for k, step in enumerate(steps):
        cy = top - k * pitch - box_h / 2
        centers.append(cy)
        opt = step["optional"]
        box = FancyBboxPatch(
            (cx - box_w / 2, cy - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.6, edgecolor=opt_ec if opt else core_ec,
            facecolor=opt_fc if opt else core_fc,
            linestyle="--" if opt else "-", zorder=2,
        )
        ax.add_patch(box)
        txt = _box_text(step)
        head, *rest = txt.split("\n")
        ax.text(cx, cy + (0.14 if rest else 0), head, ha="center",
                va="center", fontsize=9.5, fontweight="bold",
                color="#555" if opt else "#143a63", zorder=3)
        if rest:
            ax.text(cx, cy - 0.20, rest[0], ha="center", va="center",
                    fontsize=7.6, color="#666", family="monospace", zorder=3)

    # arrows between consecutive boxes
    for k in range(n - 1):
        opt = steps[k + 1]["optional"] or steps[k]["optional"]
        ax.add_patch(FancyArrowPatch(
            (cx, centers[k] - box_h / 2), (cx, centers[k + 1] + box_h / 2),
            arrowstyle="-|>", mutation_scale=14, lw=1.4,
            color=opt_ec if opt else core_ec,
            linestyle="--" if opt else "-", zorder=1,
        ))

    ttl = title or f"Pipeline: {Path(nb_path).stem}"
    ax.set_title(ttl, fontsize=12.5, fontweight="bold", pad=12)
    # legend
    ax.text(0.55, 0.55, "■ core", color=core_ec, fontsize=9, fontweight="bold")
    ax.text(2.1, 0.55, "▭ optional", color=opt_ec, fontsize=9, fontweight="bold")
    fig.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    import sys
    nb = sys.argv[1] if len(sys.argv) > 1 else None
    out = sys.argv[2] if len(sys.argv) > 2 else "pipeline_schematic.png"
    if nb:
        render_pipeline(nb, out_png=out)
        print("wrote", out)
