"""Plot line families beta·k=t inside [0,1]^2 and save as SVG.

Pure-stdlib implementation so it runs in the current WSL environment.
"""

from __future__ import annotations

import math
import os
from typing import Iterable


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models", "beta_k_equals_t_unit_square_compare.svg")


def _support_bounds(beta: tuple[int, int]) -> tuple[float, float]:
    b1, b2 = beta
    vals = (0.0, float(b1), float(b2), float(b1 + b2))
    return min(vals), max(vals)


def _line_segment_in_unit_square(beta: tuple[int, int], t: float, tol: float = 1.0e-9):
    b1, b2 = float(beta[0]), float(beta[1])
    points: list[tuple[float, float]] = []

    def _add_point(x: float, y: float) -> None:
        if -tol <= x <= 1.0 + tol and -tol <= y <= 1.0 + tol:
            x_clip = min(1.0, max(0.0, x))
            y_clip = min(1.0, max(0.0, y))
            pt = (x_clip, y_clip)
            for qx, qy in points:
                if abs(qx - pt[0]) <= 5.0e-8 and abs(qy - pt[1]) <= 5.0e-8:
                    return
            points.append(pt)

    if abs(b2) > tol:
        _add_point(0.0, t / b2)
        _add_point(1.0, (t - b1) / b2)
    if abs(b1) > tol:
        _add_point(t / b1, 0.0)
        _add_point((t - b2) / b1, 1.0)

    if len(points) < 2:
        return None

    best_pair = None
    best_dist = -1.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            x0, y0 = points[i]
            x1, y1 = points[j]
            dist = math.hypot(x1 - x0, y1 - y0)
            if dist > best_dist:
                best_dist = dist
                best_pair = ((x0, y0), (x1, y1))
    return best_pair


def _sample_line_values(beta: tuple[int, int], count: int = 7) -> list[float]:
    support_lo, support_hi = _support_bounds(beta)
    vals: list[float] = []
    for idx in range(count):
        frac = 0.10 + 0.80 * (idx / max(1, count - 1))
        vals.append(support_lo + frac * (support_hi - support_lo))
    return vals


def _svg_line(x0: float, y0: float, x1: float, y1: float, *, stroke: str, width: float, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
        f'stroke="{stroke}" stroke-width="{width:.2f}"{dash_attr} />'
    )


def _svg_polygon(points: list[tuple[float, float]], *, fill: str, stroke: str = "none", width: float = 0.0) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    stroke_attr = f' stroke="{stroke}" stroke-width="{width:.2f}"' if stroke != "none" else ""
    return f'<polygon points="{pts}" fill="{fill}"{stroke_attr} />'


def _svg_circle(cx: float, cy: float, r: float, *, fill: str, stroke: str = "none", width: float = 0.0) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="{width:.2f}"' if stroke != "none" else ""
    return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}"{stroke_attr} />'


def _svg_text(x: float, y: float, text: str, *, size: int = 14, anchor: str = "middle", fill: str = "#111") -> str:
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}">{safe}</text>'


def _svg_arrow(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    stroke: str,
    width: float,
    head_len: float = 10.0,
    head_half_width: float = 4.5,
) -> str:
    dx = x1 - x0
    dy = y1 - y0
    norm = math.hypot(dx, dy)
    if norm <= 1.0e-12:
        return ""
    ux = dx / norm
    uy = dy / norm
    px = -uy
    py = ux
    bx = x1 - head_len * ux
    by = y1 - head_len * uy
    head = [
        (x1, y1),
        (bx + head_half_width * px, by + head_half_width * py),
        (bx - head_half_width * px, by - head_half_width * py),
    ]
    return "\n".join(
        [
            _svg_line(x0, y0, bx, by, stroke=stroke, width=width),
            _svg_polygon(head, fill=stroke),
        ]
    )


def _panel_to_svg(beta: tuple[int, int], title: str, panel_x: float, panel_y: float, panel_w: float, panel_h: float) -> str:
    margin = 34.0
    x0 = panel_x + margin
    y0 = panel_y + margin
    plot_w = panel_w - 2.0 * margin
    plot_h = panel_h - 2.0 * margin - 26.0
    plot_y = y0 + 26.0

    def px(x: float) -> float:
        return x0 + x * plot_w

    def py(y: float) -> float:
        return plot_y + (1.0 - y) * plot_h

    parts: list[str] = []
    parts.append(f'<rect x="{panel_x:.2f}" y="{panel_y:.2f}" width="{panel_w:.2f}" height="{panel_h:.2f}" fill="white" stroke="#d0d0d0" />')
    parts.append(_svg_text(panel_x + panel_w / 2.0, panel_y + 18.0, title, size=15))

    support_lo, support_hi = _support_bounds(beta)
    slope = -float(beta[0]) / float(beta[1])
    meta = f"Aβ={support_lo:.0f}, Bβ={support_hi:.0f}, slope={slope:.5f}"
    parts.append(_svg_text(panel_x + panel_w / 2.0, panel_y + 34.0, meta, size=11, fill="#444"))

    # square
    parts.append(f'<rect x="{px(0):.2f}" y="{py(1):.2f}" width="{plot_w:.2f}" height="{plot_h:.2f}" fill="none" stroke="black" stroke-width="1.4" />')

    # grid
    for tick in (0.0, 0.5, 1.0):
        gx = px(tick)
        gy = py(tick)
        parts.append(_svg_line(gx, py(0.0), gx, py(1.0), stroke="#e0e0e0", width=0.8, dash="4 4"))
        parts.append(_svg_line(px(0.0), gy, px(1.0), gy, stroke="#e0e0e0", width=0.8, dash="4 4"))
        parts.append(_svg_text(gx, py(0.0) + 16.0, f"{tick:g}", size=10, fill="#555"))
        parts.append(_svg_text(px(0.0) - 12.0, gy + 3.0, f"{tick:g}", size=10, anchor="end", fill="#555"))

    # beta normal-direction arrow at upper-right corner
    beta_norm = math.hypot(float(beta[0]), float(beta[1]))
    ux = float(beta[0]) / beta_norm
    uy = float(beta[1]) / beta_norm
    anchor_x = px(0.84)
    anchor_y = py(0.88)
    arrow_len = 48.0
    end_x = anchor_x + arrow_len * ux
    end_y = anchor_y - arrow_len * uy
    parts.append(_svg_arrow(anchor_x, anchor_y, end_x, end_y, stroke="#222222", width=1.8, head_len=9.0, head_half_width=4.0))
    parts.append(_svg_text(anchor_x - 6.0, anchor_y - 8.0, "β", size=12, anchor="end", fill="#222222"))

    # family lines
    for t in _sample_line_values(beta):
        seg = _line_segment_in_unit_square(beta, t)
        if seg is None:
            continue
        (sx0, sy0), (sx1, sy1) = seg
        parts.append(_svg_line(px(sx0), py(sy0), px(sx1), py(sy1), stroke="#4C72B0", width=1.2))

    sampling_t = support_lo + 0.5
    same_special_line = abs(sampling_t - 0.5) <= 1.0e-12

    if same_special_line:
        seg = _line_segment_in_unit_square(beta, 0.5)
        if seg is not None:
            (sx0, sy0), (sx1, sy1) = seg
            x0p, y0p, x1p, y1p = px(sx0), py(sy0), px(sx1), py(sy1)
            parts.append(_svg_line(x0p, y0p, x1p, y1p, stroke="#8172B2", width=3.2))
            parts.append(_svg_circle(x0p, y0p, 3.6, fill="#8172B2"))
            parts.append(_svg_circle(x1p, y1p, 3.6, fill="#8172B2"))
            label_x = min(x0p, x1p) + 16.0
            label_y = min(y0p, y1p) + 18.0
            parts.append(_svg_text(label_x, label_y, "t=0.5 = Aβ+0.5", size=10, anchor="start", fill="#8172B2"))
    else:
        # t=0.5
        seg = _line_segment_in_unit_square(beta, 0.5)
        if seg is not None:
            (sx0, sy0), (sx1, sy1) = seg
            parts.append(_svg_line(px(sx0), py(sy0), px(sx1), py(sy1), stroke="#C44E52", width=2.0))

        # sampling t = A_beta + 0.5
        seg = _line_segment_in_unit_square(beta, sampling_t)
        if seg is not None:
            (sx0, sy0), (sx1, sy1) = seg
            parts.append(_svg_line(px(sx0), py(sy0), px(sx1), py(sy1), stroke="#55A868", width=1.8, dash="6 4"))

    # mini legend
    leg_x = px(0.05)
    leg_y = py(0.96)
    parts.append(_svg_line(leg_x, leg_y, leg_x + 18, leg_y, stroke="#4C72B0", width=1.2))
    parts.append(_svg_text(leg_x + 24, leg_y + 4, "line family", size=10, anchor="start"))
    if same_special_line:
        parts.append(_svg_line(leg_x, leg_y + 16, leg_x + 18, leg_y + 16, stroke="#8172B2", width=3.0))
        parts.append(_svg_text(leg_x + 24, leg_y + 20, "t=0.5=Aβ+0.5", size=10, anchor="start", fill="#8172B2"))
    else:
        parts.append(_svg_line(leg_x, leg_y + 16, leg_x + 18, leg_y + 16, stroke="#C44E52", width=2.0))
        parts.append(_svg_text(leg_x + 24, leg_y + 20, "t=0.5", size=10, anchor="start"))
        parts.append(_svg_line(leg_x, leg_y + 32, leg_x + 18, leg_y + 32, stroke="#55A868", width=1.8, dash="6 4"))
        parts.append(_svg_text(leg_x + 24, leg_y + 36, "t=Aβ+0.5", size=10, anchor="start"))

    return "\n".join(parts)


def _build_svg() -> str:
    panel_w = 470.0
    panel_h = 430.0
    left = 18.0
    top = 58.0
    gap = 12.0

    betas: list[tuple[tuple[int, int], str]] = [
        ((1, 128), "β=(1,128)"),
        ((127, 128), "β=(127,128)"),
        ((-1, 128), "β=(-1,128)"),
        ((1, -128), "β=(1,-128)"),
        ((25, 128), "β=(25,128)"),
        ((128, -25), "β=(128,-25)"),
    ]

    num_panels = len(betas)
    cols = min(3, num_panels)
    rows = (num_panels + cols - 1) // cols
    width = int(left * 2 + cols * panel_w + (cols - 1) * gap)
    height = int(top + rows * panel_h + (rows - 1) * gap + 24)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        _svg_text(width / 2.0, 28.0, "Line families β·k=t clipped to [0,1]^2", size=22),
        _svg_text(
            width / 2.0,
            48.0,
            "蓝色：等间隔示意线族；红色：t=0.5；绿色虚线：t=Aβ+0.5（sampling 起始线）",
            size=12,
            fill="#444",
        ),
    ]

    for idx, (beta, title) in enumerate(betas):
        row = idx // cols
        col = idx % cols
        panel_x = left + col * (panel_w + gap)
        panel_y = top + row * (panel_h + gap)
        parts.append(_panel_to_svg(beta, title, panel_x, panel_y, panel_w, panel_h))

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    svg = _build_svg()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
