"""Generate Fig 6 — Conclusions text figure for poster."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap
from pathlib import Path

# --- Font configuration ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

OUTPUT = Path("outputs/figures/poster/fig6_conclusions.png")
DPI = 600

BULLETS = [
    "Predictions from a numerical model are used to interpret observed "
    "post-eruption temperature declines at high-temperature vents as the "
    "response of Axial\u2019s hydrothermal system to eruption-induced changes in "
    "crustal permeability and/or to alterations in subseafloor hydrothermal "
    "fluid pathways caused by dike intrusion.",

    "Breaches in the upflow-zone barrier produce more rapid temperature "
    "changes than either a global reduction in permeability or an adjacent "
    "dike intrusion.",

    "Subsequent recovery of vent temperatures may result from the sealing of "
    "upflow-zone breaches through mineral precipitation, from an increase in "
    "global permeability as the volcano reinflates, or from convective heat "
    "transfer from the cooling dike.",
]


def add_justified_bullet(fig, ax, renderer, ax_bbox, text, y_start, fontsize=22,
                         bullet_char="\u2022"):
    """Render a single bold, justified bullet paragraph. Returns y after last line."""
    # Measure line height
    sample = ax.text(0, 0, "Tg", fontsize=fontsize, family='sans-serif',
                     fontweight='bold', transform=ax.transAxes)
    sb = sample.get_window_extent(renderer)
    line_height = (sb.height * 1.35) / ax_bbox.height
    sample.remove()

    # Measure bullet+space width
    bt = ax.text(0, 0, bullet_char + "\u2009", fontsize=fontsize, family='sans-serif',
                 fontweight='bold', transform=ax.transAxes)
    bb = bt.get_window_extent(renderer)
    bullet_width = bb.width / ax_bbox.width
    bt.remove()

    # Wrap text to fit available width
    text_width_in = (1.0 - bullet_width) * (ax_bbox.width / fig.dpi)
    char_width_in = fontsize / 72 * 0.58  # bold chars are wider
    wrap_chars = int(text_width_in / char_width_in)
    lines = textwrap.wrap(text, width=wrap_chars)

    y = y_start
    for i, line in enumerate(lines):
        text_x = bullet_width

        if i == 0:
            ax.text(0.0, y, bullet_char, fontsize=fontsize, family='sans-serif',
                    fontweight='bold', transform=ax.transAxes, va='top', ha='left')

        words = line.split()
        is_last_line = (i == len(lines) - 1)

        if not is_last_line and len(words) > 1:
            # Justified: measure each word, distribute gaps evenly
            word_widths = []
            for word in words:
                t = ax.text(0, 0, word, fontsize=fontsize, family='sans-serif',
                            fontweight='bold', transform=ax.transAxes)
                wb = t.get_window_extent(renderer)
                word_widths.append(wb.width / ax_bbox.width)
                t.remove()

            total_word_w = sum(word_widths)
            available = 1.0 - text_x
            remaining = available - total_word_w
            gap = remaining / (len(words) - 1)

            # Safety: if gap is negative or tiny, fall back to left-aligned
            if gap < 0.002:
                ax.text(text_x, y, line, fontsize=fontsize, family='sans-serif',
                        fontweight='bold', transform=ax.transAxes, va='top', ha='left')
            else:
                x = text_x
                for j, word in enumerate(words):
                    ax.text(x, y, word, fontsize=fontsize, family='sans-serif',
                            fontweight='bold', transform=ax.transAxes, va='top', ha='left')
                    x += word_widths[j] + gap
        else:
            # Last line: left-aligned
            ax.text(text_x, y, line, fontsize=fontsize, family='sans-serif',
                    fontweight='bold', transform=ax.transAxes, va='top', ha='left')

        y -= line_height

    return y


def main():
    fig = plt.figure(figsize=(10.5, 7.5), dpi=DPI)

    # Title
    fig.text(0.5, 0.95, "Conclusions", fontsize=24, fontweight='bold',
             ha='center', va='top', family='sans-serif')

    # Text area — wide margins for readability
    # Text width matches Fig 5 caption (6.94 inches)
    fig_w = fig.get_size_inches()[0]
    text_w = 6.94 / fig_w
    text_left = (1.0 - text_w) / 2
    text_ax = fig.add_axes([text_left, 0.05, text_w, 0.85])
    text_ax.axis('off')

    renderer = fig.canvas.get_renderer()
    ax_bbox = text_ax.get_window_extent(renderer)

    y = 1.0
    for bullet_text in BULLETS:
        y = add_justified_bullet(fig, text_ax, renderer, ax_bbox, bullet_text, y,
                                 fontsize=22)
        y -= 0.03  # gap between bullets

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
