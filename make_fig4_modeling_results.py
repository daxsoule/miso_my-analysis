"""Generate Fig 4 — Modeling Results (FISHES initial permeability) for poster."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import textwrap
from pathlib import Path

# --- Font configuration ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

IMAGE_PATH = Path("images/models/model_ini_permeability_vel_vec_label.tif")
OUTPUT = Path("outputs/figures/poster/fig4_modeling_results.png")
DPI = 600
CAPTION_FONTSIZE = 22


def add_caption_justified(fig, ax, renderer, ax_bbox, text, caption_fontsize):
    """Render justified caption text in the given axes."""
    caption_width_in = ax_bbox.width / fig.dpi
    char_width_in = caption_fontsize / 72 * 0.55
    wrap_chars = int(caption_width_in / char_width_in)
    lines = textwrap.wrap(text, width=wrap_chars)

    # Measure line height
    sample = ax.text(0, 0, "Tg", fontsize=caption_fontsize, family='sans-serif',
                     transform=ax.transAxes)
    sample_bbox = sample.get_window_extent(renderer)
    line_height = (sample_bbox.height * 1.35) / ax_bbox.height
    sample.remove()

    for i, line in enumerate(lines):
        y = 1.0 - i * line_height
        words = line.split()

        if i < len(lines) - 1 and len(words) > 1:
            word_widths = []
            for word in words:
                t = ax.text(0, 0, word, fontsize=caption_fontsize, family='sans-serif',
                            transform=ax.transAxes)
                wb = t.get_window_extent(renderer)
                word_widths.append(wb.width / ax_bbox.width)
                t.remove()

            total_word_width = sum(word_widths)
            remaining = 1.0 - total_word_width
            gap = remaining / (len(words) - 1)

            x = 0.0
            for j, word in enumerate(words):
                ax.text(x, y, word, fontsize=caption_fontsize, family='sans-serif',
                        transform=ax.transAxes, va='top', ha='left')
                x += word_widths[j] + gap
        else:
            ax.text(0.0, y, line, fontsize=caption_fontsize, family='sans-serif',
                    transform=ax.transAxes, va='top', ha='left')


def main():
    img = mpimg.imread(IMAGE_PATH)

    fig = plt.figure(figsize=(8.94, 12.08), dpi=DPI)

    # Title
    fig.text(0.5, 0.97, "Modeling Results", fontsize=24, fontweight='bold',
             ha='center', va='top', family='sans-serif')

    # Image area — aspect='auto' to fill axes width
    img_ax = fig.add_axes([0.005, 0.42, 0.99, 0.53])
    img_ax.imshow(img, aspect='auto')
    img_ax.axis('off')

    # Caption area — fixed width matching Fig 5 (6.94 inches)
    fig_w = fig.get_size_inches()[0]
    caption_w_in = 6.94  # same pixel width as Fig 5 caption
    inset_frac = (fig_w - caption_w_in) / 2 / fig_w
    caption_ax = fig.add_axes([inset_frac, 0.01, caption_w_in / fig_w, 0.40])
    caption_ax.axis('off')

    renderer = fig.canvas.get_renderer()
    ax_bbox = caption_ax.get_window_extent(renderer)

    caption = (
        "Initial crustal permeability for the FISHES 2-D hydrothermal "
        "circulation model (Lewis and Lowell, 2009). Layers 2A and 2B "
        "follow the seismic-velocity structure beneath the International "
        "District (Arnulf et al., 2018). Basal temperature decreases from "
        "400 \u00b0C (left) to 300 \u00b0C (right). High-permeability upflow zones "
        "(UF-2A, UF-2B) are bounded by a low-permeability barrier. Arrows "
        "indicate flow velocity 15 years after the start of the baseline "
        "simulation."
    )

    add_caption_justified(fig, caption_ax, renderer, ax_bbox, caption, CAPTION_FONTSIZE)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=DPI, pad_inches=0.0, facecolor='white')
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
