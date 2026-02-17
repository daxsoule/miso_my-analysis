"""
Generate Fig 0: Deployed MISO Loggers photo mosaic.

2x2 grid of ROV photos with steel blue border, panel labels (a-d),
and a justified caption below.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

# Poster styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
})

TITLE_SIZE = 24
CAPTION_SIZE = 20
LABEL_SIZE = 20
DPI = 600
BORDER_COLOR = 'steelblue'
BORDER_WIDTH = 4

# Source images — mapped to panels (a)-(b)
IMG_DIR = Path(__file__).parent / 'images'
PHOTOS = [
    IMG_DIR / 'sulis_20250822044456_J2-1730_miso.jpg',   # (a)
    IMG_DIR / 'HOBO_Recover.png',                         # (b)
]
LABELS = ['(a)', '(b)']

OUTPUT = Path(__file__).parent / 'outputs' / 'figures' / 'poster' / 'fig0_photo_mosaic.png'

CAPTION = (
    "These photos show MISO/HOBO loggers deployed at hydrothermal vents "
    "at Axial Seamount. (a) MISO logger 2023-004 deployed at the main "
    "sulfide structure in the CASM vent field (b) MISO 2017-006 recovered "
    "at El Guapo 2023."
)


def add_caption_justified(fig, caption_ax, text, fontsize=CAPTION_SIZE):
    """Add fully justified caption using renderer-based word spacing in axes coords."""
    import textwrap

    renderer = fig.canvas.get_renderer()
    ax_bbox = caption_ax.get_window_extent(renderer)

    caption_width_in = ax_bbox.width / fig.dpi
    char_width_in = fontsize / 72 * 0.50
    wrap_chars = int(caption_width_in / char_width_in)
    lines = textwrap.wrap(text, width=wrap_chars)

    # Measure line height
    sample = caption_ax.text(0, 0, "Tg", fontsize=fontsize, family='sans-serif',
                             transform=caption_ax.transAxes)
    sample_bbox = sample.get_window_extent(renderer)
    line_height = (sample_bbox.height * 1.35) / ax_bbox.height
    sample.remove()

    for i, line in enumerate(lines):
        y = 1.0 - i * line_height
        words = line.split()

        if i < len(lines) - 1 and len(words) > 1:
            word_widths = []
            for word in words:
                t = caption_ax.text(0, 0, word, fontsize=fontsize, family='sans-serif',
                                    transform=caption_ax.transAxes)
                wb = t.get_window_extent(renderer)
                word_widths.append(wb.width / ax_bbox.width)
                t.remove()

            total_word_width = sum(word_widths)
            remaining = 1.0 - total_word_width
            gap = remaining / (len(words) - 1)

            x = 0.0
            for j, word in enumerate(words):
                caption_ax.text(x, y, word, fontsize=fontsize, family='sans-serif',
                                transform=caption_ax.transAxes, va='top', ha='left')
                x += word_widths[j] + gap
        else:
            caption_ax.text(0.0, y, line, fontsize=fontsize, family='sans-serif',
                            transform=caption_ax.transAxes, va='top', ha='left')


def main():
    fig = plt.figure(figsize=(8.625, 6.325))

    # Grid: 1 row of 2 photos + caption space below
    gs = gridspec.GridSpec(1, 2, wspace=0.02,
                           left=0.02, right=0.98, top=0.92, bottom=0.28)

    # Title
    fig.suptitle('Deployed MISO Loggers', fontsize=TITLE_SIZE,
                 fontweight='bold', y=0.975)

    for idx, (photo_path, label) in enumerate(zip(PHOTOS, LABELS)):
        ax = fig.add_subplot(gs[0, idx])

        img = mpimg.imread(str(photo_path))
        # Center-crop to ~4:3 aspect ratio for uniform panels
        h, w = img.shape[:2]
        target_ratio = 4 / 3
        current_ratio = w / h
        if current_ratio > target_ratio:
            # Too wide — crop sides
            new_w = int(h * target_ratio)
            offset = (w - new_w) // 2
            img = img[:, offset:offset + new_w]
        elif current_ratio < target_ratio:
            # Too tall — crop top/bottom
            new_h = int(w / target_ratio)
            offset = (h - new_h) // 2
            img = img[offset:offset + new_h, :]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        # Steel blue border
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_COLOR)
            spine.set_linewidth(BORDER_WIDTH)

        # Panel label
        ax.text(0.04, 0.95, label, transform=ax.transAxes,
                fontsize=LABEL_SIZE, fontweight='bold', color='white',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                          alpha=0.6, edgecolor='none'))

    # Caption area
    caption_ax = fig.add_axes([0.04, 0.005, 0.92, 0.24])
    caption_ax.axis('off')

    fig.canvas.draw()
    add_caption_justified(fig, caption_ax, CAPTION)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == '__main__':
    main()
