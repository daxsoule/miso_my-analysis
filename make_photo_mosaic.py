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
    'font.sans-serif': ['Helvetica', 'Arial'],
})

TITLE_SIZE = 24
CAPTION_SIZE = 18
LABEL_SIZE = 20
DPI = 600
BORDER_COLOR = 'steelblue'
BORDER_WIDTH = 4

# Source images — mapped to panels (a)-(d)
IMG_DIR = Path(__file__).parent / 'images'
PHOTOS = [
    IMG_DIR / 'sulis_20250822044456_J2-1730_miso.jpg',   # (a)
    IMG_DIR / 'HOBO_Recover.png',                         # (b)
    IMG_DIR / 'MISO_001_Install.png',                     # (c)
    IMG_DIR / 'sulis_20250821221931_J2-1729_miso.png',    # (d)
]
LABELS = ['(a)', '(b)', '(c)', '(d)']

OUTPUT = Path(__file__).parent / 'outputs' / 'figures' / 'poster' / 'fig0_photo_mosaic.png'

CAPTION = (
    "Axial Seamount is among the most intensively monitored "
    "submarine volcanoes in the world. Long-term observations "
    "since temperature time series and investigate mechanisms "
    "driving these variations using a numerical model of "
    "hydrothermal circulation. These images (a-d) show the "
    "recovery of MISO temperature loggers at Axial Seamount."
)


def add_caption_justified(fig, text, y_pos, fontsize=CAPTION_SIZE,
                          x_left=0.075, x_right=0.925):
    """Add fully justified caption using renderer-based word spacing."""
    renderer = fig.canvas.get_renderer()
    max_width = (x_right - x_left) * fig.get_size_inches()[0] * fig.dpi

    words = text.split()
    lines = []
    current_line = []
    current_width = 0

    # Measure space width
    t = fig.text(0, 0, ' ', fontsize=fontsize, visible=False)
    bb = t.get_window_extent(renderer)
    space_width = bb.width
    t.remove()

    for word in words:
        t = fig.text(0, 0, word, fontsize=fontsize, visible=False)
        bb = t.get_window_extent(renderer)
        word_width = bb.width
        t.remove()

        test_width = current_width + word_width + (space_width if current_line else 0)
        if test_width > max_width and current_line:
            lines.append(current_line)
            current_line = [word]
            current_width = word_width
        else:
            current_line.append(word)
            current_width = test_width

    if current_line:
        lines.append(current_line)

    # Render each line
    line_height = fontsize * 1.3 / (fig.get_size_inches()[1] * fig.dpi)
    y = y_pos

    for i, line_words in enumerate(lines):
        is_last = (i == len(lines) - 1)

        if is_last or len(line_words) == 1:
            fig.text(x_left, y, ' '.join(line_words), fontsize=fontsize,
                     va='top', ha='left', fontfamily='sans-serif')
        else:
            # Measure total word width
            total_word_width = 0
            word_widths = []
            for w in line_words:
                t = fig.text(0, 0, w, fontsize=fontsize, visible=False)
                bb = t.get_window_extent(renderer)
                word_widths.append(bb.width)
                total_word_width += bb.width
                t.remove()

            remaining = max_width - total_word_width
            gap = remaining / (len(line_words) - 1) if len(line_words) > 1 else 0
            gap_frac = gap / (fig.get_size_inches()[0] * fig.dpi)

            x = x_left
            for w in line_words:
                t = fig.text(x, y, w, fontsize=fontsize, va='top', ha='left',
                             fontfamily='sans-serif')
                bb = t.get_window_extent(renderer)
                x += bb.width / (fig.get_size_inches()[0] * fig.dpi) + gap_frac

        y -= line_height


def main():
    fig = plt.figure(figsize=(7.5, 8.5))

    # Grid: 2 rows of photos + caption space below
    gs = gridspec.GridSpec(2, 2, hspace=0.02, wspace=0.02,
                           left=0.02, right=0.98, top=0.94, bottom=0.22)

    # Title
    fig.suptitle('Deployed MISO Loggers', fontsize=TITLE_SIZE,
                 fontweight='bold', y=0.975)

    for idx, (photo_path, label) in enumerate(zip(PHOTOS, LABELS)):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])

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

    # Draw canvas so renderer is available for caption
    fig.canvas.draw()
    add_caption_justified(fig, CAPTION, y_pos=0.19, x_left=0.04, x_right=0.96)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == '__main__':
    main()
