"""Render the frame with its coordinate system visible, so it can be read not guessed.

ACTION6 is the most-used action in the recorded corpus — 165 of 596 decisions — and
it takes grid coordinates. The model is shown a 512x512 image and must submit
0..63 grid indices, so it eyeballs the mapping. The corpus shows what that
produces: 16% of submissions land on the diagonal x==y against a 1.6% chance rate,
25% have both coordinates as multiples of four against 6%. Models snap to round
numbers and to the middle. A play-along probe reached the same conclusion from the
inside and built its own ASCII renderer with row and column rulers, saying neither
the integer arrays nor the PNG were "practical for precise spatial reasoning".

Rulers are labelled in **grid** coordinates because that is what the action takes:
x is the column, y is the row, both 0-based, matching ``action_requirements``.

This is interface cost, not task difficulty. It says nothing about what any object
is, where the goal lies, or which cells matter — only where the coordinate system
is, which the model is already required to use.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from llm_repl_agent.observations import pixels_to_png


MARGIN = 24
# Cell size chosen so a 64-cell grid plus the ruler margin stays inside 512 pixels:
# 64*7 + 24 = 472. Image tokens are billed by tiled resolution rather than file size,
# and most providers tile at 512, so 536 would have crossed a boundary and could cost
# four tiles where one will do. The plain 8x render remains on disk unchanged.
CELL_PIXELS = 7
LABEL_EVERY = 5
GRIDLINE_EVERY = 10
RULER_BACKGROUND = (24, 24, 24)
RULER_INK = (235, 235, 235)
GRIDLINE_BLEND = 0.28


def annotation_descriptor(scale: int = CELL_PIXELS) -> dict[str, object]:
    """What the annotation is, for the model-visible contract."""

    return {
        "rulers": "grid coordinates on the top and left margins",
        "label_every": LABEL_EVERY,
        "gridline_every": GRIDLINE_EVERY,
        "margin_pixels": MARGIN,
        "x_axis": "column, labelled along the top",
        "y_axis": "row, labelled down the left",
        "cell_pixels": scale,
        "note": (
            "image pixel (px, py) is grid cell "
            f"x=(px-{MARGIN})//{scale}, y=(py-{MARGIN})//{scale}"
        ),
    }


def annotate(pixels: np.ndarray, *, scale: int = CELL_PIXELS) -> bytes:
    """Return the rendered frame with coordinate rulers and light gridlines.

    The underlying render is ``pixels_to_png`` unchanged, so what the model sees is
    the same frame the workspace holds, with the coordinate system drawn around it.
    """

    grid = np.asarray(pixels)
    rows, columns = grid.shape[-2:]
    with Image.open(BytesIO(pixels_to_png(grid, scale=scale))) as rendered:
        frame = rendered.convert("RGB").copy()

    # Gridlines blend toward the complement of whatever is under them, so they stay
    # visible on a white background and on a black one, and a single-cell feature
    # under a line is dimmed rather than hidden. Blending toward a fixed colour fails
    # on backgrounds near that colour - the first version drew white on white.
    overlay = np.asarray(frame).astype(np.float32)
    for column in range(0, columns, GRIDLINE_EVERY):
        x = column * scale
        overlay[:, x] = overlay[:, x] * (1 - GRIDLINE_BLEND) + (255 - overlay[:, x]) * GRIDLINE_BLEND
    for row in range(0, rows, GRIDLINE_EVERY):
        y = row * scale
        overlay[y, :] = overlay[y, :] * (1 - GRIDLINE_BLEND) + (255 - overlay[y, :]) * GRIDLINE_BLEND
    frame = Image.fromarray(overlay.astype(np.uint8), mode="RGB")

    canvas = Image.new(
        "RGB", (frame.width + MARGIN, frame.height + MARGIN), RULER_BACKGROUND
    )
    canvas.paste(frame, (MARGIN, MARGIN))
    draw = ImageDraw.Draw(canvas)

    for column in range(columns):
        x = MARGIN + column * scale
        tick = 4 if column % LABEL_EVERY else 8
        draw.line([(x, MARGIN - tick), (x, MARGIN - 1)], fill=RULER_INK)
        if column % LABEL_EVERY == 0:
            draw.text((x + 1, 2), str(column), fill=RULER_INK)
    for row in range(rows):
        y = MARGIN + row * scale
        tick = 4 if row % LABEL_EVERY else 8
        draw.line([(MARGIN - tick, y), (MARGIN - 1, y)], fill=RULER_INK)
        if row % LABEL_EVERY == 0:
            draw.text((2, y + 1), str(row), fill=RULER_INK)

    stream = BytesIO()
    canvas.save(stream, format="PNG")
    return stream.getvalue()
