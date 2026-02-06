"""Generate the SOLWEIG QGIS plugin icon using Pillow.

A fun sun-with-sunglasses over a city skyline with a thermometer.
Palette: dark charcoal buildings, golden sun, red thermometer accent, white details.
Run once to produce icon.png, then delete this script.
"""

import math
import os

from PIL import Image, ImageDraw

# =============================================================================
# Palette — at most 4 colours + sky gradient
# =============================================================================
DARK = (38, 42, 48)  # charcoal — buildings, ground, sunglasses
MID = (58, 64, 72)  # lighter charcoal — alternate buildings
GOLD = (255, 210, 50)  # sun, rays, window glow
RED = (230, 60, 50)  # thermometer, antenna blink, grin
WHITE = (255, 255, 255)  # thermometer body, text, highlights

SKY_TOP = (255, 107, 53)  # warm orange  (gradient only)
SKY_MID = (247, 201, 72)  # golden       (gradient only)
SKY_BOT = (135, 206, 235)  # light blue   (gradient only)


def lerp_color(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def draw_icon(size=128):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size / 64  # scale factor relative to 64px design

    # --- Sky gradient background ---
    for y in range(size):
        t = y / size
        color = lerp_color(SKY_TOP, SKY_MID, t / 0.5) if t < 0.5 else lerp_color(SKY_MID, SKY_BOT, (t - 0.5) / 0.5)
        d.line([(0, y), (size - 1, y)], fill=color)

    # Rounded corners mask
    mask = Image.new("L", (size, size), 0)
    mask_d = ImageDraw.Draw(mask)
    mask_d.rounded_rectangle([0, 0, size - 1, size - 1], radius=int(8 * s), fill=255)
    img.putalpha(mask)
    d = ImageDraw.Draw(img)

    # --- Sun glow ---
    glow_img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    glow_d = ImageDraw.Draw(glow_img)
    cx_sun, cy_sun = int(32 * s), int(13 * s)
    for radius in range(int(22 * s), 0, -1):
        t = radius / (22 * s)
        alpha = int(55 * (1 - t))
        glow_d.ellipse(
            [cx_sun - radius, cy_sun - radius, cx_sun + radius, cy_sun + radius],
            fill=(*GOLD[:3], alpha),
        )
    img = Image.alpha_composite(img, glow_img)
    d = ImageDraw.Draw(img)

    # --- Sun body ---
    sr = int(9 * s)
    d.ellipse(
        [cx_sun - sr, cy_sun - sr, cx_sun + sr, cy_sun + sr],
        fill=GOLD,
        outline=(220, 180, 30),
        width=max(1, int(s)),
    )

    # --- Sun rays ---
    ray_inner = int(10.5 * s)
    ray_outer = int(13 * s)
    ray_w = max(1, int(1.8 * s))
    for angle_deg in range(0, 360, 45):
        if angle_deg == 180:
            continue
        a = math.radians(angle_deg)
        x1 = cx_sun + int(ray_inner * math.sin(a))
        y1 = cy_sun - int(ray_inner * math.cos(a))
        x2 = cx_sun + int(ray_outer * math.sin(a))
        y2 = cy_sun - int(ray_outer * math.cos(a))
        d.line([(x1, y1), (x2, y2)], fill=GOLD, width=ray_w)

    # --- Sunglasses (DARK) ---
    gw, gh = int(3.2 * s), int(2.4 * s)
    lx, ly = int(29 * s), int(12 * s)
    rx, ry = int(35 * s), int(12 * s)
    d.ellipse([lx - gw, ly - gh, lx + gw, ly + gh], fill=DARK)
    d.ellipse([rx - gw, ry - gh, rx + gw, ry + gh], fill=DARK)
    # Bridge + arms
    bw = max(1, int(1.2 * s))
    d.line([(lx + gw - int(s), ly), (rx - gw + int(s), ry)], fill=DARK, width=bw)
    aw = max(1, int(s))
    d.line([(lx - gw, ly - int(0.5 * s)), (lx - gw - int(2.5 * s), ly - int(2 * s))], fill=DARK, width=aw)
    d.line([(rx + gw, ry - int(0.5 * s)), (rx + gw + int(2.5 * s), ry - int(2 * s))], fill=DARK, width=aw)
    # Lens shine
    shr = int(1.2 * s)
    d.ellipse(
        [lx - int(1.5 * s) - shr, ly - int(0.8 * s) - shr, lx - int(1.5 * s) + shr, ly - int(0.8 * s) + shr],
        fill=(255, 255, 255, 70),
    )
    d.ellipse(
        [rx - int(1.5 * s) - shr, ry - int(0.8 * s) - shr, rx - int(1.5 * s) + shr, ry - int(0.8 * s) + shr],
        fill=(255, 255, 255, 70),
    )

    # --- Cheeky grin (RED) ---
    d.arc([int(28.5 * s), int(14.5 * s), int(35.5 * s), int(19 * s)], 0, 180, fill=RED, width=max(1, int(1.3 * s)))

    # --- Heat shimmer wavy lines (RED, semi-transparent) ---
    heat_color = (*RED[:3], 90)
    for hx in [int(15 * s), int(38 * s), int(52 * s)]:
        for yy in range(int(22 * s), int(37 * s), int(1.5 * s) or 1):
            offset = int(2 * s * math.sin(yy * 0.25))
            for dx in range(max(1, int(0.8 * s))):
                d.point((hx + offset + dx, yy), fill=heat_color)

    # --- City skyline (DARK / MID only, GOLD windows) ---
    # Buildings extend to y=64 (full bottom) so rounded mask clips them cleanly
    buildings = [
        # (x, y, w, h, color)
        (4, 32, 8, 32, DARK),
        (13, 37, 10, 27, MID),
        (31, 30, 9, 34, DARK),
        (41, 42, 10, 22, MID),
        (52, 35, 8, 29, DARK),
    ]

    for bx, by, bw, bh, color in buildings:
        x1, y1 = int(bx * s), int(by * s)
        x2, y2 = int((bx + bw) * s), int((by + bh) * s)
        d.rectangle([x1, y1, x2, y2], fill=color)

        # Windows — GOLD only, varying alpha for life
        wy = y1 + int(2 * s)
        win_idx = 0
        while wy + int(2 * s) < y2 - int(1 * s):
            wx = x1 + int(1.5 * s)
            while wx + int(2 * s) < x2 - int(0.5 * s):
                alpha = [210, 140, 180, 100, 220, 160][win_idx % 6]
                d.rectangle(
                    [wx, wy, wx + int(2 * s), wy + int(2 * s)],
                    fill=(*GOLD[:3], alpha),
                )
                wx += int(3.5 * s)
                win_idx += 1
            wy += int(4 * s)

    # --- Tree (DARK trunk, MID canopy — stays monochromatic) ---
    trunk_x = int(25.5 * s)
    trunk_w = max(1, int(1 * s))
    d.rectangle(
        [trunk_x - trunk_w, int(48 * s), trunk_x + trunk_w, int(64 * s)],
        fill=DARK,
    )
    tree_r = int(4.5 * s)
    d.ellipse(
        [trunk_x - tree_r, int(45 * s) - tree_r, trunk_x + tree_r, int(45 * s) + tree_r],
        fill=MID,
    )
    d.ellipse(
        [trunk_x - int(3 * s), int(46.5 * s) - int(3 * s), trunk_x + int(1 * s), int(46.5 * s) + int(3 * s)],
        fill=(68, 75, 84),  # slightly lighter MID variant
    )
    d.ellipse(
        [trunk_x - int(1 * s), int(46.5 * s) - int(3 * s), trunk_x + int(3.5 * s), int(46.5 * s) + int(3 * s)],
        fill=MID,
    )

    # --- Antenna on building 3 (DARK pole, RED blink) ---
    ant_x = int(35.5 * s)
    d.line([(ant_x, int(30 * s)), (ant_x, int(25 * s))], fill=MID, width=max(1, int(1.2 * s)))
    br = int(1.2 * s)
    d.ellipse([ant_x - br, int(25 * s) - br, ant_x + br, int(25 * s) + br], fill=RED)

    # --- Ground strip (DARK, semi-transparent) — full width to bottom edge ---
    d.rectangle([0, int(56 * s), size - 1, size - 1], fill=(*DARK[:3], 100))

    # --- Thermometer (WHITE body, RED mercury) ---
    tx = int(57 * s)
    ty_top, ty_bot = int(3 * s), int(19 * s)
    tw = int(1.8 * s)
    bulb_r = int(3 * s)

    # White body
    d.rounded_rectangle(
        [tx - tw, ty_top, tx + tw, ty_bot],
        radius=tw,
        fill=WHITE,
        outline=(*DARK[:3], 120),
        width=max(1, int(0.5 * s)),
    )
    # Red mercury
    d.rounded_rectangle(
        [tx - int(1 * s), int(6 * s), tx + int(1 * s), ty_bot],
        radius=int(1 * s),
        fill=RED,
    )
    # Red bulb
    d.ellipse(
        [tx - bulb_r, ty_bot - int(1 * s), tx + bulb_r, ty_bot + bulb_r + int(1 * s)],
        fill=RED,
        outline=(*DARK[:3], 120),
        width=max(1, int(0.5 * s)),
    )
    # Tick marks (DARK)
    for tick_y in [int(6 * s), int(9 * s), int(12 * s), int(15 * s)]:
        d.line(
            [(tx - tw - int(1 * s), tick_y), (tx - tw + int(0.5 * s), tick_y)],
            fill=(*DARK[:3], 150),
            width=max(1, int(0.5 * s)),
        )

    return img


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))

    icon_128 = draw_icon(128)
    icon_128.save(os.path.join(here, "icon_128.png"))
    print("Saved icon_128.png")

    icon_64 = icon_128.resize((64, 64), Image.Resampling.LANCZOS)
    icon_64.save(os.path.join(here, "icon.png"))
    print("Saved icon.png (64x64)")
