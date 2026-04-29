import argparse
from dataclasses import dataclass
from typing import List

import numpy as np

from terrain_generator import TerrainGenerator


@dataclass
class Box2D:
    x: float
    y: float
    sx: float
    sy: float


def overlaps(a: Box2D, b: Box2D, margin: float) -> bool:
    ax0, ax1 = a.x - 0.5 * a.sx - margin, a.x + 0.5 * a.sx + margin
    ay0, ay1 = a.y - 0.5 * a.sy - margin, a.y + 0.5 * a.sy + margin
    bx0, bx1 = b.x - 0.5 * b.sx - margin, b.x + 0.5 * b.sx + margin
    by0, by1 = b.y - 0.5 * b.sy - margin, b.y + 0.5 * b.sy + margin
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def sample_non_overlapping_boxes(
    rng: np.random.Generator,
    count: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    size_min: float,
    size_max: float,
    height_min: float,
    height_max: float,
    margin: float,
    max_trials_per_box: int,
) -> List[tuple[Box2D, float]]:
    boxes: List[tuple[Box2D, float]] = []
    for _ in range(count):
        placed = False
        for _ in range(max_trials_per_box):
            sx = float(rng.uniform(size_min, size_max))
            sy = float(rng.uniform(size_min, size_max))
            sz = float(rng.uniform(height_min, height_max))
            x = float(rng.uniform(x_min + 0.5 * sx, x_max - 0.5 * sx))
            y = float(rng.uniform(y_min + 0.5 * sy, y_max - 0.5 * sy))
            candidate = Box2D(x=x, y=y, sx=sx, sy=sy)
            if any(overlaps(candidate, old, margin) for old, _ in boxes):
                continue
            boxes.append((candidate, sz))
            placed = True
            break
        if not placed:
            break
    return boxes


def main():
    parser = argparse.ArgumentParser(description="Generate non-overlapping random box terrain using TerrainGenerator.AddBox.")
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--x-min", type=float, default=-8.0)
    parser.add_argument("--x-max", type=float, default=8.0)
    parser.add_argument("--y-min", type=float, default=-3.0)
    parser.add_argument("--y-max", type=float, default=15.0)
    parser.add_argument("--size-min", type=float, default=0.12)
    parser.add_argument("--size-max", type=float, default=0.55)
    parser.add_argument("--height-min", type=float, default=1.0)
    parser.add_argument("--height-max", type=float, default=1.5)
    parser.add_argument("--margin", type=float, default=0.02, help="Extra no-overlap clearance in XY plane.")
    parser.add_argument("--max-trials-per-box", type=int, default=300)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    sampled = sample_non_overlapping_boxes(
        rng=rng,
        count=args.count,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        size_min=args.size_min,
        size_max=args.size_max,
        height_min=args.height_min,
        height_max=args.height_max,
        margin=args.margin,
        max_trials_per_box=args.max_trials_per_box,
    )

    tg = TerrainGenerator()
    for box, sz in sampled:
        z = 0.5 * sz
        tg.AddBox(position=[box.x, box.y, z], euler=[0.0, 0.0, 0.0], size=[box.sx, box.sy, sz])
    tg.Save()

    print(f"requested={args.count} generated={len(sampled)}")
    if len(sampled) < args.count:
        print("warning: space is crowded; reduce --count or box sizes, or enlarge area.")


if __name__ == "__main__":
    main()
