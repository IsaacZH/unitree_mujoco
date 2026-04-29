import argparse
import json
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]


def load_npz_meta(path: Path) -> dict | None:
    """Load optional JSON metadata from terrain_dump-style npz."""
    if path.suffix.lower() != ".npz":
        return None

    with np.load(path) as data:
        if "meta_json" not in data:
            return None
        raw = data["meta_json"]

    if getattr(raw, "shape", None) == ():
        text = str(raw.item())
    else:
        text = str(raw)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def load_height_array(path: Path, npz_key: str | None, tile_index: int | None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            if npz_key is None:
                keys = list(data.keys())
                if len(keys) != 1:
                    raise ValueError(
                        f"NPZ has keys {keys}; please pass --npz-key to select one."
                    )
                arr = data[keys[0]]
            else:
                if npz_key not in data:
                    raise KeyError(f"Key '{npz_key}' not found in {path}.")
                arr = data[npz_key]
    else:
        raise ValueError(f"Unsupported input format: {path}. Use .npy or .npz")

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3:
        if tile_index is None:
            raise ValueError(
                f"Height array is 3D with shape {arr.shape}; please pass --tile-index."
            )
        if tile_index < 0 or tile_index >= arr.shape[0]:
            raise IndexError(
                f"--tile-index {tile_index} out of range [0, {arr.shape[0] - 1}]"
            )
        arr = arr[tile_index]
    if arr.ndim != 2:
        raise ValueError(f"Height array must be 2D, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Height array contains NaN/Inf.")
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Isaac Lab height array (.npy/.npz) to MuJoCo 16-bit PNG hfield."
    )
    parser.add_argument("--input", required=True, help="Input .npy or .npz height array path.")
    parser.add_argument("--npz-key", default=None, help="Key for .npz input.")
    parser.add_argument(
        "--tile-index",
        type=int,
        default=None,
        help="If selected array is 3D [N,H,W], choose tile index N.",
    )
    parser.add_argument("--output-png", required=True, help="Output 16-bit PNG path.")
    parser.add_argument("--output-meta", required=True, help="Output JSON metadata path.")
    parser.add_argument(
        "--cell-size",
        type=float,
        default=None,
        help="Grid size in meters (x). If omitted, auto-use npz meta horizontal_scale when available.",
    )
    parser.add_argument(
        "--cell-size-y",
        type=float,
        default=None,
        help="Optional Y grid size in meters; defaults to --cell-size.",
    )
    parser.add_argument(
        "--negative-depth",
        type=float,
        default=None,
        help="MuJoCo hfield size[3]. If omitted, auto-derived by --align-mode.",
    )
    parser.add_argument(
        "--height-scale",
        type=float,
        default=None,
        help="Multiply input heights by this factor. If omitted, auto-use npz meta vertical_scale when available.",
    )
    parser.add_argument(
        "--align-mode",
        choices=["min", "zero"],
        default="zero",
        help="How to place terrain in world-z when --negative-depth is omitted: min -> set lowest point to z=0, zero -> keep z=0 level aligned.",
    )
    parser.add_argument(
        "--z-offset",
        type=float,
        default=0.0,
        help="Additional world-space Z offset added to generated geom position.",
    )
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Flip array vertically before saving image.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_png = Path(args.output_png)
    output_meta = Path(args.output_meta)
    npz_meta = load_npz_meta(input_path)

    height_scale = args.height_scale
    if height_scale is None:
        height_scale = float(npz_meta.get("vertical_scale", 1.0)) if npz_meta else 1.0

    cell_size = args.cell_size
    if cell_size is None:
        cell_size = float(npz_meta.get("horizontal_scale", 0.05)) if npz_meta else 0.05

    z = load_height_array(input_path, args.npz_key, args.tile_index)
    z = z * float(height_scale)
    if args.flip_y:
        z = np.flipud(z)

    z_min = float(np.min(z))
    z_max = float(np.max(z))
    z_span = z_max - z_min
    if z_span <= 1e-12:
        raise ValueError("Height array is nearly constant; cannot build a useful hfield.")

    if args.negative_depth is None:
        if args.align_mode == "zero":
            neg_depth = max(0.0, -z_min)
        else:
            neg_depth = 0.0
    else:
        neg_depth = float(args.negative_depth)
    if neg_depth < 0.0:
        raise ValueError("--negative-depth must be >= 0")
    if neg_depth > z_span:
        raise ValueError(
            f"--negative-depth ({neg_depth}) must be <= z range ({z_span})."
        )

    # Normalize to [0, 65535] for MuJoCo image-backed hfield.
    z_norm = (z - z_min) / z_span
    z_u16 = np.clip(np.round(z_norm * 65535.0), 0, 65535).astype(np.uint16)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_png), z_u16):
        raise RuntimeError(f"Failed to write PNG: {output_png}")

    cell_x = float(cell_size)
    cell_y = float(cell_size if args.cell_size_y is None else args.cell_size_y)
    h, w = z.shape
    size_x = (w * cell_x) / 2.0
    size_y = (h * cell_y) / 2.0

    # Match physical heights: bottom = geom_z - neg_depth = z_min
    # top = geom_z + pos_height = z_max
    geom_z = z_min + neg_depth + float(args.z_offset)
    pos_height = z_max - geom_z

    rel_png = output_png.name
    mjcf_snippet = (
        f'<hfield name="lab_hfield" size="{size_x:.6f} {size_y:.6f} {pos_height:.6f} {neg_depth:.6f}" '
        f'file="../{rel_png}"/>\n'
        f'<geom type="hfield" hfield="lab_hfield" pos="0 0 {geom_z:.6f}" quat="1 0 0 0"/>'
    )

    meta = {
        "input": str(input_path),
        "npz_meta_used": bool(npz_meta),
        "height_scale_used": float(height_scale),
        "align_mode": args.align_mode,
        "shape": [int(h), int(w)],
        "z_min": z_min,
        "z_max": z_max,
        "z_span": z_span,
        "cell_size_x": cell_x,
        "cell_size_y": cell_y,
        "mujoco": {
            "hfield_name": "lab_hfield",
            "size": [size_x, size_y, pos_height, neg_depth],
            "geom_pos": [0.0, 0.0, geom_z],
            "geom_quat": [1.0, 0.0, 0.0, 0.0],
            "asset_file_relative_hint": f"../{rel_png}",
        },
        "mjcf_snippet": mjcf_snippet,
    }

    output_meta.parent.mkdir(parents=True, exist_ok=True)
    output_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved:")
    print(f"  PNG : {output_png}")
    print(f"  META: {output_meta}")
    print("\nUse this MJCF snippet:")
    print(mjcf_snippet)


if __name__ == "__main__":
    main()
