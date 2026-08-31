#!/usr/bin/env python3
"""Standalone SAM3 image segmentation CLI with externally mounted weights.

The container never downloads model weights. Mount a compatible SAM3 checkpoint
into the container (default: /models/sam3.pt) and pass it to the upstream model
builder with Hugging Face loading disabled.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 text-prompt instance segmentation on one image using a local checkpoint."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input RGB image.")
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Text prompt. Repeat to segment multiple concepts with one image/model load.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional UTF-8 file with one prompt per non-empty line.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/output"))
    parser.add_argument(
        "--mask-out",
        type=Path,
        default=None,
        help="For exactly one prompt, also write the selected mask to this exact path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/models/sam3.pt"),
        help="Mounted SAM3 checkpoint. The container never downloads weights.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, usually cuda or cuda:0.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--selection",
        choices=("top-score", "largest", "union"),
        default="top-score",
        help="How to choose primary_mask.png when a prompt matches multiple instances.",
    )
    parser.add_argument("--crop-padding", type=float, default=0.05)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write empty outputs instead of failing when no instance matches.",
    )
    return parser.parse_args()


def read_prompts(args: argparse.Namespace) -> list[str]:
    prompts = [p.strip() for p in args.prompt if p.strip()]
    if args.prompt_file is not None:
        if not args.prompt_file.is_file():
            raise FileNotFoundError(args.prompt_file)
        prompts.extend(
            line.strip()
            for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not prompts:
        raise ValueError("Provide at least one --prompt or --prompt-file.")
    return prompts


def slugify(value: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_.-")
    return (slug or "prompt")[:max_len]


def to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_masks(value, image_shape: tuple[int, int]) -> np.ndarray:
    masks = to_numpy(value)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise ValueError(f"Unexpected SAM3 masks shape: {masks.shape}")
    if tuple(masks.shape[1:]) != image_shape:
        raise ValueError(f"SAM3 mask shape {masks.shape[1:]} != image shape {image_shape}")
    return masks.astype(bool, copy=False)


def normalize_vector(value, length: int, name: str) -> np.ndarray:
    array = to_numpy(value).reshape(-1)
    if len(array) != length:
        raise ValueError(f"SAM3 {name} length {len(array)} != masks {length}")
    return array


def normalize_boxes(value, length: int) -> np.ndarray:
    boxes = to_numpy(value).reshape(-1, 4)
    if len(boxes) != length:
        raise ValueError(f"SAM3 boxes length {len(boxes)} != masks {length}")
    return boxes


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def choose_mask(masks: np.ndarray, scores: np.ndarray, mode: str) -> tuple[np.ndarray, int | None]:
    if mode == "union":
        return np.any(masks, axis=0), None
    if mode == "largest":
        index = int(np.argmax(masks.reshape(len(masks), -1).sum(axis=1)))
    else:
        index = int(np.argmax(scores))
    return masks[index], index


def save_rgba_products(
    rgb: Image.Image,
    mask: np.ndarray,
    output_dir: Path,
    crop_padding: float,
) -> dict[str, str | None]:
    rgba = np.asarray(rgb.convert("RGBA")).copy()
    rgba[..., 3] = mask.astype(np.uint8) * 255
    full_path = output_dir / "cutout_rgba.png"
    Image.fromarray(rgba, mode="RGBA").save(full_path)

    crop_path: Path | None = None
    ys, xs = np.nonzero(mask)
    if len(xs):
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        pad = int(round(max(x1 - x0, y1 - y0) * crop_padding))
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1 = min(mask.shape[1], x1 + pad)
        y1 = min(mask.shape[0], y1 + pad)
        crop_path = output_dir / "crop_rgba.png"
        Image.fromarray(rgba[y0:y1, x0:x1], mode="RGBA").save(crop_path)

    return {
        "cutout_rgba": str(full_path),
        "crop_rgba": str(crop_path) if crop_path else None,
    }


def run_prompt(
    processor: Sam3Processor,
    state: dict,
    prompt: str,
    rgb: Image.Image,
    output_dir: Path,
    selection: str,
    crop_padding: float,
    allow_empty: bool,
) -> dict[str, object]:
    output = processor.set_text_prompt(state=state, prompt=prompt)
    masks = normalize_masks(output["masks"], (rgb.height, rgb.width))
    scores = normalize_vector(output["scores"], len(masks), "scores").astype(float)
    boxes = normalize_boxes(output["boxes"], len(masks)).astype(float)

    order = np.argsort(-scores)
    masks, scores, boxes = masks[order], scores[order], boxes[order]
    output_dir.mkdir(parents=True, exist_ok=True)
    instances_dir = output_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    instances: list[dict[str, object]] = []
    for index, (mask, score, box) in enumerate(zip(masks, scores, boxes)):
        path = instances_dir / f"{index:03d}.png"
        save_mask(mask, path)
        instances.append(
            {
                "index": index,
                "score": float(score),
                "box_xyxy": [float(v) for v in box],
                "area_pixels": int(mask.sum()),
                "mask": str(path),
            }
        )

    if len(masks) == 0:
        if not allow_empty:
            raise RuntimeError(f"SAM3 found no instances for prompt {prompt!r}")
        primary = np.zeros((rgb.height, rgb.width), dtype=bool)
        selected_index = None
    else:
        primary, selected_index = choose_mask(masks, scores, selection)

    primary_path = output_dir / "primary_mask.png"
    save_mask(primary, primary_path)
    rgba_paths = save_rgba_products(rgb, primary, output_dir, crop_padding)
    result: dict[str, object] = {
        "prompt": prompt,
        "matched_instances": len(instances),
        "selection": selection,
        "selected_instance_index": selected_index,
        "primary_mask": str(primary_path),
        **rgba_paths,
        "instances": instances,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> int:
    args = parse_args()
    prompts = read_prompts(args)

    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"SAM3 checkpoint not found: {args.checkpoint}. "
            "Mount it, e.g. -v /host/models:/models:ro"
        )
    if args.mask_out is not None and len(prompts) != 1:
        raise ValueError("--mask-out requires exactly one prompt")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be in [0, 1]")
    if args.crop_padding < 0:
        raise ValueError("--crop-padding must be >= 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; run Docker with --gpus all")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb = Image.open(args.image).convert("RGB")

    model = build_sam3_image_model(
        device=args.device,
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
    )
    processor = Sam3Processor(
        model,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )

    state = processor.set_image(rgb)
    results: list[dict[str, object]] = []
    for index, prompt in enumerate(prompts):
        if index > 0:
            processor.reset_all_prompts(state)
        result = run_prompt(
            processor,
            state,
            prompt,
            rgb,
            args.output_dir / f"prompt_{index:02d}_{slugify(prompt)}",
            args.selection,
            args.crop_padding,
            args.allow_empty,
        )
        results.append(result)

    if args.mask_out is not None:
        source = Path(str(results[0]["primary_mask"]))
        args.mask_out.parent.mkdir(parents=True, exist_ok=True)
        args.mask_out.write_bytes(source.read_bytes())
        results[0]["mask_out"] = str(args.mask_out)

    manifest = {
        "input_image": str(args.image),
        "image_size": [rgb.width, rgb.height],
        "device": args.device,
        "checkpoint": str(args.checkpoint),
        "weights_source": "external mount",
        "network_weight_download": False,
        "results": results,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"manifest": str(manifest_path), "prompts": len(prompts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
