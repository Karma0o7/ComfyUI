"""
SaveImagePromotable: a pass-through SaveImage variant with accumulating previews
and a "promote/lock" feature.

Modes:
- Pass-through (default): saves incoming images, emits preview UI, returns the
  input tensor as output. With `accumulate=True`, the frontend appends previews
  to a gallery instead of replacing it.
- Locked: when `promoted_asset_ref` is a non-empty JSON ref to a saved asset,
  the node skips saving, loads the referenced image, and outputs that image.
  The frontend is expected to write the ref into the widget when the user
  clicks the "lock" UI on a preview.

Caching: IS_CHANGED returns a stable key derived from the ref (+ file mtime)
when locked, so re-queues with the same lock are cache hits and upstream
ancestors are skipped. Unlocked, IS_CHANGED returns False to defer to normal
input-signature caching.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo

import folder_paths
import node_helpers
from comfy.cli_args import args


def _parse_promoted_ref(promoted_asset_ref: str) -> dict | None:
    if not promoted_asset_ref:
        return None
    try:
        ref = json.loads(promoted_asset_ref)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(ref, dict):
        return None
    filename = ref.get("filename")
    if not isinstance(filename, str) or not filename:
        return None
    subfolder = ref.get("subfolder", "") or ""
    asset_type = ref.get("type", "output") or "output"
    if not isinstance(subfolder, str) or not isinstance(asset_type, str):
        return None
    # Reject anything that could escape the base directory.
    if os.path.isabs(subfolder) or ".." in subfolder.split(os.sep):
        return None
    if os.path.isabs(filename) or ".." in filename.split(os.sep):
        return None
    return {"filename": filename, "subfolder": subfolder, "type": asset_type}


def _resolve_ref_path(ref: dict) -> str | None:
    asset_type = ref["type"]
    if asset_type == "output":
        base = folder_paths.get_output_directory()
    elif asset_type == "input":
        base = folder_paths.get_input_directory()
    elif asset_type == "temp":
        base = folder_paths.get_temp_directory()
    else:
        return None
    path = os.path.join(base, ref["subfolder"], ref["filename"])
    # Defense-in-depth: ensure the resolved path stays inside the base dir.
    base_real = os.path.realpath(base)
    path_real = os.path.realpath(path)
    if not path_real.startswith(base_real + os.sep) and path_real != base_real:
        return None
    if not os.path.isfile(path_real):
        return None
    return path_real


def _load_image_tensor(path: str) -> torch.Tensor:
    img = node_helpers.pillow(Image.open, path)
    output_images: list[torch.Tensor] = []
    w: int | None = None
    h: int | None = None
    for frame in ImageSequence.Iterator(img):
        frame = node_helpers.pillow(ImageOps.exif_transpose, frame)
        image = frame.convert("RGB")
        if not output_images:
            w, h = image.size
        if image.size != (w, h):
            continue
        arr = np.array(image).astype(np.float32) / 255.0
        output_images.append(torch.from_numpy(arr)[None,])
    if not output_images:
        raise RuntimeError(f"Failed to decode any frames from {path}")
    return torch.cat(output_images, dim=0)


class SaveImagePromotable:
    """Pass-through SaveImage with accumulating previews and promote/lock.

    Inputs:
        images: IMAGE tensor to save + pass through (ignored when locked).
        filename_prefix: STRING prefix for saved files.
        accumulate: BOOLEAN — when True, frontend appends previews to gallery.
        promoted_asset_ref: STRING — JSON ref written by the frontend on lock.
            Empty string means "not locked, normal pass-through".
    Output:
        IMAGE — input pass-through, or the loaded promoted image when locked.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Images to save and pass through. Ignored when a promoted asset is locked."
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {"default": "ComfyUI", "tooltip": "Prefix for saved files."},
                ),
                "accumulate": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When enabled, previews append to a per-node gallery instead of replacing it.",
                    },
                ),
            },
            "optional": {
                "promoted_asset_ref": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "JSON ref to a saved asset. Set by the UI; do not edit manually.",
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "image"
    DESCRIPTION = "Saves images, shows accumulating previews, and passes the input through. A promoted (locked) preview overrides pass-through to output the chosen image."

    def _save_images(self, images, filename_prefix, prompt, extra_pnginfo):
        full_output_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results: list[dict] = []
        for batch_number, image in enumerate(images):
            arr = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            metadata: PngInfo | None = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for key in extra_pnginfo:
                        metadata.add_text(key, json.dumps(extra_pnginfo[key]))
            filename_with_batch = filename.replace("%batch_num%", str(batch_number))
            out_name = f"{filename_with_batch}_{counter:05}_.png"
            img.save(
                os.path.join(full_output_folder, out_name),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )
            results.append(
                {"filename": out_name, "subfolder": subfolder, "type": self.type}
            )
            counter += 1
        return results

    def execute(
        self,
        images,
        filename_prefix="ComfyUI",
        accumulate=False,  # noqa: ARG002
        promoted_asset_ref="",
        prompt=None,
        extra_pnginfo=None,
    ):
        ref = _parse_promoted_ref(promoted_asset_ref)
        if ref is not None:
            path = _resolve_ref_path(ref)
            if path is not None:
                tensor = _load_image_tensor(path)
                tensor = tensor.to(device=images.device, dtype=images.dtype)
                return {
                    "ui": {"images": [ref]},
                    "result": (tensor,),
                }
            # Ref is set but stale (file deleted / failed validation): fall
            # through to pass-through so the user gets a working graph rather
            # than an execution error.

        saved = self._save_images(images, filename_prefix, prompt, extra_pnginfo)
        return {"ui": {"images": saved}, "result": (images,)}

    @classmethod
    def IS_CHANGED(
        cls,
        images,  # noqa: ARG003
        filename_prefix="ComfyUI",
        accumulate=False,  # noqa: ARG003
        promoted_asset_ref="",
        prompt=None,  # noqa: ARG003
        extra_pnginfo=None,  # noqa: ARG003
    ):
        ref = _parse_promoted_ref(promoted_asset_ref)
        if ref is None:
            return False
        path = _resolve_ref_path(ref)
        if path is None:
            return f"PROMOTED::MISSING::{promoted_asset_ref}"
        try:
            stat = os.stat(path)
            sig = f"{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            sig = "NOSTAT"
        return f"PROMOTED::{promoted_asset_ref}::{sig}"


NODE_CLASS_MAPPINGS = {
    "SaveImagePromotable": SaveImagePromotable,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImagePromotable": "Save Image (Promotable, PoC)",
}
