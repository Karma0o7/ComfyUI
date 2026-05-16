import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()

with patch.dict("sys.modules", {"nodes": mock_nodes, "server": mock_server}):
    from comfy_extras import nodes_save_image_promotable as mod


def _make_image(width=8, height=4):
    return torch.rand(1, height, width, 3)


def _write_png(path: str, width=8, height=4):
    arr = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


class TestParseRef:
    def test_empty(self):
        assert mod._parse_promoted_ref("") is None

    def test_invalid_json(self):
        assert mod._parse_promoted_ref("{not json") is None

    def test_non_object(self):
        assert mod._parse_promoted_ref('"a string"') is None
        assert mod._parse_promoted_ref("[]") is None

    def test_missing_filename(self):
        assert mod._parse_promoted_ref('{"subfolder":"x","type":"output"}') is None

    def test_path_traversal_filename(self):
        ref = json.dumps(
            {"filename": "../etc/passwd", "subfolder": "", "type": "output"}
        )
        assert mod._parse_promoted_ref(ref) is None

    def test_path_traversal_subfolder(self):
        ref = json.dumps({"filename": "x.png", "subfolder": "../..", "type": "output"})
        assert mod._parse_promoted_ref(ref) is None

    def test_absolute_filename(self):
        ref = json.dumps({"filename": "/etc/passwd", "subfolder": "", "type": "output"})
        assert mod._parse_promoted_ref(ref) is None

    def test_valid(self):
        ref = json.dumps({"filename": "x.png", "subfolder": "sub", "type": "output"})
        parsed = mod._parse_promoted_ref(ref)
        assert parsed == {"filename": "x.png", "subfolder": "sub", "type": "output"}

    def test_defaults_applied(self):
        ref = json.dumps({"filename": "x.png"})
        parsed = mod._parse_promoted_ref(ref)
        assert parsed == {"filename": "x.png", "subfolder": "", "type": "output"}


class TestResolveRefPath:
    def test_unknown_type(self, tmp_path):
        with patch.object(
            mod.folder_paths, "get_output_directory", return_value=str(tmp_path)
        ):
            assert (
                mod._resolve_ref_path(
                    {"filename": "x.png", "subfolder": "", "type": "garbage"}
                )
                is None
            )

    def test_missing_file(self, tmp_path):
        with patch.object(
            mod.folder_paths, "get_output_directory", return_value=str(tmp_path)
        ):
            assert (
                mod._resolve_ref_path(
                    {"filename": "missing.png", "subfolder": "", "type": "output"}
                )
                is None
            )

    def test_resolves_file(self, tmp_path):
        target = tmp_path / "img.png"
        _write_png(str(target))
        with patch.object(
            mod.folder_paths, "get_output_directory", return_value=str(tmp_path)
        ):
            resolved = mod._resolve_ref_path(
                {"filename": "img.png", "subfolder": "", "type": "output"}
            )
        assert resolved is not None
        assert os.path.realpath(resolved) == os.path.realpath(str(target))

    def test_resolves_file_in_subfolder(self, tmp_path):
        sub = tmp_path / "nested"
        sub.mkdir()
        target = sub / "img.png"
        _write_png(str(target))
        with patch.object(
            mod.folder_paths, "get_output_directory", return_value=str(tmp_path)
        ):
            resolved = mod._resolve_ref_path(
                {"filename": "img.png", "subfolder": "nested", "type": "output"}
            )
        assert resolved is not None
        assert os.path.realpath(resolved) == os.path.realpath(str(target))

    def test_symlink_escape_rejected(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        secret = outside / "secret.png"
        _write_png(str(secret))
        base = tmp_path / "base"
        base.mkdir()
        link = base / "link.png"
        os.symlink(str(secret), str(link))
        with patch.object(
            mod.folder_paths, "get_output_directory", return_value=str(base)
        ):
            resolved = mod._resolve_ref_path(
                {"filename": "link.png", "subfolder": "", "type": "output"}
            )
        assert resolved is None


class TestNodeContract:
    def test_input_types_shape(self):
        inp = mod.SaveImagePromotable.INPUT_TYPES()
        assert set(inp["required"].keys()) == {
            "images",
            "filename_prefix",
            "accumulate",
        }
        assert set(inp["optional"].keys()) == {"promoted_asset_ref"}
        assert set(inp["hidden"].keys()) == {"prompt", "extra_pnginfo"}
        assert inp["required"]["accumulate"][0] == "BOOLEAN"
        assert inp["required"]["accumulate"][1]["default"] is False

    def test_class_metadata(self):
        cls = mod.SaveImagePromotable
        assert cls.RETURN_TYPES == ("IMAGE",)
        assert cls.RETURN_NAMES == ("images",)
        assert cls.OUTPUT_NODE is True
        assert cls.FUNCTION == "execute"
        assert "SaveImagePromotable" in mod.NODE_CLASS_MAPPINGS
        assert mod.NODE_CLASS_MAPPINGS["SaveImagePromotable"] is cls


class TestExecutePassthrough:
    def test_passthrough_saves_and_returns_input(self, tmp_path):
        node = mod.SaveImagePromotable()
        node.output_dir = str(tmp_path)
        images = _make_image()

        with (
            patch.object(mod.args, "disable_metadata", True),
            patch.object(mod.folder_paths, "get_save_image_path") as get_path,
        ):
            get_path.return_value = (str(tmp_path), "ComfyUI", 1, "", "ComfyUI")
            result = node.execute(
                images,
                filename_prefix="ComfyUI",
                accumulate=False,
                promoted_asset_ref="",
            )

        assert "ui" in result
        assert "result" in result
        assert torch.equal(result["result"][0], images)
        assert len(result["ui"]["images"]) == 1
        saved_name = result["ui"]["images"][0]["filename"]
        assert os.path.isfile(os.path.join(str(tmp_path), saved_name))

    def test_stale_ref_falls_through_to_passthrough(self, tmp_path):
        node = mod.SaveImagePromotable()
        node.output_dir = str(tmp_path)
        images = _make_image()
        ref = json.dumps(
            {"filename": "does_not_exist.png", "subfolder": "", "type": "output"}
        )

        with (
            patch.object(mod.args, "disable_metadata", True),
            patch.object(mod.folder_paths, "get_save_image_path") as get_path,
            patch.object(
                mod.folder_paths, "get_output_directory", return_value=str(tmp_path)
            ),
        ):
            get_path.return_value = (str(tmp_path), "ComfyUI", 1, "", "ComfyUI")
            result = node.execute(images, promoted_asset_ref=ref)

        assert torch.equal(result["result"][0], images)


class TestExecuteLocked:
    def test_locked_outputs_loaded_image(self, tmp_path):
        target = tmp_path / "promoted.png"
        _write_png(str(target), width=8, height=4)
        ref = json.dumps(
            {"filename": "promoted.png", "subfolder": "", "type": "output"}
        )
        node = mod.SaveImagePromotable()
        node.output_dir = str(tmp_path)
        upstream = _make_image(width=8, height=4)

        with patch.object(
            mod.folder_paths, "get_output_directory", return_value=str(tmp_path)
        ):
            result = node.execute(upstream, promoted_asset_ref=ref)

        assert result["ui"]["images"] == [
            {"filename": "promoted.png", "subfolder": "", "type": "output"}
        ]
        out = result["result"][0]
        assert out.shape == upstream.shape
        assert out.dtype == upstream.dtype
        assert not torch.equal(out, upstream)

    def test_locked_does_not_save(self, tmp_path):
        target = tmp_path / "promoted.png"
        _write_png(str(target))
        ref = json.dumps(
            {"filename": "promoted.png", "subfolder": "", "type": "output"}
        )
        node = mod.SaveImagePromotable()
        node.output_dir = str(tmp_path)
        images = _make_image()

        with (
            patch.object(
                mod.folder_paths, "get_output_directory", return_value=str(tmp_path)
            ),
            patch.object(node, "_save_images") as save_mock,
        ):
            node.execute(images, promoted_asset_ref=ref)

        save_mock.assert_not_called()


class TestIsChanged:
    def test_unlocked_returns_false(self):
        assert (
            mod.SaveImagePromotable.IS_CHANGED(images=None, promoted_asset_ref="")
            is False
        )

    def test_locked_missing_file(self, tmp_path):
        ref = json.dumps({"filename": "missing.png", "subfolder": "", "type": "output"})
        with patch.object(
            mod.folder_paths, "get_output_directory", return_value=str(tmp_path)
        ):
            key = mod.SaveImagePromotable.IS_CHANGED(
                images=None, promoted_asset_ref=ref
            )
        assert isinstance(key, str)
        assert key.startswith("PROMOTED::MISSING::")

    def test_locked_stable_key(self, tmp_path):
        target = tmp_path / "p.png"
        _write_png(str(target))
        ref = json.dumps({"filename": "p.png", "subfolder": "", "type": "output"})
        with patch.object(
            mod.folder_paths, "get_output_directory", return_value=str(tmp_path)
        ):
            k1 = mod.SaveImagePromotable.IS_CHANGED(images=None, promoted_asset_ref=ref)
            k2 = mod.SaveImagePromotable.IS_CHANGED(images=None, promoted_asset_ref=ref)
        assert k1 == k2
        assert k1.startswith("PROMOTED::")

    def test_locked_key_changes_when_file_changes(self, tmp_path):
        target = tmp_path / "p.png"
        _write_png(str(target), width=8, height=4)
        ref = json.dumps({"filename": "p.png", "subfolder": "", "type": "output"})
        with patch.object(
            mod.folder_paths, "get_output_directory", return_value=str(tmp_path)
        ):
            k1 = mod.SaveImagePromotable.IS_CHANGED(images=None, promoted_asset_ref=ref)
        os.utime(str(target), (1234567890, 1234567890))
        with patch.object(
            mod.folder_paths, "get_output_directory", return_value=str(tmp_path)
        ):
            k2 = mod.SaveImagePromotable.IS_CHANGED(images=None, promoted_asset_ref=ref)
        assert k1 != k2
