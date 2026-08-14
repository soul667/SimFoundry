# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 8 must resolve each object's base iteration from stage 5's manifest.

Regression cover: stage 5's iteration numbering can have gaps (a detection pass
that finds no masks consumes an index without writing artifacts), but stage 8
assumed contiguous numbering and read ``post_object_removal/iter_{idx - 1}.png``,
crashing with FileNotFoundError on the first scene with a skipped iteration.
Stage 5 now records ``base_iter`` in ``obj_cat_list/iter_{idx}.json`` and stage
8/8b resolve it via ``resolve_base_iteration``.
"""

import json

from simfoundry.pipeline.stage_utils import resolve_base_iteration


def _write_manifest(scene_dir, idx, payload):
    manifest_dir = scene_dir / "obj_cat_list"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"iter_{idx}.json").write_text(json.dumps(payload))


def _write_removal_images(scene_dir, indices):
    removal_dir = scene_dir / "post_object_removal"
    removal_dir.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        (removal_dir / f"iter_{idx}.png").write_bytes(b"")


def test_gap_in_numbering_resolves_to_recorded_base(tmp_path):
    # Iteration 2 was skipped: object 3 was detected on iteration 1's output.
    _write_manifest(tmp_path, 3, {"base_iter": 1})
    assert resolve_base_iteration(str(tmp_path), 3) == 1


def test_first_object_detected_on_source_frame(tmp_path):
    _write_manifest(tmp_path, 0, {"base_iter": None})
    assert resolve_base_iteration(str(tmp_path), 0) is None


def test_contiguous_numbering_matches_legacy(tmp_path):
    _write_manifest(tmp_path, 2, {"base_iter": 1})
    assert resolve_base_iteration(str(tmp_path), 2) == 1


def test_manifest_without_base_iter_derives_from_artifacts(tmp_path):
    # Data recorded before base_iter existed: derive the base from the last iteration
    # below idx that wrote a post-removal image — correct even when the data has gaps.
    _write_manifest(tmp_path, 3, {"removed_obj_phrase": "cup"})
    _write_removal_images(tmp_path, [0, 1, 3])
    assert resolve_base_iteration(str(tmp_path), 3) == 1


def test_legacy_contiguous_data_matches_old_behavior(tmp_path):
    _write_manifest(tmp_path, 3, {"removed_obj_phrase": "cup"})
    _write_removal_images(tmp_path, [0, 1, 2, 3])
    assert resolve_base_iteration(str(tmp_path), 3) == 2


def test_missing_manifest_derives_from_artifacts(tmp_path):
    _write_removal_images(tmp_path, [0, 1, 3])
    assert resolve_base_iteration(str(tmp_path), 3) == 1
    assert resolve_base_iteration(str(tmp_path), 0) is None


def test_no_manifest_no_artifacts_is_source_frame(tmp_path):
    assert resolve_base_iteration(str(tmp_path), 0) is None
