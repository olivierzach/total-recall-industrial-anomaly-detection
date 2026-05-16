from pathlib import Path

import numpy as np
from PIL import Image

from hybrid_memory_demo.run_btad_demo import _collect_btad_splits, _parse_btad_name


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)


def test_parse_btad_name_extracts_component_and_status():
    component, status = _parse_btad_name("data/btad/test/img/02_ko_0034.png")

    assert component == "02"
    assert status == "ko"


def test_collect_btad_splits_builds_known_and_unknown_protocol(tmp_path: Path):
    _write_rgb(tmp_path / "train" / "img" / "01_ok_0001.png")
    _write_rgb(tmp_path / "train" / "img" / "02_ok_0001.png")
    _write_rgb(tmp_path / "train" / "img" / "03_ok_0001.png")

    _write_rgb(tmp_path / "test" / "img" / "01_ok_0002.png")
    _write_rgb(tmp_path / "test" / "img" / "01_ko_0003.png")
    _write_rgb(tmp_path / "test" / "img" / "01_ko_0004.png")
    _write_rgb(tmp_path / "test" / "img" / "02_ko_0005.png")
    _write_rgb(tmp_path / "test" / "img" / "03_ko_0006.png")

    nominal_train, support_paths, eval_rows, unknown_components = _collect_btad_splits(
        btad_root=tmp_path,
        known_components={"01", "02"},
        support_per_class=1,
        nominal_train_cap=0,
        seed=0,
    )

    assert len(nominal_train) == 3
    assert sorted(support_paths) == ["component_01", "component_02"]
    assert any(row["ground_truth_status"] == "normal" for row in eval_rows)
    assert any(row["ground_truth_status"] == "known_failure" and row["ground_truth_label"] == "component_01" for row in eval_rows)
    assert any(row["ground_truth_status"] == "unknown_anomaly" and row["ground_truth_label"] == "component_03" for row in eval_rows)
    assert unknown_components == ["03"]
