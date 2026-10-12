"""Preprocessing tests on tiny SYNTHETIC files that mimic the raw PEMS-SF format (test fixtures only, never used for results)."""
import numpy as np
import pytest

from preprocessing import preprocess as pp

N_SENS, N_SLOTS = 3, 6


def _write_vec(path, arr):
    path.write_text("[" + " ".join(str(int(a)) for a in arr) + "]\n")


def _write_days(path, days):
    with open(path, "w") as fh:
        for d in days:
            fh.write("[" + "; ".join(" ".join(f"{v:.4f}" for v in row) for row in d) + "]\n")


@pytest.fixture()
def raw_dir(tmp_path):
    rng = np.random.default_rng(0)
    labels14 = np.array([(1 + d) % 7 + 1 for d in range(14)])            # 2,3,4,5,6,7,1,2,... (1 = Sunday)
    base = np.where(np.isin(labels14, [1, 7]), 0.01, 0.05)               # weekends clearly lower
    chrono14 = np.stack([np.clip(base[d] + rng.random((N_SENS, N_SLOTS)) * 0.01, 0, 1) for d in range(14)]).astype(np.float32)
    chrono = np.concatenate([chrono14, chrono14[:3]])                    # 3 exact duplicates, like the real files
    lab = np.concatenate([labels14, labels14[:3]])
    n = len(chrono)
    perm = rng.permutation(n) + 1                                        # stored[i] = chrono[perm[i]-1]
    stored, lab_stored = chrono[perm - 1], lab[perm - 1]
    n_train = 11
    _write_days(tmp_path / "PEMS_train", stored[:n_train]); _write_days(tmp_path / "PEMS_test", stored[n_train:])
    _write_vec(tmp_path / "PEMS_trainlabels", lab_stored[:n_train]); _write_vec(tmp_path / "PEMS_testlabels", lab_stored[n_train:])
    _write_vec(tmp_path / "randperm", perm); _write_vec(tmp_path / "stations_list", [400000, 400001, 400002])
    return tmp_path, chrono14, labels14


def test_parse_matrix_line():
    m = pp.parse_matrix_line("[0.1 0.2 0.3; 0.4 0.5 0.6]\n")
    assert m.shape == (2, 3) and m[1, 2] == pytest.approx(0.6)


def test_chronology_restored_and_duplicates_removed(raw_dir):
    path, chrono14, labels14 = raw_dir
    ds = pp.build_dataset(path, N_SENS, N_SLOTS)
    assert ds["report"]["stored_days"] == 17
    assert ds["report"]["distinct_days"] == 14 and ds["report"]["duplicate_days_removed"] == 3
    assert ds["cube"].shape == (14, N_SENS, N_SLOTS)
    np.testing.assert_allclose(ds["cube"], chrono14, atol=1e-4)           # exact chronological order recovered
    assert ds["report"]["weekday_chain_score_restored_order"] == 1.0


def test_weekday_encoding_inferred_from_data(raw_dir):
    path, _, labels14 = raw_dir
    ds = pp.build_dataset(path, N_SENS, N_SLOTS)
    assert ds["report"]["weekday_encoding"]["sunday_label"] in (1, 7)    # lowest-occupancy label
    # python weekday: label 2 must be Monday (0) when label 1 is Sunday
    first_dow = int(ds["dow"][0])
    assert first_dow == (labels14[0] - ds["report"]["weekday_encoding"]["sunday_label"] - 1) % 7


def test_shape_validation_rejects_bad_matrix(tmp_path):
    (tmp_path / "bad").write_text("[1 2 3; 4 5 6]\n")
    with pytest.raises(ValueError):
        pp.load_days(tmp_path / "bad", n_sensors=3, n_slots=3)


def test_clean_values_repairs_invalid_but_keeps_zero():
    cube = np.full((1, 2, 6), 0.05, dtype=np.float32)
    cube[0, 0, 2] = np.nan
    cube[0, 1, 3] = 1.7
    cube[0, 1, 0] = 0.0                                                  # genuine zero stays
    out, rep = pp.clean_values(cube)
    assert rep["invalid_values_repaired"] == 2 and np.isfinite(out).all() and out.max() <= 1
    assert out[0, 1, 0] == 0.0


def test_find_duplicates():
    a = np.random.default_rng(1).random((4, 2, 3)).astype(np.float32)
    a = np.concatenate([a, a[:1]])
    assert pp.find_duplicates(a).tolist() == [False, False, False, False, True]
