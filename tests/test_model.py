import sys, pathlib, json, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from model import load


def test_paper_cell_set_and_split():
    df = load()
    assert len(df) == 124 and df.life.between(140, 2300).all()
    sizes = df.split.value_counts().to_dict()
    assert sizes == {"train": 42, "primary_test": 42, "secondary_test": 40}
    assert (df[df.split == "secondary_test"].batch == "2018-04-12").all()     # later batch is held out whole


def test_no_cell_in_two_splits():
    df = load()
    assert df.groupby("cell_id").split.nunique().max() == 1 or df.groupby(["batch", "cell_id"]).split.nunique().max() == 1


def test_results_reproduce_and_beat_baselines():
    subprocess.run([sys.executable, str(ROOT / "src" / "model.py")], check=True, capture_output=True)
    r = json.load(open(ROOT / "results" / "results.json"))
    best = r["models"][r["chosen_by_train_cv"]]
    for sp in ("primary_test", "secondary_test"):
        assert best[sp]["mape_pct"] < r["models"]["naive_mean"][sp]["mape_pct"]
        assert best[sp]["mape_pct"] < r["models"]["capacity_only"][sp]["mape_pct"]
    assert r["corr_logvar_loglife"] < -0.9
