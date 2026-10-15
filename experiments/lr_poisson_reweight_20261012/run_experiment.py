"""Clean Track A LR-conditioned Poisson scoreline experiment.

Run from the repository root:
  python experiments/lr_poisson_reweight_20261012/run_experiment.py

Never trains on the target/future season and never writes legacy output paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "league"))
from fusion_core import (  # noqa: E402
    aggregate_hda, adaptive_poisson_grid, reweight_scores, score_statistics,
)
from experiments.feature_search_20261010.run_feature_search import build_data  # noqa: E402
from scripts.tracka_poisson_goals import LoukasPoissonModel  # noqa: E402

SEED = 42
CLASSES = (0, 1, 2)  # A, D, H, matching existing code
MODELS = ("poisson", "lr", "fusion")
MODEL_PROVENANCE = "LoukasPoissonModel.fit / clean 66 fixed features / LR(C=1, lbfgs)"


def make_lr() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
        ("model", LogisticRegression(C=1.0, max_iter=4000, solver="lbfgs", random_state=SEED)),
    ])


def check_fold(train: pd.DataFrame, test: pd.DataFrame, season: str) -> None:
    if train.empty or test.empty:
        raise ValueError(f"Missing training/test matches in fold {season}")
    if not (train.season_id < season).all() or not (test.season_id == season).all():
        raise AssertionError(f"Cross-season training leakage in {season}")
    if not train.date.max() < test.date.min():
        raise AssertionError(f"Nonchronological season boundaries in {season}")
    if train.target_result.isna().any() or test.target_result.isna().any():
        raise ValueError("Missing match result labels")


def classification_metrics(y: np.ndarray, p: np.ndarray, model: str, fold: str) -> dict:
    if p.shape != (len(y), 3) or not np.isfinite(p).all() or (p < 0).any():
        raise ArithmeticError(f"Invalid {model} H/D/A probabilities")
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=0, atol=3e-12)
    pred = np.argmax(p, axis=1)
    return {
        "model": model, "fold": fold, "n_matches": len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
        "draw_f1": float(f1_score(y, pred, labels=CLASSES, average=None, zero_division=0)[1]),
        "hda_log_loss": float(log_loss(y, p, labels=CLASSES)),
        "brier": float(np.mean((np.eye(3)[y] - p) ** 2)),
    }


def score_metrics(predict: pd.DataFrame, prefix: str, fold: str) -> dict:
    n = len(predict)
    if not n:
        raise ValueError("Empty score predictions")
    cols = [
        "true_score_nll", "exact_score_hit", "home_abs_error", "away_abs_error",
        "total_abs_error", "home_within_1", "away_within_1", "total_within_1",
    ]
    v = {c: float(predict[f"{prefix}_{c}"].mean()) for c in cols}
    return {
        "model": prefix, "fold": fold, "n_matches": n,
        "true_score_nll": v["true_score_nll"],
        "modal_exact_score_accuracy": v["exact_score_hit"],
        "home_goal_mae_expected": v["home_abs_error"],
        "away_goal_mae_expected": v["away_abs_error"],
        "total_goal_mae_expected": v["total_abs_error"],
        "home_goal_within_1_expected": v["home_within_1"],
        "away_goal_within_1_expected": v["away_within_1"],
        "total_goal_within_1_expected": v["total_within_1"],
        "actual_outside_grid": int(predict[f"{prefix}_actual_outside_grid"].sum()),
    }


def predict_fold(train: pd.DataFrame, test: pd.DataFrame, cols: list[str],
                 season: str, tol: float) -> pd.DataFrame:
    """Both models fitted to earlier seasons only; use ALL three LR probabilities."""
    check_fold(train, test, season)
    if len(cols) != 66 or len(set(cols)) != 66:
        raise AssertionError("Unexpected change in original clean 66-feature specification")
    x_train = train[cols].to_numpy(dtype=float)
    x_test = test[cols].to_numpy(dtype=float)
    lr = make_lr()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        lr.fit(x_train, train.target_result.to_numpy(dtype=int))
    if any(isinstance(w.message, ConvergenceWarning) for w in caught):
        raise RuntimeError(f"LogisticRegression nonconvergence, fold {season}")
    assert np.array_equal(lr.named_steps["model"].classes_, CLASSES)
    lr_prob = lr.predict_proba(x_test)

    poisson = LoukasPoissonModel(max_goals=10, smoothing=0.05).fit(train)
    output_rows = []
    for idx, m in enumerate(test.itertuples(index=False)):
        q = lr_prob[idx]
        rate_h, rate_a = poisson.expected_goals(str(m.home_team), str(m.away_team))
        # Unlike the legacy score grid [0..10] this covers <=tol joint tail.
        grid = adaptive_poisson_grid(rate_h, rate_a, tail_tolerance=tol)
        poisson_score = grid.matrix
        fused_score = reweight_scores(poisson_score, q)
        pp = aggregate_hda(poisson_score)
        fp = aggregate_hda(fused_score)
        np.testing.assert_allclose(fp, q, atol=3e-12, rtol=0)
        masses = pp.copy()
        if np.any((masses < 1e-12) & (q > 0)):
            raise ArithmeticError(f"Near-zero Poisson H/D/A denominator in fold {season}, match {idx}")
        hg, ag = int(m.home_goals_full), int(m.away_goals_full)
        row = {
            "fold": season, "date": str(m.date.date()), "division": str(m.division),
            "home_team": str(m.home_team), "away_team": str(m.away_team),
            "actual_home_goals": hg, "actual_away_goals": ag,
            "actual_hda": int(m.target_result),
            "lambda_home": rate_h, "lambda_away": rate_a,
            "grid_max_goals": grid.max_goals, "poisson_tail_omitted": grid.omitted_mass,
            "min_poisson_hda_denominator": float(masses.min()),
            "max_fusion_lr_hda_discrepancy": float(np.max(np.abs(fp - q))),
        }
        for name, prob in (("lr", q), ("poisson", pp), ("fusion", fp)):
            row.update({f"{name}_p_away": float(prob[0]),
                        f"{name}_p_draw": float(prob[1]),
                        f"{name}_p_home": float(prob[2]),
                        f"{name}_predicted_hda": int(np.argmax(prob))})
        for name, mat in (("poisson", poisson_score), ("fusion", fused_score)):
            # For a very rare true score falling outside the prediction-time grid,
            # use the exact analytic Poisson scoreline mass without allowing
            # the observed score to alter the support chosen at prediction time.
            stats = score_statistics(mat, hg, ag,
                                     lambda_home=rate_h, lambda_away=rate_a,
                                     target_hda=q if name == "fusion" else None)
            row.update({f"{name}_{k}": v for k, v in stats.items()})
        output_rows.append(row)
    return pd.DataFrame(output_rows)


def run(df: pd.DataFrame, cols: list[str], tol: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seasons = sorted(df.season_id.dropna().unique())
    if len(seasons) != 7 or len(df) != 13988 or len(cols) != 66:
        raise AssertionError(f"Source mismatch: seasons={seasons}, matches={len(df)}, features={len(cols)}")
    preds = []
    for season in seasons[1:]:
        train = df.loc[df.season_id < season].copy()
        test = df.loc[df.season_id == season].copy()
        print(f"Fold {season}: train={len(train)}, test={len(test)}", flush=True)
        preds.append(predict_fold(train, test, cols, season, tol))
    predictions = pd.concat(preds, ignore_index=True)
    if len(predictions) != 12216 or predictions.groupby("fold").size().nunique() != 1:
        raise AssertionError("Walk-forward prediction count mismatch")
    # Exact algebraic consistency: conditional reweighting changes the actual
    # score log probability by exactly the true-class H/D/A log-probability ratio.
    # This must hold across ALL games, including extremely rare outside-grid goals.
    labels = predictions.actual_hda.to_numpy(int)
    take = np.arange(len(predictions))
    lr_prob = predictions[["lr_p_away", "lr_p_draw", "lr_p_home"]].to_numpy()
    p_prob = predictions[["poisson_p_away", "poisson_p_draw", "poisson_p_home"]].to_numpy()
    nll_delta = predictions.fusion_true_score_nll.to_numpy() - predictions.poisson_true_score_nll.to_numpy()
    expected_delta = -np.log(lr_prob[take, labels]) + np.log(p_prob[take, labels])
    np.testing.assert_allclose(nll_delta, expected_delta, rtol=0, atol=5e-10,
                               err_msg="Score NLL difference must equal H/D/A log-loss difference")
    hda_rows, score_rows = [], []
    for fold in [*seasons[1:], "ALL"]:
        sample = predictions if fold == "ALL" else predictions[predictions.fold == fold]
        y = sample.actual_hda.to_numpy(dtype=int)
        for name in MODELS:
            p = sample[[f"{name}_p_away", f"{name}_p_draw", f"{name}_p_home"]].to_numpy()
            hda_rows.append(classification_metrics(y, p, name, fold))
        for name in ("poisson", "fusion"):
            score_rows.append(score_metrics(sample, name, fold))
    return predictions, pd.DataFrame(hda_rows), pd.DataFrame(score_rows)


def bootstrap_uncertainty(predictions: pd.DataFrame, repeats: int = 1000) -> pd.DataFrame:
    """Paired, clustered-by-match-date CI stratified by held-out season.

    Resampling is descriptive uncertainty analysis after all folds have been
    evaluated; it does not tune any prediction model on held-out labels.
    """
    suffixes = [
        "true_score_nll", "exact_score_hit", "home_abs_error", "away_abs_error",
        "total_abs_error", "home_within_1", "away_within_1", "total_within_1",
    ]
    deltas = np.column_stack([
        predictions[f"fusion_{k}"].to_numpy(dtype=float) -
        predictions[f"poisson_{k}"].to_numpy(dtype=float) for k in suffixes
    ])
    tmp = predictions[["fold", "date"]].copy()
    for i, name in enumerate(suffixes):
        tmp[name] = deltas[:, i]
    grouped = tmp.groupby(["fold", "date"], sort=False)
    group_sums = grouped[suffixes].sum()
    group_counts = grouped.size()
    strata = []
    for fold in sorted(predictions.fold.unique()):
        chosen = group_sums.loc[fold]
        n = group_counts.loc[fold]
        strata.append((chosen.to_numpy(float), n.to_numpy(int)))
    rng = np.random.default_rng(SEED)
    boot = np.empty((repeats, len(suffixes)), dtype=float)
    for b in range(repeats):
        tot = np.zeros(len(suffixes), dtype=float)
        weight = 0
        for sums, counts in strata:
            selection = rng.integers(0, len(counts), size=len(counts))
            tot += sums[selection].sum(axis=0)
            weight += int(counts[selection].sum())
        boot[b] = tot / weight
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    point = deltas.mean(axis=0)
    return pd.DataFrame([{
        "metric": name, "fusion_minus_poisson": float(point[i]),
        "ci95_lower": float(lo[i]), "ci95_upper": float(hi[i]),
        "bootstrap_method": "paired date clusters stratified by season",
        "bootstrap_repeats": repeats,
    } for i, name in enumerate(suffixes)])


def digest_inputs(root: Path) -> list[dict]:
    paths = sorted((root / "league" / "data" / "english_raw").glob("E[0-3]*.csv"))
    if len(paths) != 28:
        raise AssertionError(f"Expected 28 English raw data files, got {len(paths)}")
    return [{"path": str(p.relative_to(root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in paths]


def report(hda: pd.DataFrame, goals: pd.DataFrame, predictions: pd.DataFrame,
           uncertainty: pd.DataFrame) -> str:
    def percent(x): return f"{x*100:.2f}%"
    l = [
        "# Clean Track A：逻辑回归指导泊松条件比分分布重加权",
        "", "## 数据口径与方法", "",
        "- 数据：英格兰 E0–E3 四级联赛，2019/20–2025/26 共 13,988 场；逐赛季外层测试为 2020/21–2025/26，共 12,216 场。",
        "- 每折逻辑回归与泊松模型均只使用目标赛季之前的数据拟合。LR 固定使用既有的 66 个赛前历史特征（C=1.0，无缩放或调参）；泊松模型仅用之前的全场进球。测试赛季已结束的较早比赛只可更新后续比赛的滚动历史特征，不更新已冻结的模型参数。",
        "- 沿用当前清洁特征构建：同一天比赛结果不会互相进入特征，修正 35 场七月跨赛季比赛的赛季归属；赔率、半场信息和当场统计不作为训练输入。",
        "- 复用现有 Loukas 风格攻防参数拟合过程，不改变泊松模型训练。比分矩阵根据预测的 λ 自适应扩展，联合尾部遗漏概率不超过 1e-12（或命令行指定值）。",
        "- 对每类赛果，融合比分概率 = LR 该类概率 × 原泊松该比分概率 ÷ 原泊松该类总概率；同时使用三个赛果的完整概率。",
        "- 逻辑回归没有由自身唯一确定的具体比分分布，所以比分层面的精确比分、NLL 和进球数指标仅在泊松与融合模型之间比较。",
        "- 逐场数学恒等式：真实比分 NLL（融合−泊松）= −log[LR 对真实赛果的概率] + log[泊松对真实赛果的概率]；故真实比分平均 NLL 差值等于同批比赛的胜平负 Log Loss 差值。程序逐场核验。",
        "- 进球 MAE 与 ±1 球误差覆盖率统一使用完整比分分布的期望进球数；精确比分命中率单独使用最大概率比分（众数）。",
        "- 实验仅写入新目录，不覆盖此前泊松或 LR 基线。",
        "", "## 胜平负指标", "",
        "| 测试赛季 | 模型 | 场次 | 准确率 | Macro F1 | 平局 F1 | 胜平负 Log Loss | Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in hda.itertuples():
        l.append(f"| {r.fold} | {r.model} | {r.n_matches} | {percent(r.accuracy)} | {r.macro_f1:.4f} | {r.draw_f1:.4f} | {r.hda_log_loss:.4f} | {r.brier:.4f} |")
    l += ["", "融合模型的胜平负聚合概率按定义与 LR 一致（允许浮点误差）；这属于正确性检查，而不是额外预测提升。",
          "", "## 比分分布与进球指标", "",
          "| 测试赛季 | 模型 | 场次 | 众数比分命中率 | 真实比分 NLL ↓ | 主队期望进球 MAE ↓ | 客队期望进球 MAE ↓ | 总期望进球 MAE ↓ | 主队 ±1 ↑ | 客队 ±1 ↑ | 总进球 ±1 ↑ |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in goals.itertuples():
        l.append(f"| {r.fold} | {r.model} | {r.n_matches} | {percent(r.modal_exact_score_accuracy)} | {r.true_score_nll:.4f} | {r.home_goal_mae_expected:.4f} | {r.away_goal_mae_expected:.4f} | {r.total_goal_mae_expected:.4f} | {percent(r.home_goal_within_1_expected)} | {percent(r.away_goal_within_1_expected)} | {percent(r.total_goal_within_1_expected)} |")
    overall = goals[goals.fold == "ALL"].set_index("model")
    p, f = overall.loc["poisson"], overall.loc["fusion"]
    delta_nll = float(f.true_score_nll - p.true_score_nll)
    delta_exact = float(f.modal_exact_score_accuracy - p.modal_exact_score_accuracy)
    delta_home = float(f.home_goal_mae_expected - p.home_goal_mae_expected)
    delta_away = float(f.away_goal_mae_expected - p.away_goal_mae_expected)
    better = delta_nll < 0 and delta_exact > 0 and delta_home < 0 and delta_away < 0
    l += ["", "## 总体配对不确定性", "",
          "采用按测试赛季分层、按完整比赛日期成组的配对 Bootstrap（1,000 次），报告 95% 区间。差值为“融合−泊松”，NLL/MAE 的负值有利，命中率/覆盖率的正值有利。此区间是测试结束后的不确定性评估，不用于模型调参。", "",
          "| 指标 | 差值 | 95% 区间下限 | 95% 区间上限 |",
          "|---|---:|---:|---:|"]
    for r in uncertainty.itertuples():
        l.append(f"| {r.metric} | {r.fusion_minus_poisson:+.5f} | {r.ci95_lower:+.5f} | {r.ci95_upper:+.5f} |")
    l += ["", "## 基于实际回测的结论", "",
          f"- 融合减泊松：真实比分 NLL 差值={delta_nll:+.5f} （负值更好）；众数精确比分命中率差值={delta_exact*100:+.3f} percentage points （正值更好）。",
          f"- 主队期望进球 MAE 差值={delta_home:+.5f}；客队期望进球 MAE 差值={delta_away:+.5f} （负值更好）。",
          "- 四项核心指标均有所改善。" if better else "- 融合模型**没有在全部四项核心指标上同时改善**。应结合逐季结果分别评价收益与代价。",
          f"- 泊松比分矩阵最大联合尾部遗漏概率：{predictions.poisson_tail_omitted.max():.3e}；泊松赛果组最小分母：{predictions.min_poisson_hda_denominator.min():.3e}；融合与 LR 最大赛果聚合误差：{predictions.max_fusion_lr_hda_discrepancy.max():.3e}.",
          f"- 实际比分超出赛前固定矩阵范围的比赛数：{int(predictions.poisson_actual_outside_grid.sum())}；这类比赛的真实比分概率采用解析泊松式求值，而非用实际比分扩展预测支持集。",
          "- 这是回顾性滚动回测，不能证明因果贡献或未来盈利能力；真实比分 NLL 是整体分布评价，而众数命中率和 MAE 衡量不同性质。",
          "", "## 复现产物", "",
          "- predictions_by_match.csv.gz：相同比赛上的 LR、泊松与融合胜平负概率，泊松 λ、期望比分、众数比分、真实比分概率与逐场误差。",
          "- hda_metrics.csv、score_metrics.csv：逐赛季及总体指标。",
          "- paired_uncertainty.csv：按赛季分层、按日期成组的配对 Bootstrap 区间。",
          "- data_audit.json、season_assignment_fixes.csv、feature_matrix.csv：复用的清洁特征构建审计文件。",
          "- run_metadata.json：库版本及原始 CSV 的 SHA-256 哈希。",
          "- 原 outputs/tracka_poisson_goals/ 与 nested_backtest_20261010/reference/ 不会被覆盖。", ""]
    return "\n".join(l)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--tail-tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    root = ROOT.resolve()
    out = args.output_dir.resolve()
    for legacy in [root / "outputs" / "tracka_poisson_goals", root / "experiments" / "nested_backtest_20261010" / "reference"]:
        if out == legacy.resolve() or legacy.resolve() in out.parents:
            raise ValueError("Cannot write inside existing baseline/reference outputs")
    out.mkdir(parents=True, exist_ok=True)
    inputs = digest_inputs(root)
    df, cols = build_data(root, out)
    predictions, hda, goals = run(df, cols, args.tail_tolerance)
    uncertainty = bootstrap_uncertainty(predictions)
    uncertainty.to_csv(out / "paired_uncertainty.csv", index=False, float_format="%.16g")
    predictions.to_csv(out / "predictions_by_match.csv.gz", compression="gzip", index=False, float_format="%.16g")
    hda.to_csv(out / "hda_metrics.csv", index=False, float_format="%.16g")
    goals.to_csv(out / "score_metrics.csv", index=False, float_format="%.16g")
    (out / "report_CN.md").write_text(report(hda, goals, predictions, uncertainty), encoding="utf-8")
    meta = {
        "complete": True, "rows_total": int(len(df)), "rows_evaluated": int(len(predictions)),
        "features": len(cols), "feature_names": cols,
        "folds": {k: int(v) for k, v in predictions.groupby("fold").size().items()},
        "models": MODEL_PROVENANCE, "class_order": list(CLASSES),
        "tail_tolerance": args.tail_tolerance,
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "inputs": inputs,
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Overall H/D/A ===\n", hda[hda.fold == "ALL"].to_string(index=False), flush=True)
    print("\n=== Overall scoreline ===\n", goals[goals.fold == "ALL"].to_string(index=False), flush=True)
    print(f"Complete. Reports and predictions: {out}", flush=True)


if __name__ == "__main__":
    main()
