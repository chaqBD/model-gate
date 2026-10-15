"""Champion against challenger.

Known answers first — an identical champion, a strictly better one, a
strictly worse one — then the invariants that make the comparison
trustworthy (antisymmetry, row order, exposure scale), then the traps that
produce a confident wrong comparison rather than an error, each with a test
that fails without its guard.
"""

import json
import pickle
import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest

from bdp_model_gate import (
    Champion,
    ComparisonConfig,
    GateConfig,
    ModelGate,
    PerformanceConfig,
    StructuredGateContext,
    UncertaintyConfig,
)
from bdp_model_gate.comparison import (
    IMPROVED,
    NOT_BETTER_FLAG,
    WORSE_FLAG,
    compared,
    decide,
    delta_statistic,
    paired_interval,
    resampling_frame,
    resolve_comparison_metric,
)
from bdp_model_gate.exceptions import GateConfigurationError, GateValidationError
from bdp_model_gate.stats import rank_auc
from bdp_model_gate.structured import default_structured_checks
from bdp_model_gate.structured.actuarial_checks import DislocationCheck
from bdp_model_gate.structured.comparison_checks import (
    ChallengerGuardrailCheck,
    ChallengerPerformanceCheck,
    ComparisonIntegrityCheck,
    DecisionChurnCheck,
    FairnessDeltaCheck,
    OperationalDeltaCheck,
    SubgroupRegressionCheck,
    comparison_checks,
)
from bdp_model_gate.uncertainty import Interval


def auc(y_true, y_score):
    """ROC AUC without scikit-learn — `stats.rank_auc`, which
    `tests/test_uncertainty.py` holds equal to `roc_auc_score`. Most of these
    tests are about the comparison, not the metric, and must run on a core
    install: there, naming `roc_auc` is a configuration error by design."""
    return rank_auc(np.asarray(y_true), np.asarray(y_score, dtype=float))


# --------------------------------------------------------------------------
# Fixtures: scores whose quality is known by construction
# --------------------------------------------------------------------------


def _scores(y, noise, rng):
    """A score that ranks `y` well when `noise` is small."""
    return 1.0 / (1.0 + np.exp(-(3.0 * (y - 0.5) + rng.normal(scale=noise, size=len(y)))))


@pytest.fixture
def binary():
    rng = np.random.default_rng(11)
    n = 1500
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    good = _scores(y, 1.5, rng)  # roughly AUC 0.92
    poor = _scores(y, 4.0, rng)  # roughly AUC 0.70
    protected = pd.DataFrame({"region": rng.choice(["north", "south"], n)})
    return X, y, good, poor, protected


class Fixed:
    """A 'model' that returns a fixed vector — what a champion scored elsewhere
    looks like to the adapter."""

    def __init__(self, scores):
        self.scores = np.asarray(scores)

    def predict(self, X):
        return self.scores[: len(X)]


def _context(X, y, challenger, champion, **kw):
    return StructuredGateContext(
        model=Fixed(challenger),
        X=X,
        y_true=y,
        y_pred=challenger,
        champion=champion,
        task=kw.pop("task", "binary"),
        **kw,
    )


def _run(check, context):
    return check.run(context)


def _only(results):
    (result,) = results
    return result


# --------------------------------------------------------------------------
# The decision rule, on its own
# --------------------------------------------------------------------------


def _interval(low, high, point=None):
    return Interval(
        point=(low + high) / 2 if point is None else point,
        low=low,
        high=high,
        level=0.95,
        samples=1000,
    )


@pytest.mark.parametrize(
    ("low", "high", "require", "flag", "outcome"),
    [
        (0.01, 0.03, "non_inferior", "OK", "improved"),
        (-0.005, 0.02, "non_inferior", "OK", "non_inferior"),
        (-0.010, 0.02, "non_inferior", "OK", "non_inferior"),  # exactly on −margin passes
        (-0.02, 0.01, "non_inferior", "UNCERTAIN", "uncertain"),
        (-0.05, -0.02, "non_inferior", WORSE_FLAG, "worse"),
        (0.01, 0.03, "superior", "OK", "improved"),
        (-0.005, 0.0, "superior", NOT_BETTER_FLAG, "not_improved"),
        (-0.005, 0.02, "superior", "UNCERTAIN", "uncertain"),
        (-0.05, -0.02, "superior", WORSE_FLAG, "worse"),
    ],
)
def test_the_decision_table(low, high, require, flag, outcome):
    decision = decide(
        _interval(low, high),
        (low + high) / 2,
        margin=0.01,
        min_improvement=0.0,
        require=require,
        on_uncertain="review",
    )
    assert (decision.flag, decision.outcome) == (flag, outcome)
    if flag in (WORSE_FLAG, NOT_BETTER_FLAG):
        assert decision.blocking
    if flag == "UNCERTAIN":
        assert not decision.blocking


def test_a_straddle_follows_on_uncertain():
    straddle = _interval(-0.03, 0.01, point=-0.005)
    kwargs = dict(margin=0.01, min_improvement=0.0, require="non_inferior")
    assert decide(straddle, -0.005, on_uncertain="review", **kwargs).flag == "UNCERTAIN"
    assert decide(straddle, -0.005, on_uncertain="block", **kwargs).flag == WORSE_FLAG
    accepted = decide(straddle, -0.005, on_uncertain="point", **kwargs)
    assert accepted.flag == "OK" and "on_uncertain='point'" in accepted.note


def test_without_an_interval_the_point_estimate_decides_and_says_so():
    worse = decide(
        None, -0.2, margin=0.01, min_improvement=0.0, require="non_inferior", on_uncertain="review"
    )
    assert worse.flag == WORSE_FLAG
    assert "point estimate decides" in worse.note


# --------------------------------------------------------------------------
# Known answers
# --------------------------------------------------------------------------


def test_an_identical_champion_is_non_inferior_with_zero_delta(binary):
    X, y, good, _, _ = binary
    context = _context(X, y, good, Champion(predictions=good))
    result = _only(
        ChallengerPerformanceCheck(performance=PerformanceConfig(metric=auc)).run(context)
    )

    assert result.flag == "OK"
    assert result.metadata["delta"] == 0.0
    assert result.metadata["outcome"] == "non_inferior"
    churn = _only(DecisionChurnCheck().run(context))
    assert churn.metadata["flip_rate"] == 0.0


def test_a_better_challenger_is_improved_and_a_worse_one_blocks(binary):
    X, y, good, poor, _ = binary
    performance = PerformanceConfig(metric=auc, min_score=0.0)

    better = _only(
        ChallengerPerformanceCheck(performance=performance).run(
            _context(X, y, good, Champion(predictions=poor))
        )
    )
    assert better.flag == "OK" and better.metadata["outcome"] == IMPROVED
    assert better.metadata["ci_low"] > 0

    worse = _only(
        ChallengerPerformanceCheck(performance=performance).run(
            _context(X, y, poor, Champion(predictions=good))
        )
    )
    assert worse.flag == WORSE_FLAG and worse.blocking

    report = ModelGate(
        checks=default_structured_checks(GateConfig(performance=performance), include_plugins=False)
    ).run(_context(X, y, poor, Champion(predictions=good)))
    assert report.gate_status == "BLOCKED"


def test_require_superior_blocks_a_merely_equal_challenger(binary):
    X, y, good, _, _ = binary
    check = ChallengerPerformanceCheck(
        ComparisonConfig(require="superior"), PerformanceConfig(metric=auc)
    )
    result = _only(check.run(_context(X, y, good, Champion(predictions=good))))
    assert result.flag == NOT_BETTER_FLAG
    assert result.blocking


def test_an_error_metric_is_signed_so_lower_reads_as_better():
    rng = np.random.default_rng(3)
    n = 600
    y = rng.normal(100, 20, n)
    X = pd.DataFrame({"x": rng.normal(size=n)})
    close = y + rng.normal(scale=2, size=n)
    far = y + rng.normal(scale=10, size=n)

    result = _only(
        ChallengerPerformanceCheck(performance=PerformanceConfig(metric="rmse", max_error=1e9)).run(
            _context(X, y, close, Champion(predictions=far), task="regression")
        )
    )
    assert result.metadata["challenger_score"] < result.metadata["champion_score"]
    assert result.metadata["delta"] > 0
    assert result.metadata["outcome"] == IMPROVED


def test_the_relative_margin_is_a_fraction_of_the_champions_score(binary):
    X, y, good, poor, _ = binary
    result = _only(
        ChallengerPerformanceCheck(
            ComparisonConfig(margin_relative=0.05), PerformanceConfig(metric=auc)
        ).run(_context(X, y, good, Champion(predictions=poor)))
    )
    assert result.metadata["margin"] == pytest.approx(
        0.05 * result.metadata["champion_score"], abs=1e-4
    )
    assert result.metadata["margin_is_relative"] is True

    absolute = _only(
        ChallengerPerformanceCheck(
            ComparisonConfig(margin=0.002), PerformanceConfig(metric=auc)
        ).run(_context(X, y, good, Champion(predictions=poor)))
    )
    assert absolute.metadata["margin"] == 0.002
    assert absolute.metadata["margin_is_relative"] is False


def test_a_zero_margin_is_allowed_and_warned_about(binary, caplog):
    X, y, good, _, _ = binary
    with caplog.at_level("WARNING", logger="bdp_model_gate"):
        ChallengerPerformanceCheck(ComparisonConfig(margin=0.0), PerformanceConfig(metric=auc)).run(
            _context(X, y, good, Champion(predictions=good))
        )
    assert "margin is zero" in caplog.text


# --------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------


@pytest.mark.real_bootstrap
def test_swapping_the_models_negates_every_delta_and_swaps_fixes_with_breaks(binary):
    """Exact for the point estimate and the counts. The interval is mirrored
    only to within resampling noise: the resample order is derived from row
    *content*, and swapping which column holds which model changes it."""
    X, y, good, poor, _ = binary
    performance = PerformanceConfig(metric=auc)
    forward = _context(X, y, good, Champion(predictions=poor))
    backward = _context(X, y, poor, Champion(predictions=good))

    a = _only(ChallengerPerformanceCheck(performance=performance).run(forward)).metadata
    b = _only(ChallengerPerformanceCheck(performance=performance).run(backward)).metadata
    assert a["delta"] == -b["delta"]
    assert (a["challenger_score"], a["champion_score"]) == (
        b["champion_score"],
        b["challenger_score"],
    )
    assert a["ci_low"] == pytest.approx(-b["ci_high"], abs=0.01)
    assert a["ci_high"] == pytest.approx(-b["ci_low"], abs=0.01)

    churn_a = _only(DecisionChurnCheck().run(forward)).metadata
    churn_b = _only(DecisionChurnCheck().run(backward)).metadata
    assert (churn_a["fixed"], churn_a["broken"]) == (churn_b["broken"], churn_b["fixed"])


def test_row_order_does_not_move_the_comparison(binary):
    X, y, good, poor, protected = binary
    order = np.random.default_rng(0).permutation(len(X))

    def run(rows):
        context = _context(
            X.iloc[rows].reset_index(drop=True),
            y[rows],
            good[rows],
            Champion(predictions=poor[rows]),
            protected_df=protected.iloc[rows].reset_index(drop=True),
        )
        checks = comparison_checks(performance=PerformanceConfig(metric=auc))
        return {
            (r.check_name, r.flag, r.metadata.get("ci_low")) for c in checks for r in c.run(context)
        }

    assert run(np.arange(len(X))) == run(order)


def test_the_paired_interval_is_far_narrower_than_two_separate_ones(binary):
    """The reason the comparison is paired. Two models scored on the same rows
    make correlated errors, so checking whether their separate intervals
    overlap would call this real improvement UNCERTAIN."""
    X, y, good, _, _ = binary
    rng = np.random.default_rng(5)
    slightly_worse = np.clip(good + rng.normal(scale=0.08, size=len(good)), 0.001, 0.999)
    context = _context(X, y, good, Champion(predictions=slightly_worse))
    compared_ = compared(context, 0.5)
    metric = resolve_comparison_metric(auc, "binary")
    statistic, _, _ = delta_statistic(metric, compared_)
    uncertainty = UncertaintyConfig(bootstrap_samples=400)
    frame = resampling_frame(compared_)

    paired = paired_interval(statistic, frame, uncertainty)
    ours = paired_interval(lambda r: metric.score(y[r], good[r]), frame, uncertainty)
    theirs = paired_interval(lambda r: metric.score(y[r], slightly_worse[r]), frame, uncertainty)

    assert paired.low > 0, "the improvement is real"
    assert ours.low < theirs.high, "...and the separate intervals overlap anyway"
    assert paired.width < 0.5 * min(ours.width, theirs.width)


def test_exposure_weights_follow_their_rows_through_the_bootstrap():
    """A weight vector bound to the full frame would be paired with the wrong
    row on every resample. Rescaling exposure is a change of units, so it must
    not move either the point or the interval."""
    rng = np.random.default_rng(9)
    n = 400
    y = rng.gamma(2.0, 50.0, n)
    X = pd.DataFrame({"x": rng.normal(size=n)})
    exposure = rng.uniform(0.1, 1.0, n)
    close, far = y + rng.normal(scale=5, size=n), y + rng.normal(scale=25, size=n)
    performance = PerformanceConfig(metric="rmse", max_error=1e9)

    def run(scale):
        return _only(
            ChallengerPerformanceCheck(performance=performance).run(
                _context(
                    X,
                    y,
                    close,
                    Champion(predictions=far),
                    task="regression",
                    exposure=exposure * scale,
                )
            )
        ).metadata

    one, twelve = run(1.0), run(12.0)
    assert one["exposure_weighted"] is True
    weighted = np.sqrt(np.average((y - close) ** 2, weights=exposure))
    assert one["challenger_score"] == pytest.approx(weighted, abs=1e-4)
    assert (one["delta"], one["ci_low"], one["ci_high"]) == pytest.approx(
        (twelve["delta"], twelve["ci_low"], twelve["ci_high"])
    )


# --------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------


def test_a_calibration_guardrail_blocks_a_badly_calibrated_challenger(binary):
    X, y, good, _, _ = binary
    overconfident = np.where(good > 0.5, 0.995, 0.005)  # same ranking, wrecked calibration
    results = ChallengerGuardrailCheck(ComparisonConfig(guardrails={"brier": 0.01})).run(
        _context(X, y, overconfident, Champion(predictions=good))
    )
    brier = _only(results)
    assert brier.metadata["guardrail"] == "brier"
    assert brier.flag == WORSE_FLAG and brier.blocking


def test_guardrail_intervals_widen_with_the_number_of_guardrails(binary):
    X, y, good, poor, _ = binary
    context = _context(X, y, good, Champion(predictions=poor))
    one = ChallengerGuardrailCheck(ComparisonConfig(guardrails=["brier"])).run(context)[0]
    two = ChallengerGuardrailCheck(ComparisonConfig(guardrails=["brier", "ece"])).run(context)[0]
    assert two.metadata["ci_level"] > one.metadata["ci_level"]
    assert (two.metadata["ci_high"] - two.metadata["ci_low"]) > (
        one.metadata["ci_high"] - one.metadata["ci_low"]
    )


def test_a_guardrail_that_does_not_apply_says_so_rather_than_failing():
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    X = pd.DataFrame({"x": y})
    result = _only(
        ChallengerGuardrailCheck(ComparisonConfig(guardrails=["brier"])).run(
            _context(X, y, y, Champion(predictions=y), task="regression")
        )
    )
    assert result.flag == "NOT_APPLICABLE"
    assert "probabilities" in result.detail


@pytest.mark.parametrize(
    "bad",
    [
        dict(require="better"),
        dict(guardrails=["not_a_metric"]),
        dict(guardrails={"brier": -0.1}),
        dict(margin=-0.01),
        dict(margin_relative=-1),
        dict(max_latency_ratio=0),
        dict(min_segment_size=1),
    ],
)
def test_a_bad_comparison_config_fails_when_the_suite_is_built(bad):
    with pytest.raises(GateConfigurationError):
        default_structured_checks(
            GateConfig(comparison=ComparisonConfig(**bad)), include_plugins=False
        )


# --------------------------------------------------------------------------
# Who it is better for
# --------------------------------------------------------------------------


def test_a_challenger_better_overall_and_worse_for_one_group_routes_to_review(binary):
    X, y, good, poor, protected = binary
    south = (protected["region"] == "south").to_numpy()
    # Better than the champion in the north, much worse in the south.
    challenger = np.where(south, poor, good)
    champion = np.where(south, good, poor)
    champion = np.where(south, champion, _scores(y, 2.5, np.random.default_rng(2)))
    context = _context(X, y, challenger, Champion(predictions=champion), protected_df=protected)

    result = _only(SubgroupRegressionCheck(performance=PerformanceConfig(metric=auc)).run(context))
    assert result.flag == "SUBGROUP_WORSE"
    assert not result.blocking
    assert "south" in result.detail
    groups = {g["group"]: g["outcome"] for g in result.metadata["groups"]}
    assert groups["south"] == "worse" and groups["north"] != "worse"


def test_segments_can_be_feature_columns_and_numeric_ones_are_cut_into_quintiles(binary):
    X, y, good, poor, _ = binary
    result = _only(
        SubgroupRegressionCheck(
            ComparisonConfig(segments=["a"]), PerformanceConfig(metric=auc)
        ).run(_context(X, y, good, Champion(predictions=poor)))
    )
    assert {g["group"] for g in result.metadata["groups"]} == {"Q1", "Q2", "Q3", "Q4", "Q5"}


def test_an_unknown_segment_column_is_a_configuration_error(binary):
    X, y, good, poor, _ = binary
    with pytest.raises(GateConfigurationError, match="nowhere"):
        SubgroupRegressionCheck(ComparisonConfig(segments=["nowhere"])).run(
            _context(X, y, good, Champion(predictions=poor))
        )


def test_a_widened_disparity_is_reported_even_inside_the_absolute_threshold(binary):
    X, y, _, _, protected = binary
    south = (protected["region"] == "south").to_numpy()
    rng = np.random.default_rng(4)
    champion = rng.uniform(0, 1, len(y))
    challenger = np.where(south, champion * 0.6, champion)  # selects far fewer in the south

    results = FairnessDeltaCheck().run(
        _context(X, y, challenger, Champion(predictions=champion), protected_df=protected)
    )
    parity = next(r for r in results if r.metadata["statistic"] == "demographic parity difference")
    assert parity.flag == "FAIRNESS_WORSE"
    assert parity.metadata["change"] > 0.1
    assert not parity.blocking


def test_churn_counts_fixes_and_breaks_by_hand():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0] * 10)
    champion = np.array([1, 0, 0, 1, 1, 0, 0, 0] * 10, dtype=float)
    challenger = np.array([1, 1, 0, 0, 0, 0, 1, 1] * 10, dtype=float)
    X = pd.DataFrame({"x": np.arange(len(y))})
    m = _only(
        DecisionChurnCheck().run(_context(X, y, challenger, Champion(predictions=champion)))
    ).metadata
    # Per block of 8: fixed rows 1, 3, 6; broken rows 4, 7; both right 0, 2, 5.
    assert (m["fixed"], m["broken"], m["both_right"], m["both_wrong"]) == (30, 20, 30, 0)
    assert m["break_rate"] == 0.25


def test_each_model_is_cut_at_its_own_operating_threshold():
    y = np.array([1, 0] * 50)
    scores = np.where(y == 1, 0.45, 0.30)
    X = pd.DataFrame({"x": np.arange(len(y))})
    same = _only(
        DecisionChurnCheck().run(
            _context(X, y, scores, Champion(predictions=scores, decision_threshold=0.4))
        )
    )
    # At 0.5 the challenger calls everything 0; at 0.4 the champion is perfect.
    assert same.metadata["broken"] == 50


def test_the_operational_comparison_is_a_ratio(binary):
    X, y, good, _, _ = binary
    context = _context(
        X,
        y,
        good,
        Champion(predictions=good, latencies_ms=[10.0] * 50, cost_per_inference=0.001),
        latencies_ms=[40.0] * 50,
        cost_per_inference=0.0012,
    )
    by_measure = {r.metadata["measure"]: r for r in OperationalDeltaCheck().run(context)}
    assert by_measure["p95 latency"].flag == "OPERATIONAL_COST_RISK"
    assert by_measure["p95 latency"].metadata["ratio"] == 4.0
    assert by_measure["cost per inference"].flag == "OK"


def test_dislocation_reads_the_champions_predictions_when_no_baseline_is_given():
    rng = np.random.default_rng(6)
    n = 300
    old = rng.uniform(100, 200, n)
    new = old * np.where(rng.uniform(size=n) < 0.3, 1.5, 1.0)
    X = pd.DataFrame({"x": rng.normal(size=n)})
    result = _only(
        DislocationCheck().run(_context(X, old, new, Champion(predictions=old), task="regression"))
    )
    assert result.metadata["baseline_source"] == "champion"
    assert result.flag == "DISLOCATION_RISK"


# --------------------------------------------------------------------------
# Integrity: the traps
# --------------------------------------------------------------------------


def test_an_inverted_champion_is_caught_rather_than_reported_as_a_big_win(binary):
    X, y, good, _, _ = binary
    inverted = 1.0 - good
    result = _only(
        ComparisonIntegrityCheck().run(_context(X, y, good, Champion(predictions=inverted)))
    )
    assert result.flag == "COMPARISON_INTEGRITY_RISK"
    assert result.blocking
    assert "backwards" in result.detail


def test_probabilities_against_labels_is_caught_for_a_ranking_metric(binary):
    X, y, good, _, _ = binary
    labels = (good > 0.5).astype(float)
    context = _context(X, y, good, Champion(predictions=labels))

    ranking = _only(
        ComparisonIntegrityCheck(performance=PerformanceConfig(metric=auc)).run(context)
    )
    assert ranking.flag == "COMPARISON_INTEGRITY_RISK"
    assert "ranking against a cut-off" in ranking.detail

    labelled = _only(
        ComparisonIntegrityCheck(performance=PerformanceConfig(metric="accuracy")).run(context)
    )
    assert labelled.flag == "OK"


def test_labels_the_outcome_never_takes_are_caught(binary):
    X, y, good, _, _ = binary
    words = np.where(good > 0.5, "yes", "no")
    result = _only(
        ComparisonIntegrityCheck(performance=PerformanceConfig(metric="accuracy")).run(
            _context(X, y, (good > 0.5).astype(int), Champion(predictions=words))
        )
    )
    assert result.flag == "COMPARISON_INTEGRITY_RISK"
    assert "never occur in y_true" in result.detail


def test_a_champion_that_cannot_score_the_rows_blocks_and_the_rest_stand_aside(binary):
    X, y, good, _, _ = binary
    champion = Champion(Fixed(good), features=["a", "not_in_X"])
    context = _context(X, y, good, champion)

    integrity = _only(ComparisonIntegrityCheck().run(context))
    assert integrity.flag == "COMPARISON_INTEGRITY_RISK" and integrity.blocking
    assert "not_in_X" in integrity.detail
    performance = _only(ChallengerPerformanceCheck().run(context))
    assert performance.flag == "NOT_APPLICABLE"
    assert "comparison_integrity" in performance.detail


def test_classes_fitted_in_a_different_encoding_are_caught():
    sklearn = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(8)
    X = pd.DataFrame({"a": rng.normal(size=300)})
    y = (X["a"] > 0).astype(int).to_numpy()
    challenger = sklearn.LogisticRegression().fit(X, y)
    champion = sklearn.LogisticRegression().fit(X, np.where(y == 1, "yes", "no"))

    result = _only(
        ComparisonIntegrityCheck().run(
            StructuredGateContext(
                model=challenger,
                X=X,
                y_true=y,
                y_pred=challenger.predict_proba(X)[:, 1],
                champion=Champion(champion),
                task="binary",
            )
        )
    )
    assert result.flag == "COMPARISON_INTEGRITY_RISK"
    assert "different classes" in result.detail


def test_the_champion_receives_only_its_own_columns():
    sklearn = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(10)
    X = pd.DataFrame({"a": rng.normal(size=400), "new_feature": rng.normal(size=400)})
    y = (X["a"] + X["new_feature"] > 0).astype(int).to_numpy()
    old = sklearn.LogisticRegression().fit(X[["a"]], y)
    new = sklearn.LogisticRegression().fit(X, y)

    context = StructuredGateContext(
        model=new,
        X=X,
        y_true=y,
        y_pred=new.predict_proba(X)[:, 1],
        champion=Champion(old),
        task="binary",
    )
    from sklearn.metrics import roc_auc_score

    result = _only(
        ChallengerPerformanceCheck(performance=PerformanceConfig(metric="roc_auc")).run(context)
    )
    expected = roc_auc_score(y, old.predict_proba(X[["a"]])[:, 1])
    assert result.metadata["champion_score"] == pytest.approx(expected, abs=1e-4)
    assert result.metadata["outcome"] == IMPROVED


def test_without_a_champion_every_comparison_check_stands_aside(binary):
    X, y, good, _, protected = binary
    context = StructuredGateContext(
        model=Fixed(good), X=X, y_true=y, y_pred=good, protected_df=protected, task="binary"
    )
    for check in comparison_checks():
        for result in check.run(context):
            assert result.flag == "NOT_APPLICABLE"
            assert "champion" in result.detail


@pytest.mark.parametrize(
    ("champion", "message"),
    [
        ("not a champion", "must be a Champion"),
        (Champion(predictions=[0.1, 0.2]), "row-aligned"),
    ],
)
def test_the_champion_is_validated_before_any_check_runs(binary, champion, message):
    X, y, good, _, _ = binary
    with pytest.raises(GateValidationError, match=message):
        ModelGate().run(_context(X, y, good, champion))


# --------------------------------------------------------------------------
# Loading a champion
# --------------------------------------------------------------------------


class Model:
    def __init__(self, value=0.5):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


class InconsistentVersionWarning(UserWarning):
    """Named like scikit-learn's, which is what the loader matches on."""


class WarnsOnLoad(Model):
    def __setstate__(self, state):
        warnings.warn(
            InconsistentVersionWarning(
                "Trying to unpickle estimator from version 1.2 when using 1.6"
            ),
            stacklevel=1,
        )
        self.__dict__.update(state)


class WarnsSomethingElse(Model):
    def __setstate__(self, state):
        warnings.warn("an unrelated notice", UserWarning, stacklevel=1)
        self.__dict__.update(state)


class Explodes:
    def __reduce__(self):
        return (_record_execution, ())


EXECUTED = []


def _record_execution():
    EXECUTED.append(True)
    return Model()


def _pickle(path, obj):
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)
    return path


def test_a_pickle_loads_and_records_its_provenance(tmp_path):
    path = _pickle(tmp_path / "v3.pkl", Model(0.3))
    champion = Champion.from_file(path)
    described = champion.describe()

    assert champion.model.value == 0.3
    assert described["source"] == "file"
    assert described["loader"] in {"joblib", "pickle"}
    assert len(described["sha256"]) == 64
    assert described["size_bytes"] == path.stat().st_size
    assert described["sha256_pinned"] is False


def test_a_pinned_hash_that_does_not_match_is_never_unpickled(tmp_path):
    path = _pickle(tmp_path / "v3.pkl", Explodes())
    EXECUTED.clear()
    with pytest.raises(GateConfigurationError, match="does not match its pinned sha256"):
        Champion.from_file(path, sha256="0" * 64)
    assert EXECUTED == [], "the pickle ran before its hash was checked"


def test_a_pinned_hash_that_matches_loads_and_is_recorded_as_pinned(tmp_path):
    from bdp_model_gate.champion import sha256_of

    path = _pickle(tmp_path / "v3.pkl", Model())
    champion = Champion.from_file(path, sha256=sha256_of(path).upper())
    assert champion.describe()["sha256_pinned"] is True


def test_a_version_warning_while_unpickling_is_an_error(tmp_path):
    path = _pickle(tmp_path / "v3.pkl", WarnsOnLoad())
    with pytest.raises(GateConfigurationError, match="Champion\\(predictions=") as raised:
        Champion.from_file(path)
    assert "version" in str(raised.value)


def test_an_unrelated_warning_is_passed_on_and_the_load_succeeds(tmp_path):
    path = _pickle(tmp_path / "v3.pkl", WarnsSomethingElse())
    with pytest.warns(UserWarning, match="an unrelated notice"):
        champion = Champion.from_file(path)
    assert isinstance(champion.model, WarnsSomethingElse)


def test_a_pickle_that_cannot_be_read_names_the_predictions_route(tmp_path):
    path = tmp_path / "v3.pkl"
    path.write_bytes(b"not a pickle")
    with pytest.raises(GateConfigurationError, match="could not be unpickled"):
        Champion.from_file(path)


@pytest.mark.parametrize("name", ["model.pt", "model.pth"])
def test_torch_files_are_refused_with_a_pointer_to_from_loader(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"\x00")
    with pytest.raises(GateConfigurationError, match="from_loader"):
        Champion.from_file(path)


def test_a_saved_model_directory_is_refused(tmp_path):
    with pytest.raises(GateConfigurationError, match="from_loader"):
        Champion.from_file(tmp_path)


def test_an_unknown_extension_and_a_missing_file_are_refused(tmp_path):
    odd = tmp_path / "model.bin"
    odd.write_bytes(b"\x00")
    with pytest.raises(GateConfigurationError, match="no loader"):
        Champion.from_file(odd)
    with pytest.raises(GateConfigurationError, match="not found"):
        Champion.from_file(tmp_path / "missing.joblib")


def _fake_skops(monkeypatch, untrusted):
    loaded = []
    io = types.ModuleType("skops.io")
    io.get_untrusted_types = lambda file: list(untrusted)
    io.load = lambda path: loaded.append(path) or Model(0.7)
    package = types.ModuleType("skops")
    package.io = io
    monkeypatch.setitem(sys.modules, "skops", package)
    monkeypatch.setitem(sys.modules, "skops.io", io)
    return loaded


def test_skops_refuses_untrusted_types_rather_than_trusting_them_for_you(tmp_path, monkeypatch):
    loaded = _fake_skops(monkeypatch, untrusted=["evil.Module"])
    path = tmp_path / "v3.skops"
    path.write_bytes(b"\x00")
    with pytest.raises(GateConfigurationError, match="evil.Module"):
        Champion.from_file(path)
    assert loaded == []


def test_skops_loads_a_trusted_file(tmp_path, monkeypatch):
    _fake_skops(monkeypatch, untrusted=[])
    path = tmp_path / "v3.skops"
    path.write_bytes(b"\x00")
    champion = Champion.from_file(path)
    assert champion.model.value == 0.7
    assert champion.describe()["loader"] == "skops"


def test_onnx_feeds_the_declared_dtype_and_unzips_probabilities(tmp_path, monkeypatch):
    fed = {}

    class Spec:
        def __init__(self, name, type_):
            self.name, self.type = name, type_

    class Session:
        def __init__(self, path):
            pass

        def get_inputs(self):
            return [Spec("input", "tensor(float)")]

        def get_outputs(self):
            return [Spec("label", ""), Spec("probabilities", "")]

        def run(self, outputs, feed):
            fed.update(feed)
            n = len(feed["input"])
            return [np.ones(n, dtype=np.int64), [{0: 0.2, 1: 0.8}] * n]

    ort = types.ModuleType("onnxruntime")
    ort.InferenceSession = Session
    ort.__version__ = "1.99"
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    path = tmp_path / "v3.onnx"
    path.write_bytes(b"\x00")

    champion = Champion.from_file(path)
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert champion.adapter.predict(X).tolist() == [1, 1]
    assert fed["input"].dtype == np.float32
    assert champion.adapter.predict_positive_proba(X).tolist() == pytest.approx([0.8, 0.8])
    assert champion.describe()["versions"] == {"onnxruntime": "1.99"}


def test_keras_files_go_through_keras(tmp_path, monkeypatch):
    keras = types.ModuleType("keras")
    keras.models = types.SimpleNamespace(load_model=lambda path: Model(0.9))
    monkeypatch.setitem(sys.modules, "keras", keras)
    path = tmp_path / "v3.keras"
    path.write_bytes(b"\x00")
    assert Champion.from_file(path).model.value == 0.9


def make_champion():
    """A loader for `from_loader`, addressed as tests.test_comparison:make_champion."""
    return Model(0.4)


def test_a_loader_spec_is_imported_and_called():
    champion = Champion.from_loader("tests.test_comparison:make_champion")
    assert champion.model.value == 0.4
    assert champion.describe()["loader"] == "tests.test_comparison:make_champion"


@pytest.mark.parametrize(
    "spec", ["no_colon", "tests.test_comparison:nothing_here", "missing.module:fn"]
)
def test_a_bad_loader_spec_is_a_configuration_error(spec):
    with pytest.raises(GateConfigurationError):
        Champion.from_loader(spec)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(model=Model(), predictions=[1.0]),
        dict(predictions=[0.5], decision_threshold=1.5),
    ],
)
def test_a_champion_needs_exactly_one_source_and_a_sane_threshold(kwargs):
    with pytest.raises(GateConfigurationError):
        Champion(**kwargs)


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def test_the_report_leads_with_the_comparison(binary):
    X, y, good, poor, protected = binary
    config = GateConfig(performance=PerformanceConfig(metric=auc, min_score=0.0))
    report = ModelGate(checks=default_structured_checks(config, include_plugins=False)).run(
        _context(X, y, good, Champion(predictions=poor), protected_df=protected)
    )
    summary = report.comparison
    assert summary["outcome"] == IMPROVED
    assert summary["champion"]["source"] == "predictions"
    assert "vs champion: auc" in report.summary()
    payload = json.loads(report.to_json())
    assert payload["comparison"]["delta"] == summary["delta"]
    assert list(payload["results_by_category"])[:2] == ["validation", "comparison"]


def test_without_a_champion_the_report_carries_no_comparison(small_valid_context):
    report = ModelGate().run(small_valid_context)
    assert report.comparison is None
    assert "comparison" not in report.to_dict()
    assert "vs champion" not in report.summary()


def test_the_html_report_draws_the_comparison(binary):
    pytest.importorskip("matplotlib")
    X, y, good, poor, protected = binary
    config = GateConfig(performance=PerformanceConfig(metric=auc, min_score=0.0))
    report = ModelGate(checks=default_structured_checks(config, include_plugins=False)).run(
        _context(X, y, good, Champion(predictions=poor), protected_df=protected)
    )
    page = report.to_html()
    assert page.index("<h2>Comparison</h2>") < page.index("<h2>Performance</h2>")
    assert "Who the challenger is better for" in page
    assert "fixes and breaks" in page
    assert "vs champion" in page


def test_gating_the_champion_is_informational_and_never_moves_the_status(binary):
    sklearn = pytest.importorskip("sklearn.linear_model")
    X, y, _, _, protected = binary
    old = sklearn.LogisticRegression().fit(X[["a"]], y)
    new = sklearn.LogisticRegression().fit(X, y)
    context = StructuredGateContext(
        model=new,
        X=X,
        y_true=y,
        y_pred=new.predict_proba(X)[:, 1],
        protected_df=protected,
        champion=Champion(old),
        task="binary",
    )

    def config(gate):
        return GateConfig(
            performance=PerformanceConfig(metric="roc_auc", min_score=0.99),
            comparison=ComparisonConfig(gate_champion=gate),
        )

    plain = ModelGate(checks=default_structured_checks(config(False), include_plugins=False)).run(
        context
    )
    gated = ModelGate(
        checks=default_structured_checks(config(True), include_plugins=False), config=config(True)
    ).run(context)

    assert gated.champion_report is not None
    assert gated.champion_report.gate_status == "BLOCKED"  # min_score 0.99 is out of reach
    assert gated.gate_status == plain.gate_status
    assert gated.comparison["champion_gate"]["gate_status"] == "BLOCKED"
    assert not any(r.category == "comparison" for r in gated.champion_report.results)


def test_a_predictions_only_champion_is_not_gated_and_the_report_says_why(binary):
    X, y, good, poor, _ = binary
    config = GateConfig(comparison=ComparisonConfig(gate_champion=True))
    report = ModelGate(config=config).run(_context(X, y, good, Champion(predictions=poor)))
    assert report.champion_report is None
    assert "predictions" in report.comparison["champion_gate"]["note"]


# --------------------------------------------------------------------------
# Every task
# --------------------------------------------------------------------------


def test_multiclass_compares_labels_and_probability_matrices():
    rng = np.random.default_rng(12)
    n = 900
    classes = ["decline", "refer", "accept"]
    y = rng.choice(classes, n)
    X = pd.DataFrame({"x": rng.normal(size=n)})
    good = np.where(rng.uniform(size=n) < 0.8, y, rng.choice(classes, n))
    poor = np.where(rng.uniform(size=n) < 0.5, y, rng.choice(classes, n))
    performance = PerformanceConfig(metric="accuracy")
    context = _context(
        X, y, good, Champion(predictions=poor), task="multiclass", class_order=classes
    )
    assert (
        _only(ChallengerPerformanceCheck(performance=performance).run(context)).metadata["outcome"]
        == IMPROVED
    )
    churn = _only(DecisionChurnCheck().run(context)).metadata
    assert churn["fixed"] > churn["broken"]

    def matrix(labels):
        one_hot = np.zeros((n, 3))
        # Columns in class_order, the contract `to_class_labels` reads them by.
        one_hot[np.arange(n), [classes.index(v) for v in labels]] = 1.0
        return 0.1 + 0.7 * one_hot

    context = _context(
        X,
        y,
        matrix(good),
        Champion(predictions=matrix(poor)),
        task="multiclass",
        class_order=classes,
    )
    result = _only(ChallengerPerformanceCheck(performance=performance).run(context))
    assert result.metadata["outcome"] == IMPROVED


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_the_full_suite_runs_cleanly_with_a_champion_for_every_task(task):
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression

    rng = np.random.default_rng(13)
    n = 400
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    protected = pd.DataFrame({"region": rng.choice(["north", "south"], n)})
    latent = X["a"] + 0.5 * X["b"] + rng.normal(scale=0.5, size=n)
    if task == "binary":
        y = (latent > 0).astype(int).to_numpy()
        new = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
        old = LogisticRegression().fit(X[["a"]], y)
        y_pred, metric = (
            new.predict_proba(X)[:, 1],
            PerformanceConfig(metric="roc_auc", min_score=0.0),
        )
        extra = {}
    elif task == "multiclass":
        y = pd.cut(latent, 3, labels=["decline", "refer", "accept"]).astype(str).to_numpy()
        new = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
        old = LogisticRegression().fit(X[["a"]], y)
        y_pred, metric = (
            new.predict(X),
            PerformanceConfig(metric="balanced_accuracy", min_score=0.0),
        )
        extra = {"class_order": ["decline", "refer", "accept"]}
    else:
        y = (100 + 20 * latent).to_numpy()
        new = RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y)
        old = LinearRegression().fit(X[["a"]], y)
        y_pred, metric = new.predict(X), PerformanceConfig(metric="r2", min_score=-1e9)
        extra = {}

    report = ModelGate(
        checks=default_structured_checks(
            GateConfig(
                performance=metric,
                comparison=ComparisonConfig(guardrails=["accuracy"] if task == "binary" else []),
            ),
            include_plugins=False,
        )
    ).run(
        StructuredGateContext(
            model=new,
            X=X,
            y_true=y,
            y_pred=y_pred,
            protected_df=protected,
            champion=Champion(old),
            task=task,
            **extra,
        )
    )
    integrity = next(r for r in report.results if r.check_name == "comparison_integrity")
    assert integrity.flag == "OK", integrity.detail
    performance = next(r for r in report.results if r.check_name == "challenger_performance")
    assert performance.flag != "NOT_APPLICABLE"
    assert report.comparison is not None


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------


@pytest.fixture
def cli_files(tmp_path):
    pytest.importorskip("sklearn")
    joblib = pytest.importorskip("joblib")
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(14)
    n = 300
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (X["a"] + X["b"] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    new = LogisticRegression().fit(X, y)
    old = LogisticRegression().fit(X[["a"]], y)
    scored = X.assign(label=y, old_score=old.predict_proba(X[["a"]])[:, 1])
    paths = {
        "data": tmp_path / "validation.csv",
        "scored": tmp_path / "validation_with_champion_scores.csv",
        "model": tmp_path / "v4.joblib",
        "champion": tmp_path / "v3.joblib",
        "output": tmp_path / "report.json",
    }
    scored.drop(columns=["old_score"]).to_csv(paths["data"], index=False)
    scored.to_csv(paths["scored"], index=False)
    joblib.dump(new, paths["model"])
    joblib.dump(old, paths["champion"])
    return paths


def _cli(paths, *extra, data="data"):
    from bdp_model_gate.cli import main

    base = [
        "--data",
        str(paths[data]),
        "--target-col",
        "label",
        "--metric",
        "roc_auc",
        "--min-score",
        "0.5",
        "--output",
        str(paths["output"]),
    ]
    return main(base + list(extra))


def test_the_cli_compares_against_a_pinned_champion_file(cli_files):
    from bdp_model_gate.champion import sha256_of

    code = _cli(
        cli_files,
        "--model",
        str(cli_files["model"]),
        "--champion",
        str(cli_files["champion"]),
        "--champion-sha256",
        sha256_of(cli_files["champion"]),
    )
    report = json.loads(cli_files["output"].read_text())
    assert code in (0, 2)
    assert report["comparison"]["champion"]["sha256_pinned"] is True
    assert report["comparison"]["outcome"] == IMPROVED


def test_the_cli_refuses_a_champion_whose_hash_moved(cli_files, capsys):
    code = _cli(
        cli_files,
        "--model",
        str(cli_files["model"]),
        "--champion",
        str(cli_files["champion"]),
        "--champion-sha256",
        "0" * 64,
    )
    assert code == 1
    assert "pinned sha256" in capsys.readouterr().err


def test_the_cli_takes_champion_predictions_from_a_column_and_drops_it_from_X(cli_files):
    code = _cli(
        cli_files, "--model", str(cli_files["model"]), "--champion-col", "old_score", data="scored"
    )
    report = json.loads(cli_files["output"].read_text())
    assert code in (0, 2)
    assert report["comparison"]["champion"]["source"] == "predictions"
    contract = report["results_by_category"]["validation"]
    assert not any("old_score" in r["detail"] for r in contract)


def test_the_cli_rejects_a_hash_with_no_file_to_pin(cli_files, capsys):
    code = _cli(
        cli_files,
        "--model",
        str(cli_files["model"]),
        "--champion-col",
        "old_score",
        "--champion-sha256",
        "ab",
        data="scored",
    )
    assert code == 1
    assert "--champion-sha256" in capsys.readouterr().err


def test_a_guardrail_dict_survives_a_config_file(cli_files, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"comparison": {"guardrails": {"brier": 0.01}, "require": "superior"}})
    )
    _cli(
        cli_files,
        "--model",
        str(cli_files["model"]),
        "--champion-col",
        "old_score",
        "--config",
        str(config),
        data="scored",
    )
    report = json.loads(cli_files["output"].read_text())
    guardrails = [
        r
        for r in report["results_by_category"]["comparison"]
        if r["check_name"] == "challenger_guardrails"
    ]
    assert guardrails[0]["metadata"]["guardrail"] == "brier"
    assert report["comparison"]["require"] == "superior"
