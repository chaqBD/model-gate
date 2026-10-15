"""Champion against challenger — is the replacement better, and for whom?

Every check here needs `context.champion` and reports NOT_APPLICABLE without
one. The challenger still runs the full gate: these checks are added on top
and never replace it. A challenger that beats a bad champion is still
BLOCKED if it fails fairness or validation on its own terms — "better than
what we had" is not a pass.

Seven questions:

    ComparisonIntegrityCheck    Can these two models be compared at all?
    ChallengerPerformanceCheck  Is the challenger at least as good, on the
                                metric the gate scores?
    ChallengerGuardrailCheck    Did a secondary metric regress?
    SubgroupRegressionCheck     Better on average, and worse for someone?
    FairnessDeltaCheck          Did a disparity widen?
    DecisionChurnCheck          How many decisions change, and how many of
                                those were right before?
    OperationalDeltaCheck       Is it slower or dearer to run?

The first three block. The other four route to a person: whether a segment
that got worse, or a book that moved, is acceptable is a judgement, and that
is what `NEEDS_REVIEW` is for.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._logging import get_logger
from ..classes import favourable_mask
from ..comparison import (
    Compared,
    ComparisonMetric,
    bonferroni,
    compared,
    decide,
    delta_statistic,
    favourable_outcome,
    margin_for,
    paired_interval,
    resampling_frame,
    resolve_comparison_metric,
    segment_labels,
    validate_comparison_config,
)
from ..config import ComparisonConfig, FairnessConfig, PerformanceConfig, UncertaintyConfig
from ..core.base import BaseCheck, CheckResult
from ..exceptions import GateConfigurationError
from ..metrics import to_class_labels
from ..stats import rank_auc, selection_rate_difference
from ..task import BINARY, CLASSIFICATION_TASKS, MULTICLASS, REGRESSION
from ..uncertainty import ABOVE, Uncertainty

logger = get_logger("comparison_checks")

_NO_CHAMPION = (
    "no context.champion — the comparison checks measure a challenger against the model it "
    "would replace. Pass champion=Champion.from_file(path), or --champion on the command line"
)
_UNSCORED = (
    "the champion could not be scored on context.X, so there is nothing to compare — see "
    "comparison_integrity for why"
)


class _ComparisonCheck(BaseCheck):
    category = "comparison"
    blocking = True

    def __init__(
        self,
        config: ComparisonConfig | None = None,
        performance: PerformanceConfig | None = None,
        uncertainty: UncertaintyConfig | None = None,
        fairness: FairnessConfig | None = None,
    ):
        self.config = config or ComparisonConfig()
        self.performance = performance or PerformanceConfig()
        self.uncertainty = Uncertainty(uncertainty or UncertaintyConfig())
        self.fairness = fairness or FairnessConfig()
        validate_comparison_config(self.config)

    @property
    def uncertainty_config(self) -> UncertaintyConfig:
        """Read through `Uncertainty`, so every interval here uses the same
        settings the shared verdict logic does."""
        return self.uncertainty.config

    def _result(self, flag: str, detail: str, blocking: bool | None = None, **metadata):
        return CheckResult(
            self.name,
            self.category,
            flag,
            detail,
            self.blocking if blocking is None else blocking,
            metadata=metadata,
        )

    def _not_applicable(self, reason: str) -> list[CheckResult]:
        return [self._result("NOT_APPLICABLE", reason)]

    def _compared(self, context) -> Compared | None:
        """Both models' predictions, or None when the champion cannot be scored."""
        try:
            return compared(context, self.performance.decision_threshold)
        except GateConfigurationError as exc:
            logger.debug("%s: champion not scorable: %s", self.name, exc)
            return None

    def _primary(self, compared_: Compared) -> ComparisonMetric:
        setting = self.config.metric if self.config.metric is not None else self.performance.metric
        return resolve_comparison_metric(
            setting,
            compared_.task,
            average=self.performance.average,
            class_order=compared_.class_order,
            exposure=compared_.exposure,
        )

    def _preconditions(self, context) -> tuple[Compared | None, list[CheckResult] | None]:
        if getattr(context, "champion", None) is None:
            return None, self._not_applicable(_NO_CHAMPION)
        if context.y_true is None or context.y_pred is None:
            return None, self._not_applicable(
                "no y_true/y_pred — both models are scored against the same labels"
            )
        compared_ = self._compared(context)
        if compared_ is None:
            return None, self._not_applicable(_UNSCORED)
        return compared_, None


# --------------------------------------------------------------------------
# Integrity
# --------------------------------------------------------------------------


def _label_key(value: Any) -> Any:
    """A label as a comparable key: `1`, `1.0` and `np.int64(1)` are one class."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


class ComparisonIntegrityCheck(_ComparisonCheck):
    """Can these two models be compared at all?

    Every trap here produces a *confident* comparison rather than an error,
    which is why it blocks:

    - **The class order differs.** A champion fitted with
      `classes_ = ["no", "yes"]` and a challenger with `[0, 1]` — or the same
      labels encoded the other way round — gives an inverted comparison that
      looks like a large improvement.
    - **The champion's scores rank the outcome backwards.** The data-driven
      version of the same trap, for when no `classes_` is available: an AUC
      well below 0.5 means its positive class is the other one.
    - **One model supplies probabilities and the other labels.** A ranking
      metric then compares a ranking against a cut-off.
    - **The champion cannot score these rows at all** — a missing column, a
      wrong length, a framework error.

    The champion's provenance — loader, file hash, whether the hash was pinned
    — is recorded on the result either way, so the report says *which*
    incumbent the challenger was measured against.
    """

    name = "comparison_integrity"

    #: An AUC this far below chance is not noise: the scores rank the outcome
    #: backwards. Paired with the challenger being clearly better than chance,
    #: so two weak models are not mistaken for an encoding problem.
    INVERTED_AUC = 0.40

    def run(self, context) -> list[CheckResult]:
        champion = getattr(context, "champion", None)
        if champion is None:
            return self._not_applicable(_NO_CHAMPION)
        provenance = champion.describe()
        if context.y_true is None or context.y_pred is None:
            return self._not_applicable(
                "no y_true/y_pred — both models are scored against the same labels"
            )

        try:
            compared_ = compared(context, self.performance.decision_threshold)
        except GateConfigurationError as exc:
            return [
                self._result(
                    "COMPARISON_INTEGRITY_RISK",
                    f"the champion cannot be compared with the challenger: {exc}",
                    champion=provenance,
                )
            ]

        problems = [
            *self._class_order_problems(context, champion),
            *self._label_problems(compared_),
            *self._form_problems(compared_),
            *self._inversion_problems(compared_),
        ]
        metadata = {
            "champion": provenance,
            "n_rows": compared_.n,
            "challenger_predictions": compared_.challenger_kind,
            "champion_predictions": compared_.champion_kind,
            "challenger_decision_threshold": compared_.challenger_threshold,
            "champion_decision_threshold": compared_.champion_threshold,
        }
        if problems:
            return [
                self._result(
                    "COMPARISON_INTEGRITY_RISK",
                    "; ".join(problems),
                    problems=problems,
                    **metadata,
                )
            ]
        return [
            self._result(
                "OK",
                f"champion {champion.provenance.describe()} scored on the same {compared_.n} "
                f"rows ({compared_.champion_kind}, against the challenger's "
                f"{compared_.challenger_kind})",
                **metadata,
            )
        ]

    @staticmethod
    def _class_order_problems(context, champion) -> list[str]:
        challenger_classes = getattr(getattr(context, "model", None), "classes_", None)
        champion_classes = getattr(champion.model, "classes_", None)
        if challenger_classes is None or champion_classes is None:
            return []
        ours, theirs = list(challenger_classes), list(champion_classes)
        if [str(c) for c in ours] == [str(c) for c in theirs]:
            return []
        if sorted(map(str, ours)) == sorted(map(str, theirs)):
            return [
                f"the two models list the same classes in a different order (challenger "
                f"{ours}, champion {theirs}), so their probability columns do not line up"
            ]
        return [
            f"the two models were fitted on different classes (challenger {ours}, champion "
            f"{theirs}) — the label encoding changed between them"
        ]

    @staticmethod
    def _label_problems(compared_: Compared) -> list[str]:
        if compared_.task == REGRESSION or compared_.champion_kind != "labels":
            return []
        known = {_label_key(v) for v in np.unique(compared_.y_true)}
        produced = {_label_key(v) for v in np.unique(compared_.champion)}
        unknown = sorted(map(str, produced - known))
        if not unknown:
            return []
        return [
            f"the champion predicts label(s) {unknown[:5]} that never occur in y_true "
            f"({sorted(map(str, known))[:5]}) — its labels are encoded differently"
        ]

    def _form_problems(self, compared_: Compared) -> list[str]:
        if compared_.challenger_kind == compared_.champion_kind:
            return []
        try:
            metric = self._primary(compared_)
        except GateConfigurationError:
            return []
        if metric.needs_hard_labels:
            return []  # both are reduced to labels before scoring, so this is fair
        return [
            f"the challenger's predictions are {compared_.challenger_kind} and the champion's "
            f"{compared_.champion_kind}, so {metric.name} would compare a ranking against a "
            "cut-off. Supply both in the same form — Champion(predictions=...) if the "
            "champion cannot produce probabilities"
        ]

    def _inversion_problems(self, compared_: Compared) -> list[str]:
        if compared_.task != BINARY or compared_.champion_kind != "probability":
            return []
        actual = favourable_mask(compared_.y_true, [1])
        if actual.all() or not actual.any():
            return []
        champion_auc = rank_auc(actual, compared_.champion.astype(float))
        challenger_auc = (
            rank_auc(actual, compared_.challenger.astype(float))
            if compared_.challenger_kind == "probability"
            else float("nan")
        )
        if champion_auc < self.INVERTED_AUC and challenger_auc >= 1.0 - self.INVERTED_AUC:
            return [
                f"the champion's scores rank y_true backwards (AUC {champion_auc:.3f}, against "
                f"the challenger's {challenger_auc:.3f}) — its positive class is almost "
                "certainly encoded the other way round"
            ]
        return []


# --------------------------------------------------------------------------
# The headline comparison
# --------------------------------------------------------------------------


class ChallengerPerformanceCheck(_ComparisonCheck):
    """Is the challenger at least as good as the champion?

    Scored on `ComparisonConfig.metric`, which defaults to the performance
    gate's own metric, so the comparison and the absolute gate agree on what
    "good" means. Decided by the non-inferiority rule in
    `bdp_model_gate.comparison`, on a paired interval of the difference.
    """

    name = "challenger_performance"

    def run(self, context) -> list[CheckResult]:
        compared_, skipped = self._preconditions(context)
        if skipped is not None:
            return skipped
        assert compared_ is not None

        metric = self._primary(compared_)
        statistic, challenger_score, champion_score = delta_statistic(metric, compared_)
        delta = metric.sign * (challenger_score - champion_score)
        interval = paired_interval(statistic, resampling_frame(compared_), self.uncertainty_config)
        margin, relative = margin_for(self.config, champion_score)
        if margin == 0.0:
            logger.warning(
                "comparison margin is zero: two equally good models straddle it by "
                "construction, so every like-for-like retrain will read UNCERTAIN"
            )
        decision = decide(
            interval,
            delta,
            margin=margin,
            min_improvement=self.config.min_improvement,
            require=self.config.require,
            on_uncertain=self.uncertainty_config.on_uncertain,
            blocking=self.blocking,
        )
        direction = "higher" if metric.greater_is_better else "lower"
        margin_note = (
            f"{self.config.margin_relative:.0%} of the champion's score" if relative else "absolute"
        )
        detail = (
            f"{metric.name} ({direction} is better): challenger {challenger_score:.4f}, champion "
            f"{champion_score:.4f}, Δ {delta:+.4f} (positive = challenger better)"
            f"{decision.note}. Margin {margin:.4f} ({margin_note}); "
            f"require={self.config.require} → {decision.outcome.replace('_', '-')}"
        )
        return [
            self._result(
                decision.flag,
                detail,
                blocking=decision.blocking,
                metric=metric.name,
                greater_is_better=metric.greater_is_better,
                challenger_score=round(challenger_score, 4),
                champion_score=round(champion_score, 4),
                delta=round(delta, 4),
                margin=round(margin, 6),
                margin_is_relative=relative,
                min_improvement=self.config.min_improvement,
                require=self.config.require,
                outcome=decision.outcome,
                exposure_weighted=metric.weighted,
                on_uncertain=self.uncertainty_config.on_uncertain,
                **(interval.as_metadata() if interval is not None else {}),
            )
        ]


class ChallengerGuardrailCheck(_ComparisonCheck):
    """Did a secondary metric regress?

    A challenger can gain a point of AUC by becoming badly calibrated, or gain
    Gini while its level drifts. `ComparisonConfig.guardrails` names the
    metrics that must not get materially worse, each read by the same
    non-inferiority rule as the headline metric.

    Each extra guardrail is another chance for sampling noise to block a
    deploy, so the intervals are **Bonferroni-widened** across the list, and a
    guardrail blocks only on a clear regression — never for failing to
    improve, whatever `require` says. `ComparisonConfig.margin` is in the
    headline metric's units, so it never applies here: a guardrail's margin
    is its own absolute value when `guardrails` is a dict, and otherwise
    `margin_relative` of the champion's score on that metric.
    """

    name = "challenger_guardrails"

    def run(self, context) -> list[CheckResult]:
        compared_, skipped = self._preconditions(context)
        if skipped is not None:
            return skipped
        assert compared_ is not None
        if not self.config.guardrails:
            return self._not_applicable(
                "no comparison.guardrails configured — name secondary metrics that must not "
                "regress, e.g. {'brier': 0.01} beside an AUC"
            )

        level = bonferroni(self.uncertainty_config.confidence_level, len(self.config.guardrails))
        results = []
        for name in self.config.guardrails:
            try:
                metric = resolve_comparison_metric(
                    name,
                    compared_.task,
                    average=self.performance.average,
                    class_order=compared_.class_order,
                    exposure=compared_.exposure,
                )
                statistic, challenger_score, champion_score = delta_statistic(metric, compared_)
            except GateConfigurationError as exc:
                results.append(self._result("NOT_APPLICABLE", f"{name}: {exc}", guardrail=name))
                continue
            delta = metric.sign * (challenger_score - champion_score)
            interval = paired_interval(
                statistic, resampling_frame(compared_), self.uncertainty_config, level=level
            )
            margin = (
                float(self.config.guardrails[name])
                if isinstance(self.config.guardrails, dict)
                else self.config.margin_relative * abs(champion_score)
            )
            decision = decide(
                interval,
                delta,
                margin=margin,
                min_improvement=0.0,
                require="non_inferior",
                on_uncertain=self.uncertainty_config.on_uncertain,
                blocking=self.blocking,
            )
            results.append(
                self._result(
                    decision.flag,
                    f"{metric.name}: challenger {challenger_score:.4f}, champion "
                    f"{champion_score:.4f}, Δ {delta:+.4f}{decision.note} "
                    f"[{len(self.config.guardrails)} guardrail(s), interval widened to "
                    f"{level:.1%}]",
                    blocking=decision.blocking,
                    guardrail=metric.name,
                    greater_is_better=metric.greater_is_better,
                    challenger_score=round(challenger_score, 4),
                    champion_score=round(champion_score, 4),
                    delta=round(delta, 4),
                    margin=round(margin, 6),
                    outcome=decision.outcome,
                    **(interval.as_metadata() if interval is not None else {}),
                )
            )
        return results


# --------------------------------------------------------------------------
# Who it is better for
# --------------------------------------------------------------------------


def _group_columns(context, config: ComparisonConfig) -> list[tuple[str, pd.Series]]:
    columns: list[tuple[str, pd.Series]] = []
    protected = getattr(context, "protected_df", None)
    if protected is not None:
        columns.extend((str(c), protected[c].reset_index(drop=True)) for c in protected.columns)
    for name in config.segments:
        if name in context.X.columns:
            columns.append((name, context.X[name].reset_index(drop=True)))
        elif protected is not None and name in protected.columns:
            continue  # already included as a protected attribute
        else:
            raise GateConfigurationError(
                f"comparison.segments names {name!r}, which is neither a column of context.X "
                "nor of context.protected_df"
            )
    return columns


class SubgroupRegressionCheck(_ComparisonCheck):
    """Better on average, and worse for someone?

    An aggregate improvement routinely hides a segment that got worse — a
    model that gains on the large, easy segment and loses on a small one can
    improve the headline number while making things worse for real people.

    Δ is computed within every group of every protected attribute and every
    `ComparisonConfig.segments` column, with intervals Bonferroni-widened
    across all of the groups scored, so a wide report does not manufacture a
    finding. **Non-blocking**: whether a segment that got worse is acceptable
    depends on why, which is a judgement.
    """

    name = "subgroup_regression"
    blocking = False

    def run(self, context) -> list[CheckResult]:
        compared_, skipped = self._preconditions(context)
        if skipped is not None:
            return skipped
        assert compared_ is not None
        columns = _group_columns(context, self.config)
        if not columns:
            return self._not_applicable(
                "no protected_df and no comparison.segments — nothing to compare within"
            )

        metric = self._primary(compared_)
        statistic, _, _ = delta_statistic(metric, compared_)
        frame = resampling_frame(compared_)
        y = compared_.y_true
        challenger = metric.prepare(compared_, "challenger")
        champion = metric.prepare(compared_, "champion")
        weights = compared_.exposure if metric.weighted else None

        def scores(rows: np.ndarray) -> tuple[float, float]:
            w = None if weights is None else weights[rows]
            return (
                metric.score(y[rows], challenger[rows], w),
                metric.score(y[rows], champion[rows], w),
            )

        groups: list[tuple[str, str, np.ndarray]] = []
        too_small: dict[str, list[str]] = {}
        for attr, series in columns:
            labels = segment_labels(series)
            for level in sorted(labels.dropna().unique(), key=str):
                rows = np.flatnonzero((labels == level).to_numpy())
                if len(rows) < self.config.min_segment_size:
                    too_small.setdefault(attr, []).append(str(level))
                    continue
                groups.append((attr, str(level), rows))

        level = bonferroni(self.uncertainty_config.confidence_level, len(groups))
        by_attr: dict[str, list[dict[str, Any]]] = {}
        for attr, group, rows in groups:
            try:
                challenger_score, champion_score = scores(rows)
            except Exception as exc:
                by_attr.setdefault(attr, []).append(
                    {"group": group, "n": len(rows), "outcome": f"undefined ({exc})"}
                )
                continue
            delta = metric.sign * (challenger_score - champion_score)
            if not np.isfinite(delta):
                by_attr.setdefault(attr, []).append(
                    {"group": group, "n": len(rows), "outcome": "undefined"}
                )
                continue

            def within(positions: np.ndarray, rows: np.ndarray = rows) -> float:
                return statistic(rows[positions])

            interval = paired_interval(
                within,
                frame.iloc[rows].reset_index(drop=True),
                self.uncertainty_config,
                level=level,
            )
            margin, _ = margin_for(self.config, champion_score)
            low, high = (interval.low, interval.high) if interval else (delta, delta)
            outcome = "worse" if high < -margin else ("uncertain" if low < -margin else "fine")
            by_attr.setdefault(attr, []).append(
                {
                    "group": group,
                    "n": len(rows),
                    "challenger": round(challenger_score, 4),
                    "champion": round(champion_score, 4),
                    "delta": round(delta, 4),
                    "ci_low": round(low, 4),
                    "ci_high": round(high, 4),
                    "margin": round(margin, 6),
                    "outcome": outcome,
                }
            )

        results = []
        for attr, _ in columns:
            rows_ = by_attr.get(attr, [])
            scored = [r for r in rows_ if "delta" in r]
            small = too_small.get(attr, [])
            small_note = (
                f" [{len(small)} group(s) under {self.config.min_segment_size} rows not scored]"
                if small
                else ""
            )
            if not scored:
                results.append(
                    self._result(
                        "NOT_APPLICABLE",
                        f"{attr}: no group large enough to compare{small_note}",
                        attribute=attr,
                    )
                )
                continue
            worst = min(scored, key=lambda r: r["delta"])
            worse = [r for r in scored if r["outcome"] == "worse"]
            undecided = [r for r in scored if r["outcome"] == "uncertain"]
            flag = "SUBGROUP_WORSE" if worse else ("UNCERTAIN" if undecided else "OK")
            named = ", ".join(f"{r['group']} ({r['delta']:+.4f})" for r in (worse or undecided))
            if worse:
                summary = f"challenger is clearly worse for {named}"
            elif undecided:
                summary = f"cannot rule out a regression for {named}"
            else:
                summary = "no group is materially worse off"
            results.append(
                self._result(
                    flag,
                    f"{attr}: {summary}. Worst group {worst['group']}: Δ {worst['delta']:+.4f} "
                    f"[{worst['ci_low']:+.4f}, {worst['ci_high']:+.4f}] on {metric.name}, "
                    f"n={worst['n']} [{len(groups)} group(s) scored, intervals at "
                    f"{level:.1%}]{small_note}",
                    blocking=False,
                    attribute=attr,
                    metric=metric.name,
                    groups=rows_,
                    groups_not_scored=small,
                    ci_level=round(level, 4),
                )
            )
        return results

    def plot(self, context, results=None, ax=None):
        """Every group's Δ with its interval, against its own margin.

        The finding is a shape: one group sitting left of its tick while the
        rest sit right is a different conversation from everything clustered
        at zero, and the headline Δ cannot tell the two apart.
        """
        from ..plots import require_plotting
        from ..plots.style import RULE, caption, new_axes, verdict_colour

        require_plotting()
        rows = [
            (r.metadata["attribute"], g)
            for r in (results or [])
            if r.flag != "NOT_APPLICABLE"
            for g in r.metadata.get("groups", [])
            if "delta" in g
        ]
        if not rows:
            return None
        ax = new_axes(ax, figsize=(6.4, 1.4 + 0.32 * len(rows)))
        colours = {"worse": "BLOCKED", "uncertain": "NEEDS_REVIEW", "fine": "PASS"}
        for i, (_attr, g) in enumerate(rows):
            colour = verdict_colour(colours[g["outcome"]])
            ax.plot([g["ci_low"], g["ci_high"]], [i, i], color=colour, linewidth=2.2)
            ax.plot(g["delta"], i, "o", color=colour, markersize=5.5, zorder=3)
            ax.plot(-g["margin"], i, "|", color=RULE, markersize=12, markeredgewidth=2, zorder=2)
        ax.axvline(0.0, color=RULE, linewidth=1.2, linestyle="--")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{attr}={g['group']}  (n={g['n']})" for attr, g in rows])
        ax.invert_yaxis()
        ax.set_xlabel("Δ = challenger − champion (positive = challenger better)")
        ax.set_title("Who the challenger is better for")
        caption(
            ax,
            "dots are each group's Δ, bars its interval; the tick is that group's margin.\n"
            "A bar entirely left of its tick is a group the challenger is clearly worse for.",
        )
        return ax


def _tpr_gap(predicted: np.ndarray, actual: np.ndarray, groups: np.ndarray) -> float:
    """Spread in true-positive rate across groups — equal opportunity."""
    positive = actual == 1
    rates = []
    for g in pd.unique(groups):
        cell = positive & (groups == g)
        if cell.sum():
            rates.append(float(predicted[cell].mean()))
    return float(max(rates) - min(rates)) if len(rates) >= 2 else 0.0


def _relative_mean_gap(predicted: np.ndarray, groups: np.ndarray) -> float:
    """Spread in group mean prediction, relative to the overall mean."""
    overall = float(np.mean(predicted))
    if overall == 0.0:
        return float("nan")
    means = pd.Series(predicted).groupby(groups).mean()
    return float((means.max() - means.min()) / abs(overall)) if len(means) >= 2 else 0.0


class FairnessDeltaCheck(_ComparisonCheck):
    """Did a disparity widen?

    A challenger whose demographic-parity gap moves from 0.02 to 0.09 passes
    an absolute threshold of 0.10 — and has still become materially less fair
    than the model it replaces. The absolute fairness checks cannot see that;
    this one reports the change.

    The statistics, per protected attribute: demographic parity (selection-rate
    spread) for classification, plus equal opportunity (true-positive-rate
    spread) for binary; for regression, the spread in group mean prediction
    relative to the overall mean. Δ is challenger minus champion, so positive
    means **wider**, and it is flagged once its interval sits entirely above
    `ComparisonConfig.max_disparity_increase`. Non-blocking.
    """

    name = "fairness_delta"
    blocking = False

    def run(self, context) -> list[CheckResult]:
        protected = getattr(context, "protected_df", None)
        if protected is None or protected.empty:
            if getattr(context, "champion", None) is None:
                return self._not_applicable(_NO_CHAMPION)
            return self._not_applicable("no protected_df supplied")
        compared_, skipped = self._preconditions(context)
        if skipped is not None:
            return skipped
        assert compared_ is not None

        try:
            statistics = self._statistics(compared_)
        except GateConfigurationError as exc:
            return self._not_applicable(str(exc))

        results = []
        for attr in protected.columns:
            raw_groups = protected[attr].astype(str).to_numpy()
            sizes = pd.Series(raw_groups).value_counts()
            keep = np.isin(raw_groups, sizes[sizes >= self.fairness.min_group_size].index)
            if len(np.unique(raw_groups[keep])) < 2:
                results.append(
                    self._result(
                        "NOT_APPLICABLE",
                        f"{attr}: fewer than two groups of at least "
                        f"{self.fairness.min_group_size} rows",
                        attribute=attr,
                    )
                )
                continue
            rows = np.flatnonzero(keep)
            groups = raw_groups
            frame = resampling_frame(compared_, groups).iloc[rows].reset_index(drop=True)
            for label, stat in statistics:

                def delta(positions: np.ndarray, stat=stat, rows=rows, groups=groups) -> float:
                    picked = rows[positions]
                    return stat("challenger", picked, groups) - stat("champion", picked, groups)

                ours = stat("challenger", rows, groups)
                theirs = stat("champion", rows, groups)
                point = ours - theirs
                interval = self.uncertainty.interval(delta, frame)
                verdict = self.uncertainty.verdict(
                    interval,
                    self.config.max_disparity_increase,
                    point=point,
                    risk_flag="FAIRNESS_WORSE",
                    blocking=False,
                    flag_when=ABOVE,
                )
                results.append(
                    self._result(
                        verdict.flag,
                        f"{attr}: {label} challenger {ours:.4f}, champion {theirs:.4f}, "
                        f"change {self.uncertainty.measured(interval, point)} "
                        f"(positive = wider; tolerance +{self.config.max_disparity_increase})"
                        f"{verdict.note}",
                        blocking=verdict.blocking,
                        attribute=attr,
                        statistic=label,
                        challenger=round(ours, 4),
                        champion=round(theirs, 4),
                        change=round(point, 4),
                        threshold=self.config.max_disparity_increase,
                        **verdict.metadata,
                    )
                )
        return results

    def _statistics(self, compared_: Compared):
        """`[(label, fn(which, rows, groups) -> float)]` for this task."""
        if compared_.task == REGRESSION:
            values = {
                "challenger": compared_.challenger.astype(float),
                "champion": compared_.champion.astype(float),
            }

            def mean_gap(which, rows, groups):
                return _relative_mean_gap(values[which][rows], groups[rows])

            return [("relative group-mean gap", mean_gap)]

        outcomes = {
            "challenger": favourable_outcome(
                compared_.challenger, compared_, compared_.challenger_threshold
            ),
            "champion": favourable_outcome(
                compared_.champion, compared_, compared_.champion_threshold
            ),
        }

        def parity(which, rows, groups):
            return selection_rate_difference(outcomes[which][rows], groups[rows])

        statistics = [("demographic parity difference", parity)]
        if compared_.task == BINARY:
            actual = favourable_mask(compared_.y_true, [1]).astype(int)

            def opportunity(which, rows, groups):
                return _tpr_gap(outcomes[which][rows], actual[rows], groups[rows])

            statistics.append(("equal opportunity (TPR) gap", opportunity))
        return statistics


class DecisionChurnCheck(_ComparisonCheck):
    """How many decisions change, and how many of those were right before?

    Every changed decision is a **fix** (the champion was wrong, the challenger
    is right) or a **break** (the reverse). Breaks are the number to watch:
    they are the people for whom the "better" model is worse. A large break
    rate under a small net gain is a model that is *different* rather than
    better, and that is worth a person's time even when the headline metric
    improved.

    Each model is cut at its own operating threshold. Regression has no
    decision to churn — `prediction_dislocation` measures who moves, and reads
    the champion's predictions automatically. Non-blocking.
    """

    name = "decision_churn"
    blocking = False
    supported_tasks = CLASSIFICATION_TASKS

    def _decisions(self, compared_: Compared) -> tuple[np.ndarray, np.ndarray]:
        """Each model's correctness, row by row."""
        if compared_.task == MULTICLASS:
            truth = np.asarray([str(v) for v in compared_.y_true])
            ours = to_class_labels(compared_.challenger, compared_.class_order)
            theirs = to_class_labels(compared_.champion, compared_.class_order)
            return (
                np.asarray([str(v) for v in ours]) == truth,
                np.asarray([str(v) for v in theirs]) == truth,
            )
        truth01 = favourable_mask(compared_.y_true, [1])
        ours01 = favourable_outcome(compared_.challenger, compared_, compared_.challenger_threshold)
        theirs01 = favourable_outcome(compared_.champion, compared_, compared_.champion_threshold)
        return ours01.astype(bool) == truth01, theirs01.astype(bool) == truth01

    def run(self, context) -> list[CheckResult]:
        compared_, skipped = self._preconditions(context)
        if skipped is not None:
            return skipped
        assert compared_ is not None
        try:
            challenger_right, champion_right = self._decisions(compared_)
        except GateConfigurationError as exc:
            return self._not_applicable(str(exc))

        n = compared_.n
        fixed = ~champion_right & challenger_right
        broken = champion_right & ~challenger_right
        both_right = champion_right & challenger_right
        both_wrong = ~champion_right & ~challenger_right
        fix_rate, break_rate = float(fixed.mean()), float(broken.mean())

        frame = resampling_frame(compared_, fixed.astype(int), broken.astype(int))
        interval = self.uncertainty.interval(lambda rows: float(broken[rows].mean()), frame)
        verdict = self.uncertainty.verdict(
            interval,
            self.config.max_break_rate,
            point=break_rate,
            risk_flag="DECISION_CHURN",
            blocking=False,
            flag_when=ABOVE,
        )
        return [
            self._result(
                verdict.flag,
                f"{fix_rate + break_rate:.1%} of decisions change: {fix_rate:.1%} fixed "
                f"(champion wrong, challenger right), "
                f"{self.uncertainty.measured(interval, break_rate)} broken (champion right, "
                f"challenger wrong; tolerance {self.config.max_break_rate:.1%}); net "
                f"{fix_rate - break_rate:+.1%}{verdict.note}",
                blocking=verdict.blocking,
                n_rows=n,
                both_right=int(both_right.sum()),
                fixed=int(fixed.sum()),
                broken=int(broken.sum()),
                both_wrong=int(both_wrong.sum()),
                fix_rate=round(fix_rate, 4),
                break_rate=round(break_rate, 4),
                flip_rate=round(fix_rate + break_rate, 4),
                threshold=self.config.max_break_rate,
                **verdict.metadata,
            )
        ]

    def plot(self, context, results=None, ax=None):
        """Champion right/wrong against challenger right/wrong, as counts.

        The two off-diagonal cells are the whole story — fixes and breaks — and
        the net gain hides how large each one is.
        """
        from ..plots import require_plotting
        from ..plots.style import ACCENT, caption, new_axes, ring_cell, sharpen_colourbar

        _, sns = require_plotting()
        found = [r for r in (results or []) if "broken" in r.metadata]
        if not found:
            return None
        m = found[0].metadata
        counts = pd.DataFrame(
            [[m["both_right"], m["broken"]], [m["fixed"], m["both_wrong"]]],
            index=["champion right", "champion wrong"],
            columns=["challenger right", "challenger wrong"],
        )
        ax = new_axes(ax, figsize=(5.2, 3.6))
        sns.heatmap(
            counts / max(1, m["n_rows"]),
            ax=ax,
            annot=counts,
            fmt="d",
            cmap="crest",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "share of rows"},
        )
        sharpen_colourbar(ax)
        ring_cell(ax, 1, 0, ACCENT)
        ax.tick_params(labelrotation=0)
        ax.set_title("Decisions that change — fixes and breaks")
        caption(
            ax,
            "the ringed cell is the breaks: rows the champion got right and the challenger "
            "gets wrong.\nA net gain can hide a large one.",
        )
        return ax


class OperationalDeltaCheck(_ComparisonCheck):
    """Is the challenger slower or dearer to run than the champion?

    A one-point gain at four times the cost is a business decision, not a
    statistical one, so this routes to a person rather than blocking. Needs
    the same measurement for both models: `context.latencies_ms` beside
    `Champion(latencies_ms=...)`, and likewise for cost per inference.
    Point estimates — the two benchmarks are separate samples, not paired rows.
    """

    name = "operational_delta"
    blocking = False

    def run(self, context) -> list[CheckResult]:
        champion = getattr(context, "champion", None)
        if champion is None:
            return self._not_applicable(_NO_CHAMPION)

        results = []
        ours, theirs = getattr(context, "latencies_ms", None), champion.latencies_ms
        if ours is not None and theirs is not None:
            p95_ours = float(np.percentile(np.asarray(ours, dtype=float), 95))
            p95_theirs = float(np.percentile(np.asarray(theirs, dtype=float), 95))
            results.append(
                self._ratio_result(
                    "p95 latency", p95_ours, p95_theirs, self.config.max_latency_ratio, "ms"
                )
            )
        cost_ours, cost_theirs = (
            getattr(context, "cost_per_inference", None),
            champion.cost_per_inference,
        )
        if cost_ours is not None and cost_theirs is not None:
            results.append(
                self._ratio_result(
                    "cost per inference",
                    float(cost_ours),
                    float(cost_theirs),
                    self.config.max_cost_ratio,
                    "",
                )
            )
        return results or self._not_applicable(
            "no matching operational measurements — supply context.latencies_ms with "
            "Champion(latencies_ms=...), or cost_per_inference on both"
        )

    def _ratio_result(self, what: str, ours: float, theirs: float, limit: float, unit: str):
        ratio = ours / theirs if theirs > 0 else float("inf")
        flag = "OPERATIONAL_COST_RISK" if ratio > limit else "OK"
        return self._result(
            flag,
            f"{what}: challenger {ours:g}{unit}, champion {theirs:g}{unit} — {ratio:.2f}× "
            f"(tolerance {limit:g}×)",
            blocking=False,
            measure=what,
            challenger=ours,
            champion=theirs,
            ratio=round(ratio, 4) if np.isfinite(ratio) else None,
            threshold=limit,
        )


COMPARISON_CHECKS = (
    ComparisonIntegrityCheck,
    ChallengerPerformanceCheck,
    ChallengerGuardrailCheck,
    SubgroupRegressionCheck,
    FairnessDeltaCheck,
    DecisionChurnCheck,
    OperationalDeltaCheck,
)


def comparison_checks(
    config: ComparisonConfig | None = None,
    performance: PerformanceConfig | None = None,
    uncertainty: UncertaintyConfig | None = None,
    fairness: FairnessConfig | None = None,
) -> list[BaseCheck]:
    """All seven comparison checks, sharing one configuration."""
    return [cls(config, performance, uncertainty, fairness) for cls in COMPARISON_CHECKS]


__all__ = [
    "COMPARISON_CHECKS",
    "ChallengerGuardrailCheck",
    "ChallengerPerformanceCheck",
    "ComparisonIntegrityCheck",
    "DecisionChurnCheck",
    "FairnessDeltaCheck",
    "OperationalDeltaCheck",
    "SubgroupRegressionCheck",
    "comparison_checks",
]
