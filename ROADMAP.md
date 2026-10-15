# Roadmap

Planned work, with enough detail that decisions already taken do not get
re-argued. Shipped releases are in [`CHANGELOG.md`](CHANGELOG.md).

## How each release is delivered

Every entry below ships as a complete slice, not just code:

1. **Its own branch**, named in the entry.
2. **Implementation**, with the tests the 0.4.2 guards now demand — known
   answers, invariants, and a place in the model-family matrix.
3. **Examples**: a new notebook, or updates to the existing ones, executed via
   `examples/run_all.sh` so the committed outputs are true.
4. **Web**: the affected pages under `web/docs/`, and the landing page — its
   version stamp every time, its checks grid and capability claims whenever
   those change. The site is part of the release, not a follow-up; see the
   checklist in [`CONTRIBUTING.md`](CONTRIBUTING.md#the-site-ships-with-the-release).
   `tests/test_package.py` enforces the version stamp and check coverage.
5. **A pull request to `main`**, with CI and the docs build green before
   merge. Both are conventions rather than enforcement today — see
   [Before the next release](#before-the-next-release--the-repository-itself),
   which is about fixing that.

A release is not done when the code works. It is done when someone who has
never seen it can find it, read why it exists, and run it.

Current release: **0.6.0**. Next: **0.7.0**.

> **This file tracks what should happen.** What already happened lives in
> [`CHANGELOG.md`](CHANGELOG.md), and shipped entries are removed from here
> rather than marked done. The only release appearing in both is the one
> currently being built.

---

## Before the next release — the repository itself

Not a release, and not code: three findings about this repository's own
configuration, checked against the GitHub API on 2026-09-09. The first one is
live.

### `main-lock` has never protected `main`

The ruleset exists, is `active`, and applies to nothing. Its ref condition is:

```
refs/heads/"main"
```

Those double quotes are literal characters in an fnmatch pattern, so it can
only ever match a branch named `"main"` — quotes included — and the branch is
called `main`. `GET /repos/{owner}/{repo}/rules/branches/main` returns `[]`,
and the ruleset reports `current_user_can_bypass: "never"`, so the empty
answer is not bypass permission masking the rules. There are none.

Which means, right now: **direct pushes to `main` are not blocked,
force-pushes are not blocked, and no review is required.** Every release so
far went through a PR by convention, not by enforcement.

> A guard with a hole in it is worse than none, because it is trusted. That
> sentence is already in `CONTRIBUTING.md` about the version-stamp guard, and
> it applies here to `CONTRIBUTING.md` itself, which states that "direct
> pushes are blocked by a ruleset". They are not. Fix the ruleset and the
> sentence becomes true; until then it is a false claim in the contributor
> documentation.

Fix: change the condition to `refs/heads/main` — or better `~DEFAULT_BRANCH`,
which cannot be typo'd.

### There is no required-status-check rule, and that is the one that matters

Even with the pattern fixed, `main-lock` carries `deletion`,
`non_fast_forward`, `creation`, `update`, `pull_request`, `code_scanning` and
`code_quality` — and **no `required_status_checks`**. So a PR could be merged
with a red matrix.

That is not hypothetical: v0.3.0 and v0.3.1 were both tagged on commits with a
red Python 3.9 job, and both tags had to be withdrawn. Requiring a PR stops
nobody from merging a broken one.

Add `required_status_checks` naming the jobs, exactly:

- `Lint and type-check`
- `Test (Python 3.9)` … `Test (Python 3.13)`
- `Test (core install only)`
- `Build distribution`
- `Build site`

Mutation testing is not on that list and no longer could be: it moved out of
`ci.yml` into `.github/workflows/mutation.yml` and runs weekly, so it never
reports a status on a pull request. See *Mutation survivors* below.

### Two rules that cannot currently be satisfied

- **`require_code_owner_review: true`, and there is no `CODEOWNERS` file.**
  Either add one or drop the rule; a rule that names a file that does not
  exist is another guard that reads as protection and is not.
- **`required_approving_review_count: 1` on a single-maintainer repository.**
  GitHub will not let an author approve their own pull request, so this rule
  can only ever be satisfied by a second person or by an admin bypass — and a
  rule whose normal path is "bypass it" trains the reflex that defeats every
  other rule.

  The honest position for now: **required status checks are the rule that
  protects this repository; required reviews are theatre until there is a
  second maintainer.** Keep the `pull_request` rule for the merge-method and
  thread-resolution settings, set the approval count to 0, and raise it the
  day someone else has commit rights. Record the decision either way, because
  the alternative is discovering it at 11pm during a release.

### The tag ruleset is disabled, and would break tagging if enabled

`release tags` is `enforcement: disabled`, which is why `v0.5.3` and `v0.5.4`
pushed without complaint. Its rules are `deletion`, `non_fast_forward` and
`creation` against `~ALL` tags with no bypass actors — and `creation` on
`~ALL` forbids creating *any* tag, including the release tags the ruleset is
named for.

Fix before enabling: scope the condition to `refs/tags/v*`, keep `deletion`
and `non_fast_forward`, drop `creation`. That gives what the name suggests and
prevents a repeat of the withdrawn v0.3.0/v0.3.1 tags.

### Checklist

- [ ] `main-lock` condition -> `~DEFAULT_BRANCH`
- [ ] `main-lock` gains `required_status_checks` with the seven jobs above
- [ ] Decide `required_approving_review_count` (0 for now) and
      `require_code_owner_review` (drop, or add `CODEOWNERS`)
- [ ] `release tags`: scope to `refs/tags/v*`, drop `creation`, then enable
- [ ] Restore the plain "direct pushes are blocked" wording in
      `CONTRIBUTING.md` once the first two boxes are ticked — it has been
      softened to the truth in the meantime rather than left as a false claim

---

## Mutation survivors — a real backlog, now that the number means something

Not a release either. Recorded here because the figure changed for reasons
that have nothing to do with the tests, and the difference should not be
mistaken for progress or regression later.

Until 0.6.0 the mutation job measured mutants against five hand-picked test
files, and was killed by its own timeout at ~74% of the surface without
saying so. Both are fixed. Measured over the same 1,651 mutants:

| | old (5 files, truncated) | now (suite, complete) |
|---|---|---|
| kill rate | 34.6% | **54.4%** |
| survivors | 1,058 | **753** |
| never reached | 430 | 0 |

305 of those survivors were killed by assertions that already existed. The
remaining 753 are the real backlog — a comparison, threshold or boolean that
could be wrong with nothing to notice.

Worked so far, with each kill verified by applying the mutant and watching the
new test fail:

- `uncertainty.bootstrap` and `permutation_pvalue` — the `min_rows` floor, the
  kept-draw floor, `continue` vs `break` on a degenerate draw, the draw
  counter, and the tie tolerance in the p-value. 11 mutants.
- `stats` — `rank_auc` with a single positive or a single negative, one-member
  groups in `selection_rate_difference` and `correlation_ratio`, and the
  empty / constant / small-variance guards. 11 mutants.

Two families were checked and found **equivalent** rather than unkilled, and
are annotated in place so nobody spends an afternoon on them: `average_ranks`'
tie-run condition (a singleton run averages to the rank it already had) and
`benjamini_hochberg`'s upper clip bound (unreachable by construction).

Roughly by module, what is left:

| Module | Survivors |
|---|---|
| `structured.actuarial_checks` | 111 |
| `structured.calibration_checks` | 102 |
| `structured.security` | 96 |
| `structured.fairness` | 91 |
| `structured.regression_fairness` | 73 |
| `structured.validation_checks` | 44 |
| `calibration` | 38 |
| `metrics` | 31 |
| everything else | ~146 |

Take these an area at a time while hardening that area, not as a sprint. A
survivor is a question ("would anything notice?"), and the answer is sometimes
"this mutant is equivalent" — which is worth writing down rather than
papering over with a test that asserts the mutation instead of the behaviour.

Do not set `--min-kill-rate` until a few weekly runs have established where the
number actually sits.

---

## 0.7.0 — Champion vs challenger

**Branch:** `feat/champion-challenger`

The gate answers one question today: *is this model good enough?* A
replacement raises a second question that governance cares about more: *is it
better than the model already in production, and what changes for the people
it scores?* A challenger can clear every absolute threshold and still be worse
than the model it replaces. Nothing currently notices.

The seed is already here. `context.baseline_pred` and `DislocationCheck` take
the incumbent's predictions, but only as a column you computed yourself, and
only to measure who moves. This release makes the incumbent a first-class
input. You name the old model, the gate scores it on the same rows as the new
one, and the comparison produces blocking and non-blocking results like every
other check.

The names are **champion** (the model in production) and **challenger** (the
one proposed to replace it). That is the model-risk and insurance vocabulary
this library's users already speak, and it is harder to misread than
"baseline", which already means something in `metrics.py`.

### The shape

```python
from bdp_model_gate import Champion, StructuredGateContext, ModelGate

champion = Champion.from_file("models/scorecard_v3.joblib", sha256="9f2c…")

context = StructuredGateContext(
    model=challenger,  # the new model, gated in full as today
    X=X_val,
    y_true=y_val,
    y_pred=y_pred,
    protected_df=protected_val,
    champion=champion,  # activates the comparison checks
)

report = ModelGate().run(context)
print(report.summary())
# vs champion: roc_auc 0.7304 against 0.6517, Δ +0.0786 [+0.0614, +0.0955] -> improved
```

```bash
bdp-model-gate --model scorecard_v4.joblib --champion scorecard_v3.joblib \
  --champion-sha256 9f2c… --data validation.csv --target-col label \
  --output gate_report.json
```

Exit codes do not change (`0` / `2` / `1`). The comparison adds results; it
does not add a new kind of status.

### Decisions, with the reasons

**1. The challenger still runs the full gate. Comparison is added on top and
never replaces it.** A challenger that beats a bad champion is still BLOCKED
if it fails fairness, validation or compliance on its own terms. "Better than
what we had" is not a pass. The reverse also holds: a challenger that passes
every absolute check but regresses against the champion is BLOCKED by the
comparison. Status aggregation is untouched, so blocking flags still mean
`BLOCKED` and non-blocking flags still mean `NEEDS_REVIEW`.

**2. Both models are scored on the same rows, and the interval is on the
*difference*.** The champion's score from last year's model card is not a
baseline. It was measured on different data, and comparing across datasets
measures the data. So the champion is re-scored on today's validation set,
and each bootstrap resample scores **both** models on the **same** rows
(a paired bootstrap through the existing `uncertainty.bootstrap`, with the
statistic returning `metric(challenger) − metric(champion)`).

Checking whether two separate confidence intervals overlap is the tempting
alternative, and it is wrong. Two models scored on the same rows make
correlated errors, so the difference is far more precise than either score.
Overlapping intervals would call a real improvement `UNCERTAIN` almost every
time. Row-order invariance comes free from `canonical_order`.

**3. The decision rule is non-inferiority, read three ways like every other
interval.** `Δ` is signed so that positive always means "challenger better",
using the `greater_is_better` that `MetricSpec` already carries, so RMSE and
AUC read the same way. Against a margin `δ`:

| Interval of Δ | Outcome | Flag | Blocking |
|---|---|---|---|
| entirely above `+min_improvement` | improved | `OK` | — |
| entirely above `−δ` | non-inferior | `OK` | — |
| straddles `−δ` | uncertain | `UNCERTAIN` | no, and `on_uncertain` applies |
| entirely below `−δ` | worse | `CHALLENGER_WORSE` | **yes** |

The outcome is recorded in metadata; only the last two rows are flags. An
improvement is not a finding, and a non-OK flag would route every better model
to a person.

`ComparisonConfig.require` decides what counts as enough:

- `"non_inferior"` **(default)**: the challenger must not be materially worse.
  Retraining on fresh data, a cheaper model, a simpler one or one that drops a
  contested feature are all legitimate reasons to replace a model that is not
  measurably better. Requiring improvement by default would block every one.
- `"superior"`: the challenger must be measurably better, and a non-inferior
  one becomes `CHALLENGER_NOT_BETTER`, blocking. This is for teams whose change process
  says a replacement must earn its risk.

**4. The default margin is relative, because no absolute default is honest.**
`δ = 0.01` means something for AUC and nothing for RMSE on claim costs. So
`margin_relative = 0.01` (1% of the champion's score) is the default. It is
unit-free and works for every metric in `metrics.py` and for a custom one.
`margin` (absolute) overrides it when set. Setting the margin to zero is
allowed but warned about. With `δ = 0`, two identical models straddle the
line by construction, so every pure retrain reads `UNCERTAIN`.

**5. One primary metric decides; guardrails can only block on clear harm.**
The primary metric defaults to `config.performance.metric`, so the comparison
scores what the absolute gate scores. `ComparisonConfig.guardrails` lists
secondary metrics that must not regress: Brier or ECE beside AUC, and the
Lorenz Gini and A/E on a pricing model. Every extra guardrail is another
chance for sampling noise to block a deploy, so guardrail intervals are
Bonferroni-widened across the guardrail count. They block only on
`CHALLENGER_WORSE`; a straddle is `UNCERTAIN`, non-blocking.

### The checks

A new category, `comparison`, reported straight after `validation`. It is the
question a reviewer of a replacement asks first.

| Check | Blocking | What it asks |
|---|---|---|
| `ComparisonIntegrityCheck` | yes | Can these two models be compared at all? Same task; the **same class set in the same order** (a champion with `classes_ = ["no", "yes"]` and a challenger with `["yes", "no"]` produce a confident, inverted comparison and no error); the same label encoding; scores that do not rank the outcome backwards; probabilities against probabilities under a ranking metric; the champion can score the frame at all. |
| `ChallengerPerformanceCheck` | yes | Primary metric Δ with its paired interval, read by the table above. |
| `ChallengerGuardrailCheck` | on regression | Each guardrail metric's Δ, widened for multiplicity. |
| `SubgroupRegressionCheck` | no | Better on average and worse for someone? Δ per protected group and per `ComparisonConfig.segments` column. An aggregate improvement routinely hides a segment that got worse, and that is a judgement call, so it routes to review like the fairness checks. |
| `FairnessDeltaCheck` | no | Did a disparity **widen**? A challenger whose demographic-parity gap moves from 0.02 to 0.09 passes an absolute threshold of 0.10 and has still gotten materially less fair. Reports challenger vs champion for each fairness statistic the gate already computes. |
| `DecisionChurnCheck` | no | For classifiers: how many decisions flip, split into **fixes** (champion wrong, challenger right) and **breaks** (the reverse). Breaks are the number to watch: they are the people for whom a "better" model is worse, and a large break rate under a small net gain is a model that is different rather than better. Regression models already have this in `DislocationCheck`, which now reads the champion's predictions automatically. |
| `OperationalDeltaCheck` | no | p95 latency and cost per inference, challenger against champion, when both are supplied. A 1-point gain at 4× the cost is a business decision, not a statistical one. |

Every check that compares uses the same paired resamples, so the numbers in
one report agree with each other.

### Supplying the champion: files, frameworks, and when not to load it at all

The request is "point at an old model file, whatever the format". Most of that
is easy. The honest limits are worth writing down before someone discovers
them in a pipeline.

`Champion` accepts four sources, most to least convenient:

```python
Champion.from_file("v3.joblib")  # a file; format chosen by extension
Champion(model=estimator)  # an object already in memory
Champion(predict_fn=fn, predict_proba_fn=fn)  # the same callables ModelAdapter takes
Champion(predictions=y_old)  # scored elsewhere: same rows, same form as y_pred
```

`from_file` dispatches on extension and imports lazily, so the core install
still pulls in no framework:

| Extension | Loader | Notes |
|---|---|---|
| `.joblib`, `.pkl`, `.pickle` | `joblib.load` | what the CLI does today |
| `.skops` | `skops.io.load` | the safe format; untrusted types refused |
| `.onnx` | `onnxruntime.InferenceSession` | the input dtype is declared by the model, so nothing is guessed |
| `.keras`, `.h5` | `keras.models.load_model` | `.predict()` works as it does for `model=` |
| `.pt`, `.pth`, SavedModel dirs, anything else | **refused**, pointing at `loader=` | see below |

**PyTorch and TensorFlow SavedModel get a loader spec, not a guess.** A `.pt`
file is either a `state_dict`, which cannot be loaded without the class that
defines the network, or TorchScript, which can be loaded but whose tensor
dtype, device and batching are the caller's business. `model.py` already draws
this line: the library never converts a DataFrame to a tensor, because a wrong
guess produces plausible numbers rather than an error. So those formats use
the mechanism the CLI already has:

```python
Champion.from_loader("mypkg.serving:load_v3")  # returns a model or fn(DataFrame) -> array
```

```bash
--champion-loader "mypkg.serving:load_v3"
```

That same spec covers a registry (MLflow, SageMaker, Vertex). A
`load_v3` that fetches `models:/scorecard/Production` is three lines on the
caller's side, so registry URIs are not built in.

**An old pickle often will not load in the new environment.** The champion
was serialised under last year's scikit-learn, and the challenger's
environment has this year's. Unpickling across versions either fails or,
worse, warns and loads a subtly different model. A comparison against the
wrong champion is worse than none. So:

- Version warnings raised while unpickling (`InconsistentVersionWarning`) are
  treated as errors. The load fails and says why.
- The error message names the fallback: score the champion **in its own
  environment** and pass `Champion(predictions=…)`, or `--champion-col` in the
  CLI. Predictions computed on the same rows are exactly as valid for every
  check above except `OperationalDeltaCheck`.
- A comparison that was asked for and cannot run is a `GateConfigurationError`,
  never a skipped check. The rule is the one `metric="roc_auc"` already
  follows: naming something opts out of fallback.

**Loading a pickle runs code.** The champion comes from a registry the team
controls, so this is acceptable, but it should be a visible choice. The
report records it: `Champion.describe()` gives the loader used, the file's
`sha256` and size, and any framework version found in the artifact.
`sha256=` pins the artifact, and it is checked **before** anything is
deserialised: a mismatch refuses to load, so unknown bytes never run. That makes "the model we compared against is the
one actually in production" something the record proves, not something it
assumes. `.skops` is recommended in the docs for new champions.

**Different feature sets are normal.** A challenger usually adds features. `X`
carries the union; each model gets its own columns from its fitted
`feature_names_in_`, or from `Champion(features=[...])` when the model does not
record them. A column the champion needs and `X` lacks blocks through
`ComparisonIntegrityCheck`.

**Different operating points are normal too.** The champion may have been run
at a cut-off of 0.42. `Champion(decision_threshold=0.42)` means churn and
label metrics compare each model at its real threshold, not at a shared 0.5
that neither used.

### Optional: gate the champion as well

`ComparisonConfig.gate_champion = True` also runs the absolute checks against
the champion and puts both verdicts side by side. It is **informational only
and never affects status**. The point is the finding "the incumbent would be
BLOCKED today on `SubgroupCalibrationCheck`", which is worth knowing before
anyone argues the challenger is too risky to replace it. It is off by default
because it roughly doubles the run time.

### The report

- A comparison section in `to_html()`, straight after validation, with the
  headline Δ in the verdict card. Two charts, per the existing `plot()`
  convention: each group's Δ against its margin, and the 2×2 fixes/breaks
  matrix. The champion's provenance is on the integrity result.
- `report.comparison` in Python, and `"comparison"` in `to_json()`, holding the
  champion's description, the margin actually used (and whether it was
  relative), `require`, and each Δ with its interval.
- `gate_status` stays a single field, because pipelines already branch on it.

### Not in this release

- **More than two models.** Comparing several challengers is model
  *selection*, which belongs in a training pipeline, not a pre-deployment
  gate. One champion, one challenger.
- **Online comparison** (shadow traffic, A/B results). The gate judges a
  holdout before deployment. Live comparison is monitoring, which is a
  different product.
- **Built-in registry URIs.** The loader spec covers them without an adapter
  per vendor.

### Tests the 0.4.2 guards demand

- **Known answers.** A champion identical to the challenger gives Δ = 0,
  churn = 0 and non-inferior at the default margin. A noisier champion gives
  improved, and the reverse gives `CHALLENGER_WORSE`.
- **Invariants.** Antisymmetry: swapping the two models negates every Δ and
  swaps fixes with breaks exactly — the interval only to within resampling
  noise, because the resample order is derived from row content. Row order
  cannot move a verdict. On two closely related models, the paired interval
  excludes zero where the separate intervals overlap.
- **Traps, each with a test that fails without the guard.** Inverted
  `classes_` order; a `sha256` mismatch; an `InconsistentVersionWarning` on
  load; a `.pt` path passed to `from_file`; a margin of zero.
- **Every task.** Binary, multiclass and regression through the full default
  suite with an in-memory champion; predictions-only and file-loaded champions
  in their own tests. The skops, ONNX and Keras loaders are tested against
  stand-in modules, so they run on every install.

### The rest of the slice

- **Example:** `examples/09_champion_challenger.ipynb`. It replaces a
  scorecard, shows a challenger that improves AUC while breaking decisions for
  one segment, then loads the champion from a file whose environment has
  drifted and shows the predictions fallback.
- **Web:** a new `web/docs/comparison.md`; `models.md` gains the loader table;
  the landing page's checks grid gains the `comparison` category, which
  `tests/test_package.py` will require.
- **CLI:** `--champion`, `--champion-loader`, `--champion-col`,
  `--champion-sha256`, `--champion-features`, `--champion-decision-threshold`.
- **Config:** `ComparisonConfig` in JSON/YAML/TOML like every other section.

### Where the build departed from this plan

Recorded rather than silently corrected, so the reasons survive:

- **A pinned hash is checked before loading, not by a check.** The plan had a
  mismatch block through `ComparisonIntegrityCheck` — after the pickle had
  already run. Refusing to deserialise is the only version that protects
  anything.
- **`--baseline-col` is not an alias for `--champion-col`.** An alias would
  have switched the comparison on for every existing pipeline that passes
  `--baseline-col`, including a blocking check it never asked for. That is a
  verdict change by stealth.
- **`Champion(predictions=...)` takes one argument**, in the same form as
  `y_pred`, and `--champion-proba-cols` was dropped. A second `proba=` would
  have been a second way to describe the same thing.
- **`FeatureContractCheck` is not run against the champion.** Each model is
  handed its own columns; one the champion needs and `X` lacks blocks through
  the integrity check, which is the failure that matters.
- **Improvements are `OK` with an outcome in metadata**, and the risk flags
  are `CHALLENGER_WORSE` / `CHALLENGER_NOT_BETTER` — not `REGRESSION`, which
  reads as the task.
- **Guardrails take a dict of absolute margins.** A relative margin on an ECE
  of 0.03 is 0.0003, inside its noise, so the list-only form made every
  calibration guardrail `UNCERTAIN`.
- **`max_break_rate` defaults to 10%, not 5%.** An ordinary retrain that gains
  a few points of AUC breaks a few percent of decisions while fixing more, and
  5% fired on it.
- **`--champion-features`** was added: a champion fitted without
  `feature_names_in_` otherwise has no way to say which columns are its own.

Found while building, and **not fixed on this branch** because each changes
existing verdicts and deserves its own entry:

- `PerformanceThresholdCheck` binds an exposure weight vector to the full
  frame, so on every bootstrap resample the weights land on the wrong rows. An
  exposure-weighted RMSE of 5.36 reported an interval of [131.8, 180.7]. The
  comparison passes weights per resample and is unaffected.
- `to_class_labels` reads a probability matrix's columns in `class_order`,
  while scikit-learn's `predict_proba` orders them alphabetically and
  `calibration_checks` assumes the alphabetical order. With
  `class_order=["decline", "refer", "accept"]` a 99%-accurate model scores
  0.3%.

### Checklist

- [x] `Champion` with `from_file`, `from_loader`, `model=`, `predict_fn=`,
      `predictions=`; lazy extension dispatch; version warnings as errors
- [x] Paired Δ statistic on top of `uncertainty.bootstrap`, signed by
      `greater_is_better`
- [x] `ComparisonConfig`: `metric`, `margin`, `margin_relative`,
      `min_improvement`, `require`, `guardrails`, `segments`, `gate_champion`
- [x] The seven checks above, in a new `comparison` category
- [x] `DislocationCheck` reads champion predictions when `baseline_pred` is absent
- [x] Provenance in `report.comparison`, JSON and HTML
- [x] CLI flags (`--baseline-col` deliberately left alone — see above)
- [x] Notebook 09, docs page, landing page, CHANGELOG
- [ ] Version bump to 0.7.0 and the site's version stamps, at release

## 0.8.0 — Release automation, in two stages

**Branch:** `feat/release-automation`

Publishing is manual. Two of the silent failures this project has shipped were
caught *only* by installing the published artifact, and v0.3.0 and v0.3.1 were
both tagged on commits with a red Python 3.9 job. Fitting, for a tool that
exists to gate deploys.

### The shape: publish a candidate, then promote it

Two workflows, two triggers, one artifact.

```
STAGE 1 — candidate            trigger: tag v* pushed
  guard        tag == pyproject version, and CI was green for this exact SHA
  build        sdist + wheel; suite run against the INSTALLED wheel
               -> SHA256SUMS recorded, dists uploaded as a run artifact
  testpypi     environment: testpypi, no approval          -> TestPyPI
  smoke        fresh venv, install FROM TestPyPI, run a real gate

        ... a human tests the candidate, for as long as that takes ...

STAGE 2 — promotion            trigger: GitHub Release published
  fetch        download stage 1's dists for this tag; RE-VERIFY SHA256
  pypi         environment: pypi, REQUIRED REVIEWER        -> PyPI
```

### Why two workflows rather than one with an approval gate

The obvious design is a single workflow whose last job waits on a required
reviewer. Three reasons not to.

**1. Least privilege by construction, not by approval.** Trusted Publishing
binds a publisher to a specific *workflow filename* plus environment. Register
PyPI's publisher against `release-promote.yml` **only**, and a tag push becomes
structurally incapable of reaching PyPI: no credential for it exists in that
workflow. In the single-workflow design the capability is present on every tag
push and held back by a person clicking the right button.

For a library whose entire purpose is stopping the wrong thing reaching
production, "it cannot" beats "someone must not".

**2. Promotion is a decision, and decisions need a record.** Publishing the
GitHub Release is the human act, and the release notes plus the environment
approval are the audit trail. A pending deployment inside a workflow run is
not a record anyone reads a year later.

**3. Testing a candidate takes days, and a pending run does not.** GitHub
expires pending deployments, and a run parked waiting for approval is awkward
to re-trigger. Decoupling means the candidate can sit on TestPyPI as long as it
needs to.

### The property separation puts at risk, and how stage 2 keeps it

> **PyPI must receive the exact bytes that TestPyPI was tested with.**

So **stage 2 must never rebuild.** Builds are not reproducible across runner
images and setuptools versions, and a rebuild would publish an artifact nobody
smoke-tested — which is precisely the failure mode this pipeline exists to
prevent, reintroduced by the pipeline itself.

Stage 1 records `SHA256SUMS` beside the dists. Stage 2 downloads that run's
artifact, re-verifies every digest, and **fails if the artifact is missing or a
digest differs**. No fallback to building; a missing candidate means re-running
stage 1, not improvising.

### One decision to take before writing any of it

**Does the candidate burn the real version number on TestPyPI?**

Version collisions are permanent on *both* indexes. Publish `0.6.0` to
TestPyPI, find a bug, and `0.6.0` can never be published to TestPyPI again.

The tempting escape is to publish `0.6.0.devN` to TestPyPI and `0.6.0` to PyPI
— but then the two artifacts are **not the same bytes**, and the property above
is gone. You cannot have both identical artifacts and a re-testable version
number.

**Take identical artifacts.** A failed candidate means bumping the patch
version and tagging again, which costs a version number and nothing else. Say
so in `CONTRIBUTING.md` so it is not rediscovered mid-release.

### Guards

- **Tag/version consistency** — fail if the tag does not match
  `pyproject.toml`. Cheap, and the whole release hangs off it.
- **Publish only on a green matrix for that commit** — query the CI conclusion
  for the tagged SHA rather than trusting that a tag implies a tested commit.
  This is the guard that v0.3.0 and v0.3.1 did not have.
- **Test the installed artifact, not the source tree.** The Python 3.9 import
  failure lived in the *published wheel*, and the shap/numpy clash only
  appeared on a clean install. Running the suite against the repo would have
  missed both.
- **`skip-existing: true` on TestPyPI, `false` on PyPI.** Re-running a
  candidate is normal; a duplicate on PyPI is a mistake and should fail loudly.
- **`--extra-index-url https://pypi.org/simple/` on the smoke test.** TestPyPI
  does not carry numpy or pandas, so `pip install --index-url` alone cannot
  resolve this package's dependencies. This trap has cost other projects an
  afternoon; it costs us one line.
- **Assert `__version__` from the installed wheel** in the smoke test, then run
  a real gate end to end and check the verdict — not just that the import
  succeeded.

### Manual prerequisites (repo admin, not code)

- [ ] Register **TestPyPI** trusted publisher: repo, `release-candidate.yml`,
      environment `testpypi`
- [ ] Register **PyPI** trusted publisher: repo, `release-promote.yml`,
      environment `pypi` — a *different* workflow filename, which is the point
- [ ] Create the `testpypi` environment (no protection)
- [ ] Create the `pypi` environment with a **required reviewer**
- [ ] Do the branch and tag ruleset work in
      [Before the next release](#before-the-next-release--the-repository-itself)
      first; only an Environment stands between a tag and PyPI, and only a
      ruleset stands between a bad commit and `main`

Two things to know: if either workflow **filename** changes, publishing breaks
until the trusted-publisher registration is updated; and **version collisions
are permanent**, per the decision above.

## Later

- **A public, subclassable `ModelAdapter` (1.0.0).** The extension point is a
  plain callable for now, which covers every case with less ceremony; a named
  class earns its place once someone needs batching, retries or auth on a
  serving layer.
- **Unstructured data** (text / image / audio).
  `bdp_model_gate.unstructured` reserves the shape and raises
  `NotImplementedError` until it lands. Deliberately *after* the statistical
  work above: broadening the modality before deepening the statistics would
  trade a defensible niche for a shallow generalist.
