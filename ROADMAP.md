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

Current release: **0.7.0**. Next: **0.7.1**.

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

## 0.7.1 — Exposure weights that follow their rows

**Branch:** `fix/exposure-weighted-intervals`

A bug in 0.6.0, found while building 0.7.0. **It changes intervals, and so
can change verdicts, on every exposure-weighted performance score.**

### What is wrong

`resolve_metric` binds `context.exposure` into the metric as
`functools.partial(fn, sample_weight=exposure)` — the full vector, in the
frame's original row order. `PerformanceThresholdCheck` then bootstraps that
metric by calling it on resampled rows: `metric.fn(y_true[rows], y_pred[rows])`.
The predictions move with the resample; the weights do not. Every draw pairs
row *i*'s error with some other row's exposure.

The point estimate is computed on all rows in order, so it is right. Only the
interval is wrong — which since 0.6.0 is what decides the verdict.

Measured:

| | point | interval reported |
|---|---|---|
| exposure-weighted RMSE, constructed case | 5.36 | **[131.8, 180.7]** — does not contain its own point |
| notebook 07's Lorenz Gini, as published | 0.5637 | [0.484, 0.595] |
| the same Gini, weights following their rows | 0.5637 | [0.522, 0.602] |

Notebook 07's verdict survives — both intervals clear the 0.25 floor — but the
published number is wrong, and a floor nearer 0.5 would have read `UNCERTAIN`
where the data says `OK`.

**Affected:** `performance_thresholds` whenever `context.exposure` is set and
the metric takes a weight — `rmse`, `mae`, `mape`, `poisson_deviance`, `r2`,
`lorenz_gini`. The regression fairness and actuarial checks index their
weights per resample and are not affected; neither is the 0.7.0 comparison,
which was written to pass weights per call.

### The fix

**Never bind a per-row array into a callable that will be resampled.**
`resolve_metric` returns the unweighted function plus a `weighted` flag, and
the caller passes `sample_weight=exposure[rows]` on each call — the shape
`ComparisonMetric.score` already has. One mechanism, shared by both, so the two
cannot drift.

### Guards, so this class of bug stays dead

- **Constant exposure is no exposure.** With every row weighted 1.0, every
  exposure-reading check must give exactly the interval it gives without
  exposure. This is the test that would have caught it, and it applies to all
  of them, not just the one that was broken.
- **Units do not matter.** Exposure × 12 (months instead of years) must not
  move a point or an interval — the invariant 0.7.0's comparison tests
  already assert, extended to the performance check.
- **A weighted interval contains its point** on a well-behaved case, asserted
  for each weight-taking metric.
- An audit of every `bootstrap` statistic for closures over un-indexed per-row
  arrays, with the result written down rather than assumed.

### The rest of the slice

- Re-execute 03 and 07; 07's Gini interval changes, as above.
- CHANGELOG under **⚠️ This release changes verdicts**, naming the metrics and
  the condition (`exposure` supplied), so a pinned pipeline knows whether it
  is affected.
- `web/docs/concepts.md#how-sure-is-it` gains a sentence on weights.

## 0.7.2 — Probability columns in the order the model wrote them

**Branch:** `fix/probability-column-order`

A bug in 0.6.0 and earlier, found while building 0.7.0. **It changes scores and
verdicts for multiclass models gated on a probability matrix with a
non-alphabetical `class_order`** — which is the ordinal case, the one
`class_order` exists for.

### What is wrong

The library holds two conventions for which class a probability-matrix column
belongs to, and uses both:

| Reads columns in | Used by |
|---|---|
| `class_order` | `to_class_labels` → `performance_thresholds` (score and the ordinal confusion chart), `disparate_impact` and `equalised_odds` for multiclass, and the 0.7.0 comparison checks on a matrix |
| sorted order (`sorted(class_order, key=str)`) | `calibration_checks`, `shap_subgroup_gap`, `counterfactual_flip`, the leakage check's model score |

scikit-learn's `predict_proba` — and LightGBM's and XGBoost's sklearn APIs —
emits columns in `model.classes_` order, which is sorted. So the second group
is right for every common model and the first is wrong whenever `class_order`
is not alphabetical. Measured: a `LogisticRegression` that is **99% accurate**
by its labels scores **0.3%** from its own `predict_proba` matrix with
`class_order=["decline", "refer", "accept"]`, because "accept" (column 0) is
read as "decline".

Nothing reports an error. It is a confident, wrong number — the failure this
project exists to prevent.

### The decision

**`class_order` says which outcome is better. It does not say how a matrix is
laid out**, and using one field for both is the root cause. A matrix's
columns are the model's classes, resolved once:

1. `model.classes_`, when the model has one — the authority for the matrix it
   produced;
2. otherwise sorted order, the convention every scikit-learn-shaped model
   follows;
3. a new, optional `StructuredGateContext.proba_classes` to state it
   explicitly, for a `predict_proba_fn` or an endpoint with its own layout.

One helper owns that resolution and every call site above uses it, so there is
one convention in the codebase rather than two. When `model.classes_` and
`class_order` name different sets of classes, validation fails before any check
runs rather than mapping one onto the other.

For a `Champion`, the same rule applies to the champion's own model — its
`classes_` may differ from the challenger's, which `comparison_integrity`
already reports.

### Guards

- **The matrix and the labels agree.** For every multiclass family in the
  model matrix, scoring `predict_proba(X)` and scoring `predict(X)` with a
  label metric must give the same answer — with a deliberately
  non-alphabetical `class_order`. This is the test that would have caught it,
  and it belongs in `test_model_matrix.py` so a new family inherits it.
- **Column relabelling.** Permuting a matrix's columns together with
  `proba_classes` must not move any number.
- **One convention.** A test that fails if anything other than the helper
  indexes a probability matrix by class.

### The rest of the slice

- Re-execute 02 and 06 (multiclass, labels today, so expected unchanged — say
  so once checked), and add a matrix-scored cell to 02 that would have shown
  the bug.
- CHANGELOG under **⚠️ This release changes verdicts**: affected when
  `y_pred` is a probability matrix *and* `class_order` is not in sorted order.
  Label predictions, binary models and regression are untouched.
- `web/docs/tasks/multiclass.md` states the column contract and
  `proba_classes`.

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
