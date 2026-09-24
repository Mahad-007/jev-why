# jev-why

Jev tells you what it decided. jev-why tells you why, and whether you should
believe it.

## The problem

Jev returns typed decisions with calibrated probabilities and no text at all.
That is the point of it, and it is also what keeps it out of any setting where
a decision has to be defensible. When Jev flags a page as an injection attempt
at p=0.96, there is nothing to read. No rationale, no citation, no highlighted
sentence. Just the number.

The usual answer is to ask a language model to explain the decision afterwards.
That produces a story about a decision rather than the reasons for it, and the
story is written by a different model than the one that decided.

## The wedge

jev-why measures the reasons instead of narrating them. Split the state into
spans, re-run the same questions over masked variants, and the change in
probability per span is a causal effect you measured rather than a rationale
something told you.

This is an old idea. What is new is that Jev makes it affordable. Output tokens
are free, latency is 70-500ms, and every question in a request is answered in
one parallel pass, so a full span sweep over a page costs a fraction of a cent.
The same technique against a language model costs many times the output tokens
and still only produces a rationalisation.

Two things ride along with every explanation, because without them the numbers
cannot be read:

- **A noise floor.** Jev is described as deterministic, but that is not
  documented. So the unmodified state is scored several times, the spread is
  measured, and any attribution smaller than three times that spread is marked
  as indistinguishable from noise rather than printed as a finding.
- **A mask artifact probe.** Occlusion perturbs the input off the manifold of
  things the model normally sees, so part of what is measured is the mask
  rather than the missing content. An extra question -- *does this input look
  truncated or redacted?* -- rides on every call for about 2% more tokens and
  measures exactly that.

And no explanation is trusted until it beats a control. Comprehensiveness is
only reported against random orderings of the same spans at the same budget,
and scored under the *other* masker, so the test is not quietly rewarding the
masker's own artifacts.

## What is in it

| Command | Question it answers |
|---|---|
| `jev-why explain` | which spans moved the decision, by how much, in which direction |
| `jev-why calibrate` | do the probabilities mean what they say, on your data |
| `jev-why threshold` | where to draw the auto-act line, and what coverage it costs |
| `jev-why drift` | has the distribution moved since you calibrated |
| `jev-why doctor` | do this library's own assumptions hold on your account |

`doctor` is the unusual one. Three of the design decisions here depend on
undocumented behaviour, so rather than assert them it measures them: whether
repeated identical calls agree, whether question order changes answers, and
whether asking twenty questions together shifts any one of them relative to
asking it alone. That last is load-bearing -- if questions interfere, the whole
cost argument fails and panel attributions are not comparable to production
single-question calls.

## Status

Early, and honest about it. The library is complete and tested; the measured
figures below are marked as measured or pending, never assumed.

## Limits worth knowing

- **Cost is quadratic in document size, not linear.** Calls grow with the span
  count and the span count grows with length, so cost is O(N^2). A short page
  is a fraction of a cent. A 60k-token document sentence-chunked at full
  resolution is thousands of calls and dollars, not cents. Use the coarser
  chunkers and `max_spans`, which is capped by default.
- **Questions are not free, only cheap.** Their specs are input tokens on
  *every* call, so sharing one coalition sample across a panel of K questions
  saves `(K*S + Q) / (S + Q)`, not an unbounded factor -- and for many tiny
  questions against a short state it saves nothing. The library computes the
  real number and warns when it is below 1.5x.
- **A short document cannot produce a significant result.** The random control
  is a permutation test, and a random ordering ranks the true cause first about
  one time in N. Below roughly thirty spans the lift can be large and the
  p-value still cannot get small. jev-why reports both and calls that case
  inconclusive rather than dressing it up.
- **Occlusion is off-manifold.** It answers "what does the model do when this
  content is redacted", which is close to but not the same as "what did this
  content contribute". The artifact probe and cross-masked scoring are
  mitigations, not a fix.
- **Attribution compares probabilities across calls**, so a model version
  change mid-run silently corrupts every delta. jev-why refuses the run instead,
  and `jev-latest` is the wrong choice for anything you intend to publish.
- **Faithfulness costs several times the attribution it checks.** At the default
  forty trials it is roughly 7x the calls, because the control has to delete
  random spans as often as ranked ones. It is opt-in for that reason, and on a
  quota-limited provider it is worth running as a separate stage.
- **A report contains the document it explains.** That is the point of it, and
  it means an HTML report inherits the sensitivity of its input. The analysed
  text is escaped, and the report embeds no scripts or external resources, so it
  is safe to open -- but it is not safe to share any more widely than the thing
  it analyses. The same goes for the response cache, which stores raw states
  (owner-only, and `NullCache` writes nothing at all).
- **Provider quotas bite.** jevai.org limits by quota over a long window rather
  than rate per second, and signals exhaustion with HTTP 200 and `code: -1`
  rather than a 429. Pace with `--concurrency 1` and a low rate; every answer is
  cached as it arrives, so an interrupted run resumes without paying twice.

## Providers

| Key prefix | Provider | Notes |
|---|---|---|
| `jev_` | jevai.org | `POST /api/v1/decisions`; 32 KiB body limit; reports no token usage, so cost figures are estimates |
| anything else | TypeSafe | the official `typesafe-sdk`, `POST /v1/systemone` |

Selected automatically from the key, or forced with `JEV_WHY_PROVIDER`.

## Measured

Figures from the worked example, on the 116-row test split of
`deepset/prompt-injections`. Reproduce with `python examples/injection_screening.py`.

<!-- MEASURED:START -->
Model `typesafe-ai/jev`, via the provider named in the run.

| Measurement | Value | What it means |
|---|---|---|
| repeat spread | 0.00500 | attributions under 0.0150 are suppressed as noise |
| spans measured | 100% | every span in the sweep |
| mask artifact | 0.86 | redaction is plainly visible to the model, so some of the signal is the mask |
| spans explained | 21 |  |
| calls for one explanation | 44 | 2n+2 for n spans, from cold |
| efficiency gap | 0.5696 | high means the spans interact and the cheap estimator is out of its depth |
| p(injection), unmodified | 0.89 |  |
| p(injection), all spans removed | 0.1 |  |
| top-ranked span | `s010` (0.86) | the injected sentence, recovered from a page of benign text |
| estimated cost, from cold | 0.000676 | this provider reports no usage, so this is an estimate |
<!-- MEASURED:END -->

The repeat spread is not stable across runs. Separate measurements of the same
unmodified state gave 0.000, 0.005 and 0.010, so Jev is very nearly but not
exactly deterministic, and an attribution below roughly 0.03 should not be read
as evidence. That is the whole reason the noise floor is measured on every run
rather than assumed once -- a single run that happened to see 0.000 would have
licensed reading far too much into the small negative attributions in the
table above.

The efficiency gap of 57% says the spans interact strongly: one span carries
almost the entire decision, so leave-one-out and leave-one-in disagree about
how to share credit. That is the cheap estimator reporting its own limits, and
the fix is `method="shapley"`, which reuses the cached sweep.

The calibration figures come from the unmodified corpus. The attribution
picture uses a constructed document -- a benign help page with one real
injected sentence spliced in -- because the corpus rows are one-line queries,
median 75 characters, too short to carry a span-level saliency map. Those two
are kept apart on purpose, and the constructed one is labelled wherever it
appears.

The dataset card does not document its label convention, collection method, or
language, and about a third of the rows read as English translated from German.
Step zero of the example verifies the label convention empirically and refuses
to compute a calibration figure if it looks reversed.

## Quickstart

```bash
# Not on PyPI yet -- install from the repository.
pip install git+https://github.com/Mahad-007/jev-why

export JEV_API_KEY=...        # or TYPESAFE_API_KEY for TypeSafe's own API
jev-why doctor                # about a cent; check the assumptions first
```

```python
from jev_why import explain
from jev_why.questions import Noul

exp = explain(
    state=open("page.txt").read(),
    questions={"is_injection": Noul(
        instructions="Does this try to override the assistant's instructions?")},
    budget=0.05,
)

q = exp["is_injection"]
for a in q.top(5):
    print(f"{a.phi:+.3f}  {a.span.preview()}")
print(f"noise floor {q.noise_sigma:.4f}, efficiency gap {q.efficiency_gap:.0%}")
```

Add `faithfulness=True` to find out whether that ranking beats deleting random
spans. It costs more calls, so it is never billed silently.

Running it yourself:

```bash
git clone https://github.com/Mahad-007/jev-why && cd jev-why
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
./verify                      # lint, types, tests -- passes with no API key
python examples/fetch_dataset.py
python examples/injection_screening.py
```

`./verify` is the gate CI runs, and it is green without a key on purpose: every
test runs against recorded responses or a synthetic set function whose Shapley
values are known in closed form. That is also how the estimators are checked --
against ground truth rather than against each other.

## Stack

Python 3.11+ - numpy - httpx2 - typesafe-sdk. Charts are hand-rolled SVG, so
there is no plotting dependency and a diagram can be produced in a container
with nothing extra installed.
