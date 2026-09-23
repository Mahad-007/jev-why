# LinkedIn draft

Angle: not "I built a thing with the new model". It is "I built the tool that
tests whether the new model can be trusted, and then spent most of the effort
trying to prove my own tool wrong." That second half is the differentiator --
almost nobody shows their own negative controls.

Figures here came from a live run and are filled from `assets/run.json` by
`scripts/update_readme.py`. The only thing still to fill is the repository URL,
which does not exist until you create it. Do not post with a PENDING left in.

---

## Draft A — the main post

Jev doesn't explain itself. That's the whole design: typed decisions, calibrated
probabilities, no text. It's why it's fast and cheap, and it's why it can't go
anywhere a decision has to be defensible.

So I stopped asking it to explain and measured the reasons instead.

Split the input into spans. Re-run the same question with each span redacted.
The change in probability is a causal effect you measured — not a story a second
model told you afterwards.

That idea is decades old. What's new is that Jev makes it cheap enough to matter:
output tokens are free, and every question in a request is answered in one
parallel pass. A full span sweep over a page runs in cents.

Here's the part I'd want a reviewer to push on, so I did it myself:

— An explanation that isn't faithful is worse than none, because it's convincing.
So comprehensiveness is only ever reported against a random ordering of the same
spans at the same budget. If deleting my top-ranked spans doesn't beat deleting
random ones, the tool says so.

— Occlusion pushes inputs off-distribution, so part of what you measure is the
redaction, not the missing content. Every run carries an extra question — "does
this input look truncated?" — for about 2% more tokens, and reports the answer.

— Jev is *described* as deterministic. That isn't documented. So the tool scores
the unmodified input several times, measures the spread, and refuses to report
any attribution smaller than three times that noise floor.

— And it scores faithfulness under a *different* masker than the one used to
produce the ranking, because otherwise the test quietly rewards the masker's own
artifacts.

Findings I didn't expect:

1. **Jev is nearly, but not exactly, deterministic.** I scored the same
unmodified input repeatedly across several runs and measured spreads of 0.000,
0.005 and 0.010. Not documented anywhere. It matters because it sets the floor:
anything under about 0.03 is not evidence, it is jitter.

   And that is exactly why the floor is measured on every run instead of once.
   The first run I did happened to see 0.000, which would have licensed reading
   real meaning into differences that later turned out to be noise.

2. **The redaction is not invisible.** I asked Jev, on every single call,
whether the input looked truncated. It said yes, 0.86. So part of what
occlusion measures here is the mask itself, not the missing content. That is a
real limitation of the method and it is in the README, not a footnote.

3. **The spans interact strongly** — around a 50% efficiency gap, which means
the cheap leave-one-out estimator is out of its depth on this document and the
tool says so rather than quietly reporting numbers that do not add up.

4. **My own tool caught my own broken run.** The provider throttled and 18 of
44 calls died. The version I had then scored those missing spans as "no effect"
and produced a confident table whose top span was not the injected sentence at
all. It now refuses: "only 76% of spans were measured; any of them could outrank
everything in this table." Rerunning filled the gaps from cache, a few calls at
a time, until it did not have to say that any more.

That fix is the most useful thing I wrote all day, and it is the whole thesis in
miniature: a confident wrong answer is the failure mode worth engineering
against.

Limits, because they matter more than the demo:
cost is O(N²) in document length, not linear. Below ~30 spans the permutation
control can't reach significance however real the effect is. And "questions are
free" is too strong — their specs are input tokens on every call.

Repo: PENDING_URL  <- paste the GitHub link once the repo exists
MIT. It runs offline against a synthetic set function whose Shapley values are
known in closed form, so you can check the estimators without an API key.

#AI #MachineLearning #Explainability

---

## Draft B — the shorter, sharper version

Everyone's asking whether Jev's confidence can be trusted in production.

I stopped speculating and built the instrument.

jev-why redacts one span at a time and measures what the probability does. The
delta is a causal effect, not a rationalisation — and because Jev's output
tokens are free, a whole document's worth costs cents.

Then I spent most of the effort trying to break it:
• every ranking is scored against random spans at the same budget
• faithfulness is measured under a *different* masker, so the test can't reward
  the masker's own artifacts
• repeated identical calls set a noise floor, and anything under it is reported
  as noise rather than insight
• an extra question on every call measures how "damaged" the redacted input
  looks to Jev itself

On a page of billing documentation with one injected sentence buried in it, the
injected sentence scored +0.86 — necessary and sufficient on its own. Every
other sentence landed between -0.02 and -0.06, near the measured noise floor.
Forty-four calls, $0.0007.

Honest limits in the README, including the one that surprised me: below about
thirty spans, the significance test can't fire however real the effect is.

PENDING_URL

---

## Carousel / image plan

1. The attribution heatmap: a benign help page, one sentence lit up.
   Caption: "one injected sentence in a page of billing docs. Nothing else moved."
2. The faithfulness curve: attributed vs random control, with the gap shaded.
   Caption: "if the blue line didn't beat the orange one, the tool would say so."
3. The reliability diagram with its prediction histogram.
   Caption: "strong ranking, weaker probabilities — measured, not asserted."
   (Needs the corpus run; skip this slide if that has not been done.)

## Comment to pin

The part I'd push back on if I were reading this: occlusion measures what the
model does when content is *redacted*, which is close to but not identical to
what that content contributed. That's a real limitation, not a footnote, and
it's why the artifact probe exists.
