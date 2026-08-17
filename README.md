# llm-eval-harness

[![CI](https://github.com/saivarun161/llm-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/saivarun161/llm-eval-harness/actions/workflows/ci.yml)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![Zero runtime dependencies](https://img.shields.io/badge/runtime%20deps-0-brightgreen.svg)](pyproject.toml)
[![Coverage 97%](https://img.shields.io/badge/coverage-97%25-brightgreen.svg)](#development)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An evaluation harness for LLM applications built on one rule: **a score without a
confidence interval is a rumour.** Every number it prints carries an interval,
every A/B comparison is paired and resampled, the judge is measured against
human labels before it is trusted, and the CI gate fails only when a regression
is bigger than the noise. It runs end to end with no API key.

The usual eval setup reports that quality went from 0.78 to 0.81 and someone
ships. On a forty-five case eval set that difference is comfortably inside the
resampling noise, and the same pipeline would have reported it after a change
that did nothing at all. The interesting question is never "did the number go
up" — it is "did it go up by more than this eval set can measure", and answering
that takes three things most harnesses leave out: pairing, resampling, and a
judge whose error rate you have actually looked at.

---

## Quickstart (60 seconds, no API keys, no network)

```bash
git clone https://github.com/saivarun161/llm-eval-harness.git
cd llm-eval-harness

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

evalharness demo
```

The demo runs five simulated systems over the bundled 45-case dataset and walks
through every claim above. Five things happen in order.

**One. Four scorers look at the same outputs and disagree.**

```
scorer              n     mean        95% interval  distribution
------------------------------------------------------------------------------
exact_match        45    0.489  [ 0.355,  0.622]  ##########..........
token_f1           45    0.626  [ 0.511,  0.740]  #############.......
semantic           45    0.765  [ 0.676,  0.846]  ###############.....
judge              45    0.780  [ 0.677,  0.867]  ################....
```

Twenty-nine points separate the strictest scorer from the most generous. Pick
your headline metric carelessly and you will spend a quarter optimising phrasing.

**Two. A real improvement, and a change that only looks like one.**

```
candidate vs baseline on 'judge'          tweaked vs baseline on 'judge'
  baseline   0.780                          baseline   0.780
  candidate  0.867                          candidate  0.813
  delta      0.087 [0.015, 0.173]           delta      0.033 [0.000, 0.089]
  verdict    IMPROVED                       verdict    INCONCLUSIVE
```

Both systems beat the baseline on the raw mean. Only one of them beat the noise.
A dashboard showing two point estimates calls them both wins.

**Three. The same answers, more words.** The `verbose` system gets exactly the
same cases right as the baseline and wraps each answer in a sentence. Exact match
reports `-0.467` — a catastrophe. The judge reports `-0.072` — a style penalty.
Same outputs, two stories, and only one of them is about quality.

**Four. The judge is measured, not assumed.**

```
Judge calibration (heuristic judge, 72 human labels)
  cut points        partial >= 0.26, correct >= 0.56
  Cohen's kappa     0.676 [0.528, 0.820] linear-weighted, substantial
  raw agreement     73.6%

  confusion (rows: human, columns: judge)
                   wrong   partial   correct
    wrong             21         4         1
    partial            4        10         1
    correct            2         7        22
```

Note what the confusion matrix admits: the judge downgrades seven answers the
humans called correct. That is the judge's real error rate, printed rather than
assumed, and it is why the interval on kappa is there too.

**Five. The gate fires on the regression and stays quiet about the noise.** A
gate that fires on both gets switched off within a fortnight — and then the next
real regression ships.

---

## Architecture

```
      dataset (JSONL, versioned)             target
      ┌──────────────────────────┐      ┌──────────────────────┐
      │ id / input / expected    │      │ module:function      │
      │ tags / metadata          │      │ or a built-in profile│
      │ + sha256 fingerprint ────┼──┐   └──────────┬───────────┘
      └──────────────────────────┘  │              │
                                    │              v
                                    │   ┌──────────────────────┐
                                    │   │ runner               │
                                    │   │  one call per case   │
                                    │   └──────────┬───────────┘
                                    │              │ predictions
                                    │              v
                                    │   ┌──────────────────────┐
                                    │   │ scorer registry      │
                                    │   │  exact_match  fuzzy  │
                                    │   │  token_f1  contains  │
                                    │   │  semantic  regex     │
                                    │   │  judge ──────────────┼──> heuristic (default)
                                    │   └──────────┬───────────┘     or a real model
                                    │              │ per-case scores      (env var)
                                    │              v
                                    │   ┌──────────────────────┐   ┌──────────────────┐
                                    │   │ statistics           │   │ calibration      │
                                    │   │  bootstrap CI        │   │  human labels    │
                                    │   │  paired delta CI     │   │  Cohen's kappa   │
                                    │   │  sign test           │   │  threshold search│
                                    │   └──────────┬───────────┘   └──────────────────┘
                                    │              │
                                    │              v
                                    │   ┌──────────────────────┐
                                    └──>│ gate                 │  fingerprints must match
                                        │  exit 0 / 1 / 2      │  or the delta is meaningless
                                        └──────────────────────┘
```

Runs are plain JSON on disk. The baseline your CI compares against is a file you
commit or fetch from an artefact store; it survives library upgrades, diffs
readably in a pull request, and can be re-analysed six months later when someone
argues about the number.

---

## Using it on your own system

**Point it at your code.** A target is any `Callable[[str], str]`:

```python
# myapp/eval_target.py
from myapp import assistant


def answer(question: str) -> str:
    return assistant.reply(question).text
```

```bash
evalharness run --target myapp.eval_target:answer \
                --dataset evals/support_v3.jsonl \
                --scorers semantic,judge \
                --out runs/candidate.json
```

**Datasets are JSONL** with an optional header line:

```json
{"__dataset__": "support", "__version__": "3.1.0"}
{"id": "refund-01", "input": "How do I get a refund?", "expected": "Open Settings > Billing and choose Request refund", "tags": ["billing"]}
```

**Scorers are one function and a decorator:**

```python
from evalharness.scorers import Verdict, register


@register("cites_a_source", "Answer must contain a doc:// citation")
def cites_a_source(case, output: str) -> Verdict:
    ok = "doc://" in output
    return Verdict(1.0 if ok else 0.0, "citation present" if ok else "no citation")
```

Once registered it is indistinguishable from a built-in: the runner scores it,
the bootstrap intervals it, the gate can gate on it.

**Wire the gate into CI:**

```yaml
- name: Evaluate this branch
  run: |
    evalharness run --target myapp.eval_target:answer \
                    --dataset evals/support_v3.jsonl \
                    --out runs/candidate.json
    evalharness gate --baseline evals/baseline.json \
                     --candidate runs/candidate.json \
                     --scorer judge \
                     --tolerance 0.02 \
                     --require-mde 0.05
```

Exit codes are `0` pass, `1` regression (or an eval set too weak to detect one),
`2` operator error — a missing file, an unknown scorer, a dataset that changed
underneath you.

### Commands

| Command | What it does |
| --- | --- |
| `evalharness demo` | The full guided walkthrough, no keys required |
| `evalharness run` | Evaluate a target, print scores with intervals, optionally save the run |
| `evalharness compare A B` | Paired A/B with a delta interval, win/loss counts and a sign test |
| `evalharness gate` | Pass/fail a build against a stored baseline |
| `evalharness calibrate` | Measure the judge against human labels; `--min-kappa` makes it a check |
| `evalharness scorers` / `datasets` | List what is registered and bundled |

Add `--json` to any of them for machine-readable output.

### Using a real model as the judge

```bash
pip install "llm-eval-harness[judge]"
export EVALHARNESS_JUDGE=model
export OPENAI_API_KEY=...            # or OPENAI_BASE_URL for a compatible endpoint
evalharness calibrate                # measure it before you trust it
```

Nothing else changes — the model judge implements the same interface, so the
same calibration command tells you whether the paid judge is actually better
than the free one *on your data*. Run it before you rewrite your gate around it.

---

## Design decisions

**The bootstrap is paired, and pairing is enforced.** Comparing two independent
intervals throws away the fact that both systems saw the same cases. Case
difficulty is shared, so resampling case *indices* once and applying them to both
runs cancels it out, and the delta interval comes out far tighter than the
difference of the two individual intervals. That only works if the two runs
really did see the same cases, so `compare` and `gate` refuse to proceed when the
dataset fingerprints differ. Otherwise a delta silently measures your eval-set
edit instead of your model change — the most convincing wrong number in the
business.

**Datasets carry both a version and a fingerprint.** The version is declared by
whoever edits the file; it is documentation and nobody remembers to bump it. The
fingerprint is a sha256 over the canonicalised cases in order; it is enforcement.
Two identifiers because they answer different questions.

**The gate has two modes, and the default is the quiet one.** `confident` (the
default) fails only when the *entire* delta interval sits below the tolerance —
it will miss a genuine regression it cannot yet resolve, and it will essentially
never cry wolf. `cautious` fails as soon as a regression that size cannot be
ruled out. The choice is about what is downstream: a flaky gate is worse than no
gate, because a flaky gate gets disabled and takes the real regressions with it.

**The gate measures its own resolving power.** Both modes are blind if the eval
set is too small to detect the tolerance in the first place, and that failure is
silent by construction — the build goes green and everyone reads it as "nothing
broke" rather than "we could not tell". So every comparison prints the minimum
detectable effect, and `--require-mde` turns it into a hard check. This is the
feature most likely to embarrass you the first time you run it against a real
eval set, which is exactly the argument for it.

**The default judge is deterministic, and that is a feature.** A judge you cannot
reproduce is a judge you cannot regression-test, and a harness whose demo needs
an API key is a harness nobody runs. The heuristic judge combines explicit
signals — fact coverage, similarity, numeric contradiction, unmatched negation,
refusal, verbosity — into a score that is stable across runs and machines. It is
not a language model and is not sold as one. The real claim is structural: the
judge sits behind an interface with a calibration command pointed at it, so
whichever judge you plug in, you find out its agreement with humans *before* you
gate a release on it. An unmeasured judge is an opinion; a measured one is an
instrument with known error bars.

**Kappa, not accuracy, for judge agreement.** Raw agreement flatters a judge
badly when labels are unbalanced: on a set that is 80% correct, a judge that
says "correct" every time scores 80% agreement — and a kappa of exactly zero.
The bundled labels use an ordinal wrong/partial/correct scale and the default
report is linear-weighted, because calling a *partial* answer correct is a
smaller mistake than calling a *wrong* answer correct, and unweighted kappa
cannot express that difference.

**Threshold search is in-sample, and says so.** `calibrate` grid-searches the two
cut points that map a continuous judge score onto the label scale. That is two
parameters fitted on the label set, so the resulting kappa is optimistic by
exactly as much as any in-sample fit. The command therefore prints kappa at the
shipped cut points alongside the searched ones, so the size of the gap is visible
rather than hidden. The shipped defaults `(0.26, 0.56)` are themselves the output
of this command against the bundled labels.

**Text is hashed with BLAKE2b, not `hash()`.** Python salts string hashing per
process. A hashing vectoriser built on the builtin would produce different
vectors in different processes, so a baseline recorded last week would not be
comparable to a run made today — silently, and only for the semantic scorer.
There is a test that runs the embedder in a subprocess with `PYTHONHASHSEED=random`
to keep that honest.

**Similarity blends cosine with asymmetric coverage.** Plain cosine punishes a
correct answer for being wordy: "Paris" and "The capital of France is Paris"
point in noticeably different directions. Coverage asks the question a
reference-based scorer actually cares about — is the expected answer in there
somewhere — and the blend of the two keeps verbose-but-correct answers scoring
like correct answers.

**A failing target scores zero; a failing scorer raises.** A system that crashes
on an input has, for evaluation purposes, answered it badly, and the run
continues with the error recorded. A scorer that raises is a bug in the harness,
and quietly recording it as zero would corrupt the result set in the one
direction nobody would think to question.

**Zero runtime dependencies.** Bootstraps, correlations and kappa are a few dozen
lines of standard library each. Pulling a numerical stack into every CI image
that wants to run an eval gate is a poor trade, and writing the statistics out
means they can be read and checked rather than trusted.

**The simulated targets are simulated on purpose.** The demo's five systems
produce reference answers deformed in specific ways — verbose, paraphrased,
truncated, confidently wrong, refused — drawn from one shared per-case difficulty
value, so a case that is hard for one system is hard for all of them. A demo that
called a real model would need a key, a network and a budget, and would produce
different numbers every run, which would make it useless for demonstrating that
the statistics are right. Your own targets are real; these exist so the harness's
own behaviour is verifiable.

---

## What this is not

- **The bundled judge is not a language model.** It reads surface features. It
  cannot tell that "two hundred" is close to 206, and there is a test that pins
  that blind spot rather than papering over it. Its measured agreement with the
  bundled human labels is κ = 0.68 — substantial, not authoritative.
- **The local embedder is not a sentence encoder.** It handles paraphrase far
  better than exact match and materially worse than a real embedding model.
  `calibrate` is how you find out what that costs on your data.
- **The bootstrap assumes your cases are a sample of something.** If your eval
  set is forty hand-picked cases that all resemble each other, the interval is
  honest about resampling noise and says nothing about your users.
- **A green gate is not proof of quality.** It is proof that a regression larger
  than your tolerance was not detected by this eval set. `--require-mde` exists
  so the difference between those two statements is visible.

---

## Development

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -e ".[dev]"

.venv/bin/python -m pytest -q --cov --cov-report=term-missing   # 152 tests, 97% coverage
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

CI runs lint, the suite on Python 3.11 and 3.12, and an end-to-end job that
points the harness at itself: the gate must block a known regression, let a
known-good change through, report the underpowered case, refuse a drifted
dataset, and the judge must still reach κ ≥ 0.6 against the human labels. If any
of those stop being true, the build goes red.

## License

MIT — see [LICENSE](LICENSE).
