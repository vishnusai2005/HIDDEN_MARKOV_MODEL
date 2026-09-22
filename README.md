# Hidden Markov Model POS Tagger — Built From First Principles

![Python](https://img.shields.io/badge/Python-3.x-blue) ![Dependencies](https://img.shields.io/badge/dependencies-standard%20library%20only-brightgreen) ![Status](https://img.shields.io/badge/status-in%20progress-yellow)

A from-scratch implementation of a **Hidden Markov Model (HMM)** for Part-of-Speech tagging — no `nltk.pos_tag()`, no `spaCy`, no pre-trained tagger. Just probability theory and Python's `collections.defaultdict`.

Most tagging tutorials start (and end) by calling a library function. This project instead builds the statistical machinery underneath that function: how a tagger learns from a labeled corpus, and how it uses context — not just word identity — to resolve ambiguity.

---

## The Problem This Toy Corpus Is Designed to Expose

POS tagging looks trivial until a word can legitimately be more than one tag. This project's training set is built specifically around two such words:

| Sentence | Tags |
|---|---|
| Mary Jane can see Will | N N M V **N** |
| Spot will see Mary | N **M** V N |
| Will Jane spot Mary | **M** N **V** N |
| Mary will pat Spot | N M V **N** |

- **"Will"/"will"** appears as a proper noun *and* as a modal verb.
- **"Spot"/"spot"** appears as a proper noun *and* as a verb.

A tagger that just looks up "the most common tag for this word" fails here. An HMM resolves the ambiguity using **transition probabilities** — what tag is likely to follow the previous tag — which is the entire point of the exercise.

## What's Actually Implemented

1. **Corpus & tag set** — 4 hand-labeled sentences, 3 tags (`N` = Noun, `M` = Modal, `V` = Verb).
2. **Parameter estimation** — single pass over the corpus building three count tables via nested `defaultdict`:
   - `start_counts` — which tag opens a sentence
   - `transition_counts[prev_tag][tag]` — including explicit `<S>` (start) and `<E>` (end) boundary tokens
   - `emission_counts[tag][word]` — lower-cased word given a tag
3. **MLE normalization** — raw counts converted to probability distributions by dividing each row by its total.
4. **Sequence probability scoring** — given a sentence *and a candidate tag sequence*, computes the joint probability by chaining `start → emission → transition → … → end`.

## Sample Output

```
Start Probabilities: {'N': 0.75, 'M': 0.25}

Transition Probabilities:
  <S> → {'N': 0.75, 'M': 0.25}
  N   → {'N': 0.111, 'M': 0.333, 'V': 0.111, '<E>': 0.444}
  M   → {'V': 0.75, 'N': 0.25}
  V   → {'N': 1.0}

Emission Probabilities:
  N → {'mary': 0.444, 'jane': 0.222, 'will': 0.111, 'spot': 0.222}
  M → {'can': 0.25, 'will': 0.75}
  V → {'see': 0.5, 'spot': 0.25, 'pat': 0.25}

Sentence: ['Will', 'can', 'spot', 'Mary']
Candidate tags: ['N', 'M', 'V', 'N']
P(sequence) = 0.00025720164609053495
```

Note what's happening in that last example: "Will" is scored as a **Noun** here (correctly, since it's the subject of the sentence), even though "will" is a **Modal** in three of the four training sentences. That's context — captured by the transition probabilities — doing its job.

## Tech Stack

Deliberately minimal: **pure Python + `collections.defaultdict`**. No ML libraries. The point of this project is to internalize the math (Markov assumption, maximum-likelihood estimation, generative sequence probability) before relying on library abstractions for it.

## Honest Audit — Current Limitations

In the interest of not overselling this: here's exactly what's *not* done yet, and why.

- **The Viterbi decoder is not implemented.** The final code cell is titled `#viterbi algorithm for POS tagging` but contains a single `print(states)` referencing a variable that was never defined — it raises a `NameError` and the cell has no actual algorithm in it. As it stands, the notebook can only **score a tag sequence you already guessed**; it cannot **find the best one automatically**, which is the actual job of a POS tagger.
- **No smoothing.** Probabilities are raw MLE counts. Any word or transition not seen in training gets a hard zero probability, and the scorer explicitly bails out with "impossible sequence." A single out-of-vocabulary word breaks the model completely.
- **No log-space computation.** Sequence probability is computed by chaining raw multiplications. For longer sentences this will silently underflow toward `0.0` well before it hits a genuine zero-probability case — a real numerical stability issue, not just a cosmetic one.
- **Toy dataset only.** Four sentences are enough to illustrate the mechanics, but nowhere near enough to train a usable tagger. There's no train/test split and no accuracy metric anywhere in the notebook.
- **No baseline comparison.** There's nothing here yet showing the HMM outperforms a trivial "most frequent tag per word" tagger — which is normally the whole argument for using an HMM in the first place.
- **A leftover debug print** (`print(previous_tag)` at the end of the counting cell) has no functional purpose — it just prints whatever tag happened to be last in the loop.

## Roadmap

- [ ] Implement the Viterbi dynamic-programming decoder (O(T·N²)) to actually predict tags for new sentences
- [ ] Add Laplace/add-k smoothing (or an unknown-word backoff strategy) for unseen emissions and transitions
- [ ] Move probability computations to log-space to avoid underflow
- [ ] Train and evaluate on a real tagged corpus (e.g., NLTK's Brown or Penn Treebank sample) with a train/test split and per-tag accuracy/confusion matrix
- [ ] Benchmark against a most-frequent-tag baseline to quantify what the HMM actually buys you

## Concepts Demonstrated

- Markov assumption & hidden vs. observed states
- Generative probabilistic sequence modeling
- Maximum-likelihood parameter estimation from counts
- The foundational math underneath every modern sequence-labeling model (CRFs, BiLSTM-CRF, Transformer-based NER/tagging) — none of which work fundamentally differently at the core, just with richer feature functions

## How to Run

```bash
git clone <this-repo-url>
cd <repo-folder>
jupyter notebook NLP_pos_tagging.ipynb
```

No `pip install` required — the entire implementation uses the Python standard library.

---

### Connect

**Vishnusai Vydhyam** — Final-year CSE (AI & ML) student, building toward ML/AI Engineer roles

[GitHub](https://github.com/vishnusai2005) · [LinkedIn](https://linkedin.com/in/vishnusai-vydhyam) · [Hugging Face](https://huggingface.co/v2005) · [X](https://x.com/VishnusaiSaii)
