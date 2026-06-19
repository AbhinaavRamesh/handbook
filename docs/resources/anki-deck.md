---
title: ML Interview Anki Deck
description: A free Anki flashcard deck with 188 ML interview questions and answers, generated from The Handbook's ML Fundamentals interview FAQ. Spaced-repetition practice for bias-variance, regularization, evaluation metrics, optimization, and more.
---

# ML Interview Anki Deck

A **free, 188-card Anki deck** built straight from The Handbook's
[ML interview FAQ](/ml-fundamentals/interview-faq/) — every `Q&A` becomes a flashcard so you can
drill the fundamentals with spaced repetition on your phone or laptop.

<div style="margin:24px 0;">
  <a class="anki-download" href="https://abhinaavramesh.github.io/handbook/downloads/handbook-ml-interview-faq.apkg" download>
    ⬇️ Download the deck (.apkg, 188 cards)
  </a>
</div>

## What's inside

- **188 cards** across **12 topic areas** of core ML, each tagged by topic.
- Generated from the interview FAQ, so the wording matches what you'll read on the site:
  - Bias-variance tradeoff, overfitting & underfitting, regularization
  - Evaluation metrics (precision/recall, AUC-ROC/PR, MSE/MAE, NDCG…)
  - Optimization (gradient descent, cross-validation), ensemble methods
  - Statistics & probability (hypothesis testing, Bayes)
  - Class imbalance, feature engineering, hyperparameter tuning
  - Vanishing gradients, transfer learning, model drift, interpretability

## How to use it

1. Install [Anki](https://apps.ankiweb.net/) (desktop) or AnkiMobile / AnkiDroid.
2. Download the `.apkg` above.
3. In Anki: **File → Import** (or just double-click the file) and pick it.
4. Study daily — Anki schedules reviews so you see each card right before you'd forget it.

::: tip Pair it with active recall
Cards are most effective *after* you've struggled with the material. Take a
[mock interview with an AI](https://abhinaavramesh.github.io/handbook/guides/prep-with-ai), let it
expose a gap, then let these cards keep it fresh.
:::

## Known limitations

To keep the deck self-contained and easy to import, the generator **strips images, Mermaid
diagrams, and block math** — those richer visuals live on the site. Cards keep prose, code blocks,
tables, and inline formatting. For anything visual, follow the topic link back to
[the FAQ](/ml-fundamentals/interview-faq/).

## Regenerate it yourself

The deck is reproducible from source — see
[`docs/scripts/generate_anki_deck.py`](https://github.com/AbhinaavRamesh/handbook/blob/main/docs/scripts/generate_anki_deck.py):

```bash
pip install genanki markdown
python docs/scripts/generate_anki_deck.py
# -> docs/public/downloads/handbook-ml-interview-faq.apkg
```

<style scoped>
.anki-download {
  display: inline-block;
  padding: 12px 22px;
  border-radius: 8px;
  background: var(--vp-c-brand-1);
  color: #fff !important;
  font-weight: 600;
  text-decoration: none;
  transition: background 0.2s;
}
.anki-download:hover { background: var(--vp-c-brand-2); }
</style>
