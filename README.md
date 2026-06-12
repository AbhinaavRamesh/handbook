# The Handbook

> The definitive technical interview resource for ML, AI, and Software Engineering roles

Complete interview preparation for SDE, MLE, ML Research, and AI Engineering roles at top tech companies. Built with [VitePress](https://vitepress.dev/).

**📖 Read it live: [abhinaavramesh.github.io/handbook](https://abhinaavramesh.github.io/handbook/)**

---

## Content Overview

| Section | Files | Description |
|---------|-------|-------------|
| **SDE Coding** | 227 | 150+ problems across 14 categories with step-by-step solutions |
| **GenAI Engineering** | 73 | LLM foundations, RAG, fine-tuning, agents, prompt engineering |
| **ML Fundamentals** | 61 | Core ML concepts, interview FAQ, theory explanations |
| **ML Coding** | 20 | From-scratch implementations in NumPy/PyTorch |
| **Behavioural** | 12 | Company-specific guides for Google, Amazon, Meta, Microsoft, AI Labs |
| **ML System Design** | 6 | Framework, real questions, key phrases |

**Total: 750+ visualizations** (diagrams, animations, architecture charts)

---

## Site Features

- **Progress tracking** — Study plans and checklists (Fast Track, Weekend Sprint, day-of checklists) have interactive checkboxes. Your progress is saved locally in your browser (no account needed) and a progress bar tracks how far you've come.
- **Full-text search** — Search across every page from the top bar.
- **Dark mode** — Toggle in the top-right.
- **AI assistant friendly** — A machine-readable index lives at [`/llms.txt`](https://abhinaavramesh.github.io/handbook/llms.txt) (curated, per the [llmstxt.org](https://llmstxt.org/) convention) and the full content at [`/llms-full.txt`](https://abhinaavramesh.github.io/handbook/llms-full.txt), so tools like ChatGPT, Claude, and Perplexity can cite the Handbook accurately.

---

## Sections

### SDE Coding
14 categories covering all major data structures and algorithms:
- Arrays, Strings, Hash Tables, Linked Lists
- Trees, Graphs, Heaps, Stacks & Queues
- Dynamic Programming, Recursion & Backtracking
- Searching & Sorting, Complexity Analysis
- Fast Track (2-week intensive plan)
- Weekend Sprint (intensive 2-day plan with 32 problems across 6 tiers)

### GenAI Engineering
Comprehensive coverage of modern AI engineering:
- LLM Foundations (transformers, attention, tokenization)
- Prompt Engineering & RAG Systems
- Fine-tuning & Evaluation
- Agents & Tools, LLMOps
- Safety & Alignment, System Design

### ML Coding (From Scratch)
15+ algorithm implementations with visualizations:
- Linear/Logistic Regression, Decision Trees
- KNN, K-Means, PCA, SVM, Naive Bayes
- Neural Networks (MLP), CNN Filters
- Random Forest, Self-Attention
- Batch Normalization, Feature Scaling
- Softmax & Cross-Entropy

### ML Fundamentals
Core concepts and interview FAQ:
- Bias-Variance, Overfitting, Regularization
- Evaluation Metrics, Cross-Validation
- Feature Engineering, Hyperparameter Tuning
- Probability & Bayes, Gradient Descent
- Transfer Learning, Model Interpretability

### Behavioural Interview
Company-specific guides with 148+ questions:
- **Google**: Googleyness, 8 core attributes
- **Amazon**: 16 Leadership Principles
- **Meta**: Jedi interview format, SPSIL method
- **Microsoft**: Growth mindset, SOAR method
- **AI Labs**: Anthropic, OpenAI, DeepMind
- **Apple, Netflix, Stripe**: Company-specific culture
- Story templates, question banks, quick reference

### ML System Design
End-to-end ML system design framework:
- 6-step design framework
- Real interview questions from FAANG
- Key phrases and vocabulary
- Experience mapping techniques

---

## Quick Start

```bash
cd docs

# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build
```

Visit `http://localhost:5173/handbook/` after starting the dev server.

---

## Project Structure

```
docs/
├── sde-coding/          # 227 files - DSA problems & solutions
├── genai/               # 73 files - GenAI engineering
├── ml-fundamentals/     # 61 files - ML theory & FAQ
├── ml-coding/           # 20 files - From-scratch implementations
├── behavioural/         # 12 files - Company interview guides
├── ml-design/           # 6 files - System design
└── .vitepress/          # VitePress configuration
```

---

## Contributing

Corrections and topic suggestions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup and conventions, [open an issue](https://github.com/AbhinaavRamesh/handbook/issues/new/choose) to report an error or request a topic, or start a thread in [Discussions](https://github.com/AbhinaavRamesh/handbook/discussions). By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## License

MIT — see [LICENSE](LICENSE).
