"""Write README, viva notes, and the analysis notebook from recorded artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.models import MODEL_DISPLAY_NAMES
from src.utils import PROJECT_ROOT, ensure_dir, load_config


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _md_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"{value:.4f}" if abs(value) < 10000 else f"{value:.2f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _artifacts() -> dict:
    metrics = PROJECT_ROOT / "results" / "metrics"
    return {
        "preprocess": _read_json(PROJECT_ROOT / "data" / "processed" / "preprocess_metadata.json"),
        "dataset": _read_json(PROJECT_ROOT / "data" / "raw" / "dataset_metadata.json"),
        "test": pd.read_csv(metrics / "test_metrics.csv"),
        "val": pd.read_csv(metrics / "validation_metrics.csv"),
        "history": pd.read_csv(metrics / "training_history.csv") if (metrics / "training_history.csv").exists() else pd.DataFrame(),
        "gen_metrics": pd.read_csv(metrics / "generation_metrics.csv"),
        "hyper": _read_json(metrics / "actual_hyperparameters.json"),
        "best": _read_json(PROJECT_ROOT / "results" / "reports" / "best_model.json"),
        "samples": _read_json(PROJECT_ROOT / "results" / "generated_text" / "all_samples.json"),
    }


def write_readme(config: dict | None = None) -> Path:
    config = config or load_config()
    art = _artifacts()
    test = art["test"].copy()
    test["display_name"] = test["model"].map(MODEL_DISPLAY_NAMES)
    val = art["val"].copy()
    val["display_name"] = val["model"].map(MODEL_DISPLAY_NAMES)
    hyper = art["hyper"]
    best = art["best"]
    prep = art["preprocess"]
    data = art["dataset"]

    sample_bits = []
    for sample in art["samples"]:
        if sample["seed_phrase"] == "ROMEO:" and sample["temperature"] == 1.0:
            snippet = sample["full_text"][:500]
            sample_bits.append(f"### {sample['display_name']}\n\n```text\n{snippet}\n```\n")

    content = f"""# Tiny Shakespeare Word-Level Autoregressive Text Generation

**Course:** ICT4403 Foundations of Generative AI  
**Assignment:** Assignment 2  
**Task:** Word-level next-token prediction and autoregressive generation with Simple RNN, Bidirectional RNN, LSTM, and GRU.

## Overview

This repository trains four recurrent neural networks on [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) to predict the next word after a 20-token context window. The same splits, vocabulary, sequence length, optimizer, and epoch budget are used for every architecture so quantitative differences are comparable.

## Objectives

- Preprocess Tiny Shakespeare with word-level tokenization.
- Build input–target windows for next-word prediction.
- Train Simple RNN, Bidirectional RNN, LSTM, and GRU under matched settings.
- Compare loss, perplexity, next-word accuracy, training time, and parameter count.
- Generate text from identical seeds at temperatures 0.5, 1.0, and 1.5.
- Analyse temperature, diversity, repetition, and bidirectional-model limitations.

## Dataset

- **Name:** {data['dataset_name']}
- **Direct source URL:** {data['source_url']}
- **Original file size:** {data['bytes']} bytes
- **Decoded characters:** {data['characters']}
- **Word-level tokens:** {prep['total_word_tokens']}
- **Unique types in the full corpus:** {prep['unique_tokens_full_corpus']}
- **Train / validation / test tokens:** {prep['train_tokens']} / {prep['val_tokens']} / {prep['test_tokens']}
- **Train-only vocabulary size (including specials):** {prep['vocabulary_size']}

Tiny Shakespeare is a public-domain concatenation of Shakespeare plays, widely used in language-modelling tutorials. This project uses **word-level** tokens rather than characters, which matches the assignment's next-word prediction requirement.

The original file is small (~1.1 MB) and is included under `data/raw/input.txt` after download so the experiment is reproducible offline. `download_data.py` still fetches and validates the canonical URL when the file is missing.

## Download procedure

```powershell
python download_data.py
```

The script skips the network request when a non-empty, decodable `data/raw/input.txt` already exists. Use `--force` to download again.

## Word-level next-token prediction

Given tokens `To be or not to be` and a window of five input tokens:

- **Input:** `To be or not to`
- **Target:** `be`

The trained models output a distribution over the vocabulary for that next token. Autoregressive generation samples a token, appends it, and repeats.

## Preprocessing methodology

1. Decode UTF-8 text and normalize line endings.
2. Remove unwanted ASCII control characters.
3. Convert newlines to spaces; collapse repeated whitespace.
4. Keep case, punctuation, speaker names, and contractions.
5. Tokenize with a regex that keeps internal apostrophes and isolates punctuation.

Details: `docs/methodology.md`.

## Vocabulary construction

Special tokens `<PAD>`, `<UNK>`, `<BOS>`, and `<EOS>` occupy the first indices. Remaining slots, up to {hyper['vocabulary_limit']} total, are the most frequent **training** types. Validation and test tokens never influence the vocabulary. Unseen types map to `<UNK>`.

Actual vocabulary size used by the models: **{prep['vocabulary_size']}**.

## Input–target sequence construction

`NextTokenDataset` yields windows of length L = {hyper['sequence_length']} with stride {hyper['window_stride']}. Target is the token immediately after the window. Splits are applied on the linear token stream **before** windowing so no sequence crosses a boundary.

Chronological 80/10/10 splitting is used because the source is a continuous literary text. Later acts/passages form a held-out continuation rather than randomly mixed lines.

## Architectures

All four models share embedding dimension {hyper['embedding_dim']}, hidden dimension {hyper['hidden_dim']}, dropout {hyper['dropout']}, and a linear classifier over the vocabulary.

| Model | PyTorch module | Next-token representation |
| --- | --- | --- |
| Simple RNN | `nn.RNN` | Last observed hidden state |
| Bidirectional RNN | `nn.RNN(bidirectional=True)` | Concatenation of final forward and backward states |
| LSTM | `nn.LSTM` | Last-layer hidden state |
| GRU | `nn.GRU` | Last-layer hidden state |

The Bidirectional RNN may read the already observed window from both ends. It never sees the target token. During generation it re-encodes the current window; it is **not** a strictly causal language model.

## Training methodology

- Optimizer: Adam
- Learning rate: {hyper['learning_rate']}
- Batch size: {hyper['batch_size']}
- Epochs: {hyper['epochs']}
- Loss: token-level cross-entropy (padding ignored)
- Gradient clipping: {hyper['gradient_clip_norm']}
- Random seed: {hyper['random_seed']}
- Device actually used: `{hyper['device']}`

The checkpoint with the lowest **validation** loss is saved for each architecture. The test set is not used for selection, early stopping, or hyperparameter search.

## Evaluation metrics and formulas

- **Cross-entropy:** mean token-level negative log-likelihood of the true next token.
- **Perplexity:** `exp(mean cross-entropy)`. Lower is better under a matched evaluation setup. It estimates how many tokens the model is choosing among, on average.
- **Next-word accuracy:** fraction of targets equal to `argmax` of the predicted distribution.
- **Training time:** `time.perf_counter()` wall-clock over the training loop.
- **Parameter count:** number of trainable tensors.

Generation-side metrics (not coherence proofs):

- Unique-token ratio = `|set(tokens)| / |tokens|`
- Distinct-1 = unique unigrams / unigram count
- Distinct-2 = unique bigrams / bigram count
- Repeated bigram ratio = bigram instances with count > 1, divided by bigram count

## Actual quantitative results

Validation:

{_md_table(val, ['display_name', 'val_loss', 'val_perplexity', 'val_accuracy', 'parameter_count', 'total_training_time_sec'])}

Test:

{_md_table(test, ['display_name', 'test_loss', 'test_perplexity', 'test_accuracy', 'parameter_count', 'total_training_time_sec'])}

Machine-readable copies: `results/metrics/test_metrics.csv`, `results/metrics/validation_metrics.csv`, `results/metrics/model_summary.json`.

## Training and validation plots

![Training loss](results/figures/training_loss_curves.png)

![Validation loss](results/figures/validation_loss_curves.png)

## Test performance comparison

![Test perplexity](results/figures/test_perplexity_comparison.png)

![Test accuracy](results/figures/test_accuracy_comparison.png)

![Training time](results/figures/training_time_comparison.png)

![Parameter count](results/figures/parameter_count_comparison.png)

## Generated-text samples

Seeds: `ROMEO:`, `KING:`, `To be, or not to be`.  
Temperatures: 0.5, 1.0, 1.5.  
Complete dump: [results/generated_text/generated_samples.md](results/generated_text/generated_samples.md).

Representative `ROMEO:` continuations at temperature 1.0:

{''.join(sample_bits)}

## Temperature analysis

![Diversity](results/figures/temperature_vs_diversity.png)

![Repetition](results/figures/temperature_vs_repetition.png)

Full write-up: [results/reports/temperature_analysis.md](results/reports/temperature_analysis.md).

Temperature divides logits before softmax. Lower T concentrates mass on high-probability tokens; higher T flattens the distribution. Whether that produces more repetition or more incoherence is an empirical question for each checkpoint and is discussed from the saved samples.

## Diversity and repetition analysis

{_md_table(art['gen_metrics'], ['model', 'temperature', 'unique_token_ratio', 'distinct_1', 'distinct_2', 'repeated_bigram_ratio', 'average_length'])}

High distinct-n is not the same as fluent Shakespeare. These numbers complement, rather than replace, reading the samples.

## Bidirectional RNN limitations

A bidirectional encoder looks at the full *observed* window, including tokens to the right of earlier positions. That can help next-token classification when the entire window is already known, but it does not match the causal process of writing one word at a time from left to right. Rolling the model out still requires re-encoding a window of past tokens and cannot be interpreted as a hidden state that never saw future context *inside that window*. For a production causal LM, unidirectional LSTM/GRU (or a causal Transformer) is the appropriate inductive bias.

## Best-performing model and evidence

**{best['display_name']}** is selected using: {best['criteria']}

- Test perplexity: {best['test_perplexity']:.4f}
- Test loss: {best['test_loss']:.4f}
- Test accuracy: {best['test_accuracy']:.4f}

See [results/reports/experimental_analysis.md](results/reports/experimental_analysis.md) for the architecture discussion.

## Installation

Python 3.10 or 3.11 is the intended target. This run used the local interpreter documented in `docs/EXECUTION_STATUS.md`.

### Windows PowerShell

```powershell
python -m venv .venv
.\\ .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python download_data.py
python main.py --mode all
```

If PowerShell blocks scripts, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` or invoke `.\\.venv\\Scripts\\python.exe` without activating.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python download_data.py
python main.py --mode all
```

GPU is optional. The code uses CUDA when `torch.cuda.is_available()` is true; otherwise it runs on CPU.

## Exact pipeline commands

```powershell
python download_data.py
python main.py --mode preprocess
python main.py --mode train
python main.py --mode evaluate
python main.py --mode generate
python main.py --mode all
```

Useful flags:

```powershell
python main.py --mode train --force-retrain
python main.py --mode all --epochs 5 --batch-size 64
python -m pytest -q
python create_submission.py
```

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── config.yaml
├── main.py
├── download_data.py
├── train.py
├── evaluate.py
├── generate.py
├── create_submission.py
├── src/
├── tests/
├── notebooks/assignment_analysis.ipynb
├── data/raw/input.txt
├── data/processed/
├── checkpoints/          # best *.pt files, recreated by training
├── results/
│   ├── metrics/
│   ├── generated_text/
│   ├── figures/
│   └── reports/
├── docs/
└── submission/
```

## Dependencies

See `requirements.txt`. Core packages: PyTorch, NumPy, pandas, matplotlib, PyYAML, pytest, Jupyter tooling, python-docx. No paid APIs or hosted models are required.

## Reproducibility

- Config lives in `config.yaml` (seed 42).
- Vocabulary and encoded split ids are stored under `data/processed/`.
- Best checkpoints store architecture, hyperparameters, epoch, and validation loss.
- Generation uses stored sampling seeds.
- Retrain with `--force-retrain` if the vocabulary or sequence length changes; incompatible checkpoints are rejected.

Checkpoints are gitignored because they are large binary files. Recreate them with `python main.py --mode train`.

## Limitations

- Small corpus and a hard 8,000-token vocabulary leave many rare words as `<UNK>`.
- Context is only {hyper['sequence_length']} tokens.
- Recurrent hidden states compress history lossily.
- Five epochs is a matched budget, not an exhaustive search.
- Generated verse often repeats or drifts.
- Bidirectional encoding of the window is not causal generation.

## Conclusion

Four recurrent models were trained and evaluated under matched word-level next-token conditions on Tiny Shakespeare. On the held-out chronological test split, **{best['display_name']}** is the strongest quantitative system (test perplexity {best['test_perplexity']:.4f}). Qualitative judgement of Shakespearean style still depends on temperature and remains imperfect for all four architectures.

## License / data note

Tiny Shakespeare is derived from public-domain Shakespeare texts as distributed by the [char-rnn](https://github.com/karpathy/char-rnn) project. Include attribution to that source when redistributing the raw file.
"""
    # Fix the accidental space in Activate path in the generated README
    content = content.replace(".\\ .venv\\Scripts\\Activate.ps1", ".\\.venv\\Scripts\\Activate.ps1")
    path = PROJECT_ROOT / "README.md"
    path.write_text(content, encoding="utf-8")
    return path


def write_viva(config: dict | None = None) -> Path:
    art = _artifacts()
    best = art["best"]
    hyper = art["hyper"]
    test = art["test"]
    rows = "\n".join(
        f"- {MODEL_DISPLAY_NAMES[r.model]}: test loss {r.test_loss:.4f}, perplexity {r.test_perplexity:.4f}, "
        f"accuracy {r.test_accuracy:.4f}, params {int(r.parameter_count)}, "
        f"train time {r.total_training_time_sec:.1f}s"
        for r in test.itertuples()
    )
    text = f"""# Viva questions and answers

Answers refer to this implementation (word-level Tiny Shakespeare, window {hyper['sequence_length']}, seed {hyper['random_seed']}, device `{hyper['device']}`).

## What is autoregressive text generation?

A token is produced given previous tokens, then appended to the context, then the next token is produced. This project samples from `softmax(logits / temperature)` rather than always taking argmax.

## Why is next-word prediction used?

It is the supervised objective for a language model: maximize the likelihood of the true following word. Sequential sampling of those predictions yields generation.

## What is a token?

Here, a token is a word, a contraction with an internal apostrophe, a number, or a punctuation mark produced by the regex tokenizer.

## How is a vocabulary constructed?

Training-token frequencies only. Specials occupy the first indices; the remaining slots up to 8,000 are the most frequent training types. Others become `<UNK>`.

## What are input–target sequences?

For window length L, the input is L consecutive tokens and the target is the next token. Example: input `To be or not to`, target `be`.

## What is an embedding?

A learned lookup table mapping token ids to dense vectors of size {hyper['embedding_dim']}.

## What is a recurrent neural network?

A network whose hidden state at time t is a function of the current embedding and the previous hidden state, so it can process a sequence with shared weights.

## What are vanishing and exploding gradients?

Products of Jacobians through time can shrink toward 0 or grow without bound. Vanishing gradients make early tokens hard to learn; exploding gradients destabilize training. We clip gradients at {hyper['gradient_clip_norm']}.

## How does an LSTM work?

An LSTM maintains a cell state and uses input, forget, and output gates. The cell can carry information across steps with additive updates, which reduces vanishing-gradient pressure relative to a vanilla RNN.

## How does a GRU work?

A GRU uses reset and update gates and a single hidden state (no separate cell). It is a lighter gated recurrent unit than LSTM.

## How does a Bidirectional RNN differ from a unidirectional RNN?

It runs a forward RNN and a backward RNN on the same observed window and concatenates their final states (hidden size effectively doubled at the classifier). Parameter count is therefore higher at the same `hidden_dim`.

## Why is bidirectional processing problematic for strict causal generation?

The backward pass sees tokens to the right of earlier positions *inside the window*. That is not the same as predicting the future from a prefix only. We still withhold the target token, but the architecture is not a causal LM.

## What is teacher forcing?

During training we always feed the ground-truth context window, not the model's previous predictions. Generation, by contrast, feeds sampled tokens.

## What is cross-entropy loss?

For next-token classification, `-log p(true token | context)`, averaged over targets. Implemented as `torch.nn.CrossEntropyLoss`.

## What is perplexity?

`exp(mean cross-entropy)`. Lower means the model assigns higher probability to the true next tokens.

## What is next-word accuracy?

The fraction of targets for which the highest-probability vocabulary item equals the true token.

## What is temperature sampling?

Divide logits by T > 0, softmax, then multinomial sample. T is not used at training time.

## Why can higher temperature produce more diverse but less coherent text?

Larger T flattens the distribution, raising the chance of low-probability tokens. That often increases distinct-n and can break local grammar.

## Why can low temperature lead to repetitive text?

Small T sharpens the distribution toward a few high-probability tokens, which can create loops.

## How can overfitting be detected?

Training loss falling while validation loss rises. We record both curves and select the best validation-loss checkpoint.

## Why are training and validation splits necessary?

Training fits parameters; validation estimates generalization for checkpoint selection without touching the final test numbers.

## Why must the test set remain separate?

To provide an unbiased estimate after all modelling decisions. We evaluate test metrics only after checkpoints are chosen.

## How are model parameters counted?

`sum(p.numel() for p in model.parameters() if p.requires_grad)`.

## Why might one model outperform another?

Gating (LSTM/GRU) can learn longer-window dependencies more stably than a vanilla RNN. Extra bidirectional capacity can help scoring a fully observed window but may not translate into better causal rollouts.

## What are the limitations of Tiny Shakespeare?

It is small, literary, highly stylized, and full of rare names. Word-level vocabularies still leave OOV types. It is not a general modern-English corpus.

## How would this system be improved?

Longer context, more epochs or better regularization, subword tokenization, a causal Transformer, nucleus sampling, and a larger corpus.

## Results from this run (for viva)

Best model by test perplexity: **{best['display_name']}** (PPL {best['test_perplexity']:.4f}).

{rows}
"""
    path = PROJECT_ROOT / "docs" / "viva_questions.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_notebook() -> Path:
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    cells = []

    def md(source: str) -> None:
        cells.append(nbf.v4.new_markdown_cell(source))

    def code(source: str) -> None:
        cells.append(nbf.v4.new_code_cell(source))

    md("# ICT4403 Foundations of Generative AI — Assignment 2\n\nWord-level autoregressive text generation on Tiny Shakespeare using Simple RNN, Bidirectional RNN, LSTM, and GRU.\n\nRun this notebook from the **project root** (the directory that contains `config.yaml`).")
    md("## 1. Assignment objective\n\nPredict the next word after a fixed context window, compare four recurrent architectures under matched training conditions, generate text at three temperatures, and justify the most effective model using held-out test metrics plus generated samples.")
    md("## 2. Dataset description and source\n\nTiny Shakespeare is a public-domain concatenation of Shakespeare plays distributed at:\n\nhttps://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
    code(
        """from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, Markdown, display

from src.utils import load_config, PROJECT_ROOT
from src.data_preprocessing import clean_text, tokenize, load_raw_text, Vocabulary
from src.dataset import NextTokenDataset
from src.models import MODEL_DISPLAY_NAMES

config = load_config()
dataset_meta = json.loads((PROJECT_ROOT / "data/raw/dataset_metadata.json").read_text(encoding="utf-8"))
prep_meta = json.loads((PROJECT_ROOT / "data/processed/preprocess_metadata.json").read_text(encoding="utf-8"))
dataset_meta, {k: prep_meta[k] for k in ["total_word_tokens", "vocabulary_size", "train_tokens", "val_tokens", "test_tokens"]}"""
    )
    md("## 3. Imports and configuration")
    code("""config["model"], config["training"], config["preprocessing"]["sequence_length"]""")
    md("## 4. Text loading")
    code(
        """raw = load_raw_text(config["paths"]["raw_text"])
print("characters:", len(raw))
print(raw[:400])"""
    )
    md("## 5. Preprocessing and word-level tokenization")
    code(
        """cleaned = clean_text(raw)
tokens = tokenize(cleaned)
print(tokens[:30])
print("token count:", len(tokens))"""
    )
    md("## 6. Vocabulary creation")
    code(
        """vocab = Vocabulary.load(PROJECT_ROOT / "data/processed/vocabulary.json")
print("vocab size", len(vocab))
print("specials", vocab.index_to_token[:4])
print("UNK id", vocab.unk_index)"""
    )
    md("## 7. Input–target sequence construction")
    code(
        """ids = vocab.encode(["To", "be", "or", "not", "to", "be"])
demo = NextTokenDataset(ids, sequence_length=5)
x, y = demo[0]
print("input", [vocab.decode_token(i) for i in x.tolist()])
print("target", vocab.decode_token(int(y)))"""
    )
    md("## 8. Dataset split verification")
    code(
        """print(prep_meta["split_strategy"])
print(prep_meta["train_tokens"], prep_meta["val_tokens"], prep_meta["test_tokens"])
assert prep_meta["train_tokens"] + prep_meta["val_tokens"] + prep_meta["test_tokens"] == prep_meta["total_word_tokens"]"""
    )
    md("## 9. Architecture descriptions\n\n- **Simple RNN:** unidirectional tanh RNN; last hidden state is classified.\n- **Bidirectional RNN:** forward and backward RNNs over the *observed* window only; concatenated finals predict the next token. Not strictly causal.\n- **LSTM / GRU:** gated unidirectional recurrences; last-layer hidden state is classified.\n\nImplementations live in `src/models.py`.")
    md("## 10. Training methodology\n\nAdam, cross-entropy, gradient clipping 1.0, five epochs unless `config.yaml` was overridden. Checkpoints are selected by validation loss. Test data is unused until after selection.")
    md("## 11. Training and validation curves")
    code(
        """history = pd.read_csv(PROJECT_ROOT / "results/metrics/training_history.csv")
display(history.head())
display(Image(str(PROJECT_ROOT / "results/figures/training_loss_curves.png")))
display(Image(str(PROJECT_ROOT / "results/figures/validation_loss_curves.png")))"""
    )
    md("## 12. Test-set evaluation")
    code(
        """test = pd.read_csv(PROJECT_ROOT / "results/metrics/test_metrics.csv")
val = pd.read_csv(PROJECT_ROOT / "results/metrics/validation_metrics.csv")
display(val)
display(test)
display(Image(str(PROJECT_ROOT / "results/figures/test_perplexity_comparison.png")))
display(Image(str(PROJECT_ROOT / "results/figures/test_accuracy_comparison.png")))"""
    )
    md("## 13. Model comparison table")
    code(
        """cols = ["model", "test_loss", "test_perplexity", "test_accuracy", "parameter_count", "total_training_time_sec"]
display(test[cols])
display(Image(str(PROJECT_ROOT / "results/figures/parameter_count_comparison.png")))
display(Image(str(PROJECT_ROOT / "results/figures/training_time_comparison.png")))"""
    )
    md("## 14. Text-generation examples")
    code(
        """samples = pd.read_csv(PROJECT_ROOT / "results/generated_text/all_samples.csv")
subset = samples[(samples["seed_phrase"] == "To be, or not to be") & (samples["temperature"] == 1.0)]
for _, row in subset.iterrows():
    display(Markdown(f\"\"\"### {row['display_name']}\\n```text\\n{row['full_text'][:800]}\\n```\"\"\"))"""
    )
    md("## 15. Temperature analysis")
    code(
        """display(Image(str(PROJECT_ROOT / "results/figures/temperature_vs_diversity.png")))
display(Image(str(PROJECT_ROOT / "results/figures/temperature_vs_repetition.png")))
print((PROJECT_ROOT / "results/reports/temperature_analysis.md").read_text(encoding="utf-8")[:1500])"""
    )
    md("## 16. Diversity and repetition metrics")
    code("""pd.read_csv(PROJECT_ROOT / "results/metrics/generation_metrics.csv")""")
    md("## 17. Discussion of the Bidirectional RNN\n\nThe bidirectional model encodes the complete observed window from both directions. That can improve next-token *classification* of a fully known window, but generation still has to hide the future by construction: we never feed the target. It should not be described as a causal language model. See the README section on bidirectional limitations.")
    md("## 18. Limitations\n\nSmall corpus, 20-token windows, 8k vocabulary, five-epoch budget, recurrent compression, OOV names, and possible repetitive decoding.")
    md("## 19. Conclusion")
    code(
        """best = json.loads((PROJECT_ROOT / "results/reports/best_model.json").read_text(encoding="utf-8"))
best"""
    )
    nb["cells"] = cells
    path = ensure_dir("notebooks") / "assignment_analysis.ipynb"
    path.write_text(nbf.writes(nb), encoding="utf-8")
    return path
