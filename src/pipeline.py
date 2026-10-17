"""End-to-end preprocess, train, evaluate, generate, and report pipeline."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from download_data import download_dataset
from src.data_preprocessing import Vocabulary, load_encoded_ids, preprocess
from src.dataset import make_dataloader
from src.generation import generate_grid
from src.metrics import evaluate_loader, generation_statistics
from src.models import MODEL_DISPLAY_NAMES, MODEL_REGISTRY, build_model
from src.trainer import build_criterion, checkpoint_compatible, load_checkpoint, train_model
from src.utils import PROJECT_ROOT, ensure_dir, get_device, load_config, save_json, set_seed
from src.visualization import generate_all_figures


def _checkpoint_path(config: dict, architecture: str) -> Path:
    return PROJECT_ROOT / config["paths"]["checkpoints_dir"] / f"{architecture}_best.pt"


def _actual_hyperparameters(config: dict, vocab_size: int, device: torch.device) -> dict:
    return {
        "embedding_dim": int(config["model"]["embedding_dim"]),
        "hidden_dim": int(config["model"]["hidden_dim"]),
        "num_layers": int(config["model"].get("num_layers", 1)),
        "dropout": float(config["model"]["dropout"]),
        "vocabulary_limit": int(config["preprocessing"]["max_vocab_size"]),
        "vocabulary_size_actual": vocab_size,
        "sequence_length": int(config["preprocessing"]["sequence_length"]),
        "window_stride": int(config["preprocessing"].get("window_stride", 1)),
        "batch_size": int(config["training"]["batch_size"]),
        "epochs": int(config["training"]["epochs"]),
        "learning_rate": float(config["training"]["learning_rate"]),
        "optimizer": config["training"]["optimizer"],
        "loss": config["training"]["loss"],
        "gradient_clip_norm": float(config["training"]["gradient_clip_norm"]),
        "random_seed": int(config["project"]["random_seed"]),
        "device": str(device),
    }


def run_preprocess(config: dict) -> dict:
    download_dataset(config)
    metadata = preprocess(config)
    print(f"Preprocessed. vocabulary_size={metadata['vocabulary_size']} tokens={metadata['total_word_tokens']}")
    return metadata


def _load_vocab(config: dict) -> Vocabulary:
    return Vocabulary.load(PROJECT_ROOT / config["paths"]["processed_dir"] / "vocabulary.json")


def run_train(config: dict, force: bool = False) -> list[dict]:
    device = get_device()
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    vocab = _load_vocab(config)
    seq_len = int(config["preprocessing"]["sequence_length"])
    stride = int(config["preprocessing"].get("window_stride", 1))
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 0))
    train_ids = load_encoded_ids("train")
    val_ids = load_encoded_ids("val")
    train_loader = make_dataloader(train_ids, seq_len, batch_size, stride, shuffle=True, num_workers=num_workers)
    val_loader = make_dataloader(val_ids, seq_len, batch_size, stride, shuffle=False, num_workers=num_workers)
    summaries = []
    ensure_dir(config["paths"]["checkpoints_dir"])
    ensure_dir(config["paths"]["metrics_dir"])

    for architecture in MODEL_REGISTRY:
        ckpt = _checkpoint_path(config, architecture)
        reuse = False
        if ckpt.exists() and not force and not config["training"].get("force_retrain"):
            try:
                existing = load_checkpoint(ckpt, map_location="cpu")
                reuse = checkpoint_compatible(existing, architecture, len(vocab), seq_len)
            except Exception:
                reuse = False
        if reuse:
            summary_path = PROJECT_ROOT / config["paths"]["metrics_dir"] / f"{architecture}_train_summary.json"
            if summary_path.exists():
                saved = json.loads(summary_path.read_text(encoding="utf-8"))
                saved["reused_checkpoint"] = True
                print(
                    f"Skipping {architecture}: compatible checkpoint and training summary already exist",
                    flush=True,
                )
                summaries.append(saved)
                continue
            print(f"Skipping {architecture}: compatible checkpoint already exists at {ckpt}", flush=True)
            summaries.append(
                {
                    "architecture": architecture,
                    "reused_checkpoint": True,
                    "checkpoint_path": str(ckpt),
                    "best_val_loss": existing.get("best_val_loss"),
                    "best_epoch": existing.get("epoch"),
                    "parameter_count": (existing.get("hyperparameters") or {}).get("parameter_count"),
                    "device": str(device),
                    "history": [],
                }
            )
            continue

        print(f"Training {architecture} on {device} ...")
        model = build_model(
            architecture,
            vocab_size=len(vocab),
            embedding_dim=int(config["model"]["embedding_dim"]),
            hidden_dim=int(config["model"]["hidden_dim"]),
            pad_index=vocab.pad_index,
            dropout=float(config["model"]["dropout"]),
            num_layers=int(config["model"].get("num_layers", 1)),
        )
        result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=int(config["training"]["epochs"]),
            learning_rate=float(config["training"]["learning_rate"]),
            clip_norm=float(config["training"]["gradient_clip_norm"]),
            device=device,
            checkpoint_path=ckpt,
            architecture=architecture,
            vocab_size=len(vocab),
            sequence_length=seq_len,
            pad_index=vocab.pad_index,
            random_seed=int(config["project"]["random_seed"]),
            ignore_pad=bool(config["evaluation"].get("ignore_pad", True)),
            log_every=int(config["training"].get("log_every", 200)),
        )
        result["reused_checkpoint"] = False
        result["display_name"] = MODEL_DISPLAY_NAMES[architecture]
        summaries.append(result)
        save_json(PROJECT_ROOT / config["paths"]["metrics_dir"] / f"{architecture}_train_summary.json", result)

    save_json(PROJECT_ROOT / config["paths"]["metrics_dir"] / "train_run_summaries.json", summaries)
    return summaries


def _history_rows(summaries: list[dict]) -> list[dict]:
    rows = []
    for summary in summaries:
        architecture = summary["architecture"]
        for record in summary.get("history") or []:
            rows.append({"model": architecture, **record})
    return rows


def run_evaluate(config: dict) -> pd.DataFrame:
    device = get_device()
    vocab = _load_vocab(config)
    seq_len = int(config["preprocessing"]["sequence_length"])
    stride = int(config["preprocessing"].get("window_stride", 1))
    batch_size = int(config["training"]["batch_size"])
    val_ids = load_encoded_ids("val")
    test_ids = load_encoded_ids("test")
    val_loader = make_dataloader(val_ids, seq_len, batch_size, stride, shuffle=False)
    test_loader = make_dataloader(test_ids, seq_len, batch_size, stride, shuffle=False)
    criterion = build_criterion(vocab.pad_index, ignore_pad=bool(config["evaluation"].get("ignore_pad", True)))

    metrics_dir = PROJECT_ROOT / config["paths"]["metrics_dir"]
    summaries_path = metrics_dir / "train_run_summaries.json"
    train_summaries = json.loads(summaries_path.read_text(encoding="utf-8")) if summaries_path.exists() else []
    train_by_arch = {item["architecture"]: item for item in train_summaries}
    for architecture in MODEL_REGISTRY:
        per_model = metrics_dir / f"{architecture}_train_summary.json"
        if architecture not in train_by_arch and per_model.exists():
            train_by_arch[architecture] = json.loads(per_model.read_text(encoding="utf-8"))
        elif architecture in train_by_arch and not train_by_arch[architecture].get("history") and per_model.exists():
            saved = json.loads(per_model.read_text(encoding="utf-8"))
            train_by_arch[architecture] = {**train_by_arch[architecture], **saved}

    val_rows = []
    test_rows = []
    history_rows = []
    model_summary = {}

    for architecture in MODEL_REGISTRY:
        ckpt_path = _checkpoint_path(config, architecture)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {architecture}: {ckpt_path}")
        checkpoint = load_checkpoint(ckpt_path, map_location=device)
        if not checkpoint_compatible(checkpoint, architecture, len(vocab), seq_len):
            raise ValueError(f"Checkpoint incompatible with current vocabulary/config: {ckpt_path}")
        model = build_model(
            architecture,
            vocab_size=len(vocab),
            embedding_dim=int(config["model"]["embedding_dim"]),
            hidden_dim=int(config["model"]["hidden_dim"]),
            pad_index=vocab.pad_index,
            dropout=float(config["model"]["dropout"]),
            num_layers=int(config["model"].get("num_layers", 1)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        val_stats = evaluate_loader(model, val_loader, criterion, device)
        test_stats = evaluate_loader(model, test_loader, criterion, device)
        train_info = train_by_arch.get(architecture, {})
        row_common = {
            "model": architecture,
            "display_name": MODEL_DISPLAY_NAMES[architecture],
            "parameter_count": int(model.parameter_count()),
            "best_epoch": checkpoint.get("epoch"),
            "best_val_loss_checkpoint": checkpoint.get("best_val_loss"),
            "total_training_time_sec": train_info.get("total_training_time_sec"),
            "epochs_completed": train_info.get("epochs_completed", config["training"]["epochs"]),
            "device": train_info.get("device", str(device)),
            "reused_checkpoint": train_info.get("reused_checkpoint", False),
        }
        val_rows.append({**row_common, "val_loss": val_stats["loss"], "val_perplexity": val_stats["perplexity"], "val_accuracy": val_stats["accuracy"]})
        test_rows.append(
            {
                **row_common,
                "test_loss": test_stats["loss"],
                "test_perplexity": test_stats["perplexity"],
                "test_accuracy": test_stats["accuracy"],
            }
        )
        history_rows.extend({"model": architecture, **record} for record in train_info.get("history") or [])
        model_summary[architecture] = {
            **row_common,
            "validation": val_stats,
            "test": test_stats,
            "hyperparameters": _actual_hyperparameters(config, len(vocab), device),
        }
        print(
            f"{architecture}: test_loss={test_stats['loss']:.4f} "
            f"ppl={test_stats['perplexity']:.2f} acc={test_stats['accuracy']:.4f}"
        )

    metrics_dir = ensure_dir(config["paths"]["metrics_dir"])
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    val_df.to_csv(metrics_dir / "validation_metrics.csv", index=False)
    test_df.to_csv(metrics_dir / "test_metrics.csv", index=False)
    if history_rows:
        pd.DataFrame(history_rows).to_csv(metrics_dir / "training_history.csv", index=False)
    save_json(metrics_dir / "model_summary.json", model_summary)
    save_json(metrics_dir / "actual_hyperparameters.json", _actual_hyperparameters(config, len(vocab), device))
    return test_df


def run_generate(config: dict) -> list[dict]:
    device = get_device()
    vocab = _load_vocab(config)
    seq_len = int(config["preprocessing"]["sequence_length"])
    gen_cfg = config["generation"]
    out_dir = ensure_dir(config["paths"]["generated_dir"])
    all_samples = []
    grouped_tokens: dict[tuple[str, float], list[list[str]]] = defaultdict(list)

    for architecture in MODEL_REGISTRY:
        ckpt_path = _checkpoint_path(config, architecture)
        checkpoint = load_checkpoint(ckpt_path, map_location=device)
        model = build_model(
            architecture,
            vocab_size=len(vocab),
            embedding_dim=int(config["model"]["embedding_dim"]),
            hidden_dim=int(config["model"]["hidden_dim"]),
            pad_index=vocab.pad_index,
            dropout=float(config["model"]["dropout"]),
            num_layers=int(config["model"].get("num_layers", 1)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        samples = generate_grid(
            model=model,
            vocab=vocab,
            seeds=gen_cfg["seeds"],
            temperatures=gen_cfg["temperatures"],
            max_new_tokens=int(gen_cfg["max_new_tokens"]),
            sequence_length=seq_len,
            device=device,
            sampling_seed=int(gen_cfg["sampling_seed"]),
            model_name=architecture,
        )
        for sample in samples:
            grouped_tokens[(architecture, sample["temperature"])].append(sample["generated_tokens"])
            all_samples.append(sample)
        print(f"Generated {len(samples)} samples for {architecture}")

    serializable = []
    for sample in all_samples:
        serializable.append(
            {
                "model": sample["model"],
                "display_name": MODEL_DISPLAY_NAMES[sample["model"]],
                "seed_phrase": sample["seed_phrase"],
                "temperature": sample["temperature"],
                "random_seed": sample["sampling_seed"],
                "generated_token_count": sample["generated_token_count"],
                "stopped_on_eos": sample["stopped_on_eos"],
                "generated_text": sample["generated_text"],
                "full_text": sample["full_text"],
            }
        )
    save_json(out_dir / "all_samples.json", serializable)
    pd.DataFrame(serializable).to_csv(out_dir / "all_samples.csv", index=False)

    metric_rows = []
    for (architecture, temperature), sequences in grouped_tokens.items():
        stats = generation_statistics(sequences)
        metric_rows.append({"model": architecture, "temperature": temperature, **stats})
    metrics_df = pd.DataFrame(metric_rows).sort_values(["model", "temperature"])
    metrics_df.to_csv(PROJECT_ROOT / config["paths"]["metrics_dir"] / "generation_metrics.csv", index=False)

    md_lines = ["# Generated text samples", "", "All samples below were produced by temperature sampling, not argmax decoding.", ""]
    for sample in serializable:
        md_lines.extend(
            [
                f"## {sample['display_name']} | seed=`{sample['seed_phrase']}` | T={sample['temperature']}",
                "",
                f"- Random seed: `{sample['random_seed']}`",
                f"- Generated token count: `{sample['generated_token_count']}`",
                "",
                "```text",
                sample["full_text"],
                "```",
                "",
            ]
        )
    (out_dir / "generated_samples.md").write_text("\n".join(md_lines), encoding="utf-8")
    save_json(out_dir / "generation_config.json", gen_cfg)
    return serializable


def _fmt(value, digits=4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _select_best_model(test_df: pd.DataFrame) -> dict:
    ranked = test_df.sort_values(["test_perplexity", "test_loss"])
    best = ranked.iloc[0]
    return {
        "model": best["model"],
        "display_name": best["display_name"],
        "criteria": (
            "Primary: lowest test perplexity (equivalently lowest test cross-entropy) "
            "on the held-out chronological test split. Secondary: test next-word accuracy "
            "and generated-text diversity/repetition. Training loss was not used for selection."
        ),
        "test_perplexity": float(best["test_perplexity"]),
        "test_loss": float(best["test_loss"]),
        "test_accuracy": float(best["test_accuracy"]),
    }


def write_reports(config: dict) -> None:
    reports_dir = ensure_dir(config["paths"]["reports_dir"])
    metrics_dir = PROJECT_ROOT / config["paths"]["metrics_dir"]
    preprocess_meta = json.loads((PROJECT_ROOT / config["paths"]["processed_dir"] / "preprocess_metadata.json").read_text(encoding="utf-8"))
    dataset_meta = json.loads((PROJECT_ROOT / "data" / "raw" / "dataset_metadata.json").read_text(encoding="utf-8"))
    test_df = pd.read_csv(metrics_dir / "test_metrics.csv")
    val_df = pd.read_csv(metrics_dir / "validation_metrics.csv")
    gen_metrics = pd.read_csv(metrics_dir / "generation_metrics.csv")
    samples = json.loads((PROJECT_ROOT / config["paths"]["generated_dir"] / "all_samples.json").read_text(encoding="utf-8"))
    hyper = json.loads((metrics_dir / "actual_hyperparameters.json").read_text(encoding="utf-8"))
    best = _select_best_model(test_df)

    def table(df: pd.DataFrame, columns: list[str]) -> str:
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join("---" for _ in columns) + " |"
        lines = [header, sep]
        for _, row in df.iterrows():
            cells = []
            for col in columns:
                value = row[col]
                if isinstance(value, float):
                    cells.append(f"{value:.4f}" if abs(value) < 1000 else f"{value:.2f}")
                else:
                    cells.append(str(value))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    display_test = test_df.copy()
    display_test["display_name"] = display_test["model"].map(MODEL_DISPLAY_NAMES)

    sample_blocks = []
    for sample in samples:
        if sample["seed_phrase"] == "To be, or not to be" and sample["temperature"] == 1.0:
            sample_blocks.append(
                f"### {sample['display_name']}\n\n```text\n{sample['full_text']}\n```\n"
            )

    experimental = f"""# Experimental analysis

Generated: {datetime.now(timezone.utc).isoformat()}

## 12.1 Objective

This assignment develops a word-level autoregressive text-generation system on Tiny Shakespeare.
Each model is trained to predict the next word after a fixed window of previously observed tokens,
then samples continuations from seed phrases at multiple temperatures.

## 12.2 Dataset

- Name: {dataset_meta.get('dataset_name')}
- Source: {dataset_meta.get('source_url')}
- Original file size: {dataset_meta.get('bytes')} bytes
- Characters (decoded): {dataset_meta.get('characters')}
- Word-level tokens after preprocessing: {preprocess_meta['total_word_tokens']}
- Unique tokens in full corpus: {preprocess_meta['unique_tokens_full_corpus']}
- Vocabulary size (train-only, including specials): {preprocess_meta['vocabulary_size']}
- Train / validation / test token counts: {preprocess_meta['train_tokens']} / {preprocess_meta['val_tokens']} / {preprocess_meta['test_tokens']}
- Split: chronological 80/10/10 on the token stream before windowing.

Preprocessing is intentionally light: control characters are removed, newlines become spaces, case is preserved, and punctuation is retained as separate tokens. Contractions keep internal apostrophes.

## 12.3 Methodology

- Tokenizer: regex word-level pattern matching alphabetic tokens with optional `'suffix`, digit runs, and isolated punctuation.
- Vocabulary: built from training tokens only; remaining types map to `<UNK>`. Special tokens are `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`.
- Sequences: for window length L={hyper['sequence_length']}, input is tokens `[i, i+L)` and the target is token `i+L`. Windows never cross split boundaries.
- Architectures: Simple RNN, Bidirectional RNN, LSTM, and GRU with embedding {hyper['embedding_dim']} and hidden size {hyper['hidden_dim']}.
- Training: Adam, learning rate {hyper['learning_rate']}, batch size {hyper['batch_size']}, {hyper['epochs']} epochs, gradient clip {hyper['gradient_clip_norm']}, dropout {hyper['dropout']}, seed {hyper['random_seed']}.
- Device actually used: `{hyper['device']}`.
- Checkpoint selection uses validation loss only. The test split is evaluated after selection.

The Bidirectional RNN reads the observed input window in both directions. It does not receive the target token. This is not a strictly causal language model; generation still proceeds by feeding previously generated tokens as the next window.

## 12.4 Quantitative results

Validation metrics:

{table(val_df, ['display_name', 'val_loss', 'val_perplexity', 'val_accuracy', 'parameter_count', 'total_training_time_sec'])}

Test metrics:

{table(display_test, ['display_name', 'test_loss', 'test_perplexity', 'test_accuracy', 'parameter_count', 'total_training_time_sec'])}

## 12.5 Generated-text comparison

Representative samples for the seed `To be, or not to be` at temperature 1.0:

{''.join(sample_blocks)}

Full samples are in `results/generated_text/generated_samples.md`.

## 12.6 Temperature effects

Generation automatic metrics:

{table(gen_metrics, ['model', 'temperature', 'unique_token_ratio', 'distinct_1', 'distinct_2', 'repeated_bigram_ratio', 'average_length'])}

Lower temperature is expected to concentrate mass on high-probability tokens (often more repetitive).
Higher temperature is expected to flatten the distribution (often more diverse and less coherent).
These are hypotheses tested against the table above and the saved samples, not guaranteed outcomes.

## 12.7 Architecture comparison

Simple RNN updates a single hidden state with a tanh recurrence and is most exposed to vanishing gradients on long-range cues, even with a short window of {hyper['sequence_length']} tokens.
LSTM adds input/forget/output gates and a cell state. GRU uses an update/reset gating design with fewer tensors than LSTM.
The Bidirectional RNN concatenates forward and backward final states, so it has a larger parameter count at the same hidden size. Extra capacity can improve next-token scoring on a complete observed window, but it does not correspond to a causal generator that can be rolled out without re-encoding the window.

## 12.8 Best-performing model

Selected model: **{best['display_name']}**

- Test perplexity: {best['test_perplexity']:.4f}
- Test loss: {best['test_loss']:.4f}
- Test accuracy: {best['test_accuracy']:.4f}

Selection criteria: {best['criteria']}

## 12.9 Limitations

- Tiny Shakespeare is a small literary corpus; many rare word types become `<UNK>`.
- Context is limited to {hyper['sequence_length']} tokens.
- Recurrent models compress history into a fixed-size state.
- Five epochs and a modest hidden size are a comparable experimental budget, not a tuned optimum.
- Generated text can repeat or drift out of character.
- Bidirectional encoding of the input window is not equivalent to causal language modelling.

## 12.10 Conclusion

Under the recorded hyperparameters and the chronological test split, **{best['display_name']}** is the most effective of the four models for this next-word prediction setup, using test perplexity as the primary quantitative criterion and inspecting generated samples for qualitative support.
"""
    (reports_dir / "experimental_analysis.md").write_text(experimental, encoding="utf-8")

    temp_report = f"""# Temperature analysis

Temperatures evaluated: 0.5, 1.0, 1.5.

Implementation: `scaled_logits = logits / temperature`, then softmax and multinomial sampling.
Argmax decoding was not used.

## Automatic metrics

{table(gen_metrics, ['model', 'temperature', 'unique_token_ratio', 'distinct_1', 'distinct_2', 'repeated_bigram_ratio'])}

## Metric formulas

- Unique-token ratio: `|set(tokens)| / |tokens|`
- Distinct-1: unique unigrams / unigram count (equals unique-token ratio)
- Distinct-2: unique bigrams / bigram count
- Repeated bigram ratio: instances belonging to bigrams with count > 1, divided by the number of bigrams

These measures do not prove coherence. High distinct-n can still be ungrammatical.

## Qualitative rubric (automated, subjective scale 1–5)

This rubric is applied by heuristic inspection of the saved samples, not by a human rater panel.

1. Coherence: local grammatical plausibility.
2. Contextual consistency: whether the continuation remains related to the seed speaker/theme.
3. Diversity: variety of word types versus looping.
4. Repetition: presence of copied phrases or token cycles.
5. Shakespeare-style resemblance: archaic pronouns, speaker labels, verse-like diction.

See `results/generated_text/generated_samples.md` for the evidence.

## Observed temperature behaviour

Compare distinct-1 and repeated-bigram ratio across 0.5, 1.0, and 1.5 for each architecture in the table.
If distinct-1 rises and repeated-bigram ratio falls as temperature increases, the run agrees with the usual sampling hypothesis.
If a model loops even at high temperature, architecture/training quality is dominating temperature.

## Conclusion

Temperature is a decoding control, not a training hyperparameter. It reshapes an already learned next-token distribution. The quantitative table and generated samples should be read together: a diverse sample can still be incoherent, and a repetitive sample can still look Shakespearean.
"""
    (reports_dir / "temperature_analysis.md").write_text(temp_report, encoding="utf-8")

    save_json(reports_dir / "best_model.json", best)
    from src.documentation import write_notebook, write_readme, write_viva

    write_readme(config)
    write_viva(config)
    write_notebook()


def run_visualize(config: dict) -> list[str]:
    metrics_dir = PROJECT_ROOT / config["paths"]["metrics_dir"]
    return generate_all_figures(
        history_csv=metrics_dir / "training_history.csv",
        test_metrics_csv=metrics_dir / "test_metrics.csv",
        generation_metrics_csv=metrics_dir / "generation_metrics.csv",
        figures_dir=config["paths"]["figures_dir"],
    )


def run_all(config: dict, force_retrain: bool = False) -> None:
    run_preprocess(config)
    run_train(config, force=force_retrain)
    run_evaluate(config)
    run_generate(config)
    run_visualize(config)
    write_reports(config)
