# Experimental analysis

Generated: 2026-10-17T19:08:05.438805+00:00

## 12.1 Objective

This assignment develops a word-level autoregressive text-generation system on Tiny Shakespeare.
Each model is trained to predict the next word after a fixed window of previously observed tokens,
then samples continuations from seed phrases at multiple temperatures.

## 12.2 Dataset

- Name: Tiny Shakespeare
- Source: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- Original file size: 1115394 bytes
- Characters (decoded): 1115394
- Word-level tokens after preprocessing: 253599
- Unique tokens in full corpus: 14297
- Vocabulary size (train-only, including specials): 8000
- Train / validation / test token counts: 202879 / 25359 / 25361
- Split: chronological 80/10/10 on the token stream before windowing.

Preprocessing is intentionally light: control characters are removed, newlines become spaces, case is preserved, and punctuation is retained as separate tokens. Contractions keep internal apostrophes.

## 12.3 Methodology

- Tokenizer: regex word-level pattern matching alphabetic tokens with optional `'suffix`, digit runs, and isolated punctuation.
- Vocabulary: built from training tokens only; remaining types map to `<UNK>`. Special tokens are `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`.
- Sequences: for window length L=20, input is tokens `[i, i+L)` and the target is token `i+L`. Windows never cross split boundaries.
- Architectures: Simple RNN, Bidirectional RNN, LSTM, and GRU with embedding 64 and hidden size 128.
- Training: Adam, learning rate 0.001, batch size 64, 5 epochs, gradient clip 1.0, dropout 0.2, seed 42.
- Device actually used: `cpu`.
- Checkpoint selection uses validation loss only. The test split is evaluated after selection.

The Bidirectional RNN reads the observed input window in both directions. It does not receive the target token. This is not a strictly causal language model; generation still proceeds by feeding previously generated tokens as the next window.

## 12.4 Quantitative results

Validation metrics:

| display_name | val_loss | val_perplexity | val_accuracy | parameter_count | total_training_time_sec |
| --- | --- | --- | --- | --- | --- |
| Simple RNN | 4.9991 | 148.2804 | 0.1495 | 1568832 | 845.2732 |
| Bidirectional RNN | 5.1001 | 164.0356 | 0.1455 | 2617664 | 1359.77 |
| LSTM | 4.8993 | 134.1932 | 0.1551 | 1643328 | 916.2384 |
| GRU | 4.8842 | 132.1839 | 0.1602 | 1618496 | 871.6444 |

Test metrics:

| display_name | test_loss | test_perplexity | test_accuracy | parameter_count | total_training_time_sec |
| --- | --- | --- | --- | --- | --- |
| Simple RNN | 5.0100 | 149.9073 | 0.1419 | 1568832 | 845.2732 |
| Bidirectional RNN | 5.0946 | 163.1450 | 0.1441 | 2617664 | 1359.77 |
| LSTM | 4.9435 | 140.2616 | 0.1446 | 1643328 | 916.2384 |
| GRU | 4.8992 | 134.1788 | 0.1480 | 1618496 | 871.6444 |

## 12.5 Generated-text comparison

Representative samples for the seed `To be, or not to be` at temperature 1.0:

### Simple RNN

```text
To be, or not to be society that is, so much a hall to another; for every never draw Namely and thy bosom: Nay doth it. QUEEN MARGARET: This son, will be gone; almost you hath jewel for you; You tyrant a eye, sick, Or all of a famous, The day his wife in lives to France; I wot he ends, like me to that what an look, Which, the words of it; The petty towards done of you to, you in France in the enemy? Therefore, to my wife; then is the vantage of the shore, Then I am too general for our accusation that slew thee? JULIET: I think the people is your two violent: The child is loved- shining, good who most warlike gentleman
```
### Bidirectional RNN

```text
To be, or not to be so that our WESTMORELAND, and the defence piece another from me every prince. FLORIZEL: Yes, conduct Richard, my lord And a word. He we spear, gentle gracious boy; almost you well, and you must die in the eye of sick. They are never not that, as you, to the bull by my noble shape, he ends the blood of son that Ravenspurgh Duke, In me, the words of your Each gentlemen with him! ruthless shall be, I in his ears made your eyes, He may my wife by my HERMIONE safe, and Mildly a and I saw their general, To the sea of his place That therefore to lose him Where presently, which my bones and discourse!- Polixenes, good who most weakness and
```
### LSTM

```text
To be, or not to be society that told: so much the hall to another? That every prince? FLORIZEL: Doth thy conduct Richard with me before it. QUEEN MARGARET: This is the of my horse. KING RICHARD III: Then he near tyrant, our general sick; Or all of a that villain on't to revenge, He lives to France; if my grief ends, like Bohemia son A son Of look too. HENRY BOLINGBROKE: He's it; come with him! Nurse: We shall you go me in the door of our people? PAULINA: And then, the messenger of the succession- place I am too general, To weep the world of my That I can have him Where presently, violent stone, I'll warrant you, as the lie who most cost and
```
### GRU

```text
To be, or not to be Show that told: so much the hall we shall be seen; never draw Namely in peace, Or Richard with me before it. QUEEN MARGARET: This son, gentle gracious boy; Plantagenet, Never becomes loving like; and in the eye of sick love me, never says that villain on't. RICHARD: Good morrow, by my noble lady, he mercy, like Bohemia son. QUEEN MARGARET: Away me, that I thought it; come with him! Nurse: Good, you well be in the door of a bawd, my wife is a friend. First Watchman: Why, poor I am too general for a very late, since shut to plague him to the people Where presently, violent him, I'll warrant them, as the lie who most cost and
```


Full samples are in `results/generated_text/generated_samples.md`.

## 12.6 Temperature effects

Generation automatic metrics:

| model | temperature | unique_token_ratio | distinct_1 | distinct_2 | repeated_bigram_ratio | average_length |
| --- | --- | --- | --- | --- | --- | --- |
| bidirectional_rnn | 0.5000 | 0.3689 | 0.3689 | 0.7248 | 0.3893 | 150.0000 |
| bidirectional_rnn | 1.0000 | 0.6867 | 0.6867 | 0.9776 | 0.0425 | 150.0000 |
| bidirectional_rnn | 1.5000 | 0.8356 | 0.8356 | 1.0000 | 0.0000 | 150.0000 |
| gru | 0.5000 | 0.4422 | 0.4422 | 0.7808 | 0.3333 | 150.0000 |
| gru | 1.0000 | 0.6778 | 0.6778 | 0.9709 | 0.0559 | 150.0000 |
| gru | 1.5000 | 0.8467 | 0.8467 | 1.0000 | 0.0000 | 150.0000 |
| lstm | 0.5000 | 0.4200 | 0.4200 | 0.7718 | 0.3490 | 150.0000 |
| lstm | 1.0000 | 0.7156 | 0.7156 | 0.9955 | 0.0089 | 150.0000 |
| lstm | 1.5000 | 0.8356 | 0.8356 | 1.0000 | 0.0000 | 150.0000 |
| simple_rnn | 0.5000 | 0.4667 | 0.4667 | 0.8076 | 0.2998 | 150.0000 |
| simple_rnn | 1.0000 | 0.6711 | 0.6711 | 0.9843 | 0.0291 | 150.0000 |
| simple_rnn | 1.5000 | 0.8622 | 0.8622 | 1.0000 | 0.0000 | 150.0000 |

Lower temperature is expected to concentrate mass on high-probability tokens (often more repetitive).
Higher temperature is expected to flatten the distribution (often more diverse and less coherent).
These are hypotheses tested against the table above and the saved samples, not guaranteed outcomes.

## 12.7 Architecture comparison

Simple RNN updates a single hidden state with a tanh recurrence and is most exposed to vanishing gradients on long-range cues, even with a short window of 20 tokens.
LSTM adds input/forget/output gates and a cell state. GRU uses an update/reset gating design with fewer tensors than LSTM.
The Bidirectional RNN concatenates forward and backward final states, so it has a larger parameter count at the same hidden size. Extra capacity can improve next-token scoring on a complete observed window, but it does not correspond to a causal generator that can be rolled out without re-encoding the window.

## 12.8 Best-performing model

Selected model: **GRU**

- Test perplexity: 134.1788
- Test loss: 4.8992
- Test accuracy: 0.1480

Selection criteria: Primary: lowest test perplexity (equivalently lowest test cross-entropy) on the held-out chronological test split. Secondary: test next-word accuracy and generated-text diversity/repetition. Training loss was not used for selection.

## 12.9 Limitations

- Tiny Shakespeare is a small literary corpus; many rare word types become `<UNK>`.
- Context is limited to 20 tokens.
- Recurrent models compress history into a fixed-size state.
- Five epochs and a modest hidden size are a comparable experimental budget, not a tuned optimum.
- Generated text can repeat or drift out of character.
- Bidirectional encoding of the input window is not equivalent to causal language modelling.

## 12.10 Conclusion

Under the recorded hyperparameters and the chronological test split, **GRU** is the most effective of the four models for this next-word prediction setup, using test perplexity as the primary quantitative criterion and inspecting generated samples for qualitative support.
