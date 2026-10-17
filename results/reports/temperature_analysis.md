# Temperature analysis

Temperatures evaluated: 0.5, 1.0, 1.5.

Implementation: `scaled_logits = logits / temperature`, then softmax and multinomial sampling.
Argmax decoding was not used.

## Automatic metrics

| model | temperature | unique_token_ratio | distinct_1 | distinct_2 | repeated_bigram_ratio |
| --- | --- | --- | --- | --- | --- |
| bidirectional_rnn | 0.5000 | 0.3689 | 0.3689 | 0.7248 | 0.3893 |
| bidirectional_rnn | 1.0000 | 0.6867 | 0.6867 | 0.9776 | 0.0425 |
| bidirectional_rnn | 1.5000 | 0.8356 | 0.8356 | 1.0000 | 0.0000 |
| gru | 0.5000 | 0.4422 | 0.4422 | 0.7808 | 0.3333 |
| gru | 1.0000 | 0.6778 | 0.6778 | 0.9709 | 0.0559 |
| gru | 1.5000 | 0.8467 | 0.8467 | 1.0000 | 0.0000 |
| lstm | 0.5000 | 0.4200 | 0.4200 | 0.7718 | 0.3490 |
| lstm | 1.0000 | 0.7156 | 0.7156 | 0.9955 | 0.0089 |
| lstm | 1.5000 | 0.8356 | 0.8356 | 1.0000 | 0.0000 |
| simple_rnn | 0.5000 | 0.4667 | 0.4667 | 0.8076 | 0.2998 |
| simple_rnn | 1.0000 | 0.6711 | 0.6711 | 0.9843 | 0.0291 |
| simple_rnn | 1.5000 | 0.8622 | 0.8622 | 1.0000 | 0.0000 |

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
