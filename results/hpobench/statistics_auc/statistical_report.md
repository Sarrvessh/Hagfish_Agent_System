# Benchmark Statistical Report

Ranking metric: `auc` (lower is better)

Friedman chi-square: `9.000000`
Friedman p-value: `0.0292909`
Interpretation: significant differences detected.

## Global Ranking

|   global_position | algorithm   |   avg_rank |
|------------------:|:------------|-----------:|
|                 1 | sha         |        1.2 |
|                 2 | hyperband   |        2.4 |
|                 3 | hagfish     |        2.8 |
|                 4 | tpe         |        3.6 |

## Pairwise Wilcoxon (Holm corrected)

| algorithm_a   | algorithm_b   |   wilcoxon_stat |   p_value |   p_value_holm | significant_0_05   |
|:--------------|:--------------|----------------:|----------:|---------------:|:-------------------|
| hyperband     | sha           |               0 |    0.0625 |         0.375  | False              |
| hyperband     | tpe           |               0 |    0.0625 |         0.375  | False              |
| sha           | tpe           |               0 |    0.0625 |         0.375  | False              |
| hagfish       | sha           |               2 |    0.1875 |         0.5625 | False              |
| hagfish       | hyperband     |               4 |    0.4375 |         0.875  | False              |
| hagfish       | tpe           |               4 |    0.4375 |         0.875  | False              |