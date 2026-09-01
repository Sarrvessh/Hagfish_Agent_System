# Benchmark Statistical Report

Ranking metric: `final` (lower is better)

Friedman chi-square: `7.200000`
Friedman p-value: `0.0657891`
Interpretation: no significant omnibus difference.

## Global Ranking

|   global_position | algorithm   |   avg_rank |
|------------------:|:------------|-----------:|
|                 1 | hyperband   |        2.1 |
|                 2 | sha         |        2.1 |
|                 3 | tpe         |        2.1 |
|                 4 | hagfish     |        3.7 |

## Pairwise Wilcoxon (Holm corrected)

| algorithm_a   | algorithm_b   |   wilcoxon_stat |   p_value |   p_value_holm | significant_0_05   |
|:--------------|:--------------|----------------:|----------:|---------------:|:-------------------|
| hagfish       | hyperband     |               0 |     0.125 |           0.75 | False              |
| hagfish       | sha           |               0 |     0.125 |           0.75 | False              |
| hagfish       | tpe           |               0 |     0.125 |           0.75 | False              |
| hyperband     | sha           |               7 |     1     |           1    | False              |
| hyperband     | tpe           |               7 |     1     |           1    | False              |
| sha           | tpe           |               7 |     1     |           1    | False              |