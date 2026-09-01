# LCBench Statistical Analysis

Paired unit: LCBench instance and seed (34 instances x 10 seeds = 340 blocks).
Two-sided Wilcoxon signed-rank tests compare HAT with DEHB, ASHA, and BOHB; Holm correction is applied separately per metric.
Rank-biserial signs use HAT minus baseline; positive values mean higher, worse HAT values because both metrics are minimized.

## Pairwise tests

| metric           | comparison   |   n_pairs |   wilcoxon_statistic |     p_value |   median_hat_minus_baseline |   rank_biserial_hat_minus_baseline |   holm_p_value | significant_0_05   |
|:-----------------|:-------------|----------:|---------------------:|------------:|----------------------------:|-----------------------------------:|---------------:|:-------------------|
| final_transformed_objective | HAT vs DEHB  |       340 |                 2729 | 1.72192e-47 |                     2.70704 |                           0.905848 |    5.16575e-47 | True               |
| final_transformed_objective | HAT vs ASHA  |       340 |                   27 | 5.67809e-22 |                     0       |                           0.993143 |    1.13562e-21 | True               |
| final_transformed_objective | HAT vs BOHB  |       340 |                   27 | 5.67809e-22 |                     0       |                           0.993143 |    1.13562e-21 | True               |
| total_cost       | HAT vs DEHB  |       340 |                 9293 | 1.84951e-27 |                -22000       |                          -0.679386 |    1.84951e-27 | True               |
| total_cost       | HAT vs ASHA  |       340 |                    0 | 1.74908e-57 |                -14408.4     |                          -1        |    5.24724e-57 | True               |
| total_cost       | HAT vs BOHB  |       340 |                    0 | 1.74908e-57 |                -14408.4     |                          -1        |    5.24724e-57 | True               |

## Friedman tests

| metric           |   n_blocks |   n_algorithms |   friedman_statistic |      p_value |
|:-----------------|-----------:|---------------:|---------------------:|-------------:|
| final_transformed_objective |        340 |              4 |              603.035 | 2.2156e-130  |
| total_cost       |        340 |              4 |              474.871 | 1.33138e-102 |

## Average ranks

| metric           | algorithm   |   average_rank |
|:-----------------|:------------|---------------:|
| final_transformed_objective | dehb        |        1.30588 |
| final_transformed_objective | asha        |        2.71618 |
| final_transformed_objective | bohb        |        2.71618 |
| final_transformed_objective | hat         |        3.26176 |
| total_cost       | hat         |        1.24706 |
| total_cost       | dehb        |        2.88235 |
| total_cost       | asha        |        2.93529 |
| total_cost       | bohb        |        2.93529 |
