# LCBench Statistical Analysis

Paired unit: LCBench instance and seed (34 instances x 10 seeds = 340 blocks).
Two-sided Wilcoxon signed-rank tests compare HAT with the archived DEHB-style baseline and corrected ASHA; Holm correction is applied separately per metric.
Rank-biserial signs use HAT minus baseline; positive values mean higher, worse HAT values because both metrics are minimized.

## Pairwise tests

| metric           | comparison        |   n_pairs |   wilcoxon_statistic |     p_value |   median_hat_minus_baseline |   rank_biserial_hat_minus_baseline |   holm_p_value | significant_0_05   |
|:-----------------|:------------------|----------:|---------------------:|------------:|----------------------------:|-----------------------------------:|---------------:|:-------------------|
| final_best_error | HAT vs DEHB_STYLE |       340 |                 2729 | 1.72192e-47 |                    2.70704  |                           0.905848 |    3.44384e-47 | True               |
| final_best_error | HAT vs ASHA       |       340 |                13962 | 1.20411e-16 |                   -0.951382 |                          -0.518303 |    1.20411e-16 | True               |
| total_cost       | HAT vs DEHB_STYLE |       340 |                 9293 | 1.84951e-27 |               -22000        |                          -0.679386 |    1.84951e-27 | True               |
| total_cost       | HAT vs ASHA       |       340 |                    0 | 1.74908e-57 |                52401.9      |                           1        |    3.49816e-57 | True               |

## Friedman tests

| metric           |   n_blocks |   n_algorithms |   friedman_statistic |      p_value |
|:-----------------|-----------:|---------------:|---------------------:|-------------:|
| final_best_error |        340 |              3 |              412.947 | 2.13641e-90  |
| total_cost       |        340 |              3 |              553.506 | 6.42281e-121 |

## Average ranks

| metric           | algorithm   |   average_rank |
|:-----------------|:------------|---------------:|
| final_best_error | dehb_style  |        1.12941 |
| final_best_error | hat         |        2.23824 |
| final_best_error | asha        |        2.63235 |
| total_cost       | asha        |        1       |
| total_cost       | hat         |        2.24706 |
| total_cost       | dehb_style  |        2.75294 |
