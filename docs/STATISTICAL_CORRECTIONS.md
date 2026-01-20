---
title: Statistical Corrections - Issue #1: Multiple Comparisons Problem
date: January 20, 2026
status: CRITICAL - Publication Blocking
---

# CRITICAL ISSUE #1: Multiple Comparisons Problem

## Executive Summary

Your benchmarking protocol conducts **64 independent statistical tests** (8 datasets × 8 baselines), but reports p-values as if each is a single hypothesis test. This violates fundamental statistical principles and inflates Type I error rates.

**Impact**: 4 claimed significant results (p<0.05) are actually **NOT statistically significant** after proper correction.

**Good News**: Your empirical performance is genuinely strong. We can reframe claims around valid metrics.

---

## 1. The Problem: What Is Multiple Comparisons?

### Scenario
When you perform only 1 hypothesis test with α=0.05:
- Probability of false positive (Type I error) = 5%
- Probability of no false positive = 95%

When you perform **64 independent tests** with α=0.05:
- Expected number of false positives = 0.05 × 64 = **3.2 tests**
- Probability of at least 1 false positive = 1 - (0.95)^64 ≈ **97.4%**

### Your Situation
You report:
- Australian: p=0.0474 vs CheapGreedy ✓ (claimed significant)
- KC1: p=0.0058 vs CheapGreedy ✓ (claimed significant)
- Blood: Multiple p<0.05 vs 7 baselines ✓ (claimed significant)
- Credit_g: p=0.0388 vs CheapGreedy ✓ (claimed significant)

**Problem**: Each test was cherry-picked from 64 possible tests. At 97.4% prior probability of ≥1 false positive, claiming 4 "significant" results is statistically meaningless without correction.

---

## 2. The Solution: Multiple Comparisons Corrections

### Option A: Bonferroni Correction (Most Conservative)

**Formula**: α_corrected = α / k
- α = 0.05 (family-wise error rate)
- k = 64 (number of tests)
- **α_corrected = 0.05 / 64 = 0.000781**

**Application**: Only p-values < 0.000781 are "significant"

**Your Results After Bonferroni**:
- Australian p=0.0474: ❌ NOT significant (exceeds threshold)
- KC1 p=0.0058: ❌ NOT significant (exceeds threshold)
- Blood p=0.0001: ✅ **SURVIVES** (beats threshold!)
- Credit_g p=0.0388: ❌ NOT significant (exceeds threshold)

### Option B: Holm-Bonferroni Correction (RECOMMENDED) ⭐

**Why it's better**: Step-down procedure = less conservative than Bonferroni

**Procedure**:
1. Order p-values from smallest to largest
2. Compare p₁ to α/(k) = 0.000781
3. If p₁ < threshold: significant, continue to p₂
4. Compare p₂ to α/(k-1) = 0.000833
5. Continue until first p-value ≥ threshold (stop)
6. All remaining p-values are not significant

**Your Results After Holm-Bonferroni**:
```
Rank  Dataset             Baseline           p-value    Threshold   Decision
────────────────────────────────────────────────────────────────────────────
 1    blood_transfusion   CheapGreedy        0.000100   0.000781    ✅ SIG
 2    kc1                 CheapGreedy        0.005800   0.000833    ❌ NOT SIG
 3    blood_transfusion   SuccessiveHalving  0.008000   0.000877    ❌ NOT SIG
 4    blood_transfusion   PBT                0.011400   0.000926    ❌ NOT SIG
 5    blood_transfusion   Random             0.011800   0.000980    ❌ NOT SIG
 ...  [remaining tests]   [all fail]         [...all]   [...all]    ❌ NOT SIG
```

**Summary**: 1 result survives (Blood vs CheapGreedy at p=0.0001)

### Option C: False Discovery Rate (FDR) - Benjamini-Hochberg

**What it controls**: Expected proportion of false discoveries among declared positives

**Trade-off**: More permissive than Bonferroni, but acknowledges you'll have some false positives

**For your analysis**: Would likely show 2-3 significant results (exploratory level)

**Not recommended for**: Conservative publication claims

---

## 3. The Data: Your Uncorrected Results

All 10 comparisons with p<0.05 (uncorrected):

```
Rank  Dataset             Baseline              p-value   Bonferroni  Holm-B   Valid?
─────────────────────────────────────────────────────────────────────────────────
 1    blood_transfusion   CheapGreedy          0.000100   ✅ YES      ✅ YES   ✅ YES
 2    kc1                 CheapGreedy          0.005800   ❌ NO       ❌ NO    ❌ NO
 3    blood_transfusion   SuccessiveHalving    0.008000   ❌ NO       ❌ NO    ❌ NO
 4    blood_transfusion   PBT                  0.011400   ❌ NO       ❌ NO    ❌ NO
 5    blood_transfusion   Random               0.011800   ❌ NO       ❌ NO    ❌ NO
 6    blood_transfusion   Hyperband            0.017600   ❌ NO       ❌ NO    ❌ NO
 7    blood_transfusion   Optuna               0.033900   ❌ NO       ❌ NO    ❌ NO
 8    credit_g            CheapGreedy          0.038800   ❌ NO       ❌ NO    ❌ NO
 9    blood_transfusion   EpsilonGreedy        0.045300   ❌ NO       ❌ NO    ❌ NO
10    australian         CheapGreedy          0.047400   ❌ NO       ❌ NO    ❌ NO
```

---

## 4. Implementation: How to Fix Your Code

### A. Add Statistical Corrections to final.py

```python
from scipy.stats import ttest_ind
import numpy as np

def holm_bonferroni_correction(p_values, alpha=0.05):
    """
    Apply Holm-Bonferroni correction to p-values.
    
    Parameters
    ----------
    p_values : array-like
        List of p-values to correct
    alpha : float
        Family-wise error rate
        
    Returns
    -------
    dict with 'significant', 'thresholds', 'adjusted_p'
    """
    p_values = np.asarray(p_values)
    k = len(p_values)
    
    # Sort with original indices
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    # Compute step-down thresholds
    thresholds = alpha / (k - np.arange(k))
    
    # Apply step-down rule
    significant = np.zeros(k, dtype=bool)
    for i, (p, thresh) in enumerate(zip(sorted_p, thresholds)):
        if p < thresh:
            significant[i] = True
        else:
            break  # Stop at first non-significant
    
    # Map back to original order
    result = np.zeros(k, dtype=bool)
    result[sorted_idx[significant]] = True
    
    return {
        'significant': result,
        'thresholds': thresholds,
        'n_significant': np.sum(result)
    }

# Usage in your benchmark function:
all_pvalues = []
for dataset in datasets:
    for baseline in baselines:
        # ... run t-test ...
        p_val = ttest_ind(hagfish_results, baseline_results).pvalue
        all_pvalues.append(p_val)

# Apply correction
correction = holm_bonferroni_correction(all_pvalues, alpha=0.05)
print(f"Uncorrected significant: {np.sum(np.array(all_pvalues) < 0.05)}")
print(f"Holm-Bonferroni significant: {correction['n_significant']}")
```

### B. Update README Claims

**Remove these claims**:
```
❌ "Achieves statistical significance (p<0.05) on 4/8 datasets"
❌ "Significantly outperforms CheapGreedy (p=0.047)"
❌ "Statistically superior on KC1 (p=0.0058)"
```

**Replace with**:
```
✅ "Leads on 6/8 datasets by point estimate (Australian, Car, Phoneme, KC1, Blood, Credit_g)"
✅ "Achieves 11.9% average cost reduction vs Fixed baseline"
✅ "Occupies Pareto frontier position on 6/8 datasets"
✅ "Shows uncorrected p<0.05 on 10 comparisons (see statistical_corrections.py for correction analysis)"
```

---

## 5. The Honest Reframing

Your **empirical performance is genuinely good**. Here's how to frame it for publication:

### BEFORE (Invalid):
> "Hagfish achieves statistical significance (p<0.05) on Australian, KC1, Blood, and Credit_g datasets, demonstrating statistical superiority over multiple baselines."

### AFTER (Valid):
> "Hagfish demonstrates consistent empirical advantages across the benchmark suite:
> 
> **Empirical Metrics** (valid without correction):
> - Leads on 6/8 datasets by accuracy point estimate
> - Achieves 11.9% average cost reduction vs Fixed baseline  
> - Occupies Pareto frontier position on 6/8 datasets
> - On Blood Transfusion, achieves p<0.001 vs CheapGreedy baseline (uncorrected)
> 
> **Statistical Note**: Individual pairwise comparisons yield uncorrected p-values<0.05 on 10 comparisons. However, applying Holm-Bonferroni correction for 64 comparisons (8 datasets × 8 baselines) yields 1 significant result (Blood vs CheapGreedy). We emphasize the empirical performance metrics as primary evidence, while acknowledging the multiple comparisons correction limits strong individual comparison claims."

---

## 6. Publishing Strategy

### What Reviewers Will Notice

A statistically-aware reviewer will check:
1. "How many statistical tests?" → You'll answer: 64
2. "Did you correct for multiple comparisons?" → Currently: No ❌
3. "Why are you claiming p<0.05 validity then?" → ⚠️ Red flag

### What Reviewers Will Appreciate

If you proactively address it:
1. "We conducted 64 comparisons and applied Holm-Bonferroni correction"
2. "After correction, 1 comparison achieves significance (Blood vs CheapGreedy, p=0.0001)"
3. "We emphasize empirical metrics (Pareto, cost reduction) as primary evidence"
4. "Individual uncorrected p-values are provided for exploratory interest"

This shows: **statistical rigor, honesty, sophistication**

---

## 7. When to Use Each Correction

| Situation | Use This | Why |
|-----------|----------|-----|
| Publication, conservative claims | Holm-Bonferroni | Rigorous, less conservative than Bonferroni |
| Publication, exploratory findings | Benjamini-Hochberg (FDR) | Acknowledges false positives, more power |
| Internal use, safety check | Bonferroni | Most conservative, hard to satisfy |
| Existing field standard | Match the field | Domain conventions matter |

**Recommendation for you**: Holm-Bonferroni (good balance of rigor and power)

---

## 8. Action Items

### Immediate (Required):
- [ ] Add `holm_bonferroni_correction()` to final.py
- [ ] Generate corrected p-value table
- [ ] Update README: Remove invalid p-value claims
- [ ] Add new section: "Statistical Methodology & Multiple Comparisons"
- [ ] Commit with message: "Issue #1: Apply Holm-Bonferroni correction"

### Follow-up (Recommended):
- [ ] Create Figure: Uncorrected vs Corrected p-values visualization
- [ ] Add Supplementary Table: All 64 p-values with correction status
- [ ] Update Methods section with correction details
- [ ] Add Honest Limitations section acknowledging the correction

### Code References
File with implementation: `experiments/statistical_corrections.py`
- Class: `MultipleComparisonsCorrection`
- Method: `holm_bonferroni()`
- Analysis: `BenchmarkMultipleComparisonsAnalysis`

---

## 9. Reference Literature

**Key Papers on Multiple Comparisons**:
- Benjamini & Hochberg (1995): "Controlling the false discovery rate" (seminal FDR paper)
- Holm (1979): "A simple sequentially rejective multiple test procedure" (Holm-Bonferroni)
- Bland & Altman (1995): "Multiple significance tests: the Bonferroni method" (clinical applications)

**AutoML/HPO Specific**:
- Most recent AutoML papers report uncorrected p-values (this is actually common!)
- Recommendation: Be first in your field to do it right

---

## 10. FAQ

**Q: Does this invalidate all my results?**
A: No! Empirical metrics (Pareto, cost reduction, accuracy by point estimate) are all valid. Only individual p-value claims need correction.

**Q: What if I only report best p-values?**
A: Still invalid. You're selecting from 64 tests, so correction applies to all.

**Q: Can I just use FDR instead?**
A: Yes, but Holm-Bonferroni is more standard for this scenario.

**Q: Will this hurt my paper?**
A: Proactively addressing it shows rigor. Reviewers will respect the honesty.

**Q: What about the NAS benchmark (5 methods, 1 dataset)?**
A: Only 5 comparisons → less critical. But good practice to correct anyway (α'=0.01).

---

## 11. Summary

| Aspect | Status | Recommendation |
|--------|--------|-----------------|
| Empirical performance | ✅ STRONG | Keep all empirical claims |
| P-value claims (uncorrected) | ❌ INVALID | Remove or contextualize |
| Publication readiness | ⚠️ FIXABLE | Apply Holm-Bonferroni |
| Pareto analysis | ✅ VALID | Highlight as evidence |
| Cost reduction | ✅ VALID | Central to claims |
| Reproducibility | ✅ STRONG | Maintain as-is |

**Bottom Line**: Your work is solid empirically. Fix the statistical claims, and you're publication-ready.

---

## Appendix: Complete Corrected p-Values

See `experiments/statistical_corrections.py` output for complete table of all 64 comparisons.

---

## Implementation: Corrected Statistical Test Code

See `statistical_corrections.py` for:
- Bonferroni correction
- Holm-Bonferroni correction
- FDR correction (exploratory)
- Updated visualization with thresholds
- Honest p-value reporting table

---

## Publication Recommendation

### For Peer Review:

**Section 4.2 - Statistical Analysis:**

```
"We conducted pairwise t-tests comparing Hagfish to 8 baseline methods on 8 datasets, 
resulting in 64 independent comparisons. To control family-wise error rate, we applied 
Holm-Bonferroni correction. 

Uncorrected analyses show promising trends (Australian p=0.047, KC1 p=0.0058, 
Blood p<0.05, Credit_g p=0.0388) but do not remain significant after multiple 
comparisons correction (α_corrected = 0.00078 initial threshold).

We therefore emphasize empirical performance metrics (Pareto frontier analysis, 
accuracy leadership by point estimate) as primary evidence of effectiveness."
```

---

## Key Takeaway

Your package is **empirically strong** on 6/8 datasets. The issue isn't your algorithm—
it's that you tested too many comparisons without correction.

**Solution: Frame results honestly around:**
- Pareto frontier dominance
- Consistent cost reduction
- Accuracy leadership by point estimate
- NOT statistical significance from individual t-tests

This is actually **more compelling** for practitioners: "Our method saves 11.9% cost 
consistently across 8 datasets" > "p-value=0.047"
