#!/usr/bin/env python3
"""
Visual Guide: Multiple Comparisons Problem - Before & After

This script demonstrates the exact impact of the multiple comparisons problem
on your benchmark results, with actual numbers from your experiments.
"""

import numpy as np
from scipy.special import comb

print("=" * 100)
print("MULTIPLE COMPARISONS PROBLEM: Visual Explanation")
print("=" * 100)

print("\n" + "─" * 100)
print("SCENARIO 1: Single Test (Normal Hypothesis Testing)")
print("─" * 100)

print("""
Your Benchmark: 1 comparison (e.g., Hagfish vs Fixed on Australian dataset)

    H₀: Hagfish = Fixed (null hypothesis)
    α = 0.05 (5% false positive rate)
    
    Result: p = 0.0474 (less than 0.05)
    
    Decision: REJECT H₀ → "Significant difference detected" ✓
    
    Risk of false positive: 5%
    
    Conclusion: Safe to claim significance
""")

print("\n" + "─" * 100)
print("SCENARIO 2: YOUR ACTUAL SITUATION (64 comparisons)")
print("─" * 100)

print("""
Your Benchmark: 64 comparisons (8 datasets × 8 baselines)

    Individual p-value thresholds: α = 0.05 for each
    
    IF each test were independent:
    - Prob(no false positives) = (1 - 0.05)^64 = 0.95^64 = 0.0259 (2.59%)
    - Prob(≥1 false positive) = 1 - 0.0259 = 0.9741 (97.41%)
    
    ⚠️  EXPECTED NUMBER OF FALSE POSITIVES: 0.05 × 64 = 3.2 ⚠️
    
    Your Observed: 10 p<0.05 results
    
    Reality Check:
    - If Hagfish = all baselines (null is true everywhere)
    - We'd expect ~3.2 false "significant" results by chance
    - You found 10 "significant" results
    - How many are real? Unknown! (3-5 likely false positives)
""")

print("\n" + "─" * 100)
print("THE CORRECTION: Holm-Bonferroni")
print("─" * 100)

print(f"""
Holm-Bonferroni works by adjusting the threshold for each test:

Test #1 (smallest p-value): Compare to α/64    = 0.05/64 = 0.000781
Test #2 (if #1 rejects):    Compare to α/63    = 0.05/63 = 0.000833
Test #3 (if #2 rejects):    Compare to α/62    = 0.05/62 = 0.000887
...
Test #64 (if all prior reject): Compare to α/1 = 0.05/1  = 0.05000

STEP-DOWN RULE: Stop at first test that fails threshold

Your Results After Applying This:
│
├─ Blood vs CheapGreedy: p = 0.00010 < 0.000781 ✅ SIGNIFICANT
├─ KC1 vs CheapGreedy: p = 0.00580 > 0.000833 ❌ STOPS HERE
└─ [All remaining tests fail without further evaluation]

Final Result: 1 significant (down from 10)
""")

print("\n" + "─" * 100)
print("WHAT THIS MEANS FOR YOU")
print("─" * 100)

data = [
    ("Uncorrected p<0.05", 10, "❌ INVALID for publication"),
    ("Holm-Bonferroni significant", 1, "✅ Valid for publication"),
    ("Datasets with accuracy lead (point estimate)", 6, "✅ Always valid (no p-values)"),
    ("Datasets on Pareto frontier", 6, "✅ Always valid (empirical)"),
    ("Average cost reduction", "11.9%", "✅ Always valid (empirical)"),
]

for claim, value, validity in data:
    print(f"  {claim:<40} {str(value):<20} {validity}")

print("\n" + "─" * 100)
print("IMPLICATIONS FOR YOUR PAPER")
print("─" * 100)

print("""
OLD (INVALID):
  "Hagfish achieves statistical significance on Australian (p=0.047), 
   KC1 (p=0.0058), Blood Transfusion (p<0.05), and Credit_g (p=0.0388) 
   datasets."
  
  Problem: These p-values are not valid without multiple comparisons correction.
  Reviewers will catch this and flag as p-hacking.

NEW (VALID):
  "Hagfish demonstrates consistent empirical advantages:
  - Leads on 6/8 datasets by point estimate (Australian, Car, Phoneme, 
    KC1, Blood, Credit_g)
  - Achieves 11.9% average cost reduction vs Fixed baseline
  - Occupies Pareto frontier on 6/8 datasets
  
  Individual pairwise comparisons show promising uncorrected p-values on 
  4 datasets. After applying Holm-Bonferroni correction for 64 comparisons, 
  one comparison achieves statistical significance (Blood vs CheapGreedy, 
  p<0.001). We emphasize the empirical metrics as the primary evidence of 
  effectiveness."
  
  Why this works: Honest about correction, emphasizes stronger empirical metrics,
  shows statistical sophistication. Reviewers will respect the rigor.
""")

print("\n" + "─" * 100)
print("PROBABILITY MATH EXPLAINED")
print("─" * 100)

print(f"""
With α=0.05 and k tests:

k=1:   P(≥1 false positive) = 1 - (0.95)¹ = 0.050 (5%)
k=2:   P(≥1 false positive) = 1 - (0.95)² = 0.098 (10%)
k=5:   P(≥1 false positive) = 1 - (0.95)⁵ = 0.226 (23%)
k=10:  P(≥1 false positive) = 1 - (0.95)¹⁰ = 0.401 (40%)
k=20:  P(≥1 false positive) = 1 - (0.95)²⁰ = 0.642 (64%)
k=30:  P(≥1 false positive) = 1 - (0.95)³⁰ = 0.787 (79%)
k=50:  P(≥1 false positive) = 1 - (0.95)⁵⁰ = 0.923 (92%)
k=64:  P(≥1 false positive) = 1 - (0.95)⁶⁴ = 0.974 (97.4%) ← YOUR CASE

Your 64 comparisons have a 97.4% prior probability of finding at least
one false positive! This is why correction is critical.
""")

print("\n" + "─" * 100)
print("QUICK REFERENCE: What to Do")
print("─" * 100)

print("""
1. ✅ USE these claims (always valid):
   - "Leads on X/8 datasets by point estimate"
   - "Achieves Y% cost reduction"
   - "On Pareto frontier for Z/8 datasets"
   - "Blood dataset: p<0.001 vs CheapGreedy (Holm-Bonferroni corrected)"

2. ❌ REMOVE these claims (invalid without correction):
   - Any p<0.05 claim from individual comparisons
   - "Statistically superior" without mentioning correction
   - Uncorrected p-values presented as evidence

3. ⚠️  REFRAME these claims (needs context):
   - "Individual comparisons show uncorrected p<0.05 on 10 tests"
   - "After correction, 1 result remains significant"
   - "Empirical metrics provide stronger evidence than p-values"

See STATISTICAL_CORRECTIONS.md for complete guidance.
""")

print("\n" + "=" * 100)
print("IMPLEMENTATION")
print("=" * 100)

print("""
Use experiments/statistical_corrections.py:

    from experiments.statistical_corrections import MultipleComparisonsCorrection
    
    p_values = [all 64 p-values from benchmarks]
    mcc = MultipleComparisonsCorrection(p_values, alpha=0.05)
    holm = mcc.holm_bonferroni()
    
    print(f"Corrected significant: {holm['n_significant']}")

Run it: python experiments/statistical_corrections.py
Output: Complete analysis of all 64 comparisons with correction status
""")

print("\n" + "=" * 100)
print("✅ YOU ARE NOW PUBLICATION-READY FOR ISSUE #1")
print("=" * 100)
