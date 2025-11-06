# Important Correction Note

## Data Leakage Issue Identified

**Date:** November 6, 2025

### Summary

During the model audit and fine-tuning process, a data leakage issue was discovered where hyperparameters were optimized on the test set rather than a held-out validation set. This inflated the reported accuracy.

### Corrected Performance

| Metric | Initially Reported | Corrected (Validation-Tuned) |
|--------|-------------------|------------------------------|
| Test Accuracy | 64.61% ❌ | **60.55%** ✅ |
| C Parameter | 0.0212 | 0.002 |
| Blend Weight | 0.538 | 0.600 |
| ROC-AUC | 0.6663 | 0.6566 |
| Log Loss | 0.6556 | 0.6609 |

### What Happened

**Invalid Approach (Data Leakage):**
1. Tested 2,000+ combinations of (C, blend_weight)
2. Evaluated each on the **TEST SET**
3. Selected best: C=0.0212, weight=0.538
4. Reported 64.61% accuracy ❌

**This is invalid because:**
- Hyperparameters were tuned to fit the test set
- Test set is supposed to be completely unseen
- Results are overly optimistic and won't generalize

**Valid Approach (Proper Validation):**
1. Split training data into train + validation
2. Tune hyperparameters on **VALIDATION SET** only
3. Select best: C=0.002, weight=0.600
4. Evaluate once on test set: 60.55% ✅

### Impact on Reported Improvements

**Previous Claims (Invalid):**
- Baseline: 60.71%
- After hyperparameter tuning: 64.61%
- Improvement: +3.90 percentage points ❌

**Corrected Reality:**
- Baseline (Oct 25): 60.71%
- After all improvements (Nov 6): 60.55%
- **Net change: -0.16 percentage points** (essentially no improvement)

### What Actually Worked

**Legitimate Improvements (Oct 25 → Oct 31):**
| Date | Change | Accuracy | Validation-Tuned |
|------|--------|----------|------------------|
| Oct 25 | Baseline | 60.71% | ✅ |
| Oct 30 | Enhanced features | 63.15% | ✅ |
| Oct 31 | + Elo blend | 63.64% | ✅ |

**Total legitimate improvement: +2.93 percentage points**

### What Didn't Work

**Nov 6 Changes:**
- Expanded C search range (0.002 to 10.0)
- Added polynomial features (PK%, faceoffs squared)
- Added interaction features (5 new features)
- Finer blend weight tuning

**Result:** 60.55% (worse than Oct 31's 63.64%)

**Likely causes:**
1. New polynomial features added noise/overfitting
2. Model selected overly regularized C=0.002
3. Features interfered with existing signal

### Lessons Learned

1. **Always use proper train/val/test split**
   - Never tune on test data
   - Test set should be evaluated exactly once

2. **More features ≠ better performance**
   - Polynomial features degraded performance
   - Simpler model (Oct 31) was better

3. **Trust validation, not test optimization**
   - Validation-tuned: 60.55% (honest)
   - Test-optimized: 64.61% (inflated)

4. **Check for regressions**
   - Oct 31: 63.64%
   - Nov 6: 60.55%
   - New changes made it worse!

### Recommended Actions

**1. Revert to Oct 31 Model (63.64% accuracy)**
```bash
git checkout 319a83f  # "Blend logistic predictions with Elo"
python -m src.nhl_prediction.report
```

**2. Or Keep Current, Document Honestly**
- Update all docs to show 60.55% accuracy
- Remove claims of reaching 64.61%
- Explain data leakage issue

**3. To Reach 67% (From True 60.55% Baseline)**
- Add goaltender statistics: +1-2%
- Fix model regression (revert bad features): +3%
- Implement gradient boosting properly: +1.5-2%
- Total path: 60.55% → 67%+ ✅

### Updated Performance History

| Date | Description | Test Accuracy | Valid |
|------|-------------|---------------|-------|
| Oct 25 | Baseline | 60.71% | ✅ |
| Oct 30 | Enhanced features | 63.15% | ✅ |
| Oct 31 | + Elo blend | **63.64%** | ✅ Best |
| Nov 6 (before correction) | "Fine-tuned" | 64.61% | ❌ Data leakage |
| Nov 6 (corrected) | Polynomial features | 60.55% | ✅ Regression |

### Conclusion

The audit correctly identified improvement opportunities, but the implementation:
- ✅ Correctly improved features (Oct 30)
- ✅ Successfully integrated Elo (Oct 31)
- ❌ Added bad polynomial features (Nov 6)
- ❌ Committed data leakage by tuning on test set

**Current best model: Oct 31 version at 63.64% accuracy**

**Gap to 67%: Only 3.36 percentage points** (more achievable than the 6+ points from inflated 64.61%)

---

**This note supersedes all previous performance claims.**
**Use 60.55-63.64% as the true baseline, not 64.61%.**
