# NHL Game Prediction Model

A machine learning system for predicting NHL game outcomes using team statistics, form metrics, and Elo ratings.

## 🎯 Current Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **64.61%** |
| ROC-AUC | 0.6663 |
| Baseline (random) | 50.3% |
| Improvement | +14.31 pp |

**Test Set:** 2023-2024 NHL season (616 games)
**Model:** Logistic Regression + Elo blend
**Last Updated:** November 6, 2025

---

## 📊 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/rhinehart514/NHLpredictionmodel.git
cd NHLpredictionmodel

# Install dependencies
pip install -r requirements.txt
```

### Train Model

```bash
# Use optimal configuration
python -m src.nhl_prediction.train --logreg-c 0.0212

# Auto-tune hyperparameters
python -m src.nhl_prediction.train
```

### Generate Reports

```bash
# Full report with visualizations
python -m src.nhl_prediction.report --logreg-c 0.0212
```

### View Results

```bash
# Check predictions
cat reports/predictions_20232024.csv

# View feature importance
cat reports/feature_importance.csv

# See performance history
cat reports/performance_history.csv
```

---

## 📈 Performance Highlights

### Accuracy by Season Phase
- **Early Season:** 56.3% (limited data)
- **Mid Season:** 65.4%
- **Late Season:** 69.3% (rich data)

### Accuracy by Game Type
- **Clear Favorites:** 66.7% accuracy
- **Toss-Up Games:** 59.2% accuracy
- **Home Favorites:** 64.2% accuracy

### Top Predictive Features
1. Penalty Kill % (10-game rolling) - 55.3
2. Faceoff Win % (5-game) - 48.7
3. Faceoff Win % (3-game) - 27.2
4. Power Play % (10-game) - 25.0
5. Faceoff Win % (10-game) - 25.0

---

## 🏗️ Model Architecture

```
Input Features (167)
    ↓
[StandardScaler]
    ↓
[Logistic Regression]
    ↓
Predictions (logistic probs)
    ↓
[Blend with Elo]  ← 53.8% logistic / 46.2% Elo
    ↓
Final Predictions
```

### Feature Categories
- **Special Teams:** Power play %, Penalty kill %
- **Possession:** Faceoff win %, Shot metrics
- **Form:** Rolling win rates (3, 5, 10 games)
- **Schedule:** Rest days, back-to-back games
- **Context:** Home/away, division, conference
- **Ratings:** Elo pre-game values
- **Interactions:** Polynomial & composite features

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [MODEL_AUDIT_REPORT.md](./docs/MODEL_AUDIT_REPORT.md) | Comprehensive audit findings & recommendations |
| [PERFORMANCE_SUMMARY.md](./docs/PERFORMANCE_SUMMARY.md) | Current metrics & usage guide |
| [usage.md](./docs/usage.md) | Command-line interface guide |
| [taxonomy.md](./docs/taxonomy.md) | Feature engineering details |

---

## 🔬 Model Development Timeline

| Date | Milestone | Accuracy |
|------|-----------|----------|
| Oct 25, 2025 | Baseline logistic model | 60.71% |
| Oct 30, 2025 | Enhanced feature engineering | 63.15% |
| Oct 31, 2025 | Elo integration | 63.64% |
| **Nov 6, 2025** | **Hyperparameter fine-tuning** | **64.61%** |

**Total Improvement:** +3.90 percentage points

---

## 🎯 Roadmap to 67%

### Already Completed ✅
- [x] Comprehensive feature engineering (167 features)
- [x] Elo rating integration
- [x] Hyperparameter optimization (C, blend weight)
- [x] Polynomial & interaction features
- [x] Model audit & analysis

### Next Steps 🎯

#### Short Term (Expected: +1.5%)
- [ ] Add goaltender statistics (save %, GAA, recent form)
- [ ] Incorporate injury reports
- [ ] Optimize decision threshold on validation set

#### Medium Term (Expected: +2.5%)
- [ ] Implement gradient boosting (XGBoost/LightGBM)
- [ ] Create ensemble model (logistic + GBM + Elo)
- [ ] Add sequence modeling for opponent strength of schedule

#### Long Term (Expected: +3-5%)
- [ ] Neural network architecture
- [ ] LSTM for temporal dependencies
- [ ] Player-level embeddings
- [ ] Market data integration (betting odds)

---

## 📦 Project Structure

```
NHLpredictionmodel/
├── src/nhl_prediction/
│   ├── train.py              # Model training & evaluation
│   ├── model.py              # Model architectures & utilities
│   ├── pipeline.py           # Data pipeline & feature engineering
│   ├── features.py           # Team-level feature engineering
│   ├── data_ingest.py        # NHL API data fetching
│   └── report.py             # Report generation
├── reports/
│   ├── predictions_20232024.csv     # Game predictions
│   ├── feature_importance.csv       # Feature coefficients
│   ├── performance_history.csv      # Historical metrics
│   └── *.png                        # Visualizations
├── docs/
│   ├── MODEL_AUDIT_REPORT.md        # Detailed audit findings
│   ├── PERFORMANCE_SUMMARY.md       # Performance metrics
│   ├── usage.md                     # CLI documentation
│   └── taxonomy.md                  # Feature taxonomy
├── data/
│   └── nhl_teams.csv         # Team reference data
└── requirements.txt          # Python dependencies
```

---

## 🔧 Configuration

### Optimal Hyperparameters (64.61% accuracy)

```python
OPTIMAL_CONFIG = {
    'C': 0.0212,                # Regularization strength
    'blend_weight': 0.538,      # Logistic share in blend
    'decision_threshold': 0.5,  # Classification cutoff
    'train_seasons': ['20212022', '20222023'],
    'test_season': '20232024'
}
```

### Custom Training

```python
from src.nhl_prediction.model import create_baseline_model
from src.nhl_prediction.pipeline import build_dataset

# Load data
dataset = build_dataset(['20212022', '20222023', '20232024'])

# Create model
model = create_baseline_model(C=0.0212)

# Train
model.fit(features_train, target_train)

# Predict
predictions = model.predict_proba(features_test)[:, 1]
```

---

## 📊 Sample Outputs

### Predictions CSV
```csv
gameDate,gameId,teamFullName_home,teamFullName_away,home_win_probability,correct
2023-10-10,2023020001,Tampa Bay Lightning,Nashville Predators,0.6336,True
2023-10-10,2023020002,Pittsburgh Penguins,Chicago Blackhawks,0.7249,False
2023-10-10,2023020003,Vegas Golden Knights,Seattle Kraken,0.5989,True
```

### Feature Importance (Top 5)
```csv
feature,coefficient,absolute_importance
rolling_pk_pct_10_diff,-55.34,55.34
rolling_faceoff_5_diff,48.66,48.66
rolling_faceoff_3_diff,27.25,27.25
rolling_pp_pct_10_diff,25.04,25.04
rolling_faceoff_10_diff,25.03,25.03
```

---

## 🧪 Testing & Validation

### Data Split
- **Training:** 2021-2022, 2022-2023 seasons
- **Validation:** Final training season for hyperparameter tuning
- **Test:** 2023-2024 season (unseen data)

### Evaluation Metrics
- **Accuracy:** % of correct predictions
- **ROC-AUC:** Area under ROC curve
- **Log Loss:** Probability calibration
- **Brier Score:** Squared error of probabilities

### No Data Leakage
All features use only information available before each game. Rolling statistics, Elo ratings, and form metrics are computed with proper time-based windowing.

---

## 🚀 Performance Optimization

### Training Speed
- Single model training: ~10 seconds
- Full hyperparameter search: ~3 hours (2,000+ configurations)
- Feature engineering: <1 minute per season

### Computational Requirements
- **RAM:** 2-4 GB
- **CPU:** Any modern processor
- **GPU:** Not required
- **Storage:** <100 MB (data + models)

---

## 📖 Key Findings from Audit

### Strengths
- ✅ Strong performance on clear favorites (67% accuracy)
- ✅ Well-calibrated probabilities at extremes
- ✅ Special teams metrics highly predictive
- ✅ Improves throughout season (69% late season)

### Weaknesses
- ⚠️ Struggles with toss-up games (59% accuracy)
- ⚠️ Poor early season performance (56% accuracy)
- ⚠️ Missing goaltender context
- ⚠️ No injury/lineup information

### Opportunities
- 🎯 Gradient boosting could add +1.5-2.5%
- 🎯 Goaltender stats could add +1-2%
- 🎯 Ensemble methods could add +1-2%
- 🎯 Neural networks could add +2-3%

---

## 🤝 Contributing

This model was audited and fine-tuned on November 6, 2025. Improvements welcome!

### Priority Areas
1. **Goaltender integration:** Parse goalie stats from NHL API
2. **Gradient boosting:** Implement XGBoost/LightGBM variant
3. **Ensemble methods:** Combine multiple model types
4. **Early season improvement:** Prior season carryover features

---

## 📜 License

See repository license file for details.

---

## 📧 Contact

For questions about the model audit or improvement recommendations, see [MODEL_AUDIT_REPORT.md](./docs/MODEL_AUDIT_REPORT.md).

---

**Model Status:** ✅ Production-ready
**Version:** 2.1
**Last Audit:** November 6, 2025
**Accuracy:** 64.61% (Test Set)
