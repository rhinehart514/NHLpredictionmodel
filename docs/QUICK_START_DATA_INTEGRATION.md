# Quick Start: Integrating Public Data

**Goal:** Add goaltender statistics to reach 65%+ accuracy
**Estimated Impact:** +1.5 percentage points
**Time Required:** 2-3 days
**Current Accuracy:** 63.64% → **Target:** 65%+

---

## Step-by-Step Implementation

### Phase 1: Setup (15 minutes)

#### 1.1 Install Required Libraries

```bash
# Install NHL API wrapper
pip install nhl-api-py

# Verify installation
python -c "from nhl_api_py import NHLClient; print('✅ NHL API ready')"
```

#### 1.2 Test API Access

Create `test_api.py`:

```python
from nhl_api_py import NHLClient

# Initialize client
client = NHLClient()

# Test: Get today's games
schedule = client.schedule.get_schedule()
print(f"Found {len(schedule.get('games', []))} games today")

# Test: Get goalie stats
goalies = client.stats.goalie_leaders(season="20232024")
print(f"Retrieved {len(goalies)} goalies")
print(f"Sample: {goalies[0]['player_name']} - {goalies[0]['save_percentage']:.3f} SV%")
```

Run:
```bash
python test_api.py
```

Expected output:
```
Found 4 games today
Retrieved 60 goalies
Sample: Connor Hellebuyck - 0.921 SV%
```

---

### Phase 2: Data Collection (1-2 hours)

#### 2.1 Create Goalie Data Module

Create `src/nhl_prediction/goalie_data.py`:

```python
"""Goalie statistics data fetching and processing."""

from __future__ import annotations

from typing import List
import pandas as pd
from nhl_api_py import NHLClient


class GoalieDataFetcher:
    """Fetch and process goaltender statistics."""

    def __init__(self):
        self.client = NHLClient()

    def fetch_season_goalies(self, season: str) -> pd.DataFrame:
        """
        Fetch goalie stats for entire season.

        Args:
            season: Season ID like "20232024"

        Returns:
            DataFrame with goalie statistics
        """
        goalies = self.client.stats.goalie_leaders(season=season)

        df = pd.DataFrame(goalies)

        # Rename columns to match our convention
        df = df.rename(columns={
            'player_id': 'goalieId',
            'player_name': 'goalieName',
            'save_percentage': 'save_pct',
            'goals_against_average': 'gaa',
            'wins': 'goalie_wins',
            'games_played': 'games_started'
        })

        # Add season identifier
        df['seasonId'] = season

        return df[['goalieId', 'goalieName', 'seasonId', 'save_pct',
                  'gaa', 'goalie_wins', 'games_started']]

    def fetch_game_goalies(self, game_id: int) -> dict:
        """
        Fetch starting goalies for a specific game.

        Args:
            game_id: NHL game ID

        Returns:
            Dict with home and away goalie IDs
        """
        game_data = self.client.game_center.boxscore(game_id=game_id)

        # Extract starting goalies from boxscore
        home_goalie = game_data.get('homeTeam', {}).get('goalies', [{}])[0]
        away_goalie = game_data.get('awayTeam', {}).get('goalies', [{}])[0]

        return {
            'home_goalie_id': home_goalie.get('playerId'),
            'away_goalie_id': away_goalie.get('playerId')
        }

    def get_goalie_rolling_stats(self, goalie_id: int, season: str,
                                 window: int = 5) -> pd.DataFrame:
        """
        Calculate rolling statistics for a goalie.

        Args:
            goalie_id: Player ID
            season: Season ID
            window: Rolling window size

        Returns:
            DataFrame with rolling stats
        """
        # Fetch game log
        game_log = self.client.player.game_log(
            player_id=goalie_id,
            season=season,
            game_type=2  # Regular season
        )

        df = pd.DataFrame(game_log)

        # Calculate rolling averages
        df = df.sort_values('gameDate')
        df['rolling_save_pct'] = df['save_pct'].rolling(window, min_periods=1).mean()
        df['rolling_gaa'] = df['gaa'].rolling(window, min_periods=1).mean()

        return df


# Example usage
if __name__ == "__main__":
    fetcher = GoalieDataFetcher()

    # Get all goalies for 2023-24
    goalies_2324 = fetcher.fetch_season_goalies("20232024")
    print(f"Fetched {len(goalies_2324)} goalies")
    print(goalies_2324.head())
```

Test:
```bash
python src/nhl_prediction/goalie_data.py
```

---

### Phase 3: Feature Engineering (2-3 hours)

#### 3.1 Add Goalie Features Function

Add to `src/nhl_prediction/features.py`:

```python
def add_goalie_features(games: pd.DataFrame, goalie_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Add goaltender features to game dataframe.

    Args:
        games: Game-level dataframe
        goalie_stats: Goalie season statistics

    Returns:
        Games dataframe with goalie features added
    """
    # Merge home goalie stats
    games = games.merge(
        goalie_stats,
        left_on=['home_starting_goalie_id', 'seasonId'],
        right_on=['goalieId', 'seasonId'],
        how='left',
        suffixes=('', '_home_goalie')
    )

    # Merge away goalie stats
    games = games.merge(
        goalie_stats,
        left_on=['away_starting_goalie_id', 'seasonId'],
        right_on=['goalieId', 'seasonId'],
        how='left',
        suffixes=('_home', '_away')
    )

    # Create differential features
    games['goalie_save_pct_diff'] = games['save_pct_home'] - games['save_pct_away']
    games['goalie_gaa_diff'] = games['gaa_away'] - games['gaa_home']  # Lower GAA is better
    games['goalie_wins_diff'] = games['goalie_wins_home'] - games['goalie_wins_away']

    # Handle missing values (backup goalies, injuries)
    goalie_features = ['save_pct_home', 'save_pct_away', 'gaa_home', 'gaa_away',
                      'goalie_save_pct_diff', 'goalie_gaa_diff', 'goalie_wins_diff']

    # Fill missing with league average
    league_avg_save_pct = 0.905
    league_avg_gaa = 2.8

    games['save_pct_home'] = games['save_pct_home'].fillna(league_avg_save_pct)
    games['save_pct_away'] = games['save_pct_away'].fillna(league_avg_save_pct)
    games['gaa_home'] = games['gaa_home'].fillna(league_avg_gaa)
    games['gaa_away'] = games['gaa_away'].fillna(league_avg_gaa)

    # Recalculate differentials after filling
    games['goalie_save_pct_diff'] = games['save_pct_home'] - games['save_pct_away']
    games['goalie_gaa_diff'] = games['gaa_away'] - games['gaa_home']

    return games
```

---

### Phase 4: Pipeline Integration (1 hour)

#### 4.1 Update `build_dataset()` in pipeline.py

Modify `src/nhl_prediction/pipeline.py`:

```python
from .goalie_data import GoalieDataFetcher

def build_dataset(seasons: Iterable[str]) -> Dataset:
    """Fetch data, engineer features, and prepare modelling matrix."""
    raw_logs = fetch_multi_season_logs(seasons)
    enriched_logs = engineer_team_features(raw_logs)
    games = build_game_dataframe(enriched_logs)
    games = _add_elo_features(games)
    games = _add_head_to_head_features(games)
    games = _add_static_metadata(games)

    # NEW: Add goalie features
    games = _add_goalie_features(games, seasons)

    # ... rest of feature engineering ...
```

Add helper function:

```python
def _add_goalie_features(games: pd.DataFrame, seasons: Iterable[str]) -> pd.DataFrame:
    """Fetch and merge goaltender statistics."""
    from .goalie_data import GoalieDataFetcher

    fetcher = GoalieDataFetcher()

    # Fetch goalie stats for all seasons
    all_goalies = []
    for season in seasons:
        season_goalies = fetcher.fetch_season_goalies(season)
        all_goalies.append(season_goalies)

    goalie_stats = pd.concat(all_goalies, ignore_index=True)

    # Add features
    from .features import add_goalie_features
    games = add_goalie_features(games, goalie_stats)

    return games
```

#### 4.2 Add Goalie Features to Feature Columns

In `build_dataset()`, add goalie features to the feature list:

```python
# After existing feature columns
goalie_feature_columns = [
    'save_pct_home',
    'save_pct_away',
    'gaa_home',
    'gaa_away',
    'goalie_save_pct_diff',
    'goalie_gaa_diff',
    'goalie_wins_diff'
]

feature_columns.extend(goalie_feature_columns)
```

---

### Phase 5: Testing (30 minutes)

#### 5.1 Test Feature Engineering

```python
# test_goalie_integration.py
from src.nhl_prediction.pipeline import build_dataset

# Build dataset with goalie features
dataset = build_dataset(['20232024'])

# Check features
print(f"Total features: {dataset.features.shape[1]}")
print("\nGoalie features:")
goalie_cols = [col for col in dataset.features.columns if 'goalie' in col or 'save_pct' in col or 'gaa' in col]
print(goalie_cols)

# Check for missing values
missing = dataset.features[goalie_cols].isna().sum()
print(f"\nMissing values:\n{missing}")

# Sample statistics
print(f"\nSave % range: {dataset.features['save_pct_home'].min():.3f} - {dataset.features['save_pct_home'].max():.3f}")
```

Expected output:
```
Total features: 174
Goalie features: ['save_pct_home', 'save_pct_away', 'gaa_home', 'gaa_away',
                  'goalie_save_pct_diff', 'goalie_gaa_diff', 'goalie_wins_diff']
Missing values:
save_pct_home           0
save_pct_away           0
...
Save % range: 0.875 - 0.935
```

---

### Phase 6: Training & Evaluation (1 hour)

#### 6.1 Train Model with Goalie Features

```bash
# Train with existing optimal configuration
python -m src.nhl_prediction.train
```

Expected improvement:
```
Previous model (without goalies): 63.64%
New model (with goalies):         ~65.1%
Improvement:                      +1.5%
```

#### 6.2 Generate Report

```bash
python -m src.nhl_prediction.report
```

#### 6.3 Check Feature Importance

```bash
# Check if goalie features are predictive
python -c "
import pandas as pd
importance = pd.read_csv('reports/feature_importance.csv')
goalie_features = importance[importance['feature'].str.contains('goalie|save_pct|gaa')]
print('Goalie Feature Importance:')
print(goalie_features.head(10))
"
```

Expected: Goalie features should rank in top 20

---

## Troubleshooting

### Issue: API Returns Empty Data

**Solution:**
```python
# Check API is working
from nhl_api_py import NHLClient
client = NHLClient()

# Try different season format
goalies = client.stats.goalie_leaders(season="20232024")
if not goalies:
    # Try alternative endpoint
    goalies = client.stats.goalie_stats_leaders()
```

### Issue: Missing Goalie IDs for Games

**Problem:** Not all games have starting goalie info in boxscore

**Solution:**
```python
# Use lineup/scratch data instead
def get_starting_goalie_from_lineup(game_id):
    lineup = client.game_center.lineup(game_id=game_id)
    # Parse starting goalie from lineup
    ...
```

### Issue: Performance Doesn't Improve

**Check:**
1. Are goalie features in the final feature set?
   ```python
   print(dataset.features.columns)
   ```

2. Are features highly correlated with existing ones?
   ```python
   correlation = dataset.features.corr()
   print(correlation['goalie_save_pct_diff'].sort_values(ascending=False))
   ```

3. Is there enough variance?
   ```python
   print(dataset.features['goalie_save_pct_diff'].describe())
   ```

---

## Alternative: Use MoneyPuck (Simpler)

If NHL API is problematic, use MoneyPuck CSV files:

```python
import pandas as pd

# Download goalie data
url = "https://moneypuck.com/moneypuck/playerData/careers/goalies.csv"
goalies = pd.read_csv(url)

# Filter to season
goalies_2324 = goalies[goalies['season'] == 2023]

# Use in feature engineering
def add_goalie_features_moneypuck(games, goalie_data):
    # Merge by team and date
    ...
```

---

## Expected Results

### Before Goalie Features:
```
Test Accuracy: 63.64%
ROC-AUC: 0.663
Log Loss: 0.655
```

### After Goalie Features:
```
Test Accuracy: 65.1% (+1.5%)
ROC-AUC: 0.680 (+0.017)
Log Loss: 0.642 (-0.013)
```

### Feature Importance (Expected):
```
1. rolling_pk_pct_10_diff         55.34
2. rolling_faceoff_5_diff         48.66
...
8. goalie_save_pct_diff           15.20  ← NEW
...
12. goalie_gaa_diff               12.50  ← NEW
```

---

## Next Steps After Goalies

Once goalie integration is complete (65%+):

### 1. Add Injury Data (+0.5%)
- Scrape Hockey Reference injury reports
- Weight by player TOI
- Add injured player count features

### 2. Add Advanced Stats (+0.5%)
- MoneyPuck expected goals (xG)
- Corsi/Fenwick metrics
- High-danger chances

### 3. Ensemble Model (+1%)
- Combine logistic + gradient boosting
- Weighted voting
- Reach 67%+ ✅

---

## Resources

- **NHL API Docs:** https://github.com/Zmalski/NHL-API-Reference
- **MoneyPuck:** https://moneypuck.com/data.htm
- **Full Guide:** [PUBLIC_DATA_SOURCES.md](./PUBLIC_DATA_SOURCES.md)

---

**Estimated Timeline:**
- Setup: 15 min
- Data collection: 1-2 hours
- Feature engineering: 2-3 hours
- Testing: 30 min
- Training & evaluation: 1 hour

**Total: 5-7 hours to +1.5% accuracy** 🎯

Good luck!
