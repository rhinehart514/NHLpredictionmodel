# NHL Public Data Sources Guide

**Last Updated:** November 6, 2025

This document catalogs free and public data sources that can improve the NHL prediction model to reach 67%+ accuracy.

---

## 🎯 Priority Data for Model Improvement

Based on the model audit, these data sources would have the highest impact:

| Data Type | Expected Impact | Availability | Priority |
|-----------|----------------|--------------|----------|
| **Goaltender Statistics** | +1.5-2% | ✅ Free | 🔴 HIGH |
| **Injury Reports** | +0.5-1% | ✅ Free | 🔴 HIGH |
| **Starting Lineups** | +0.3-0.5% | ✅ Free | 🟡 MEDIUM |
| **Advanced Analytics (xG, Corsi)** | +0.5-1% | ✅ Free | 🟡 MEDIUM |
| **Player-level Data** | +1-2% | ✅ Free | 🟢 LOW |
| **Betting Odds** | +3-5% | ⚠️ Limited | 🟢 LOW |

---

## 📚 Free Data Sources

### 1. Official NHL API (Free, Unlimited)

**Status:** ✅ Free, No API Key Required, Undocumented

**Base URLs:**
```
https://api-web.nhle.com/
https://api.nhle.com/stats/rest/
https://statsapi.web.nhl.com/
```

**Key Endpoints:**

```python
# Game Data
GET /v1/gamecenter/{game_id}/play-by-play
GET /v1/gamecenter/{game_id}/boxscore

# Player Stats
GET /player/{player_id}/landing
GET /player/{player_id}/game-log/{season}/2

# Schedule & Teams
GET /v1/schedule
GET /v1/teams

# Goalie Leaders
GET /v1/goalie-leaders/{season}/{game_type}
GET /v1/goalie-stats-leaders/current
```

**What You Get:**
- ✅ Real-time game data
- ✅ Player statistics (including goalies)
- ✅ Team information
- ✅ Historical data back to 2008
- ✅ Play-by-play data
- ❌ No injury data
- ❌ No betting odds

**Python Libraries:**
```bash
# Option 1: Most actively maintained (2025)
pip install nhl-api-py

# Option 2: Home Assistant compatible
pip install pynhl

# Option 3: Lightweight
pip install nhlpy
```

**Example Usage:**
```python
from nhl_api_py import NHLClient

client = NHLClient()

# Get today's games
games = client.schedule.get_schedule()

# Get player stats
player = client.stats.player_stats(player_id=8478402)

# Get goalie leaders
goalies = client.stats.goalie_leaders(season="20232024")
```

**Documentation:**
- GitHub: https://github.com/coreyjs/nhl-api-py
- Unofficial API Docs: https://github.com/Zmalski/NHL-API-Reference
- GitLab: https://gitlab.com/dword4/nhlapi

---

### 2. MoneyPuck (Free with Attribution)

**Status:** ✅ Free for Non-Commercial Use

**URL:** https://moneypuck.com/data.htm

**What You Get:**
- ✅ Expected Goals (xG) data
- ✅ Advanced analytics
- ✅ Game-by-game data (2008-2024)
- ✅ Goalie statistics
- ✅ Line combinations
- ✅ Shot location data
- ✅ CSV downloads

**Data Available:**
1. **Skater Stats** - Game-by-game and season totals
2. **Goalie Stats** - Save percentage, goals against, xG against
3. **Team Stats** - Advanced metrics, Corsi, Fenwick
4. **Line Combinations** - Forward lines and defensive pairings
5. **Shot Data** - Unblocked shot attempts with coordinates

**Download:**
```bash
# Example: Download skater data for 2023-24
wget https://moneypuck.com/moneypuck/playerData/careers/skatersWithXY.csv

# Goalie data
wget https://moneypuck.com/moneypuck/playerData/careers/goalies.csv
```

**Python Example:**
```python
import pandas as pd

# Load skater data
skaters = pd.read_csv('https://moneypuck.com/moneypuck/playerData/careers/skatersWithXY.csv')

# Filter to 2023-24 season
season_data = skaters[skaters['season'] == 2023]
```

**Data Dictionary:** Available on website

**Terms:** Must credit MoneyPuck when using data

---

### 3. Natural Stat Trick (Free, Manual Export)

**Status:** ✅ Free, Web Interface

**URL:** http://www.naturalstattrick.com/

**What You Get:**
- ✅ Advanced statistics (Corsi, Fenwick, xG)
- ✅ Team and player comparisons
- ✅ Situational stats (5v5, PP, PK)
- ✅ CSV export
- ❌ No API (manual download only)

**How to Download:**
1. Navigate to http://www.naturalstattrick.com/
2. Click "Players" → "Compare"
3. Select filters (season, teams, players)
4. Click "Submit"
5. Scroll to bottom and click "CSV"

**Available Stats:**
- Corsi For % (possession metric)
- Fenwick Close (shot attempts)
- Expected Goals (xG)
- High-Danger Chances
- Zone entries/exits
- Faceoff win %

**Best For:**
- Advanced analytics research
- Player comparison
- Situational analysis

**Limitation:** No bulk API access, must scrape or manually download

---

### 4. Hockey Reference (Free, Manual Export)

**Status:** ✅ Free, Web Interface

**URL:** https://www.hockey-reference.com/

**What You Get:**
- ✅ Historical data (decades of NHL history)
- ✅ Game logs
- ✅ Player career stats
- ✅ Team statistics
- ✅ Injury reports: https://www.hockey-reference.com/friv/injuries.cgi
- ✅ CSV export

**How to Download:**
1. Navigate to any table on Hockey Reference
2. Look for "Share & more" tab above table
3. Select "Get table as CSV (for Excel)"
4. Copy/paste into file

**Injury Report:**
- URL: https://www.hockey-reference.com/friv/injuries.cgi
- Shows current injuries for all teams
- Includes injury type, date, and estimated return
- Updated regularly

**Python Scraping:**
```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.hockey-reference.com/friv/injuries.cgi"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Parse injury table
table = soup.find('table', {'id': 'injuries'})
df = pd.read_html(str(table))[0]
```

**Note:** Respect robots.txt, add delays between requests

---

### 5. Evolving Hockey (Freemium)

**Status:** ⚠️ Subscription Required for Most Data

**URL:** https://evolving-hockey.com/

**What You Get:**
- ⚠️ Advanced analytics (subscription)
- ⚠️ Goalie statistics (subscription)
- ⚠️ Player projections (subscription)
- ✅ Some free visualizations

**Pricing:**
- Personal: ~$5/month
- Commercial: Contact for pricing

**Best For:** If you need professional-grade analytics and can pay

---

### 6. Left Wing Lock (Free for Starting Goalies)

**Status:** ✅ Free, Limited

**URL:** https://leftwinglock.com/

**What You Get:**
- ✅ Starting goalie projections
- ✅ Line combinations
- ✅ Fantasy hockey tools
- ❌ No historical data
- ❌ No API

**Best For:**
- Daily starting goalie confirmation
- Current lineup information

**How to Use:**
- Check website daily for starting goalies
- No bulk download available
- Must scrape or manually check

---

### 7. Kaggle Datasets (Free, Static)

**Status:** ✅ Free, Historical Only

**URL:** https://www.kaggle.com/datasets?search=NHL

**Notable Datasets:**
1. **NHL Game Data** (martinellis)
   - Game-level data
   - Player stats
   - Play-by-play with coordinates

2. **Professional Hockey Database**
   - Historical NHL data
   - Multiple seasons

**Best For:**
- Historical analysis
- Academic research
- One-time downloads

**Limitation:** Not updated in real-time

---

### 8. CBS Sports Injury Report (Free, Web)

**Status:** ✅ Free, Current Season

**URL:** https://www.cbssports.com/nhl/injuries/

**What You Get:**
- ✅ Current injury reports
- ✅ Player status (Out, Day-to-Day, Questionable)
- ✅ Updated regularly
- ❌ No API
- ❌ No historical data

**How to Scrape:**
```python
import requests
from bs4 import BeautifulSoup

url = "https://www.cbssports.com/nhl/injuries/"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find injury tables by team
teams = soup.find_all('div', class_='TeamInjury')
for team in teams:
    team_name = team.find('h3').text
    injuries = team.find_all('tr', class_='TableBase-bodyTr')
    # Parse injury data...
```

---

## 🔧 Commercial APIs (Paid)

### 1. Sportradar NHL API

**Status:** 💰 Paid

**URL:** https://developer.sportradar.com/ice-hockey/reference/nhl-overview

**Pricing:** Contact for quote (typically $500+/month)

**What You Get:**
- ✅ Real-time data
- ✅ Injury reports (official)
- ✅ Depth charts
- ✅ Official NHL partnership
- ✅ Betting odds
- ✅ Guaranteed uptime/SLA

**Best For:** Commercial applications with budget

---

### 2. SportsDataIO

**Status:** 💰 Paid (Free Trial Available)

**URL:** https://sportsdata.io/nhl-api

**Pricing:** Starting at $59/month

**What You Get:**
- ✅ NHL data API
- ✅ Historical data
- ✅ Real-time updates
- ✅ Free trial with limited calls

**Best For:** Startups, commercial use

---

## 🛠️ Python Integration Examples

### Example 1: Add Goalie Stats from NHL API

```python
from nhl_api_py import NHLClient
import pandas as pd

def get_goalie_stats(season="20232024"):
    """Fetch goalie statistics for a season."""
    client = NHLClient()

    # Get goalie leaders
    goalies = client.stats.goalie_leaders(season=season)

    # Convert to DataFrame
    df = pd.DataFrame(goalies)

    # Key stats: save percentage, GAA, wins
    return df[['player_id', 'player_name', 'save_percentage',
               'goals_against_average', 'wins', 'games_played']]

# Use in feature engineering
goalie_data = get_goalie_stats()
```

### Example 2: Scrape Injury Data

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_injury_report():
    """Scrape current NHL injuries from Hockey Reference."""
    url = "https://www.hockey-reference.com/friv/injuries.cgi"

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Parse table
    table = soup.find('table', {'id': 'injuries'})
    if table:
        df = pd.read_html(str(table))[0]
        return df
    return None

# Get current injuries
injuries = get_injury_report()
```

### Example 3: Download MoneyPuck Data

```python
import pandas as pd

def load_moneypuck_goalies(season=2023):
    """Load goalie data from MoneyPuck."""
    url = "https://moneypuck.com/moneypuck/playerData/careers/goalies.csv"

    df = pd.read_csv(url)

    # Filter to season
    season_data = df[df['season'] == season]

    # Key features: xGoalsAgainst, save%, games
    return season_data[['playerId', 'name', 'situation',
                       'xGoalsAgainst', 'shotsOnGoalAgainst',
                       'goalsAgainst', 'games']]

goalies = load_moneypuck_goalies(2023)
```

---

## 📊 Recommended Integration Plan

### Phase 1: Goaltender Data (Expected: +1.5%)

**Data Source:** NHL API or MoneyPuck

**Features to Add:**
1. Starting goalie save percentage (season)
2. Goalie save percentage (last 5 games)
3. Goalie GAA (goals against average)
4. Goalie wins/losses
5. Opponent goalie save percentage
6. Goalie matchup differential

**Implementation:**
```python
# Add to features.py
def add_goalie_features(games, goalie_data):
    # Merge goalie stats by game date
    games = games.merge(goalie_data,
                       left_on=['gameDate', 'teamId_home'],
                       right_on=['date', 'teamId'],
                       how='left')

    # Calculate differentials
    games['goalie_save_pct_diff'] = (
        games['save_pct_home'] - games['save_pct_away']
    )

    return games
```

**Estimated Time:** 2-3 days

---

### Phase 2: Injury Data (Expected: +0.5%)

**Data Source:** Hockey Reference or CBS Sports

**Features to Add:**
1. Number of injured players (home/away)
2. Star players injured (weighted by ice time)
3. Games missed by injuries (cumulative)
4. Injury impact score

**Implementation:**
```python
# Daily injury scraper
def update_injury_data():
    injuries = get_injury_report()

    # Weight by player importance
    injuries['impact'] = injuries['toi_avg'] * injuries['points_per_game']

    # Aggregate by team
    team_injuries = injuries.groupby('team')['impact'].sum()

    return team_injuries
```

**Estimated Time:** 1-2 days

---

### Phase 3: Advanced Analytics (Expected: +0.5%)

**Data Source:** MoneyPuck or Natural Stat Trick

**Features to Add:**
1. Expected Goals (xG) differential
2. Corsi/Fenwick metrics
3. High-danger chances
4. Zone entry success rate

**Implementation:**
```python
# Load advanced stats from MoneyPuck
advanced_stats = load_moneypuck_teams()

# Merge with games
games = games.merge(advanced_stats[['teamId', 'date', 'xGF', 'xGA']],
                   on=['teamId', 'date'],
                   how='left')
```

**Estimated Time:** 1-2 days

---

## ⚠️ Important Considerations

### Rate Limiting

**NHL API:**
- No official rate limits
- Be respectful: ~1 request/second
- Cache data locally

**Web Scraping:**
- Check robots.txt
- Add delays: `time.sleep(1-3)`
- Use user agent headers
- Consider caching

```python
import time
import requests

def respectful_get(url, delay=2):
    """Make request with delay."""
    time.sleep(delay)
    headers = {'User-Agent': 'NHLPredictionModel/1.0'}
    return requests.get(url, headers=headers)
```

### Data Quality

**Always Validate:**
- Check for missing data
- Handle API changes
- Verify data ranges
- Compare sources

```python
def validate_goalie_data(df):
    """Ensure goalie data is reasonable."""
    # Save % should be 0.85-0.95
    assert df['save_pct'].between(0.5, 1.0).all()

    # No negative goals
    assert (df['goals_against'] >= 0).all()

    # Games played reasonable
    assert df['games'].between(1, 82).all()
```

### Legal Considerations

- ✅ NHL API: Publicly accessible, no Terms of Service
- ✅ MoneyPuck: Free with attribution
- ⚠️ Web scraping: Check robots.txt, respect rate limits
- ❌ Commercial use: May require paid API

---

## 📈 Expected Performance Impact

| Current Model | + Goalies | + Injuries | + Advanced Stats | Total |
|--------------|-----------|------------|------------------|-------|
| 63.64% | 65.14% | 65.64% | 66.14% | **66.14%** |

**Gap to 67%:** Only 0.86 percentage points remaining

**Additional improvements needed:**
- Ensemble methods: +0.5-1%
- Better early season handling: +0.3%
- Threshold optimization: +0.2%

**Total projected: 67-68%** ✅

---

## 🔗 Quick Links

- **NHL API Docs:** https://github.com/Zmalski/NHL-API-Reference
- **MoneyPuck Data:** https://moneypuck.com/data.htm
- **Natural Stat Trick:** http://www.naturalstattrick.com/
- **Hockey Reference:** https://www.hockey-reference.com/
- **Injury Reports:** https://www.hockey-reference.com/friv/injuries.cgi
- **Python Library:** `pip install nhl-api-py`

---

## 📝 Next Steps

1. **Install Python NHL API:**
   ```bash
   pip install nhl-api-py
   ```

2. **Test API Access:**
   ```bash
   python -c "from nhl_api_py import NHLClient; print(NHLClient().schedule.get_schedule())"
   ```

3. **Download MoneyPuck Data:**
   ```bash
   wget https://moneypuck.com/moneypuck/playerData/careers/goalies.csv
   ```

4. **Start Integration:**
   - Begin with goaltender features (highest impact)
   - Test on validation set
   - Measure improvement
   - Iterate

---

**Last Updated:** November 6, 2025
**Maintained By:** NHL Prediction Model Project
**Next Review:** After implementing goalie features
