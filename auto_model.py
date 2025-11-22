# simple_model.py
# Historical-based betting model with opponent-adjusted form + improved probability math
# Clean ASCII-only output for email & Telegram

import os
import sys
import requests
import pandas as pd
from datetime import datetime as dt, UTC
from unidecode import unidecode
from dotenv import load_dotenv

# Force unbuffered UTF-8 output
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

load_dotenv()

# =========================
# CONFIG
# =========================
API_KEY = os.getenv("APIFOOTBALL_API_KEY")
if not API_KEY:
    raise ValueError("Missing APIFOOTBALL_API_KEY in .env")

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# Auto season (football seasons start ~July)
TODAY = dt.now(UTC).date()
SEASON = TODAY.year if TODAY.month >= 7 else TODAY.year - 1

# Included competitions
LEAGUE_GROUPS = {
    "England": [
        39,  # Premier League
        40,  # Championship
        41,  # League One
        42,  # League Two
        45,  # FA Cup
        48,  # EFL Cup
    ],
    "Scotland": [179, 180],  # Premiership, Championship
    "Germany": [78],         # Bundesliga
    "Europe": [2, 3, 848],   # UCL, UEL, UECL
}

# Model settings
LAST_N_GAMES = 10          # Look at last 10 games for each team
H2H_LAST_N = 5             # Look at last 5 head-to-head meetings
MIN_BTTS_PROB = 45.0       # Show BTTS picks above 45%
MIN_OVER_PROB = 50.0       # Show Over 2.5 picks above 50%
MIN_WIN_PROB = 55.0        # Show Win picks above 55%

# Accuracy improvements
HOME_ADVANTAGE_BOOST = 10.0  # Base home advantage
H2H_WEIGHT = 0.20            # 20% h2h, 80% recent form
RECENCY_DECAY = 0.88         # Exponential decay for older games

# Opponent strength tiers (based on league position)
OPPONENT_WEIGHTS = {
    "top": 1.4,      # Top 25% of table
    "upper": 1.15,   # 25-50%
    "lower": 0.90,   # 50-75%
    "bottom": 0.70,  # Bottom 25%
}

# Cache for league standings
STANDINGS_CACHE = {}

# =========================
# HELPERS
# =========================
def fmt_time_eu(iso_str: str) -> str:
    try:
        ts = pd.to_datetime(iso_str, utc=True).tz_convert("Europe/London")
        return ts.strftime("%a %H:%M")
    except Exception:
        return iso_str

def confidence_label(pct: float) -> str:
    if pct >= 70: return "High"
    if pct >= 55: return "Medium"
    return "Low"

def weighted_average(values, weights):
    """Calculate weighted average with recency decay"""
    if not values or not weights:
        return 0.0
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_weight

# =========================
# STANDINGS & OPPONENT STRENGTH
# =========================
def fetch_league_standings(league_id: int):
    """Fetch current league standings and cache them"""
    if league_id in STANDINGS_CACHE:
        return STANDINGS_CACHE[league_id]
    
    params = {"league": league_id, "season": SEASON}
    try:
        r = requests.get(f"{BASE_URL}/standings", params=params, headers=HEADERS, timeout=12)
        standings = r.json().get("response", [])
        if not standings:
            return {}
        
        # Extract team positions
        standings_dict = {}
        teams = standings[0].get("league", {}).get("standings", [[]])[0]
        total_teams = len(teams)
        
        for team in teams:
            team_id = team["team"]["id"]
            rank = team["rank"]
            standings_dict[team_id] = {
                "rank": rank,
                "total": total_teams,
            }
        
        STANDINGS_CACHE[league_id] = standings_dict
        return standings_dict
    except Exception:
        return {}

def get_opponent_tier(team_id: int, league_id: int, standings: dict) -> str:
    """Determine opponent strength tier based on league position"""
    if not standings or team_id not in standings:
        return "upper"  # Default to upper-mid if unknown
    
    rank = standings[team_id]["rank"]
    total = standings[team_id]["total"]
    position_pct = rank / total
    
    if position_pct <= 0.25:
        return "top"
    elif position_pct <= 0.50:
        return "upper"
    elif position_pct <= 0.75:
        return "lower"
    else:
        return "bottom"

# =========================
# API FETCHERS
# =========================
def fetch_fixtures_for_today():
    all_rows = []
    for region, leagues in LEAGUE_GROUPS.items():
        for lid in leagues:
            params = {"league": lid, "season": SEASON, "date": str(TODAY)}
            try:
                r = requests.get(f"{BASE_URL}/fixtures", params=params, headers=HEADERS, timeout=12)
                r.raise_for_status()
                for f in r.json().get("response", []):
                    all_rows.append({
                        "fixture_id": f["fixture"]["id"],
                        "home_id": f["teams"]["home"]["id"],
                        "away_id": f["teams"]["away"]["id"],
                        "home_team": f["teams"]["home"]["name"],
                        "away_team": f["teams"]["away"]["name"],
                        "league_id": lid,
                        "league": f["league"]["name"],
                        "region": region,
                        "kickoff": f["fixture"]["date"],
                    })
            except Exception:
                continue
    return pd.DataFrame(all_rows)

def fetch_last_n_games(team_id: int, league_id: int, standings: dict, last_n: int = LAST_N_GAMES, venue: str = None):
    """
    Fetch last N finished games for a team with opponent strength weighting
    """
    params = {
        "team": team_id,
        "league": league_id,
        "season": SEASON,
        "last": last_n * 3,
    }
    if venue:
        params["venue"] = venue
    
    try:
        r = requests.get(f"{BASE_URL}/fixtures", params=params, headers=HEADERS, timeout=12)
        fixtures = r.json().get("response", [])
        
        results = []
        for f in fixtures:
            if f["fixture"]["status"]["short"] not in ["FT", "AET", "PEN"]:
                continue
            
            is_home = f["teams"]["home"]["id"] == team_id
            opponent_id = f["teams"]["away"]["id"] if is_home else f["teams"]["home"]["id"]
            
            gf = f["goals"]["home"] if is_home else f["goals"]["away"]
            ga = f["goals"]["away"] if is_home else f["goals"]["home"]
            
            if gf is None or ga is None:
                continue
            
            # Get opponent strength
            opp_tier = get_opponent_tier(opponent_id, league_id, standings)
            opp_weight = OPPONENT_WEIGHTS.get(opp_tier, 1.0)
            
            results.append({
                "won": gf > ga,
                "btts": gf > 0 and ga > 0,
                "over25": (gf + ga) >= 3,
                "scored": gf > 0,
                "conceded": ga > 0,
                "gf": gf,
                "ga": ga,
                "opponent_weight": opp_weight,
            })
            
            if len(results) >= last_n:
                break
        
        return results
    except Exception:
        return []

def fetch_head_to_head(home_id: int, away_id: int, last_n: int = H2H_LAST_N):
    """Fetch last N head-to-head meetings between two teams"""
    params = {
        "h2h": f"{home_id}-{away_id}",
        "last": last_n * 2,
    }
    try:
        r = requests.get(f"{BASE_URL}/fixtures/headtohead", params=params, headers=HEADERS, timeout=12)
        fixtures = r.json().get("response", [])
        
        results = []
        for f in fixtures:
            if f["fixture"]["status"]["short"] not in ["FT", "AET", "PEN"]:
                continue
            
            home_goals = f["goals"]["home"]
            away_goals = f["goals"]["away"]
            
            if home_goals is None or away_goals is None:
                continue
            
            was_home_id = f["teams"]["home"]["id"]
            
            results.append({
                "home_won": (home_goals > away_goals and was_home_id == home_id) or 
                           (away_goals > home_goals and was_home_id == away_id),
                "away_won": (away_goals > home_goals and was_home_id == home_id) or 
                           (home_goals > away_goals and was_home_id == away_id),
                "btts": home_goals > 0 and away_goals > 0,
                "over25": (home_goals + away_goals) >= 3,
            })
            
            if len(results) >= last_n:
                break
        
        return results
    except Exception:
        return []

def calculate_probabilities(home_id: int, away_id: int, league_id: int):
    """Calculate BTTS, Over 2.5, and Win probabilities with improved math"""
    
    # Fetch league standings for opponent weighting
    standings = fetch_league_standings(league_id)
    
    # Get last N games - SPLIT BY VENUE
    home_home_games = fetch_last_n_games(home_id, league_id, standings, venue="home")
    away_away_games = fetch_last_n_games(away_id, league_id, standings, venue="away")
    
    # Fallback to all games if venue-specific is insufficient
    if len(home_home_games) < 5:
        home_home_games = fetch_last_n_games(home_id, league_id, standings)
    if len(away_away_games) < 5:
        away_away_games = fetch_last_n_games(away_id, league_id, standings)
    
    # Get head-to-head history
    h2h_games = fetch_head_to_head(home_id, away_id)
    
    # === IMPROVED BTTS CALCULATION ===
    # P(BTTS) = P(Home Scores) × P(Away Scores)
    def calc_scoring_probability(games):
        """Probability a team scores (opponent-weighted, recency-weighted)"""
        if not games:
            return 0.75  # Default assumption
        weights = [(RECENCY_DECAY ** i) * g["opponent_weight"] for i, g in enumerate(games)]
        values = [1.0 if g["scored"] else 0.0 for g in games]
        return weighted_average(values, weights)
    
    home_scores_prob = calc_scoring_probability(home_home_games)
    away_scores_prob = calc_scoring_probability(away_away_games)
    
    # Proper BTTS probability = both must score
    btts_prob_recent = home_scores_prob * away_scores_prob * 100
    
    # === IMPROVED OVER 2.5 CALCULATION ===
    def calc_weighted_over(games):
        if not games:
            return 0.0
        weights = [(RECENCY_DECAY ** i) * g["opponent_weight"] for i, g in enumerate(games)]
        values = [1.0 if g["over25"] else 0.0 for g in games]
        return weighted_average(values, weights) * 100
    
    home_over_pct = calc_weighted_over(home_home_games)
    away_over_pct = calc_weighted_over(away_away_games)
    over_prob_recent = (home_over_pct + away_over_pct) / 2
    
    # === OPPONENT-ADJUSTED WIN CALCULATION ===
    def calc_weighted_wins(games):
        if not games:
            return 0.0
        weights = [(RECENCY_DECAY ** i) * g["opponent_weight"] for i, g in enumerate(games)]
        values = [1.0 if g["won"] else 0.0 for g in games]
        return weighted_average(values, weights) * 100
    
    home_win_pct = calc_weighted_wins(home_home_games)
    away_win_pct = calc_weighted_wins(away_away_games)
    
    # === H2H Percentages ===
    h2h_btts_pct = (sum(g["btts"] for g in h2h_games) / len(h2h_games) * 100) if h2h_games else None
    h2h_over_pct = (sum(g["over25"] for g in h2h_games) / len(h2h_games) * 100) if h2h_games else None
    h2h_home_win_pct = (sum(g["home_won"] for g in h2h_games) / len(h2h_games) * 100) if h2h_games else None
    h2h_away_win_pct = (sum(g["away_won"] for g in h2h_games) / len(h2h_games) * 100) if h2h_games else None
    
    # === Weighted Blend: 80% recent form, 20% H2H ===
    if h2h_btts_pct is not None and len(h2h_games) >= 3:
        btts_prob = btts_prob_recent * (1 - H2H_WEIGHT) + h2h_btts_pct * H2H_WEIGHT
        over_prob = over_prob_recent * (1 - H2H_WEIGHT) + h2h_over_pct * H2H_WEIGHT
        home_win_prob = home_win_pct * (1 - H2H_WEIGHT) + h2h_home_win_pct * H2H_WEIGHT
        away_win_prob = away_win_pct * (1 - H2H_WEIGHT) + h2h_away_win_pct * H2H_WEIGHT
    else:
        btts_prob = btts_prob_recent
        over_prob = over_prob_recent
        home_win_prob = home_win_pct
        away_win_prob = away_win_pct
    
    # === Apply Dynamic Home Advantage Based on Opponent ===
    # Stronger opponent = less home advantage
    opponent_tier = get_opponent_tier(away_id, league_id, standings)
    if opponent_tier == "bottom":
        home_boost = HOME_ADVANTAGE_BOOST * 1.3  # 13% vs weak teams
    elif opponent_tier == "lower":
        home_boost = HOME_ADVANTAGE_BOOST * 1.1  # 11%
    elif opponent_tier == "upper":
        home_boost = HOME_ADVANTAGE_BOOST * 0.9  # 9%
    else:  # top
        home_boost = HOME_ADVANTAGE_BOOST * 0.6  # 6% vs top teams
    
    home_win_prob = min(95.0, home_win_prob + home_boost)
    away_win_prob = max(5.0, away_win_prob - (home_boost * 0.3))
    
    return {
        "btts_prob": round(btts_prob, 2),
        "over_prob": round(over_prob, 2),
        "home_win_prob": round(home_win_prob, 2),
        "away_win_prob": round(away_win_prob, 2),
        "home_games_count": len(home_home_games),
        "away_games_count": len(away_away_games),
        "h2h_count": len(h2h_games),
    }

# =========================
# MAIN
# =========================
def main():
    lines = []
    lines.append(f"Running improved historical model for fixtures on {TODAY}")
    lines.append(f"(Opponent-adjusted form + improved probability math)")
    lines.append("")
    lines.append(f"Fetching fixtures on {TODAY}...")

    df_fx = fetch_fixtures_for_today()
    lines.append(f"Fixtures found: {len(df_fx)}")
    lines.append("")

    if df_fx.empty:
        print("\n".join(lines))
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Process each fixture
    btts_rows, over_rows, win_rows = [], [], []

    for idx, fx in df_fx.iterrows():
        hid, aid = int(fx["home_id"]), int(fx["away_id"])
        lid = int(fx["league_id"])
        
        # Calculate probabilities
        probs = calculate_probabilities(hid, aid, lid)
        
        fixture_str = unidecode(f"{fx['home_team']} vs {fx['away_team']}")
        league_str = unidecode(str(fx["league"]))
        ko_str = fmt_time_eu(fx["kickoff"])
        
        # BTTS picks
        if probs["btts_prob"] >= MIN_BTTS_PROB:
            btts_rows.append({
                "kickoff": ko_str,
                "fixture": fixture_str,
                "league": league_str,
                "region": fx["region"],
                "prob": probs["btts_prob"],
                "confidence": confidence_label(probs["btts_prob"]),
            })
        
        # Over 2.5 picks
        if probs["over_prob"] >= MIN_OVER_PROB:
            over_rows.append({
                "kickoff": ko_str,
                "fixture": fixture_str,
                "league": league_str,
                "region": fx["region"],
                "prob": probs["over_prob"],
                "confidence": confidence_label(probs["over_prob"]),
            })
        
        # Win picks
        if probs["home_win_prob"] >= MIN_WIN_PROB or probs["away_win_prob"] >= MIN_WIN_PROB:
            if probs["home_win_prob"] > probs["away_win_prob"]:
                pick_team = unidecode(str(fx["home_team"]))
                win_prob = probs["home_win_prob"]
            else:
                pick_team = unidecode(str(fx["away_team"]))
                win_prob = probs["away_win_prob"]
            
            win_rows.append({
                "kickoff": ko_str,
                "fixture": fixture_str,
                "league": league_str,
                "region": fx["region"],
                "pick_team": pick_team,
                "prob": win_prob,
                "confidence": confidence_label(win_prob),
            })
        
        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df_fx)} fixtures...")

    df_btts = pd.DataFrame(btts_rows)
    df_over = pd.DataFrame(over_rows)
    df_win = pd.DataFrame(win_rows)
    
    # Remove U21 & EFL Trophy
    for df in (df_btts, df_over, df_win):
        if not df.empty:
            df.drop(df[df["fixture"].str.contains("U21", case=False, na=False)].index, inplace=True)
            df.drop(df[df["league"].str.contains("EFL Trophy", case=False, na=False)].index, inplace=True)

    # ===== OUTPUT SECTIONS =====
    total_fixtures = len(df_fx)
    total_btts = len(df_btts)
    total_win = len(df_win)
    total_over = len(df_over)

    lines.append("=======================================")
    lines.append(f"⚽ Improved Historical Model - {TODAY}")
    lines.append(f"Fixtures: {total_fixtures} | BTTS: {total_btts} | Win: {total_win} | Over 2.5: {total_over}")
    lines.append("=======================================")
    lines.append("")

    def conf_emoji(pct):
        if pct >= 80:
            return "🔥🔥"
        elif pct >= 70:
            return "🔥"
        elif pct >= 55:
            return "⚖️"
        else:
            return "❄️"

    def fmt_btts_line(r):
        return f"• {r['kickoff']:<8} | {r['fixture']:<28} | {r['prob']:>5.1f}% {conf_emoji(r['prob'])} | {r['league']}"

    def fmt_over_line(r):
        return f"• {r['kickoff']:<8} | {r['fixture']:<28} | {r['prob']:>5.1f}% {conf_emoji(r['prob'])} | {r['league']}"

    def fmt_win_line(r):
        return f"• {r['kickoff']:<8} | {r['fixture']:<28} | {r['pick_team']:<16} | {r['prob']:>5.1f}% {conf_emoji(r['prob'])} | {r['league']}"

    def print_section_header(title):
        lines.append("=======================================")
        lines.append(title)
        lines.append("=======================================")

    def add_list_or_none(df, formatter, limit, none_msg="No qualifying picks"):
        if df is None or df.empty:
            lines.append(none_msg)
            lines.append("")
            return
        for _, r in df.head(limit).iterrows():
            lines.append(formatter(r))
        lines.append("")

    # --- BTTS ---
    print_section_header(f"BTTS - England (≥{MIN_BTTS_PROB}%)")
    eng_btts = df_btts[df_btts["region"] == "England"].sort_values(["prob"], ascending=False)
    add_list_or_none(eng_btts, fmt_btts_line, 15)

    print_section_header(f"BTTS - Scotland (≥{MIN_BTTS_PROB}%)")
    sco_btts = df_btts[df_btts["region"] == "Scotland"].sort_values(["prob"], ascending=False)
    add_list_or_none(sco_btts, fmt_btts_line, 10)

    print_section_header(f"BTTS - Germany (≥{MIN_BTTS_PROB}%)")
    ger_btts = df_btts[df_btts["region"] == "Germany"].sort_values(["prob"], ascending=False)
    add_list_or_none(ger_btts, fmt_btts_line, 10)

    print_section_header(f"BTTS - Europe (≥{MIN_BTTS_PROB}%)")
    eur_btts = df_btts[df_btts["region"] == "Europe"].sort_values(["prob"], ascending=False)
    add_list_or_none(eur_btts, fmt_btts_line, 10)

    # --- WIN PICKS ---
    print_section_header(f"Win Picks - England (≥{MIN_WIN_PROB}%)")
    eng_win = df_win[df_win["region"] == "England"].sort_values(["prob"], ascending=False)
    add_list_or_none(eng_win, fmt_win_line, 15)

    print_section_header(f"Top Combined Win Picks (≥{MIN_WIN_PROB}%)")
    all_win = df_win.sort_values(["prob"], ascending=False)
    add_list_or_none(all_win, fmt_win_line, 20)

    # --- OVER 2.5 ---
    print_section_header(f"Top Combined Over 2.5 Picks (≥{MIN_OVER_PROB}%)")
    all_over = df_over.sort_values(["prob"], ascending=False)
    add_list_or_none(all_over, fmt_over_line, 20)

    # Footer
    lines.append("=======================================")
    lines.append("Model run complete ✅")
    lines.append(f"Generated: {dt.now(UTC).strftime('%a %d %b %Y %H:%M UTC')}")
    lines.append("=======================================")
    
    with open("output_debug.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    return df_btts, df_win, df_over

if __name__ == "__main__":
    main()