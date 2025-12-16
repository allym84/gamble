# xg_poisson_model.py
# Professional xG-based Poisson model using real expected goals data
# Mathematically consistent probabilities across all markets

import os
import sys
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime as dt, UTC
from unidecode import unidecode
from dotenv import load_dotenv

# Force unbuffered UTF-8 output
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

load_dotenv()

def fetch_team_league(team_id: int, season: int):
    """Fetch which league a team primarily plays in"""
    # Try to find their league from fixtures this season
    params = {"team": team_id, "season": season, "last": 5}
    try:
        r = requests.get(f"{BASE_URL}/fixtures", params=params, headers=HEADERS, timeout=12)
        fixtures = r.json().get("response", [])
        if fixtures:
            # Return the most common league they play in
            leagues = [f["league"]["id"] for f in fixtures]
            return max(set(leagues), key=leagues.count)
    except Exception:
        pass
    return None

def get_division_strength(league_id: int) -> float:
    """Get relative strength of a division"""
    return DIVISION_STRENGTH.get(league_id, 0.70)  # Default to Championship level

# =========================
# CONFIG
# =========================
API_KEY = os.getenv("APIFOOTBALL_API_KEY")
if not API_KEY:
    raise ValueError("Missing APIFOOTBALL_API_KEY in .env")

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# Auto season
TODAY = dt.now(UTC).date()
SEASON = TODAY.year if TODAY.month >= 7 else TODAY.year - 1

# Leagues
LEAGUE_GROUPS = {
    "England": [39, 40, 41, 42, 45, 48],
    "Scotland": [179, 180],
    "Germany": [78],
    "Europe": [2, 3, 848],
}

# Model settings
LAST_N_GAMES = 10
HOME_ADVANTAGE = 0.25      # Add 0.25 xG for home team
RECENCY_DECAY = 0.88       # Exponential decay
MAX_GOALS = 8              # For Poisson calculations

# Display thresholds
MIN_BTTS_PROB = 45.0
MIN_OVER_PROB = 50.0
MIN_WIN_PROB = 55.0

# Opponent strength tiers (based on league position)
OPPONENT_WEIGHTS = {
    "top": 1.35,
    "upper": 1.10,
    "lower": 0.92,
    "bottom": 0.75,
}

# Divisional strength adjustments for cup games
DIVISION_STRENGTH = {
    39: 1.00,   # Premier League (baseline)
    40: 0.70,   # Championship
    41: 0.45,   # League One
    42: 0.30,   # League Two
    179: 0.80,  # Scottish Premiership
    180: 0.55,  # Scottish Championship
    78: 0.95,   # Bundesliga
    2: 1.05,    # Champions League (top teams)
    3: 1.00,    # Europa League
    848: 0.95,  # Europa Conference League
}

# Cup competitions where divisional adjustments apply
CUP_COMPETITIONS = [45, 48]  # FA Cup, EFL Cup

# Cache
STANDINGS_CACHE = {}

# =========================
# POISSON HELPERS
# =========================
def poisson_pmf(lam, k):
    """Poisson probability mass function"""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def calculate_poisson_probabilities(xg_home, xg_away):
    """
    Calculate all market probabilities from expected goals using Poisson distribution.
    This ensures mathematical consistency across all markets.
    """
    # Build probability matrix for all scorelines
    p_home_win = 0.0
    p_away_win = 0.0
    p_draw = 0.0
    p_btts = 0.0
    p_over25 = 0.0
    p_over15 = 0.0
    p_under25 = 0.0
    
    scoreline_probs = []
    
    for home_goals in range(MAX_GOALS + 1):
        p_home = poisson_pmf(xg_home, home_goals)
        for away_goals in range(MAX_GOALS + 1):
            p_away = poisson_pmf(xg_away, away_goals)
            prob = p_home * p_away
            
            scoreline_probs.append({
                "score": f"{home_goals}-{away_goals}",
                "prob": prob * 100
            })
            
            # Win/Draw
            if home_goals > away_goals:
                p_home_win += prob
            elif away_goals > home_goals:
                p_away_win += prob
            else:
                p_draw += prob
            
            # BTTS
            if home_goals > 0 and away_goals > 0:
                p_btts += prob
            
            # Over/Under 2.5
            total = home_goals + away_goals
            if total > 2:
                p_over25 += prob
            else:
                p_under25 += prob
            
            # Over 1.5
            if total > 1:
                p_over15 += prob
    
    # Normalize (shouldn't be needed but just in case)
    total = p_home_win + p_draw + p_away_win
    if total > 0:
        p_home_win /= total
        p_draw /= total
        p_away_win /= total
    
    # Sort scorelines by probability
    scoreline_probs.sort(key=lambda x: x["prob"], reverse=True)
    
    return {
        "home_win": p_home_win * 100,
        "draw": p_draw * 100,
        "away_win": p_away_win * 100,
        "btts": p_btts * 100,
        "over25": p_over25 * 100,
        "over15": p_over15 * 100,
        "under25": p_under25 * 100,
        "top_scorelines": scoreline_probs[:5]  # Top 5 most likely
    }

def weighted_average(values, weights):
    """Calculate weighted average"""
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
    """Fetch and cache league standings"""
    if league_id in STANDINGS_CACHE:
        return STANDINGS_CACHE[league_id]
    
    params = {"league": league_id, "season": SEASON}
    try:
        r = requests.get(f"{BASE_URL}/standings", params=params, headers=HEADERS, timeout=12)
        standings = r.json().get("response", [])
        if not standings:
            return {}
        
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
    """Determine opponent strength tier"""
    if not standings or team_id not in standings:
        return "upper"
    
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

def fetch_xg_data_for_team(team_id: int, league_id: int, standings: dict, last_n: int = LAST_N_GAMES, venue: str = None):
    """
    Fetch last N games and extract xG data for both attack and defense.
    Returns opponent-weighted, recency-weighted xG rates.
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
        # Get fixtures
        r = requests.get(f"{BASE_URL}/fixtures", params=params, headers=HEADERS, timeout=12)
        fixtures = r.json().get("response", [])
        
        xg_for_list = []
        xg_against_list = []
        weights = []
        
        for idx, f in enumerate(fixtures):
            if f["fixture"]["status"]["short"] not in ["FT", "AET", "PEN"]:
                continue
            
            fixture_id = f["fixture"]["id"]
            is_home = f["teams"]["home"]["id"] == team_id
            opponent_id = f["teams"]["away"]["id"] if is_home else f["teams"]["home"]["id"]
            
            # Fetch xG statistics for this fixture
            stat_params = {"fixture": fixture_id}
            r_stats = requests.get(f"{BASE_URL}/fixtures/statistics", params=stat_params, headers=HEADERS, timeout=12)
            stats_response = r_stats.json().get("response", [])
            
            if not stats_response or len(stats_response) < 2:
                continue
            
            # Extract xG for both teams
            team_stats = None
            opponent_stats = None
            
            for team_stat in stats_response:
                if team_stat["team"]["id"] == team_id:
                    team_stats = team_stat["statistics"]
                else:
                    opponent_stats = team_stat["statistics"]
            
            if not team_stats or not opponent_stats:
                continue
            
            # Extract xG values
            xg_for = None
            xg_against = None
            
            for stat in team_stats:
                if stat["type"] == "expected_goals" and stat["value"] is not None:
                    try:
                        xg_for = float(stat["value"])
                    except (ValueError, TypeError):
                        pass
            
            for stat in opponent_stats:
                if stat["type"] == "expected_goals" and stat["value"] is not None:
                    try:
                        xg_against = float(stat["value"])
                    except (ValueError, TypeError):
                        pass
            
            if xg_for is None or xg_against is None:
                continue
            
            # Get opponent strength weight
            opp_tier = get_opponent_tier(opponent_id, league_id, standings)
            opp_weight = OPPONENT_WEIGHTS.get(opp_tier, 1.0)
            
            # Recency weight
            recency_weight = RECENCY_DECAY ** len(xg_for_list)
            
            # Combined weight
            combined_weight = recency_weight * opp_weight
            
            xg_for_list.append(xg_for)
            xg_against_list.append(xg_against)
            weights.append(combined_weight)
            
            if len(xg_for_list) >= last_n:
                break
        
        # Calculate weighted averages
        avg_xg_for = weighted_average(xg_for_list, weights) if xg_for_list else 1.2
        avg_xg_against = weighted_average(xg_against_list, weights) if xg_against_list else 1.2
        
        return {
            "xg_for": avg_xg_for,
            "xg_against": avg_xg_against,
            "games_count": len(xg_for_list)
        }
    
    except Exception as e:
        print(f"Error fetching xG for team {team_id}: {e}")
        return {
            "xg_for": 1.2,
            "xg_against": 1.2,
            "games_count": 0
        }

def calculate_match_xg(home_id: int, away_id: int, league_id: int):
    """
    Calculate expected goals for both teams in upcoming match.
    Uses opponent-weighted, recency-weighted xG from recent matches.
    Applies divisional adjustments for cup games.
    """
    standings = fetch_league_standings(league_id)
    
    # Check if this is a cup game
    is_cup = league_id in CUP_COMPETITIONS
    
    # If cup game, get each team's actual division
    home_division_strength = 1.0
    away_division_strength = 1.0
    
    if is_cup:
        home_league = fetch_team_league(home_id, SEASON)
        away_league = fetch_team_league(away_id, SEASON)
        
        if home_league:
            home_division_strength = get_division_strength(home_league)
        if away_league:
            away_division_strength = get_division_strength(away_league)
    
    # Fetch xG data for home team at home
    home_data = fetch_xg_data_for_team(home_id, league_id, standings, venue="home")
    
    # Fetch xG data for away team away
    away_data = fetch_xg_data_for_team(away_id, league_id, standings, venue="away")
    
    # Fallback to all games if insufficient venue-specific data
    if home_data["games_count"] < 5:
        home_data = fetch_xg_data_for_team(home_id, league_id, standings)
    if away_data["games_count"] < 5:
        away_data = fetch_xg_data_for_team(away_id, league_id, standings)
    
    # Calculate expected goals for this match
    home_xg = home_data["xg_for"] * (away_data["xg_against"] / 1.2)
    away_xg = away_data["xg_for"] * (home_data["xg_against"] / 1.2)
    
    # Apply divisional adjustments for cup games
    if is_cup:
        # Adjust xG based on division strength
        home_xg = home_xg * home_division_strength
        away_xg = away_xg * away_division_strength
    
    # Apply home advantage
    home_xg = home_xg * (1 + HOME_ADVANTAGE)
    
    # Ensure reasonable bounds
    home_xg = max(0.3, min(4.0, home_xg))
    away_xg = max(0.3, min(4.0, away_xg))
    
    return {
        "home_xg": round(home_xg, 2),
        "away_xg": round(away_xg, 2),
        "home_games": home_data["games_count"],
        "away_games": away_data["games_count"],
    }

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

# =========================
# MAIN
# =========================
def main():
    lines = []
    lines.append(f"xG Poisson Model - {TODAY}")
    lines.append("")

    df_fx = fetch_fixtures_for_today()
    
    if df_fx.empty:
        lines.append("No fixtures found")
        print("\n".join(lines))
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Process each fixture
    btts_rows, over_rows, win_rows = [], [], []
    predictions_for_tracking = {}  # For results tracker
    
    for idx, fx in df_fx.iterrows():
        hid, aid = int(fx["home_id"]), int(fx["away_id"])
        lid = int(fx["league_id"])
        
        # Calculate match xG
        xg_data = calculate_match_xg(hid, aid, lid)
        
        # Calculate all probabilities using Poisson
        probs = calculate_poisson_probabilities(xg_data["home_xg"], xg_data["away_xg"])
        
        fixture_str = unidecode(f"{fx['home_team']} vs {fx['away_team']}")
        league_str = unidecode(str(fx["league"]))
        ko_str = fmt_time_eu(fx["kickoff"])
        
        data_quality = f"xG:{xg_data['home_xg']:.2f}-{xg_data['away_xg']:.2f} ({xg_data['home_games']}+{xg_data['away_games']})"
        
        # Save for results tracking
        predictions_for_tracking[str(fx["fixture_id"])] = {
            "home_team": unidecode(str(fx["home_team"])),
            "away_team": unidecode(str(fx["away_team"])),
            "league": league_str,
            "home_win_prob": round(probs["home_win"], 2),
            "away_win_prob": round(probs["away_win"], 2),
            "draw_prob": round(probs["draw"], 2),
            "btts_prob": round(probs["btts"], 2),
            "over25_prob": round(probs["over25"], 2),
            "home_xg": xg_data["home_xg"],
            "away_xg": xg_data["away_xg"],
        }
        
        # BTTS picks
        if probs["btts"] >= MIN_BTTS_PROB:
            btts_rows.append({
                "kickoff": ko_str,
                "fixture": fixture_str,
                "league": league_str,
                "region": fx["region"],
                "prob": probs["btts"],
                "confidence": confidence_label(probs["btts"]),
                "xg": data_quality,
            })
        
        # Over 2.5 picks
        if probs["over25"] >= MIN_OVER_PROB:
            over_rows.append({
                "kickoff": ko_str,
                "fixture": fixture_str,
                "league": league_str,
                "region": fx["region"],
                "prob": probs["over25"],
                "confidence": confidence_label(probs["over25"]),
                "xg": data_quality,
            })
        
        # Win picks
        if probs["home_win"] >= MIN_WIN_PROB or probs["away_win"] >= MIN_WIN_PROB:
            if probs["home_win"] > probs["away_win"]:
                pick_team = unidecode(str(fx["home_team"]))
                win_prob = probs["home_win"]
            else:
                pick_team = unidecode(str(fx["away_team"]))
                win_prob = probs["away_win"]
            
            win_rows.append({
                "kickoff": ko_str,
                "fixture": fixture_str,
                "league": league_str,
                "region": fx["region"],
                "pick_team": pick_team,
                "prob": win_prob,
                "confidence": confidence_label(win_prob),
                "xg": data_quality,
                "top_score": probs["top_scorelines"][0]["score"] if probs["top_scorelines"] else "N/A",
            })
        
        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"Processing... {idx + 1}/{len(df_fx)}")

    df_btts = pd.DataFrame(btts_rows) if btts_rows else pd.DataFrame()
    df_over = pd.DataFrame(over_rows) if over_rows else pd.DataFrame()
    df_win = pd.DataFrame(win_rows) if win_rows else pd.DataFrame()
    
    # Remove U21 & EFL Trophy
    for df in (df_btts, df_over, df_win):
        if not df.empty and "fixture" in df.columns:
            df.drop(df[df["fixture"].str.contains("U21", case=False, na=False)].index, inplace=True)
        if not df.empty and "league" in df.columns:
            df.drop(df[df["league"].str.contains("EFL Trophy", case=False, na=False)].index, inplace=True)

    # ===== CREATE TOP PICKS =====
    # Combine all markets and rank by probability * confidence
    top_picks = []
    
    # Add win picks
    for _, r in df_win.iterrows():
        top_picks.append({
            "kickoff": r["kickoff"],
            "fixture": r["fixture"],
            "league": r["league"],
            "market": f"Win: {r['pick_team']}",
            "prob": r["prob"],
            "score": r["top_score"],
            "xg": r["xg"],
            "sort_score": r["prob"],  # Higher is better
        })
    
    # Add BTTS picks (only high confidence)
    for _, r in df_btts[df_btts["prob"] >= 60].iterrows():
        top_picks.append({
            "kickoff": r["kickoff"],
            "fixture": r["fixture"],
            "league": r["league"],
            "market": "BTTS: Yes",
            "prob": r["prob"],
            "score": "N/A",
            "xg": r["xg"],
            "sort_score": r["prob"] * 0.95,  # Slightly lower weight than wins
        })
    
    # Add Over 2.5 picks (only high confidence)
    for _, r in df_over[df_over["prob"] >= 60].iterrows():
        top_picks.append({
            "kickoff": r["kickoff"],
            "fixture": r["fixture"],
            "league": r["league"],
            "market": "Over 2.5",
            "prob": r["prob"],
            "score": "N/A",
            "xg": r["xg"],
            "sort_score": r["prob"] * 0.95,  # Slightly lower weight than wins
        })
    
    # Sort by probability and take top 10
    df_top = pd.DataFrame(top_picks).sort_values("sort_score", ascending=False).head(10) if top_picks else pd.DataFrame()

    # ===== OUTPUT =====
    lines.append("=======================================")
    lines.append(f"⚽ PREDICTIONS - {TODAY}")
    lines.append("=======================================")
    lines.append("")

    # Define helper functions first
    def conf_emoji(pct):
        if pct >= 80: return "🔥🔥"
        elif pct >= 70: return "🔥"
        elif pct >= 55: return "⚖️"
        else: return "❄️"

    def fmt_btts_line(r):
        return f"{r['kickoff']} | {r['fixture']:<28} | {r['prob']:>5.1f}% {conf_emoji(r['prob'])}"

    def fmt_over_line(r):
        return f"{r['kickoff']} | {r['fixture']:<28} | {r['prob']:>5.1f}% {conf_emoji(r['prob'])}"

    def fmt_win_line(r):
        return f"{r['kickoff']} | {r['fixture']:<28} | {r['pick_team']:<16} {r['prob']:>5.1f}% {conf_emoji(r['prob'])} [{r['top_score']}]"

    def print_section_header_sub(title):
        lines.append("")
        lines.append(title)
        lines.append("---------------------------------------")

    def add_list_or_none(df, formatter, limit, none_msg="None"):
        if df is None or df.empty:
            lines.append(none_msg)
            lines.append("")
            return
        for _, r in df.head(limit).iterrows():
            lines.append(formatter(r))
        lines.append("")

    # ===== TOP PICKS SECTION (SHOW FIRST) =====
    def print_section_header(title):
        lines.append("=======================================")
        lines.append(title)
        lines.append("=======================================")

    print_section_header("🔥 TOP 10 PICKS")
    if not df_top.empty:
        for idx, r in df_top.iterrows():
            score_str = f" [{r['score']}]" if r['score'] != "N/A" else ""
            lines.append(f"{idx+1}. {r['kickoff']} | {r['fixture']}")
            lines.append(f"   {r['market']} - {r['prob']:.1f}% {conf_emoji(r['prob'])}{score_str}")
            lines.append("")
    else:
        lines.append("No picks")
        lines.append("")

    # BTTS
    print_section_header_sub(f"BTTS - England")
    if not df_btts.empty and "region" in df_btts.columns:
        eng_btts = df_btts[df_btts["region"] == "England"].sort_values(["prob"], ascending=False)
        add_list_or_none(eng_btts, fmt_btts_line, 15)
    else:
        lines.append("None")
        lines.append("")

    print_section_header_sub(f"BTTS - Scotland")
    if not df_btts.empty and "region" in df_btts.columns:
        sco_btts = df_btts[df_btts["region"] == "Scotland"].sort_values(["prob"], ascending=False)
        add_list_or_none(sco_btts, fmt_btts_line, 10)
    else:
        lines.append("None")
        lines.append("")

    print_section_header_sub(f"BTTS - Germany")
    if not df_btts.empty and "region" in df_btts.columns:
        ger_btts = df_btts[df_btts["region"] == "Germany"].sort_values(["prob"], ascending=False)
        add_list_or_none(ger_btts, fmt_btts_line, 10)
    else:
        lines.append("None")
        lines.append("")

    print_section_header_sub(f"BTTS - Europe")
    if not df_btts.empty and "region" in df_btts.columns:
        eur_btts = df_btts[df_btts["region"] == "Europe"].sort_values(["prob"], ascending=False)
        add_list_or_none(eur_btts, fmt_btts_line, 10)
    else:
        lines.append("None")
        lines.append("")

    # WIN
    print_section_header_sub(f"Win - England")
    if not df_win.empty and "region" in df_win.columns:
        eng_win = df_win[df_win["region"] == "England"].sort_values(["prob"], ascending=False)
        add_list_or_none(eng_win, fmt_win_line, 15)
    else:
        lines.append("None")
        lines.append("")

    print_section_header_sub(f"Win - Combined")
    if not df_win.empty and "region" in df_win.columns:
        all_win = df_win.sort_values(["prob"], ascending=False)
        add_list_or_none(all_win, fmt_win_line, 20)
    else:
        lines.append("None")
        lines.append("")

    # OVER 2.5
    print_section_header_sub(f"Over 2.5 - Combined")
    if not df_over.empty and "region" in df_over.columns:
        all_over = df_over.sort_values(["prob"], ascending=False)
        add_list_or_none(all_over, fmt_over_line, 20)
    else:
        lines.append("None")
        lines.append("")

    lines.append("=======================================")
    lines.append("✅ Complete")
    lines.append("=======================================")
    
    with open("output_debug.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    
    # Save predictions for tracking (optional - comment out if results_tracker.py not available)
    # try:
    #     from results_tracker import save_predictions
    #     save_predictions(TODAY, predictions_for_tracking)
    # except Exception as e:
    #     print(f"Note: Could not save predictions for tracking: {e}")
    
    return df_btts, df_win, df_over

if __name__ == "__main__":
    main()