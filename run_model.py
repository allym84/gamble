# xg_poisson_model.py
# Professional xG-based Poisson model with caching, validation, and quality checks
# Version 2.0 - Optimized and production-ready

import os
import sys
import math
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime as dt, UTC
from unidecode import unidecode
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
HOME_ADVANTAGE = 0.25
RECENCY_DECAY = 0.88
MAX_GOALS = 8

# Display thresholds
MIN_BTTS_PROB = 45.0
MIN_OVER_PROB = 50.0
MIN_WIN_PROB = 55.0

# Data quality thresholds
MIN_GAMES_FOR_HIGH_CONFIDENCE = 8
MIN_GAMES_FOR_MEDIUM_CONFIDENCE = 5

# Opponent strength tiers
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
    2: 1.05,    # Champions League
    3: 1.00,    # Europa League
    848: 0.95,  # Europa Conference League
}

CUP_COMPETITIONS = [45, 48]

# Cache files
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
XG_CACHE_FILE = os.path.join(CACHE_DIR, f"xg_cache_{TODAY}.json")
STANDINGS_CACHE_FILE = os.path.join(CACHE_DIR, f"standings_cache_{TODAY}.json")
TEAM_LEAGUE_CACHE_FILE = os.path.join(CACHE_DIR, f"team_league_cache_{TODAY}.json")

# Global caches
xg_cache = {}
standings_cache = {}
team_league_cache = {}

# =========================
# CACHE MANAGEMENT
# =========================
def load_cache(filepath):
    """Load cache from file"""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load cache from {filepath}: {e}")
    return {}

def save_cache(filepath, data):
    """Save cache to file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning(f"Could not save cache to {filepath}: {e}")

def load_all_caches():
    """Load all caches at startup"""
    global xg_cache, standings_cache, team_league_cache
    xg_cache = load_cache(XG_CACHE_FILE)
    standings_cache = load_cache(STANDINGS_CACHE_FILE)
    team_league_cache = load_cache(TEAM_LEAGUE_CACHE_FILE)
    logger.info(f"Loaded caches: {len(xg_cache)} xG entries, {len(standings_cache)} standings, {len(team_league_cache)} team leagues")

def save_all_caches():
    """Save all caches at end"""
    save_cache(XG_CACHE_FILE, xg_cache)
    save_cache(STANDINGS_CACHE_FILE, standings_cache)
    save_cache(TEAM_LEAGUE_CACHE_FILE, team_league_cache)
    logger.info("Saved all caches")

# =========================
# POISSON HELPERS
# =========================
def poisson_pmf(lam, k):
    """Poisson probability mass function"""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def calculate_poisson_probabilities(xg_home, xg_away):
    """Calculate all market probabilities from expected goals"""
    p_home_win = 0.0
    p_away_win = 0.0
    p_draw = 0.0
    p_btts = 0.0
    p_over25 = 0.0
    
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
            
            if home_goals > away_goals:
                p_home_win += prob
            elif away_goals > home_goals:
                p_away_win += prob
            else:
                p_draw += prob
            
            if home_goals > 0 and away_goals > 0:
                p_btts += prob
            
            if (home_goals + away_goals) > 2:
                p_over25 += prob
    
    # Normalize
    total = p_home_win + p_draw + p_away_win
    if total > 0:
        p_home_win /= total
        p_draw /= total
        p_away_win /= total
    
    scoreline_probs.sort(key=lambda x: x["prob"], reverse=True)
    
    return {
        "home_win": p_home_win * 100,
        "draw": p_draw * 100,
        "away_win": p_away_win * 100,
        "btts": p_btts * 100,
        "over25": p_over25 * 100,
        "top_scorelines": scoreline_probs[:5]
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
# VALIDATION
# =========================
def validate_xg(xg_value, team_name, context=""):
    """Validate xG value and warn if suspicious"""
    if xg_value < 0.2:
        logger.warning(f"⚠️  Very low xG for {team_name} ({context}): {xg_value:.2f}")
        return False
    if xg_value > 3.5:
        logger.warning(f"⚠️  Very high xG for {team_name} ({context}): {xg_value:.2f}")
        return False
    return True

def get_data_quality(games_count):
    """Return data quality label based on sample size"""
    if games_count >= MIN_GAMES_FOR_HIGH_CONFIDENCE:
        return "HIGH"
    elif games_count >= MIN_GAMES_FOR_MEDIUM_CONFIDENCE:
        return "MEDIUM"
    else:
        return "LOW"

# =========================
# STANDINGS & OPPONENT STRENGTH
# =========================
def fetch_league_standings(league_id: int):
    """Fetch and cache league standings"""
    cache_key = str(league_id)
    
    if cache_key in standings_cache:
        return standings_cache[cache_key]
    
    params = {"league": league_id, "season": SEASON}
    try:
        logger.info(f"Fetching standings for league {league_id}")
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
        
        standings_cache[cache_key] = standings_dict
        return standings_dict
    except Exception as e:
        logger.error(f"Error fetching standings for league {league_id}: {e}")
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

def fetch_team_league(team_id: int, season: int):
    """Fetch which league a team primarily plays in (cached)"""
    cache_key = str(team_id)
    
    if cache_key in team_league_cache:
        return team_league_cache[cache_key]
    
    params = {"team": team_id, "season": season, "last": 5}
    try:
        logger.info(f"Fetching primary league for team {team_id}")
        r = requests.get(f"{BASE_URL}/fixtures", params=params, headers=HEADERS, timeout=12)
        fixtures = r.json().get("response", [])
        if fixtures:
            leagues = [f["league"]["id"] for f in fixtures]
            primary_league = max(set(leagues), key=leagues.count)
            team_league_cache[cache_key] = primary_league
            return primary_league
    except Exception as e:
        logger.error(f"Error fetching league for team {team_id}: {e}")
    return None

def get_division_strength(league_id: int) -> float:
    """Get relative strength of a division"""
    return DIVISION_STRENGTH.get(league_id, 0.70)

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
            except Exception as e:
                logger.error(f"Error fetching fixtures for league {lid}: {e}")
                continue
    return pd.DataFrame(all_rows)

def fetch_xg_data_for_team(team_id: int, league_id: int, standings: dict, last_n: int = LAST_N_GAMES, venue: str = None):
    """Fetch last N games and extract xG data (cached)"""
    cache_key = f"{team_id}_{league_id}_{venue}_{last_n}"
    
    if cache_key in xg_cache:
        return xg_cache[cache_key]
    
    params = {
        "team": team_id,
        "league": league_id,
        "season": SEASON,
        "last": last_n * 3,
    }
    if venue:
        params["venue"] = venue
    
    try:
        logger.info(f"Fetching xG data for team {team_id} (league {league_id}, venue {venue})")
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
            
            # Fetch xG statistics
            stat_params = {"fixture": fixture_id}
            r_stats = requests.get(f"{BASE_URL}/fixtures/statistics", params=stat_params, headers=HEADERS, timeout=12)
            stats_response = r_stats.json().get("response", [])
            
            if not stats_response or len(stats_response) < 2:
                continue
            
            team_stats = None
            opponent_stats = None
            
            for team_stat in stats_response:
                if team_stat["team"]["id"] == team_id:
                    team_stats = team_stat["statistics"]
                else:
                    opponent_stats = team_stat["statistics"]
            
            if not team_stats or not opponent_stats:
                continue
            
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
        
        result = {
            "xg_for": avg_xg_for,
            "xg_against": avg_xg_against,
            "games_count": len(xg_for_list)
        }
        
        xg_cache[cache_key] = result
        return result
    
    except Exception as e:
        logger.error(f"Error fetching xG for team {team_id}: {e}")
        return {
            "xg_for": 1.2,
            "xg_against": 1.2,
            "games_count": 0
        }

def calculate_match_xg(home_id: int, away_id: int, home_name: str, away_name: str, league_id: int):
    """Calculate expected goals with validation and quality checks"""
    standings = fetch_league_standings(league_id)
    
    # Check if cup game
    is_cup = league_id in CUP_COMPETITIONS
    
    home_division_strength = 1.0
    away_division_strength = 1.0
    
    if is_cup:
        home_league = fetch_team_league(home_id, SEASON)
        away_league = fetch_team_league(away_id, SEASON)
        
        if home_league:
            home_division_strength = get_division_strength(home_league)
        if away_league:
            away_division_strength = get_division_strength(away_league)
    
    # Fetch xG data
    home_data = fetch_xg_data_for_team(home_id, league_id, standings, venue="home")
    away_data = fetch_xg_data_for_team(away_id, league_id, standings, venue="away")
    
    # Fallback to all games if insufficient data
    if home_data["games_count"] < 5:
        home_data = fetch_xg_data_for_team(home_id, league_id, standings)
    if away_data["games_count"] < 5:
        away_data = fetch_xg_data_for_team(away_id, league_id, standings)
    
    # Calculate xG
    home_xg = home_data["xg_for"] * (away_data["xg_against"] / 1.2)
    away_xg = away_data["xg_for"] * (home_data["xg_against"] / 1.2)
    
    # Apply divisional adjustments
    if is_cup:
        home_xg *= home_division_strength
        away_xg *= away_division_strength
    
    # Apply home advantage
    home_xg *= (1 + HOME_ADVANTAGE)
    
    # Bounds
    home_xg = max(0.3, min(4.0, home_xg))
    away_xg = max(0.3, min(4.0, away_xg))
    
    # Validate
    validate_xg(home_xg, home_name, "home")
    validate_xg(away_xg, away_name, "away")
    
    # Data quality
    data_quality = get_data_quality(min(home_data["games_count"], away_data["games_count"]))
    
    return {
        "home_xg": round(home_xg, 2),
        "away_xg": round(away_xg, 2),
        "home_games": home_data["games_count"],
        "away_games": away_data["games_count"],
        "data_quality": data_quality,
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
    # Load caches
    load_all_caches()
    
    lines = []
    lines.append(f"xG Poisson Model - {TODAY}")
    lines.append("")

    logger.info("Fetching fixtures...")
    df_fx = fetch_fixtures_for_today()
    
    if df_fx.empty:
        lines.append("No fixtures found")
        print("\n".join(lines))
        save_all_caches()
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    logger.info(f"Processing {len(df_fx)} fixtures...")
    
    # Process fixtures
    btts_rows, over_rows, win_rows = [], [], []
    predictions_for_tracking = {}
    low_quality_count = 0
    
    for idx, fx in df_fx.iterrows():
        hid, aid = int(fx["home_id"]), int(fx["away_id"])
        lid = int(fx["league_id"])
        
        # Calculate xG
        xg_data = calculate_match_xg(hid, aid, fx["home_team"], fx["away_team"], lid)
        
        # Track low quality
        if xg_data["data_quality"] == "LOW":
            low_quality_count += 1
        
        # Calculate probabilities
        probs = calculate_poisson_probabilities(xg_data["home_xg"], xg_data["away_xg"])
        
        fixture_str = unidecode(f"{fx['home_team']} vs {fx['away_team']}")
        league_str = unidecode(str(fx["league"]))
        ko_str = fmt_time_eu(fx["kickoff"])
        
        data_quality_str = f"xG:{xg_data['home_xg']:.2f}-{xg_data['away_xg']:.2f}"
        if xg_data["data_quality"] == "LOW":
            data_quality_str += " ⚠️ LOW DATA"
        
        # Save for tracking
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
                "xg": data_quality_str,
                "quality": xg_data["data_quality"],
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
                "xg": data_quality_str,
                "quality": xg_data["data_quality"],
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
                "xg": data_quality_str,
                "top_score": probs["top_scorelines"][0]["score"] if probs["top_scorelines"] else "N/A",
                "quality": xg_data["data_quality"],
            })
        
        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{len(df_fx)}")

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
    top_picks = []
    
    if not df_win.empty:
        for _, r in df_win.iterrows():
            top_picks.append({
                "kickoff": r["kickoff"],
                "fixture": r["fixture"],
                "league": r["league"],
                "market": r["pick_team"],
                "prob": r["prob"],
                "score": r["top_score"],
                "sort_score": r["prob"],
                "quality": r["quality"],
            })
    
    if not df_btts.empty and "prob" in df_btts.columns:
        for _, r in df_btts[df_btts["prob"] >= 60].iterrows():
            top_picks.append({
                "kickoff": r["kickoff"],
                "fixture": r["fixture"],
                "league": r["league"],
                "market": "BTTS",
                "prob": r["prob"],
                "score": "N/A",
                "sort_score": r["prob"] * 0.95,
                "quality": r["quality"],
            })
    
    if not df_over.empty and "prob" in df_over.columns:
        for _, r in df_over[df_over["prob"] >= 60].iterrows():
            top_picks.append({
                "kickoff": r["kickoff"],
                "fixture": r["fixture"],
                "league": r["league"],
                "market": "Over 2.5",
                "prob": r["prob"],
                "score": "N/A",
                "sort_score": r["prob"] * 0.95,
                "quality": r["quality"],
            })
    
    df_top = pd.DataFrame(top_picks).sort_values("sort_score", ascending=False).head(10) if top_picks else pd.DataFrame()

    # ===== OUTPUT =====
    # Summary stats
    high_conf = sum(1 for _, r in df_top.iterrows() if r["prob"] >= 70) if not df_top.empty else 0
    medium_conf = sum(1 for _, r in df_top.iterrows() if 55 <= r["prob"] < 70) if not df_top.empty else 0
    
    lines.append("=======================================")
    lines.append(f"⚽ PREDICTIONS - {TODAY}")
    lines.append(f"📊 {len(df_fx)} fixtures | {high_conf} high confidence | {medium_conf} medium")
    if low_quality_count > 0:
        lines.append(f"⚠️  {low_quality_count} predictions with limited data")
    lines.append("=======================================")
    lines.append("")

    def conf_emoji(pct):
        if pct >= 80: return "🔥🔥"
        elif pct >= 70: return "🔥"
        elif pct >= 55: return "⚖️"
        else: return "❄️"

    def fmt_btts_line(r):
        quality_flag = " ⚠️" if r.get("quality") == "LOW" else ""
        return f"{r['kickoff']} | {r['fixture']:<30} {r['prob']:>5.1f}% {conf_emoji(r['prob'])}{quality_flag}"

    def fmt_over_line(r):
        quality_flag = " ⚠️" if r.get("quality") == "LOW" else ""
        return f"{r['kickoff']} | {r['fixture']:<30} {r['prob']:>5.1f}% {conf_emoji(r['prob'])}{quality_flag}"

    def fmt_win_line(r):
        quality_flag = " ⚠️" if r.get("quality") == "LOW" else ""
        return f"{r['kickoff']} | {r['fixture']:<30} {r['pick_team']:<16} {r['prob']:>5.1f}% {conf_emoji(r['prob'])} [{r['top_score']}]{quality_flag}"

    def print_section_header(title):
        lines.append("=======================================")
        lines.append(title)
        lines.append("=======================================")

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

    # TOP 10
    print_section_header("🔥 TOP 10 PICKS")
    if not df_top.empty:
        for idx, r in df_top.iterrows():
            score_str = f" [{r['score']}]" if r['score'] != "N/A" else ""
            quality_flag = " ⚠️ Low Data" if r["quality"] == "LOW" else ""
            lines.append(f"{idx+1}. {r['kickoff']} | {r['fixture']}")
            lines.append(f"   {r['market']} - {r['prob']:.1f}% {conf_emoji(r['prob'])}{score_str}{quality_flag}")
            lines.append(f"   {r['league']}")
            lines.append("")
    else:
        lines.append("No picks")
        lines.append("")

    # BTTS
    print_section_header_sub("BTTS - England")
    if not df_btts.empty and "region" in df_btts.columns:
        eng_btts = df_btts[df_btts["region"] == "England"].sort_values(["prob"], ascending=False)
        add_list_or_none(eng_btts, fmt_btts_line, 15)
    else:
        lines.append("None")
        lines.append("")

    print_section_header_sub("BTTS - Scotland")
    if not df_btts.empty and "region" in df_btts.columns:
        sco_btts = df_btts[df_btts["region"] == "Scotland"].sort_values(["prob"], ascending=False)
        add_list_or_none(sco_btts, fmt_btts_line, 10)
    else:
        lines.append("None")
        lines.append("")

    print_section_header_sub("BTTS - Germany")
    if not df_btts.empty and "region" in df_btts.columns:
        ger_btts = df_btts[df_btts["region"] == "Germany"].sort_values(["prob"], ascending=False)
        add_list_or_none(ger_btts, fmt_btts_line, 10)
    else:
        lines.append("None")
        lines.append("")

    print_section_header_sub("BTTS - Europe")
    if not df_btts.empty and "region" in df_btts.columns:
        eur_btts = df_btts[df_btts["region"] == "Europe"].sort_values(["prob"], ascending=False)
        add_list_or_none(eur_btts, fmt_btts_line, 10)
    else:
        lines.append("None")
        lines.append("")

    # WIN
    print_section_header_sub("Win - England")
    if not df_win.empty and "region" in df_win.columns:
        eng_win = df_win[df_win["region"] == "England"].sort_values(["prob"], ascending=False)
        add_list_or_none(eng_win, fmt_win_line, 15)
    else:
        lines.append("None")
        lines.append("")

    print_section_header_sub("Win - Combined")
    if not df_win.empty and "region" in df_win.columns:
        all_win = df_win.sort_values(["prob"], ascending=False)
        add_list_or_none(all_win, fmt_win_line, 20)
    else:
        lines.append("None")
        lines.append("")

    # OVER 2.5
    print_section_header_sub("Over 2.5 - Combined")
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
    
    # Save caches
    save_all_caches()
    logger.info("Model run complete")
    
    return df_btts, df_win, df_over

if __name__ == "__main__":
    main()