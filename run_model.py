# run_model.py
# Smart Elo + Poisson with odds + value picks
# Clean ASCII-only output for email & Telegram

import os
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime as dt, timedelta, UTC
from unidecode import unidecode
from dotenv import load_dotenv

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

# Included competitions (FULL NAMES kept from API; we exclude EFL Trophy and U21 fixtures)
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

# Model knobs
HFA = 1.10                 # home-field advantage multiplier
MAX_GOALS = 8              # Poisson grid size
ELO_START = 1500
HOME_ELO_BONUS = 60
ELO_FORM_BOOST = 12        # 12*(wins-losses)
ELO_GD_FACTOR  = 10.0      # 10*(GF - GA)

def elo_weight(played_total: int) -> float:
    if played_total <= 5:  return 0.75
    if played_total <= 15: return 0.55
    return 0.35

# Odds / value settings
BOOKMAKER_ID = 8           # Bet365
VALUE_THRESHOLD = 10.0     # show edges >= 10%

# =========================
# HELPERS
# =========================
def confidence_label(pct: float) -> str:
    if pct >= 80: return "High"
    if pct >= 60: return "Medium"
    return "Low"

def fmt_time_eu(iso_str: str) -> str:
    # "Wed 19:45" Europe/London
    try:
        ts = pd.to_datetime(iso_str, utc=True).tz_convert("Europe/London")
        return ts.strftime("%a %H:%M")
    except Exception:
        return iso_str

def poisson_pmf(lam, k):
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def poisson_match_probs(lh, la):
    p_home = p_draw = p_away = p_btts = p_over25 = 0.0
    for i in range(MAX_GOALS + 1):
        pi = poisson_pmf(lh, i)
        for j in range(MAX_GOALS + 1):
            pj = poisson_pmf(la, j)
            p = pi * pj
            if i > j: p_home += p
            elif i == j: p_draw += p
            else: p_away += p
            if i > 0 and j > 0: p_btts += p
            if i + j >= 3: p_over25 += p
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total; p_draw /= total; p_away /= total
    return p_home, p_draw, p_away, p_btts, p_over25

def implied_prob_from_odds(od):
    try:
        od = float(od)
        return 100.0 / od if od > 0 else None
    except (TypeError, ValueError):
        return None

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
                        "league": f["league"]["name"],  # full name from API
                        "region": region,
                        "kickoff": f["fixture"]["date"],
                    })
            except Exception:
                continue
    return pd.DataFrame(all_rows)

def fetch_team_stats(team_id: int, league_id: int):
    try:
        params = {"team": team_id, "league": league_id, "season": SEASON}
        r = requests.get(f"{BASE_URL}/teams/statistics", params=params, headers=HEADERS, timeout=12)
        js = r.json().get("response", {})
        if not js:
            return None

        team_name  = unidecode(js["team"]["name"])
        league_name = unidecode(js["league"]["name"])
        fixtures   = js.get("fixtures", {})
        played     = fixtures.get("played", {}).get("total", 0) or 0
        wins       = fixtures.get("wins", {}).get("total", 0) or 0
        losses     = fixtures.get("loses", {}).get("total", 0) or 0

        goals   = js.get("goals", {})
        gf_home = float(goals.get("for", {}).get("average", {}).get("home", 0) or 0)
        ga_home = float(goals.get("against", {}).get("average", {}).get("home", 0) or 0)
        gf_away = float(goals.get("for", {}).get("average", {}).get("away", 0) or 0)
        ga_away = float(goals.get("against", {}).get("average", {}).get("away", 0) or 0)

        # simple form-elo proxy
        elo = ELO_START + ELO_FORM_BOOST * (wins - losses) + ELO_GD_FACTOR * ((gf_home + gf_away) - (ga_home + ga_away))

        return {
            "team_id": team_id,
            "team": team_name,
            "league_id": league_id,
            "league": league_name,
            "played": int(played),
            "gf_home": gf_home,
            "ga_home": ga_home,
            "gf_away": gf_away,
            "ga_away": ga_away,
            "elo": float(elo),
        }
    except Exception:
        return None

def fetch_odds(fixture_id: int):
    """Bet365 odds for Win, BTTS Yes, Over 2.5."""
    params = {"fixture": fixture_id, "bookmaker": BOOKMAKER_ID}
    try:
        r = requests.get(f"{BASE_URL}/odds", params=params, headers=HEADERS, timeout=12)
        resp = r.json().get("response", [])
        if not resp:
            return {}
        bets = resp[0].get("bookmakers", [])[0].get("bets", [])
        out = {}
        for b in bets:
            name = b.get("name", "")
            for v in b.get("values", []):
                if "Winner" in name:
                    if v["value"] == "Home": out["win_home"] = float(v["odd"])
                    if v["value"] == "Away": out["win_away"] = float(v["odd"])
                if "Both Teams Score" in name and v["value"] == "Yes":
                    out["btts"] = float(v["odd"])
                if "Over/Under" in name and v["value"] == "Over 2.5":
                    out["over25"] = float(v["odd"])
        return out
    except Exception:
        return {}

# =========================
# MAIN
# =========================
def main():
    lines = []
    lines.append(f"Running model for fixtures on {TODAY}")
    lines.append("")
    lines.append(f"Fetching fixtures on {TODAY}...")

    df_fx = fetch_fixtures_for_today()
    lines.append(f"Fixtures found: {len(df_fx)}")
    lines.append("")

    if df_fx.empty:
        print("\n".join(lines))
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Build team stats
    team_ids = sorted(list(set(df_fx["home_id"]).union(set(df_fx["away_id"]))))

    stat_rows = []
    for tid in team_ids:
        lid = df_fx.loc[(df_fx["home_id"] == tid) | (df_fx["away_id"] == tid), "league_id"].iloc[0]
        s = fetch_team_stats(int(tid), int(lid))
        if s:
            stat_rows.append(s)
    df_stats = pd.DataFrame(stat_rows)
    if df_stats.empty:
        print("\n".join(lines + ["No team statistics returned from API"]))
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # League base means (home/away goals)
    lg_home_avg = df_stats.groupby("league_id")["gf_home"].mean().to_dict()
    lg_away_avg = df_stats.groupby("league_id")["gf_away"].mean().to_dict()

    # Strengths
    def strengths(row):
        lid = int(row["league_id"])
        row["att_h"] = row["gf_home"] / max(lg_home_avg.get(lid, 1e-6), 1e-6)
        row["def_h"] = row["ga_home"] / max(lg_home_avg.get(lid, 1e-6), 1e-6)
        row["att_a"] = row["gf_away"] / max(lg_away_avg.get(lid, 1e-6), 1e-6)
        row["def_a"] = row["ga_away"] / max(lg_away_avg.get(lid, 1e-6), 1e-6)
        return row

    df_stats = df_stats.apply(strengths, axis=1)
    stats_by_team = {int(r["team_id"]): r for _, r in df_stats.iterrows()}

    # Per-fixture model
    match_rows, btts_rows, over_rows = [], [], []

    for _, fx in df_fx.iterrows():
        hid, aid = int(fx["home_id"]), int(fx["away_id"])
        if hid not in stats_by_team or aid not in stats_by_team:
            continue

        hs, as_ = stats_by_team[hid], stats_by_team[aid]
        lid = int(fx["league_id"])

        # Dynamic Elo weight by played volume
        wE = elo_weight(int(hs["played"] + as_["played"]) // 2)

        # Elo expectations
        elo_h = float(hs["elo"]) + HOME_ELO_BONUS
        elo_a = float(as_["elo"])
        exp_h = 1.0 / (1.0 + 10 ** ((elo_a - elo_h) / 400.0))
        exp_a = 1.0 - exp_h

        # Baseline lambdas
        lh_base = max(0.05, lg_home_avg.get(lid, 1.25)) * hs["att_h"] * as_["def_a"] * HFA
        la_base = max(0.05, lg_away_avg.get(lid, 1.10)) * as_["att_a"] * hs["def_h"]

        # Blend Elo into lambdas
        lh = float(np.clip(lh_base * ((1 - wE) + wE * exp_h), 0.05, 4.5))
        la = float(np.clip(la_base * ((1 - wE) + wE * exp_a), 0.05, 4.5))

        # Poisson matrix
        p_h, p_d, p_a, p_b, p_o = poisson_match_probs(lh, la)

        # Odds (Bet365)
        odds = fetch_odds(int(fx["fixture_id"]))

        # Strings
        fixture_str = unidecode(f"{fx['home_team']} vs {fx['away_team']}")
        league_str  = unidecode(str(fx["league"]))
        ko_str = fmt_time_eu(fx["kickoff"])

        match_rows.append({
            "fixture_id": int(fx["fixture_id"]),
            "fixture": fixture_str,
            "home_team": unidecode(str(fx["home_team"])),
            "away_team": unidecode(str(fx["away_team"])),
            "league": league_str,
            "region": fx["region"],
            "kickoff": ko_str,
            "home_win_prob": round(p_h * 100, 2),
            "away_win_prob": round(p_a * 100, 2),
            "win_prob": round(max(p_h, p_a) * 100, 2),
            "pick_team": unidecode(str(fx["home_team"] if p_h > p_a else fx["away_team"])),
            "confidence": confidence_label(max(p_h, p_a) * 100),
            "odds_win_home": odds.get("win_home"),
            "odds_win_away": odds.get("win_away"),
        })

        btts_rows.append({
            "fixture_id": int(fx["fixture_id"]),
            "fixture": fixture_str,
            "league": league_str,
            "region": fx["region"],
            "kickoff": ko_str,
            "btts_prob": round(p_b * 100, 2),
            "confidence": confidence_label(p_b * 100),
            "odds_btts": odds.get("btts"),
        })

        over_rows.append({
            "fixture_id": int(fx["fixture_id"]),
            "fixture": fixture_str,
            "league": league_str,
            "region": fx["region"],
            "kickoff": ko_str,
            "over25_prob": round(p_o * 100, 2),
            "confidence": confidence_label(p_o * 100),
            "odds_over25": odds.get("over25"),
        })

    df_match = pd.DataFrame(match_rows)
    df_btts  = pd.DataFrame(btts_rows)
    df_over  = pd.DataFrame(over_rows)

    # Remove U21 & EFL Trophy for quality
    for df in (df_match, df_btts, df_over):
        if not df.empty:
            df.drop(df[df["fixture"].str.contains("U21", case=False, na=False)].index, inplace=True)
            df.drop(df[df["league"].str.contains("EFL Trophy", case=False, na=False)].index, inplace=True)

    # ===== VALUE PICKS =====
    value_rows = []

    # Win market (home/away)
    for _, r in df_match.iterrows():
        ih = implied_prob_from_odds(r.get("odds_win_home"))
        ia = implied_prob_from_odds(r.get("odds_win_away"))

        if ih is not None:
            edge = r["home_win_prob"] - ih
            if edge >= VALUE_THRESHOLD:
                value_rows.append({
                    "kickoff": r["kickoff"],
                    "fixture": r["fixture"],
                    "market": "Win (Home)",
                    "prob": r["home_win_prob"],
                    "value_edge": round(edge, 2),
                    "league": r["league"],
                })
        if ia is not None:
            edge = r["away_win_prob"] - ia
            if edge >= VALUE_THRESHOLD:
                value_rows.append({
                    "kickoff": r["kickoff"],
                    "fixture": r["fixture"],
                    "market": "Win (Away)",
                    "prob": r["away_win_prob"],
                    "value_edge": round(edge, 2),
                    "league": r["league"],
                })

    # BTTS
    for _, r in df_btts.iterrows():
        ib = implied_prob_from_odds(r.get("odds_btts"))
        if ib is None: continue
        edge = r["btts_prob"] - ib
        if edge >= VALUE_THRESHOLD:
            value_rows.append({
                "kickoff": r["kickoff"],
                "fixture": r["fixture"],
                "market": "BTTS",
                "prob": r["btts_prob"],
                "value_edge": round(edge, 2),
                "league": r["league"],
            })

    # Over 2.5
    for _, r in df_over.iterrows():
        io = implied_prob_from_odds(r.get("odds_over25"))
        if io is None: continue
        edge = r["over25_prob"] - io
        if edge >= VALUE_THRESHOLD:
            value_rows.append({
                "kickoff": r["kickoff"],
                "fixture": r["fixture"],
                "market": "Over 2.5",
                "prob": r["over25_prob"],
                "value_edge": round(edge, 2),
                "league": r["league"],
            })

    df_value = pd.DataFrame(value_rows)

    # ===== SECTION BUILDERS (ASCII only, consistent) =====
    def print_section_header(title):
        lines.append(title)

    def add_list_or_none(df, formatter, limit, none_msg="No qualifying picks"):
        if df is None or df.empty:
            lines.append(none_msg)
            lines.append("")
            return
        for _, r in df.head(limit).iterrows():
            lines.append(formatter(r))
        lines.append("")

    def fmt_pct(x):
        try:
            return f"{float(x):.2f}%"
        except Exception:
            return f"{x}%"

    def fmt_btts_line(r):
        return f"{r['kickoff']} {r['fixture']} - {fmt_pct(r['btts_prob'])} - {r['league']}"

    def fmt_win_line(r):
        return f"{r['kickoff']} {r['fixture']} - {r['pick_team']} - {fmt_pct(r['win_prob'])} - {r['league']}"

    def fmt_over_line(r):
        return f"{r['kickoff']} {r['fixture']} - {fmt_pct(r['over25_prob'])} - {r['league']}"

    def fmt_value_line(r):
        return f"{r['kickoff']} {r['fixture']} - {r['market']} - {fmt_pct(r['prob'])} - +{fmt_pct(r['value_edge'])} - {r['league']}"

    # ---- BTTS regionals ----
    eng_btts = df_btts[(df_btts["region"] == "England") & (df_btts["confidence"].isin(["High", "Medium"]))] \
                .sort_values(["kickoff", "btts_prob"], ascending=[True, False])
    sco_btts = df_btts[(df_btts["region"] == "Scotland") & (df_btts["confidence"].isin(["High", "Medium"]))] \
                .sort_values(["kickoff", "btts_prob"], ascending=[True, False])
    ger_btts = df_btts[(df_btts["region"] == "Germany") & (df_btts["confidence"].isin(["High", "Medium"]))] \
                .sort_values(["kickoff", "btts_prob"], ascending=[True, False])

    print_section_header("BTTS - England (Top 10, Medium+)")
    add_list_or_none(eng_btts, fmt_btts_line, 10)

    print_section_header("BTTS - Scotland (Top 5, Medium+)")
    add_list_or_none(sco_btts, fmt_btts_line, 5)

    print_section_header("BTTS - Germany (Top 5, Medium+)")
    add_list_or_none(ger_btts, fmt_btts_line, 5)

    # ---- WIN: England top 10 and Combined top 10 ----
    win_eng = df_match[(df_match["region"] == "England") & (df_match["confidence"].isin(["High", "Medium"]))] \
                .sort_values(["kickoff", "win_prob"], ascending=[True, False])
    print_section_header("Top 10 Win Picks - England (Medium+)")
    add_list_or_none(win_eng, fmt_win_line, 10)

    win_comb = df_match[df_match["confidence"].isin(["High", "Medium"])] \
                .sort_values(["kickoff", "win_prob"], ascending=[True, False])
    print_section_header("Top 10 Combined Win Picks (Medium+)")
    add_list_or_none(win_comb, fmt_win_line, 10)

    # ---- OVER 2.5: Combined top 10 ----
    over_comb = df_over[df_over["confidence"].isin(["High", "Medium"])] \
                .sort_values(["kickoff", "over25_prob"], ascending=[True, False])
    print_section_header("Top 10 Combined Over 2.5 Picks (Medium+)")
    add_list_or_none(over_comb, fmt_over_line, 10)

    # ---- VALUE PICKS last ----
    print_section_header("Value Picks (10+% edge)")
    if df_value.empty:
        lines.append("No value picks today")
        lines.append("")
    else:
        val_sorted = df_value.sort_values(["kickoff", "value_edge"], ascending=[True, False]).head(20)
        for _, r in val_sorted.iterrows():
            lines.append(fmt_value_line(r))
        lines.append("")

    lines.append("Model run complete")

    print("\n".join(lines))
    return df_btts, df_match, df_over, df_value


if __name__ == "__main__":
    main()
