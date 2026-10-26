import os
import pandas as pd
from datetime import date
from sklearn.linear_model import LinearRegression
from elo import run_elo, rating_for

# --- Injury settings: these must match injury_features.py ---
WINDOW = 10          # look at each team's previous 10 games
MIN_GAMES = 5        # need at least 5 earlier games this season (same rule as training)
ROTATION_MIN = 15    # "rotation player" = averages 15+ minutes over that window
# Must match the best row in train_model.py's output. Use ["PTS_OUT"] if "+ Points out" won.
INJURY_FEATURES = ["PTS_OUT"]

raw = pd.read_csv("nba_games_raw.csv", parse_dates=["GAME_DATE"])
raw["WIN"] = (raw["WL"] == "W").astype(int)
df = pd.read_csv("model_data.csv", parse_dates=["GAME_DATE"])
games_table = pd.read_csv("games_table.csv", parse_dates=["GAME_DATE"])

use_injuries = (os.path.exists("players_raw.csv")
                and all(f"{c}_{s}" in df.columns for c in INJURY_FEATURES for s in ("home", "away")))
if use_injuries:
    players = pd.read_csv("players_raw.csv")
    players["GAME_DATE"] = pd.to_datetime(players["GAME_DATE"]).dt.normalize()
    players["MIN"] = pd.to_numeric(players["MIN"], errors="coerce")
    players["PTS"] = pd.to_numeric(players["PTS"], errors="coerce").fillna(0)
    players = players[players["MIN"] > 0].drop_duplicates(["PLAYER_ID", "TEAM_ID", "GAME_ID"])
else:
    print("Injury data not found, so this will predict WITHOUT injuries.")
    print("(Run pull_players.py, injury_features.py and build_features.py to enable them.)\n")

base = ["PM_L5", "PM_L10", "PM_L20", "WINPCT_SEASON", "REST", "B2B", "ELO"]
if use_injuries:
    base += INJURY_FEATURES
features = [c + s for c in base for s in ["_home", "_away"]]

# Train on every game we have
model = LinearRegression().fit(df[features], df["MARGIN"])

# Map team abbreviations (e.g. "LAL") to team IDs
latest = raw.sort_values("GAME_DATE").groupby("TEAM_ID").last()
abbr_to_id = {abbr: tid for tid, abbr in latest["TEAM_ABBREVIATION"].items()}


def season_label(d):
    y = d.year if d.month >= 9 else d.year - 1
    return f"{y}-{str(y + 1)[-2:]}"


def team_state(abbr, game_date, elo_state):
    tid = abbr_to_id[abbr]
    games = raw[(raw["TEAM_ID"] == tid) & (raw["GAME_DATE"] < game_date)]
    games = games.sort_values("GAME_DATE")
    pm = games["PLUS_MINUS"]
    this_season = games[games["SEASON_LABEL"] == season_label(game_date)]
    rest = min((game_date - games["GAME_DATE"].iloc[-1]).days, 7)
    return {
        "PM_L5": pm.tail(5).mean(),
        "PM_L10": pm.tail(10).mean(),
        "PM_L20": pm.tail(20).mean(),
        "WINPCT_SEASON": this_season["WIN"].mean() if len(this_season) else 0.5,
        "REST": rest,
        "B2B": int(rest == 1),
        "ELO": rating_for(elo_state, tid, season_label(game_date)),
        "PTS_OUT": 0.0,
        "MIN_OUT": 0.0,
        "last_game": games["GAME_DATE"].iloc[-1],
    }


def rotation(abbr, game_date):
    """The team's rotation players, using the same definition as injury_features.py."""
    tid = abbr_to_id[abbr]
    sched = raw[(raw["TEAM_ID"] == tid) & (raw["SEASON_LABEL"] == season_label(game_date))
                & (raw["GAME_DATE"] < game_date)].sort_values("GAME_DATE").drop_duplicates("GAME_ID")
    if len(sched) < MIN_GAMES:
        return None
    recent = sched.tail(WINDOW)
    w = len(recent)
    g = players[(players["TEAM_ID"] == tid) & (players["GAME_ID"].isin(recent["GAME_ID"]))]
    tot = g.groupby("PLAYER_ID").agg(NAME=("PLAYER_NAME", "first"), MIN=("MIN", "sum"), PTS=("PTS", "sum"))
    tot["MPG"] = tot["MIN"] / w          # minutes per team game (a missed game counts as 0)
    tot["PPG"] = tot["PTS"] / w
    tot = tot[tot["MPG"] >= ROTATION_MIN]
    # Drop anyone whose most recent game before this date was for a different team (traded away)
    keep = []
    for pid in tot.index:
        past = players[(players["PLAYER_ID"] == pid) & (players["GAME_DATE"] < game_date)]
        keep.append(past.sort_values("GAME_DATE")["TEAM_ID"].iloc[-1] == tid)
    tot = tot[pd.Series(keep, index=tot.index, dtype=bool)]
    return tot.sort_values("MPG", ascending=False)


def ask_out(abbr, game_date):
    """Show the rotation and ask who is out. Returns (points out, minutes out, names)."""
    rot = rotation(abbr, game_date)
    if rot is None:
        print(f"\n{abbr}: fewer than {MIN_GAMES} games played this season, so no injuries are counted (same as in training).")
        return 0.0, 0.0, []
    print(f"\n{abbr} rotation (averages over the last {WINDOW} games):")
    for i, (_, p) in enumerate(rot.iterrows(), 1):
        print(f"  {i}) {p['NAME']:25s} {p['MPG']:5.1f} min  {p['PPG']:5.1f} pts")
    answer = input(f"Who is OUT for {abbr}? Numbers separated by commas, or press Enter for nobody: ").strip()
    picked = []
    for token in answer.replace(" ", "").split(","):
        if token.isdigit() and 1 <= int(token) <= len(rot):
            picked.append(int(token) - 1)
        elif token:
            print(f"  (ignored '{token}', not a number on the list)")
    out = rot.iloc[sorted(set(picked))]
    return out["PPG"].sum(), out["MPG"].sum(), out["NAME"].tolist()


print("Teams:", " ".join(sorted(abbr_to_id)))
home = input("Home team (e.g. LAL): ").strip().upper()
away = input("Away team (e.g. BOS): ").strip().upper()
d = input("Game date YYYY-MM-DD (Enter for today): ").strip()
game_date = pd.Timestamp(d) if d else pd.Timestamp(date.today())

# Rebuild Elo ratings using only games played before the chosen date
_, elo_state = run_elo(games_table[games_table["GAME_DATE"] < game_date])

h = team_state(home, game_date, elo_state)
a = team_state(away, game_date, elo_state)


def predict_margin(h, a):
    row = {f"{c}_home": h[c] for c in base}
    row.update({f"{c}_away": a[c] for c in base})
    return model.predict(pd.DataFrame([row])[features])[0]


no_injury_margin = predict_margin(h, a)          # the same game assuming nobody is out

if use_injuries:
    for name, s in ((home, h), (away, a)):
        pts, mins, names = ask_out(name, game_date)
        s["PTS_OUT"], s["MIN_OUT"] = pts, mins
        if names:
            print(f"  -> {name} missing {pts:.1f} pts and {mins:.1f} min per game: {', '.join(names)}")

margin = predict_margin(h, a)

winner, loser = (home, away) if margin > 0 else (away, home)
print(f"\nPrediction: {winner} beats {loser} by about {abs(margin):.1f} points")
print(f"(Home margin: {margin:+.1f})")
print(f"Elo ratings: {home} {h['ELO']:.0f}, {away} {a['ELO']:.0f}")
if use_injuries:
    print(f"Injury effect: with nobody out the home margin would be {no_injury_margin:+.1f}; "
          f"with the injuries you entered it is {margin:+.1f} (shift of {margin - no_injury_margin:+.1f}).")

for name, s in [(home, h), (away, a)]:
    if (game_date - s["last_game"]).days > 14:
        print(f"Warning: the latest {name} game in your data is {s['last_game'].date()}. Refresh the data.")


# python pull_games.py
# python pull_players.py
# python injury_features.py
# python build_features.py
# python train_model.py
# python predict.py