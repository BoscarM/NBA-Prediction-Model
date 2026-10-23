import pandas as pd
from datetime import date
from sklearn.linear_model import LinearRegression
from elo import run_elo, rating_for

raw = pd.read_csv("nba_games_raw.csv", parse_dates=["GAME_DATE"])
raw["WIN"] = (raw["WL"] == "W").astype(int)
df = pd.read_csv("model_data.csv", parse_dates=["GAME_DATE"])
games_table = pd.read_csv("games_table.csv", parse_dates=["GAME_DATE"])

base = ["PM_L5", "PM_L10", "PM_L20", "WINPCT_SEASON", "REST", "B2B", "ELO"]
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
        "last_game": games["GAME_DATE"].iloc[-1],
    }


print("Teams:", " ".join(sorted(abbr_to_id)))
home = input("Home team (e.g. LAL): ").strip().upper()
away = input("Away team (e.g. BOS): ").strip().upper()
d = input("Game date YYYY-MM-DD (Enter for today): ").strip()
game_date = pd.Timestamp(d) if d else pd.Timestamp(date.today())

# Rebuild Elo ratings using only games played before the chosen date
_, elo_state = run_elo(games_table[games_table["GAME_DATE"] < game_date])

h = team_state(home, game_date, elo_state)
a = team_state(away, game_date, elo_state)

row = {f"{c}_home": h[c] for c in base}
row.update({f"{c}_away": a[c] for c in base})
margin = model.predict(pd.DataFrame([row])[features])[0]

winner, loser = (home, away) if margin > 0 else (away, home)
print(f"\nPrediction: {winner} beats {loser} by about {abs(margin):.1f} points")
print(f"(Home margin: {margin:+.1f})")
print(f"Elo ratings: {home} {h['ELO']:.0f}, {away} {a['ELO']:.0f}")

for name, s in [(home, h), (away, a)]:
    if (game_date - s["last_game"]).days > 14:
        print(f"Warning: the latest {name} game in your data is {s['last_game'].date()}. Refresh the data.")