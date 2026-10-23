import os
import pandas as pd
from elo import run_elo

raw = pd.read_csv("nba_games_raw.csv")
raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])
raw = raw.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

raw["WIN"] = (raw["WL"] == "W").astype(int)
g = raw.groupby("TEAM_ID")

# Rolling average margin over the previous 5, 10, and 20 games.
# shift(1) excludes the current game, which prevents data leakage.
for n, minp in [(5, 3), (10, 5), (20, 10)]:
    raw[f"PM_L{n}"] = g["PLUS_MINUS"].transform(
        lambda s: s.shift(1).rolling(n, min_periods=minp).mean()
    )

# Season-to-date win percentage before this game (0.5 for a team's first game of a season)
raw["WINPCT_SEASON"] = (
    raw.groupby(["TEAM_ID", "SEASON_LABEL"])["WIN"]
    .transform(lambda s: s.shift(1).expanding().mean())
    .fillna(0.5)
)

# Rest days (capped at 7) and back-to-back flag
raw["REST"] = g["GAME_DATE"].diff().dt.days.clip(upper=7)
raw["B2B"] = (raw["REST"] == 1).astype(int)

# Split into home and away rows, then join into one row per game
is_home = raw["MATCHUP"].str.contains("vs.")
feat_cols = ["PM_L5", "PM_L10", "PM_L20", "WINPCT_SEASON", "REST", "B2B"]

home = raw[is_home][["GAME_ID", "GAME_DATE", "TEAM_ID", "SEASON_LABEL", "PLUS_MINUS"] + feat_cols]
away = raw[~is_home][["GAME_ID", "TEAM_ID"] + feat_cols]

home = home.rename(columns={"TEAM_ID": "HOME_TEAM_ID", **{c: c + "_home" for c in feat_cols}})
away = away.rename(columns={"TEAM_ID": "AWAY_TEAM_ID", **{c: c + "_away" for c in feat_cols}})

games = home.merge(away, on="GAME_ID")
games["MARGIN"] = games["PLUS_MINUS"]
games = games.dropna(subset=["MARGIN"])

# Elo ratings, computed over EVERY game in order (before any rows are dropped)
games, _ = run_elo(games)

# Injury features (created by injury_features.py). Skipped if that file doesn't exist yet.
if os.path.exists("injury_features.csv"):
    inj = pd.read_csv("injury_features.csv", usecols=["TEAM_ID", "GAME_ID", "MIN_OUT", "PTS_OUT"])
    for side, id_col in [("home", "HOME_TEAM_ID"), ("away", "AWAY_TEAM_ID")]:
        f = inj.rename(columns={"TEAM_ID": id_col, "MIN_OUT": f"MIN_OUT_{side}", "PTS_OUT": f"PTS_OUT_{side}"})
        games = games.merge(f, on=["GAME_ID", id_col], how="left")
    inj_cols = [f"{c}_{s}" for c in ["MIN_OUT", "PTS_OUT"] for s in ["home", "away"]]
    # Early-season games (too little history to know the rotation) get 0 = "nothing known to be missing"
    games[inj_cols] = games[inj_cols].fillna(0)
    print("Injury features added.")
else:
    print("No injury_features.csv found, skipping injury features.")

games.to_csv("games_table.csv", index=False)   # predict.py uses this to rebuild Elo

# Drop games without enough history
needed = [c + s for c in feat_cols for s in ["_home", "_away"]]
model_df = games.dropna(subset=needed).sort_values("GAME_DATE").reset_index(drop=True)

model_df.to_csv("model_data.csv", index=False)
print(model_df.shape)