import numpy as np
import pandas as pd

WINDOW = 10          # compare against the team's previous 10 games
MIN_GAMES = 5        # need at least 5 earlier games this season to know the rotation
ROTATION_MIN = 15    # "rotation player" = averaged 15+ minutes per game over that window

sched = pd.read_csv("nba_games_raw.csv",
                    usecols=["TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "SEASON_LABEL"])
sched["GAME_DATE"] = pd.to_datetime(sched["GAME_DATE"]).dt.normalize()
sched = sched.drop_duplicates(["TEAM_ID", "GAME_ID"])      # guard against duplicated rows

players = pd.read_csv("players_raw.csv")
players["GAME_DATE"] = pd.to_datetime(players["GAME_DATE"]).dt.normalize()
players["MIN"] = pd.to_numeric(players["MIN"], errors="coerce")
players["PTS"] = pd.to_numeric(players["PTS"], errors="coerce")

if players["MIN"].isna().mean() > 0.05:
    raise ValueError("More than 5% of the MIN column could not be read as numbers. "
                     "Send me players.head() and players.dtypes and I'll fix the reader.")

players = players[players["MIN"] > 0].copy()      # keep only players who actually played
players = players.drop_duplicates(["PLAYER_ID", "TEAM_ID", "GAME_ID"])   # guard against duplicated rows
players["PTS"] = players["PTS"].fillna(0)
names = players.drop_duplicates("PLAYER_ID").set_index("PLAYER_ID")["PLAYER_NAME"].to_dict()

# Each player's appearance history (date, team), used to spot players who were traded away
history = {pid: (g["GAME_DATE"].values, g["TEAM_ID"].values)
           for pid, g in players.sort_values("GAME_DATE").groupby("PLAYER_ID")}


def still_with_team(pid, team, date):
    """False if the player's most recent appearance before `date` was for a different team."""
    dates, teams = history[pid]
    k = np.searchsorted(dates, date, side="left") - 1
    return k < 0 or teams[k] == team


by_team = {t: g for t, g in players.groupby("TEAM_ID")}
rows = []

for (team, season), s in sched.groupby(["TEAM_ID", "SEASON_LABEL"]):
    s = s.sort_values("GAME_DATE").reset_index(drop=True)
    n = len(s)
    gp = by_team.get(team)
    if gp is None:
        continue
    gp = gp[gp["GAME_ID"].isin(s["GAME_ID"])]
    if gp.empty:
        continue

    game_pos = {g: i for i, g in enumerate(s["GAME_ID"])}
    pids = gp["PLAYER_ID"].unique()
    pid_pos = {p: i for i, p in enumerate(pids)}

    # Matrices: one row per player, one column per team game (0 = did not play)
    mins = np.zeros((len(pids), n))
    pts = np.zeros((len(pids), n))
    r = gp["PLAYER_ID"].map(pid_pos).to_numpy()
    c = gp["GAME_ID"].map(game_pos).to_numpy()
    mins[r, c] = gp["MIN"].to_numpy()
    pts[r, c] = gp["PTS"].to_numpy()
    cum_m = np.hstack([np.zeros((len(pids), 1)), mins.cumsum(axis=1)])
    cum_p = np.hstack([np.zeros((len(pids), 1)), pts.cumsum(axis=1)])

    for j in range(MIN_GAMES, n):
        if mins[:, j].sum() == 0:          # no box-score data for this game: skip it
            continue
        w = min(WINDOW, j)
        # Averages use ONLY the previous w games (columns j-w .. j-1), never game j itself
        mpg = (cum_m[:, j] - cum_m[:, j - w]) / w
        ppg = (cum_p[:, j] - cum_p[:, j - w]) / w
        out = (mpg >= ROTATION_MIN) & (mins[:, j] == 0)
        date = s.loc[j, "GAME_DATE"].to_datetime64()
        out_idx = [i for i in np.where(out)[0] if still_with_team(pids[i], team, date)]
        rows.append({
            "TEAM_ID": team, "TEAM": s.loc[j, "TEAM_ABBREVIATION"], "GAME_ID": s.loc[j, "GAME_ID"],
            "GAME_DATE": s.loc[j, "GAME_DATE"].date(),
            "MIN_OUT": mpg[out_idx].sum(), "PTS_OUT": ppg[out_idx].sum(), "N_OUT": len(out_idx),
            "OUT_PLAYERS": "; ".join(names[pids[i]] for i in out_idx),
        })

inj = pd.DataFrame(rows)
inj.to_csv("injury_features.csv", index=False)

print("Team-games with injury features:", len(inj))
print(f"Share of team-games with at least one rotation player out: {(inj['N_OUT'] > 0).mean():.1%}")
print(inj[["MIN_OUT", "PTS_OUT", "N_OUT"]].describe().round(2))
print("\nThe 8 team-games with the most scoring missing (do these look like real star absences?):")
print(inj.nlargest(8, "PTS_OUT")[["GAME_DATE", "TEAM", "PTS_OUT", "OUT_PLAYERS"]].to_string(index=False))
print("\n5 random team-games with someone out:")
print(inj[inj["N_OUT"] > 0].sample(5, random_state=1)[["GAME_DATE", "TEAM", "MIN_OUT", "OUT_PLAYERS"]].to_string(index=False))