import time
import pandas as pd
from nba_api.stats.endpoints import playergamelogs

# Pull the same seasons that are in your team-games file so everything lines up
seasons = sorted(pd.read_csv("nba_games_raw.csv", usecols=["SEASON_LABEL"])["SEASON_LABEL"].unique())
print("Seasons to pull:", seasons)

wanted = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID", "GAME_DATE", "MIN", "PTS"]

frames = []
failed = []
for season in seasons:
    print("Pulling", season)
    for attempt in range(4):
        try:
            result = playergamelogs.PlayerGameLogs(
                season_nullable=season,
                season_type_nullable="Regular Season",
            )
            df = result.get_data_frames()[0]
            missing = [c for c in wanted if c not in df.columns]
            if missing:
                raise ValueError(f"expected columns missing: {missing}. Columns are: {df.columns.tolist()}")
            df = df[wanted].copy()
            df["SEASON_LABEL"] = season
            frames.append(df)
            break
        except Exception as e:
            print("  retry", attempt + 1, e)
            time.sleep(10)
    else:
        failed.append(season)
    time.sleep(2)  # be polite to the server

if not frames:
    raise SystemExit("Nothing downloaded. Copy the retry messages above and send them to me.")

players = pd.concat(frames, ignore_index=True)
players.to_csv("players_raw.csv", index=False)

print("\nShape:", players.shape)
print("Rows per season:")
print(players.groupby("SEASON_LABEL").size())
print(players.dtypes)
print(players.head())
if failed:
    print("\nWARNING - these seasons failed, run the script again:", failed)