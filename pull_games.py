import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

# Seasons to pull, e.g. "2023-24". Add or remove as you like.
seasons = [f"{y}-{str(y + 1)[-2:]}" for y in range(2012, 2027)]

frames = []
for season in seasons:
    print("Pulling", season)
    for attempt in range(3):
        try:
            result = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable="Regular Season",
            )
            df = result.get_data_frames()[0]
            df["SEASON_LABEL"] = season
            frames.append(df)
            break
        except Exception as e:
            print("  retry", attempt + 1, e)
            time.sleep(5)
    time.sleep(1)  # be polite to the server

all_games = pd.concat(frames, ignore_index=True)
all_games.to_csv("nba_games_raw.csv", index=False)
print(all_games.shape)
print(all_games.columns.tolist())