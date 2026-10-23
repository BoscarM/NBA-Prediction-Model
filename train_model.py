import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from elo import HOME_ADV

df = pd.read_csv("model_data.csv", parse_dates=["GAME_DATE"])

old_base = ["PM_L5", "PM_L10", "PM_L20", "WINPCT_SEASON", "REST", "B2B"]


def cols(base):
    return [c + s for c in base for s in ["_home", "_away"]]


feature_sets = {"Previous + Elo": cols(old_base + ["ELO"])}
has_injuries = "PTS_OUT_home" in df.columns
if has_injuries:
    feature_sets["+ Points out"] = cols(old_base + ["ELO", "PTS_OUT"])
    feature_sets["+ Points & minutes out"] = cols(old_base + ["ELO", "PTS_OUT", "MIN_OUT"])

# Test on every season from 2023-24 onward. Each time, train only on seasons BEFORE the test season.
test_seasons = [s for s in sorted(df["SEASON_LABEL"].unique()) if s >= "2023-24"]

for test_season in test_seasons:
    train = df[df["SEASON_LABEL"] < test_season]
    test = df[df["SEASON_LABEL"] == test_season]
    home_won = test["MARGIN"] > 0

    print(f"\n=== Test season {test_season}: {len(train)} train games, {len(test)} test games ===")
    print(f"Baseline MAE (always 0):          {mean_absolute_error(test['MARGIN'], [0] * len(test)):.2f}")
    naive_elo = (test["ELO_home"] + HOME_ADV) > test["ELO_away"]
    print(f"Baseline: higher Elo (+home adv): {(naive_elo == home_won).mean():.1%}")

    for name, feats in feature_sets.items():
        model = LinearRegression().fit(train[feats], train["MARGIN"])
        preds = model.predict(test[feats])
        mae = mean_absolute_error(test["MARGIN"], preds)
        acc = ((preds > 0) == home_won).mean()
        print(f"{name:24s} MAE: {mae:.2f}   Winner accuracy: {acc:.1%}")

if has_injuries:
    # Sanity check: what did the model learn about injuries, using all the data?
    feats = feature_sets["+ Points out"]
    full = LinearRegression().fit(df[feats], df["MARGIN"])
    print("\nWhat the model learned (change in predicted home margin per 1 point of missing scoring):")
    for name, coef in zip(feats, full.coef_):
        if "OUT" in name:
            print(f"  {name:14s} {coef:+.3f}")
    print("Expected: PTS_OUT_home negative (home team missing scorers hurts the home margin), PTS_OUT_away positive.")