START = 1500      # every team starts here
K = 20            # how fast ratings move after each game
HOME_ADV = 100    # Elo points of home-court advantage (about 3 points of margin)
CARRY = 0.75      # share of a rating kept into the next season (rest regresses to 1500)


def _expected(diff):
    return 1 / (1 + 10 ** (-diff / 400))


def rating_for(state, team_id, season):
    """A team's rating going into a game in `season`, with offseason regression applied."""
    rating, last_season = state.get(team_id, (START, None))
    if last_season is not None and last_season != season:
        rating = CARRY * rating + (1 - CARRY) * START
    return rating


def run_elo(games):
    """Walk through games oldest to newest. Each game's ELO_home / ELO_away are the
    ratings BEFORE that game, so a game's own result never leaks into its features.
    Returns (games with Elo columns, final ratings state)."""
    games = games.sort_values(["GAME_DATE", "GAME_ID"]).copy()
    state = {}
    elo_h, elo_a = [], []

    for r in games.itertuples():
        h = rating_for(state, r.HOME_TEAM_ID, r.SEASON_LABEL)
        a = rating_for(state, r.AWAY_TEAM_ID, r.SEASON_LABEL)
        elo_h.append(h)
        elo_a.append(a)

        diff = h + HOME_ADV - a                       # home rating advantage
        expected_home = _expected(diff)               # chance the home team wins
        home_won = 1.0 if r.MARGIN > 0 else 0.0

        # Bigger wins move ratings more, but upsets-by-favorites count for less
        winner_diff = diff if home_won else -diff
        mult = ((abs(r.MARGIN) + 3) ** 0.8) / (7.5 + 0.006 * winner_diff)
        delta = K * mult * (home_won - expected_home)

        state[r.HOME_TEAM_ID] = (h + delta, r.SEASON_LABEL)
        state[r.AWAY_TEAM_ID] = (a - delta, r.SEASON_LABEL)

    games["ELO_home"] = elo_h
    games["ELO_away"] = elo_a
    return games, state