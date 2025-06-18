import pandas as pd
import numpy as np

import wald_wolfowitz
import streak_converter
import streaks_base
import scipy.stats

# Data from https://www.kaggle.com/datasets/mexwell/nba-shots
MIN_SEASON=2004
MAX_SEASON=2024

def fix_index(shots):
    """
    the shot data from kaggle is not (always) in chronological order. furthermore,
    the NBA game ID cannot be used for sorting. it's roughly chronological but not always. 
    """

    shots["SORTABLE_DATE"] = pd.to_datetime(shots.GAME_DATE)
    shots = shots.sort_values(by=["SEASON_1", "SORTABLE_DATE", "QUARTER", "MINS_LEFT", "SECS_LEFT"], 
                                ascending=[True, True,True,False,False]).reset_index(drop=True)
    return shots

def get_all_shots():
    all_shots = []
    for season in range(MIN_SEASON, MAX_SEASON+1):
        shots =  pd.read_csv(f"kaggle_data/NBA_{season}_Shots.csv")
        all_shots.append(fix_index(shots))
    # to make the index unique after concatting them, need to redo it 
    # (there may be a more natural way of doing this)
    concated = pd.concat(all_shots)
    return fix_index(concated)

def get_streaks_for_season(season, do_percentile_rank=True):
    """
    this gets player-game level streakiness data for a single season of the NBA.
    """

    shots =  pd.read_csv(f"kaggle_data/NBA_{season}_Shots.csv")
    # I didn't realize the NBA was returning shots
    # in reverse order. this wouldn't change the number of streaks, or number 
    # of makes/misses, but raw_data field would be backwards.
    shots = fix_index(shots)

    combined_df = pd.DataFrame(columns=["player_id", "game_id", "player_name", "makes", 
                                    "misses", "total_streaks", "raw_data"])
    combined_df = combined_df.set_index(["player_id", "game_id"])


    for key, results in shots.groupby(["PLAYER_ID", "GAME_ID"]):
        streak_data = "".join(results["SHOT_MADE"].replace(True, "W").replace(False, "L"))
        as_streaks = streak_converter.convert_to_streaks(False, streak_data)

        combined_df.loc[key, :] = [results["PLAYER_NAME"].iloc[0], as_streaks['makes'], as_streaks['misses'],
                                    as_streaks['total_streaks'], as_streaks['raw_data']]

    combined_df['expected_streaks'] = wald_wolfowitz.get_expected_streaks(combined_df.makes, combined_df.misses)

    # to calculate variance > 0, expected streaks needs to be more than 2 (which makes sense)
    # makes + misses needs to be > 1.
    can_calc = ((combined_df.makes + combined_df.misses) > 1) & (combined_df.expected_streaks > 2)
    calc_df = combined_df[can_calc].copy()
    uncalc_df = combined_df[~can_calc].copy()

    calc_df['variance'] = wald_wolfowitz.get_variance(calc_df.makes, 
                                                        calc_df.misses, 
                                                        calc_df.expected_streaks)
    calc_df['z_score'] = (calc_df.total_streaks - calc_df.expected_streaks) / ((calc_df.variance)**.5)
    
    uncalc_df['variance'] = np.nan
    uncalc_df['z_score'] = np.nan

    clean_df = pd.concat([calc_df, uncalc_df])
    clean_df['season'] = season

    clean_df['ww_percentile'] = 100 * scipy.stats.norm.cdf(clean_df.z_score.tolist())

    if do_percentile_rank:
        # may as well calculate percentile ranks while we're here
        clean_df['exact_percentile_rank'] = None

        for row_id, data in clean_df.iterrows():
            # I haven't vectorized get_percentile_rank so I have to loop over every row =(
            clean_df.loc[row_id, "exact_percentile_rank"] = wald_wolfowitz.get_percentile_rank(data['makes'], data['misses'], data['total_streaks'])
            
        clean_df['z_from_percentile_rank'] = scipy.stats.norm.ppf(
            (clean_df['exact_percentile_rank'].values/ 100).tolist())

    return clean_df

def get_all_seasons(min_year=MIN_SEASON, max_year=MAX_SEASON):
    """
    this gets all player-game streakiness data for a range of seasons.
    """
    all_seasons = []
    for x in range(min_year, max_year + 1):
        season = str(x)
        pkl_name = f"streak_cache/{season}-streaks.pkl"
        try:
            season = pd.read_pickle(pkl_name)
        except:
            print(f"parsing {season}")
            season = get_streaks_for_season(season)
            season.to_pickle(pkl_name)

        all_seasons.append(season)
    return pd.concat(all_seasons)

def get_all_player_games(min_year=MIN_SEASON, max_year=MAX_SEASON):
    return get_all_seasons(min_year, max_year)

def get_all_player_streaks():
    """
    This gets streakiness data for every player over the course of their career.
    """
    all_seasons = get_all_seasons()
    streak_helper = streaks_base.StreaksBase()
    stats_df = pd.DataFrame(columns=["player_id", "player_name", "makes", 
                                "misses", "total_streaks", "raw_data"])
        
    for key, results in all_seasons.groupby(["player_id"]):
        all_shots = "".join(results["raw_data"])
        streak_data = streak_converter.convert_to_streaks(False, all_shots, False)
        stats_df.loc[len(stats_df)] = [key[0], results["player_name"].iloc[0], streak_data['makes'],  streak_data['misses'],
                                    streak_data['total_streaks'], streak_data['raw_data']
                                    ]
    stats_df.set_index("player_id")
    df_with_stats = streak_helper.calc_stats(stats_df)
    return df_with_stats

### FIXME (this and some other functions should probbly go to a new file)
### there is a performance issue with this function.
### with the fix to the index (and shot order), it might be a little better.
def add_sequence_data(base_shots):
    """
    takes a df with all shots to be analyzed
    adds data about shot sequence and in-game FG% for each player
    for looking at how that correlates with overall shooting performance
    """
    all_shots = base_shots.copy()

    all_shots['makes_in_game']  = None  # avoiding name collision with 'makes' key in career summary
    all_shots['misses_in_game'] = None  # likewise
    all_shots['shot_seq']       = None  # sequential (ordinal) shot in game by this player
    all_shots['shoot_pct']      = None  # shooting percentage in game by this player
    all_shots['last_5']         = None  # shooting percentage over previous 5 shots by player

    # there is almost certainly a better way of doing this with a more
    # functional programming approach and pandas built-ins (eg the rolling() function)
    for key, shots in base_shots.groupby(["PLAYER_ID", "GAME_ID"]):
        make_count = 0
        miss_count = 0
        shot_seq = 1        # NOTE: 1 indexed
        shoot_pct = 0
        shots_in_game = []

        for key2, shot in shots.iterrows():
            # the key should be unique and in the right order now.

            condition = key2 ## this should work now

            all_shots.loc[condition, 'makes_in_game'] = make_count
            all_shots.loc[condition, 'misses_in_game'] = miss_count
            all_shots.loc[condition, 'shot_seq'] = shot_seq
            all_shots.loc[condition, 'shoot_pct'] = shoot_pct

            # TODO? replace with rolling(5).sum()?
            # problem is it needs to be for each game, not across games
            if len(shots_in_game) > 4:
                last5_shots = shots_in_game[-5:]
                all_shots.loc[condition, 'last_5'] = sum(last5_shots) / 5

            if shot.SHOT_MADE:
                make_count += 1
                shots_in_game.append(1)
            else:
                miss_count += 1
                shots_in_game.append(0)
            shoot_pct = make_count / (make_count + miss_count)
            shot_seq += 1
    return all_shots