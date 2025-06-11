import pandas as pd
import numpy as np

import wald_wolfowitz
import streak_converter
import streaks_base
import scipy.stats

def get_streaks_for_season(season):
    """
    this gets player-game level streakiness data for a single season of the NBA.
    """

    shots =  pd.read_csv(f"kaggle_data/NBA_{season}_Shots.csv")

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

    # may as well calculate percentile ranks while we're here
    clean_df['exact_percentile_rank'] = None

    for row_id, data in clean_df.iterrows():
        # I haven't vectorized get_percentile_rank so I have to loop over every row =(
        clean_df.loc[row_id, "exact_percentile_rank"] = wald_wolfowitz.get_percentile_rank(data['makes'], data['misses'], data['total_streaks'])
        
    clean_df['z_from_percentile_rank'] = scipy.stats.norm.ppf(
        (clean_df['exact_percentile_rank'].values/ 100).tolist())

    return clean_df

def get_all_seasons(min_year=2004, max_year=2024):
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