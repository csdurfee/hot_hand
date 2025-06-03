import re
import time

import pandas as pd
import numpy as np

from streaks_base import StreaksBase

from nba_api.stats.static import players
from nba_api.stats.endpoints import shotchartdetail

PLAYER_STREAK_FILE = "player_streaks.pkl"
SLEEP_TIME = 2 #seconds between API requests


class PlayerStreaks(StreaksBase):
    def __init__(self, shot_type="FGA", season=None):
        self.shot_type = shot_type
        if season:
            self.season = season
        else:
            self.season = self.get_this_season()


    def get_this_season(self):
        return "2024-25"

    def get_all_player_ids(self):
        """
        This should return a dict() of id to player name 
        for all active players in the current season.
        """
        all_active =  players.get_active_players()
        return dict((x['id'], x['full_name']) for x in all_active)

    def get_shots_for_player_id(self, player_id):
        response = shotchartdetail.ShotChartDetail(
            team_id = 0,
            player_id = player_id,
            season_nullable = self.season,
            context_measure_simple = self.shot_type, 
            season_type_all_star='Regular Season'
        )
        shot_details = response.get_data_frames()
        return shot_details[0].SHOT_MADE_FLAG
    
    def get_make_lengths(self, streak):
        streak_data = self.convert_to_streaks(None, streak)
        return streak_data['make_lengths']
    
    def get_miss_lengths(self, streak):
        streak_data = self.convert_to_streaks(None, streak)
        return streak_data['miss_lengths']

    def get_pickle_filename(self):
        return f"{self.season}_{self.shot_type}_{PLAYER_STREAK_FILE}"

    def get_dataframe(self):
        """
        try to load the dataframe from disk. if it does not exist, return base dataframe.
        """
        try:
            dataframe = pd.read_pickle(self.get_pickle_filename())
            return dataframe
        except:
            print("no saved file, creating")
            return self.get_stats_dataframe()


    def get_data(self):
        """
        this will return streak data for all NBA players for the current season.
        """
        df = self.get_dataframe()

        player_ids = self.get_all_player_ids()
        for player_id, player_name in player_ids.items():
            ## see if we've already done this row.
            ## once we have all the data, this shouldn't fire any additional requests
            if player_id not in df.index:
                player_data = self.get_shots_for_player_id(player_id)
                streak_data = self.convert_to_streaks(player_data)
                df.loc[player_id] = [player_name, streak_data['makes'], streak_data['misses'],
                                    streak_data['total_streaks'], streak_data['raw_data']
                                    ]
                print(f"{player_id}, ",)
                # save after every iteration
                df.to_pickle(self.get_pickle_filename())
                time.sleep(SLEEP_TIME)

        return df
    
    def get_data_with_stats(self):
        df = self.get_data()
        return self.calc_stats(df)