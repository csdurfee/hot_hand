import re
import pandas as pd
import numpy as np

from streaks_base import StreaksBase

from nba_api.stats import endpoints as nba_endpoints
from nba_api.live.nba.endpoints import playbyplay

class FreeThrowStreaks(StreaksBase):
    def __init__(self):
        self.SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
        #self.game_ids = None

        # getting game id's is really really slow
        try:
            self.game_ids = pd.read_pickle("game_ids.pkl")
        except:
            self.game_ids = None

    def get_game_ids(self):
        if self.game_ids is None:
            game_ids = []
            for s in self.SEASONS:
                game_ids.extend(
                    nba_endpoints.teamgamelogs.TeamGameLogs(season_nullable=s).get_data_frames()[0]["GAME_ID"]
                )
            game_ids_list = list(set(game_ids))
            self.game_ids = pd.Series(sorted(game_ids_list))
            print("got game ids")
            self.game_ids.to_pickle("game_ids.pkl")

        return self.game_ids
    
    def get_base_dataframe(self):
        df = pd.DataFrame(columns=["game_id", "time", "player_id", "result"])
        return df
    
    def get_pickle_filename(self):
        return f"freethrows.pkl"

    def get_dataframe(self):
        """
        try to load the dataframe from disk. if it does not exist, return base dataframe.
        """
        try:
            dataframe = pd.read_pickle(self.get_pickle_filename())
            return dataframe
        except:
            print("no saved file, creating")
            return self.get_base_dataframe()
        
    def get_freethrow_data(self, game_id):
        play_by_play = playbyplay.PlayByPlay(game_id=game_id)
        actions = pd.DataFrame(play_by_play.get_dict()['game']['actions'])
        return actions[actions.actionType=="freethrow"].copy()

    def get_data(self):
        df = self.get_dataframe()
        game_ids = self.get_game_ids()
        _loop_counter = 0
        game_ids = set(df.game_id)
        for game_id in game_ids:
            if game_id not in game_ids:
                ft_data = self.get_freethrow_data(game_id)
                for counter, row in ft_data.iterrows():
                    add_data = [game_id, row["timeActual"], row["personId"], row["shotResult"]]
                    df.loc[len(df)] = add_data
                df.to_pickle(self.get_pickle_filename())
                _loop_counter += 1
                if (_loop_counter % 10) == 0:
                    print(f"{game_id},",)
        return df
    
    def get_data_with_stats(self):
        raw_data = self.get_data()
        df = self.get_stats_dataframe()
        for player_id, shots in raw_data.groupby("player_id"):
            ## FIXME: get player name
            player_name = "FIXME"
            make_miss = shots.result.replace("Made", "W").replace("Missed", "L")
            streak_data = self.convert_to_streaks(make_miss)
            df.loc[player_id] =[player_name, streak_data['makes'], streak_data['misses'],
                                    streak_data['total_streaks'], streak_data['raw_data']
                                    ]
        return self.calc_stats(df)