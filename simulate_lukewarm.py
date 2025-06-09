import collections
import functools
import operator

import pandas as pd
import numpy as np

import streaky_players
import wald_wolfowitz
import streak_converter

class SimulationBase:
    def handle_season(self, results):
        # returns streakiness stats for a player's season.
        makes = sum(results > 0)
        misses = sum(results == 0)
        if (makes + misses) > 3:
            expected = wald_wolfowitz.get_expected_streaks(makes, misses)
            streak_data = streak_converter.convert_to_streaks(results)
            total_streaks = streak_data['total_streaks']
            variance = wald_wolfowitz.get_variance(makes, misses, expected)
            if variance > 0:
                z_score = (total_streaks - expected) / np.sqrt(variance)
                return (total_streaks, expected, variance, z_score)
            else:
                return None
        else:
            return None # not enough data!

class SimulateLukewarm(SimulationBase):
    def __init__(self, df):
        self.df = df # this is all shots in a season.
        self.player_type = streaky_players.LukewarmPlayer
        self.player_cache = {}

        # calculate fg_percentages.
        makes = self.df.groupby("player_id")["makes"].sum()
        misses = self.df.groupby("player_id")["misses"].sum()

        self.fg_percentage = makes / (makes + misses)


    def get_player(self, player_id):
        if player_id in self.player_cache:
            return self.player_cache[player_id]
        else:
            player_fg_percentage = self.fg_percentage[player_id]
            player = self.player_type(shooting_percentage=player_fg_percentage)

            self.player_cache[player_id] = player
            return player

    def sim_season(self):
        """
        simulates entire season (every player, every game) using actual shot counts and fg% by player
        """
        game_results = collections.defaultdict(list)
        player_games = []
        
        # loop over every player_id + game_id combo in 'shots'
        for key, results in self.df.groupby(["player_id", "game_id"]):
            player_id = key[0]
            num_shots = (results.makes + results.misses).values[0]
            # sim the number of shots for that player_id, game_id combo
            sim_player = self.get_player(player_id)
            shot_results = [sim_player.take_shot() for x in range(num_shots)]
            # add the results of those simulated shots to a list of makes/misses for that player_id
            game_results[player_id].append(shot_results)

            player_games.append(shot_results)

            sim_player.end_game() # streaks don't persist between games (unless overridden in StreakyPlayer)


        season_results = []
        for player_id in game_results.keys():
            # each game results is an array, this combines them into one big array.
            makes_misses = functools.reduce(operator.add, game_results[player_id])
            
            player_stats = self.handle_season(pd.Series(makes_misses))

            if player_stats:
                season_results.append(player_stats)

        sim_summary =  pd.DataFrame(season_results, columns=["actual", "expected", "variance", "z_score"])

        self.game_results = game_results # for debug
        self.player_games = player_games
        return sim_summary