import re

import numpy as np
import pandas as pd
import scipy.stats

import wald_wolfowitz
import streak_converter

class StreaksBase:
    """
    Base class for streak analysis
    """
    def get_stats_dataframe(self):
        df = pd.DataFrame(columns=["player_id", "player_name", "makes", 
                                "misses", "total_streaks", "raw_data"])
        return df.set_index("player_id")    

    def convert_to_streaks(self, bools, as_str=None):
        return streak_converter.convert_to_streaks(bools, as_str)

    def get_expected_streaks(self, makes, misses):
        return wald_wolfowitz.get_expected_streaks(makes, misses)

    def get_variances(self, makes, misses, expected_streaks):
        return wald_wolfowitz.get_variance(makes, misses, expected_streaks)

    def calc_stats(self, df, percentile_ranks=False):
        df['expected_streaks'] = self.get_expected_streaks(df.makes, df.misses)
        df['variance'] = self.get_variances(df.makes, df.misses, df.expected_streaks)
        df['sd'] = np.sqrt(df.variance)
        df['z_score'] = (df.total_streaks - df.expected_streaks) / df.sd

        df['ww_percentile'] = 100 * scipy.stats.norm.cdf(df.z_score.tolist())

        if percentile_ranks:
            # get percentile rank data
            # this will be unbelievably slow on large datasets (makes + misses > 50, say)
            # and the difference from the WW percentile will be minimal.x
            df['exact_percentile_rank'] = None

            for row_id, data in df.iterrows():
                df.loc[row_id, "exact_percentile_rank"] = wald_wolfowitz.get_percentile_rank(data['makes'], data['misses'], data['total_streaks'])
                
            df['z_from_percentile_rank'] = scipy.stats.norm.ppf(
                (df['exact_percentile_rank'].values/ 100).tolist())

        #clean_df = df[(df.total_streaks > 0) & (df.variance > 0)].copy()
        clean_df = df.copy()

        return clean_df