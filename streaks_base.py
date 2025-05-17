import re

import numpy as np
import pandas as pd

class StreaksBase:
    """
    Base class for streak analysis
    """
    def get_stats_dataframe(self):
        df = pd.DataFrame(columns=["player_id", "player_name", "makes", 
                                "misses", "total_streaks", "raw_data"])
        return df.set_index("player_id")    

    def convert_to_streaks(self, bools, as_str=None):
        if not as_str:
            as_make_miss = bools.replace(0, "L").replace(1, "W")
            as_str = "".join(as_make_miss)

        make_streaks  = re.findall(r"(W{1,}+)", as_str)
        miss_streaks = re.findall(r"(L{1,}+)", as_str)
        total_streaks = len(make_streaks) + len(miss_streaks)

        make_lengths = pd.Series(map(len, make_streaks))
        miss_lengths = pd.Series(map(len, miss_streaks))

        return {
            'raw_data': as_str,
            'make_lengths': make_lengths.value_counts(),
            'miss_lengths': miss_lengths.value_counts(),
            'make_streaks': make_streaks,
            'miss_streaks': miss_streaks,
            'total_streaks': total_streaks,
            'makes': as_str.count("W"),
            'misses': as_str.count("L")
            }

    def get_expected_streaks(self, makes, misses):
        """
        Use Wald_Wolfowitz test to compute expected number of streaks.
        """
        return  (2 * (makes * misses) / (makes + misses)) + 1

    def get_variances(self, makes, misses, expected_streaks):
        # https://en.wikipedia.org/wiki/Wald%E2%80%93Wolfowitz_runs_test#Definition
        numerator = (expected_streaks - 1) * (expected_streaks - 2)
        denominator = (makes + misses - 1)
        return numerator / denominator

    def calc_stats(self, df):
        df['expected_streaks'] = self.get_expected_streaks(df.makes, df.misses)
        df['variance'] = self.get_variances(df.makes, df.misses, df.expected_streaks)
        df['sd'] = np.sqrt(df.variance)
        df['z_score'] = (df.total_streaks - df.expected_streaks) / df.sd

        #clean_df = df[(df.total_streaks > 0) & (df.variance > 0)].copy()
        clean_df = df.copy()

        return clean_df