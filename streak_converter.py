import re
import pandas as pd
import functools

w_pattern = re.compile(r"(W{1,}+)")
l_pattern = re.compile(r"(L{1,}+)")

def convert_to_streaks(bools, as_str=None, need_lengths=True):
    if not as_str:
        as_make_miss = bools.replace(0, "L") \
                            .replace(1, "W")
        as_str = "".join(as_make_miss)

    # yeah, this is pretty inefficient.
    make_streaks  = w_pattern.findall(as_str)
    miss_streaks =  l_pattern.findall(as_str)
    total_streaks = len(make_streaks) + len(miss_streaks)

    ret = {
        'raw_data': as_str,
        #'make_lengths': make_lengths.value_counts(),
        #'miss_lengths': miss_lengths.value_counts(),
        'make_streaks': make_streaks,
        'miss_streaks': miss_streaks,
        'total_streaks': total_streaks,
        'makes': as_str.count("W"),
        'misses': as_str.count("L")
        }
    
    if need_lengths:
        ret['make_lengths'] = pd.Series(map(len, make_streaks)).value_counts()
        ret['miss_lengths'] = pd.Series(map(len, miss_streaks)).value_counts()
    return ret