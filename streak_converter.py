import re
import pandas as pd

def convert_to_streaks(bools, as_str=None):
    if not as_str:
        as_make_miss = bools.replace(0, "L") \
                            .replace(1, "W")
        as_str = "".join(as_make_miss)

    # yeah, this is pretty inefficient.
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