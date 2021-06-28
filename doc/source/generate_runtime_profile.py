"""This script lists the top n functions by cumulative runtime.

Before running this script, run the cProfile on the script being analyzed.

To run cProfile on a script, use:

'python -m cProfile -s "cumulative" -o "rstats" mc.py'
"""
import pstats
from pstats import SortKey

with open("../../performance_listing.txt", "w") as f:
    p = pstats.Stats('../../restats', stream=f)
    p.strip_dirs()
    p.sort_stats(SortKey.CUMULATIVE).print_stats(500)
