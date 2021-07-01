"""Decorator for timing an arbitrary function and logging the runtime.

Joseph Wakim
July 1, 2021
"""

from typing import Callable, Any
from time import process_time
from pathlib import Path

import pandas as pd


def decorator_timed_path(output_dir: str):
    """Allows an output directory path to be passed into the decorator.

    Parameters
    ----------
    output_dir : str
        Path at which to save the runtime for the decorated function
    """
    def decorator_timed(func: Callable[[Any], Any]):
        """Create a log of runtime for a function.

        This decorator creates an output file by the name of the function and
        stores the runtime of that function when executed.

        Parameters
        ----------
        func : Callable[[Any], Any]
            Function on which the wrapper is applied
        """
        def wrapper(*args, **kwargs):
            """Run the function and save the runtime.
            """
            outdir = Path(output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            t_start = process_time()
            results = func(*args, **kwargs)
            t_end = process_time()
            runtime = t_end - t_start
            completion_time = pd.Timestamp.now()
            data = pd.DataFrame.from_dict(
                {"runtime": [runtime], "completion_time": [completion_time]}
            )
            file_name = func.__name__ + "_runtime"
            file_path = outdir / file_name
            write_header = not file_path.exists()
            data.to_csv(
                file_path, header=write_header, index=False, mode='a'
            )
            return results
        return wrapper
    return decorator_timed
