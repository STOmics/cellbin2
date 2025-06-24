import time
import tracemalloc

from cellbin2.utils import clog

CONSTANT = 1024
UNITS = ('B', 'KiB', 'MiB', 'GiB', 'TiB')
ALL_UNITS = {i: 1 if i == 'B' else 1 / (CONSTANT ** UNITS.index(i)) for i in UNITS}
DEFAULT_UNIT = 'MiB'
DECIMAL = 3


def process_decorator(unit=DEFAULT_UNIT):
    if unit not in UNITS:
        raise Exception(f"Only accept {UNITS}")

    def dec(func):
        def wrapper(*args, **kwargs):
            tracemalloc.start()
            start_time = time.time()
            result = func(*args, **kwargs)
            mem_cur, mem_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            clog.info(f"{func.__qualname__} memory peak: {round(mem_peak * ALL_UNITS[unit], DECIMAL)} {unit}")
            end_time = time.time()
            clog.info(f"{func.__qualname__} took {end_time - start_time:.4f} seconds to execute.")
            return result

        return wrapper

    return dec
