import time
import tracemalloc
import functools
from cellbin2.utils import clog

CONSTANT = 1024
UNITS = ('B', 'KiB', 'MiB', 'GiB', 'TiB')
ALL_UNITS = {i: 1 if i == 'B' else 1 / (CONSTANT ** UNITS.index(i)) for i in UNITS}
DEFAULT_UNIT = 'MiB'
DECIMAL = 3

# 全局状态管理
_TRACEMALLOC_STARTED = False
_GLOBAL_PEAK_MEMORY = 0  # 记录全局内存峰值（字节）


def process_decorator(unit=DEFAULT_UNIT):
    """

    Args:
        unit (str):
    """
    if unit not in UNITS:
        raise Exception(f"Only accept {UNITS}")

    def dec(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            global _TRACEMALLOC_STARTED, _GLOBAL_PEAK_MEMORY
            track_self = True
            if not _TRACEMALLOC_STARTED:
                tracemalloc.start()
                _TRACEMALLOC_STARTED = True
                track_self = False
                _GLOBAL_START_TIME = time.time()

            #  snapshot_before
            snapshot_before = None
            if track_self:
                snapshot_before = tracemalloc.take_snapshot()

            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            current_mem, peak_mem = tracemalloc.get_traced_memory()
            _GLOBAL_PEAK_MEMORY = max(_GLOBAL_PEAK_MEMORY, peak_mem)

            # Diff Algorithm
            if track_self and snapshot_before:
                snapshot_after = tracemalloc.take_snapshot()
                stats = snapshot_after.compare_to(snapshot_before, 'lineno')
                display_mem = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

                # log
                mem_value = round(display_mem * ALL_UNITS[unit], DECIMAL)
                clog.info(f"{func.__qualname__} memory peak: {mem_value} {unit}")
                clog.info(f"{func.__qualname__} took {end_time - start_time:.4f} seconds to execute.")

            # end    'pipeline'
            if func.__name__ == 'pipeline':
                tracemalloc.stop()
                _TRACEMALLOC_STARTED = False
                clog.info(f"{func.__qualname__} memory peak: {round(_GLOBAL_PEAK_MEMORY * ALL_UNITS[unit], DECIMAL)} {unit}")
                clog.info(f"{func.__qualname__} took {end_time - _GLOBAL_START_TIME:.4f} seconds to execute.")
                # clog.info(f"GLOBAL MEMORY PEAK: {round(_GLOBAL_PEAK_MEMORY * ALL_UNITS[unit], DECIMAL)} {unit}")

            return result

        return wrapper

    return dec