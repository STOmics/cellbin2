import sys

from pathlib import Path

cb_path = str(Path(__file__).parents[2])
sys.path.append(cb_path)

from cellbin2.utils import clog


def test_clog():
    clog.log2file("/media/Data/dzh/data/cellbin2/test/log")
    clog.info("info")
    clog.debug("debug")
    clog.warning("warning")
    clog.error("error")
    clog.critical("critical")
