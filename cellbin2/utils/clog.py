# Log Management
import os
import os.path as o_p
import logging
# from coloredlogs import ColoredFormatter
import sys
import time
import io

PROG_VERSION = '0.1'
PROG_DATE = '2021-07-25'

usage = '''
Version %s  by Chen Bichao  %s
Usage: 
import clog
clog.log2file()

clog.info("Test info.")
clog.warning("Logging warning.")
''' % (PROG_VERSION, PROG_DATE)

# Logging levels
ROOT = logging.root
CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG

# create a dict for level name strings
level_name = {
    CRITICAL: "CRITICAL",
    ERROR: "ERROR",
    WARNING: "WARNING",
    INFO: "INFO",
    DEBUG: "DEBUG"
}


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"  # add green escape sequence
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    # format = "[%(levelname).4s %(asctime)s p%(process)s %(funcName)s %(filename)s:%(lineno)s] %(message)s"
    format = "[%(levelname).4s %(asctime)s p%(process)s %(funcName)s %(filename)s:%(lineno)s]"
    # new format string for independently configuring msg color
    msg_format = " %(message)s "

    FORMATS = {
        logging.DEBUG: green + format + reset + grey + msg_format + reset,
        logging.INFO: green + format + reset + grey + msg_format + reset,
        logging.WARNING: yellow + format + reset + grey + msg_format + reset,
        logging.ERROR: red + format + reset + grey + msg_format + reset,
        logging.CRITICAL: bold_red + format + reset + grey + msg_format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class CLogger(object):
    """ Custom logger class to format and instantiate logger. """
    file_handler = None

    def __init__(self):
        """ Initialize logger, default logger has one StreamHandler. """
        self.logger = logging.getLogger(__name__)
        self.stream_handler = logging.StreamHandler(sys.stdout)
        self.stream_handler.setFormatter(CustomFormatter())
        self.logger.addHandler(self.stream_handler)
        self.logfile = ''

    def log2file(self, out_dir='', filename=''):
        """ Save logging to file. """
        if len(self.logfile) == 0:
            self.logfile = self._set_logfile(out_dir, filename)
            self.file_handler = logging.FileHandler(self.logfile, 'a')
            # if not len(self.logger.handlers):
            self.file_handler.setFormatter(self.st_formatter())
            self.logger.addHandler(self.file_handler)
        else:
            print("log file is set.")
            print(self.logfile)

    @staticmethod
    def _set_logfile(out_dir='', filename=''):
        """ Set logging file. """
        if len(out_dir) == 0:
            out_dir = o_p.join(o_p.dirname(o_p.dirname(__file__.replace('\\', '/'))), 'log')
        os.makedirs(out_dir, exist_ok=True)

        if len(filename) == 0:
            filename = time.strftime("%Y%m%d-%H-%M-%S", time.localtime()) + '.log'
        return o_p.join(out_dir, filename)

    def get_logfile(self):
        """ Get current logging file. """
        for hdl in self.logger.handlers:
            if isinstance(hdl, logging.FileHandler):
                return hdl.baseFilename
        return ''

    @staticmethod
    def st_formatter():
        """ Logging formatter. """
        # Sample: [INFO 20210723-15:41:55 p12027 <module> stlog.py:153] This is a debug message
        fmt_str = "[%(levelname).4s %(asctime)s p%(process)s %(funcName)s %(filename)s:%(lineno)s] %(message)s"
        return logging.Formatter(fmt=fmt_str, datefmt="%Y%m%d-%H-%M-%S", style='%')
        # return ColoredFormatter(fmt_str)

    def set_level(self, level=DEBUG):
        """ Set logger level. """
        try:
            self.logger.setLevel(level)
        except Exception as e:
            print(e)
        self.logger.debug("Logging level is set to: {}".format(level_name[level]))


class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)


# Set logging methods
cl = CLogger()
logger = cl.logger

get_logfile = cl.get_logfile
log2file = cl.log2file
set_level = cl.set_level

# set logging base level to INFO
set_level(INFO)

debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical
tqdm_out = TqdmToLogger(logger)
exception = logger.exception

