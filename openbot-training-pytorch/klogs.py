'''
Name: klogs.py
Description: A simple logging module for python
Author: Kevin Mahon
Date: 2023-08-25
Date Modified: 2023-08-25
'''
import logging
import argparse

def add_to_directory(path : str) -> None: 
    ''' 
    Adds kLogs to all python files in a directory
    Args:
        path (str): path to directory
    Returns:
        None
    '''
    import os 
    for file in os.listdir(path):
        if ".py" in file and file != "klogs.py":
            print(f"Adding kLogs to {file}")
            with open(file, "r+") as f:
                lines = f.readlines()
                lines.insert(0, "from klogs import kLogger\n")
                lines.insert(1, "TAG=__name__\n")
                lines.insert(2, "log = kLogger(tag=TAG)\n")
                f.seek(0,0)
                f.writelines(lines)

# create logger in file
class kLogger():
    '''
    kLogger class - a simple logging module for python
    Supports - debug, info, warning, error, critical with colored output
             - setting log level
             - setting log outputfile
    Args:
        tag (str): tag for logger
        logfile (str): path to logfile
        loglevel (str): log level, default is DEBUG

    Methods:
        debug(message)
        info(message)
        warning(message)
        error(message)
        critical(message)
        setLevel(level)
        setFile(file)
    '''

    def __init__(self, tag : str, logfile : str = None, loglevel : str = "DEBUG"):
        self.tag = tag 
        self.logfile = logfile
        self.loglevel = loglevel
        self.logger = logging.getLogger(self.tag)

        if loglevel:
            self.logger.setLevel(loglevel.upper())
        else:
            self.logger.setLevel(logging.DEBUG)

        if logfile:
            self.ch = logging.FileHandler(logfile)
        else:
            self.ch = logging.StreamHandler()
        if loglevel:
            self.ch.setLevel(loglevel.upper())
        else:
            self.ch.setLevel(logging.DEBUG)

        self.ch.setFormatter(kFormatter())
        self.logger.addHandler(self.ch)

    def debug(self, message : str) -> None:
        '''
        Print debug message
        Args:
            message (str): message to print
        Returns:
            None
        '''
        self.logger.debug(message, stacklevel=2)

    def info(self, message : str) -> None: 
        '''
        Print info message
        Args:
            message (str): message to print
        Returns:
            None
        '''
        self.logger.info(message, stacklevel=2)

    def warning(self, message : str) -> None:
        '''
        Print warning message
        Args:
            message (str): message to print
        Returns:
            None
        '''
        self.logger.warning(message, stacklevel=2)

    def error(self, message : str) -> None:
        '''
        Print error message
        Args:
            message (str): message to print
        Returns:
            None
        '''
        self.logger.error(message, stacklevel=2)

    def critical(self, message : str) -> None:
        '''
        Print critical message
        Args:
            message (str): message to print
        Returns:
            None
        '''
        self.logger.critical(message, stacklevel=2, stack_info=True)

    def setLevel(self, level : str) -> None:
        '''
        Set log level
        Args:
            level (str): log level
        Returns:
            None
        '''
        self.loglevel = level.upper()
        self.logger.setLevel(level.upper())
        self.ch.setLevel(level.upper())

    def setFile(self, file : str) -> None:
        '''
        Set log file
        Args:
            file (str): path to log file
        Returns:
            None
        '''
        self.logfile = file
        if file:
            self.logger.handlers.clear()
            self.logger = None
            self.logger = logging.getLogger(self.tag)

            if self.loglevel:
                self.logger.setLevel(self.loglevel.upper())
            else:
                self.logger.setLevel(logging.DEBUG)

            self.ch = logging.FileHandler(self.logfile)

            if self.loglevel:
                self.ch.setLevel(self.loglevel.upper())
            else:
                self.ch.setLevel(logging.DEBUG)

            self.ch.setFormatter(kFormatter())
            self.logger.addHandler(self.ch)

# create format
class kFormatter(logging.Formatter):
    '''
    kFormatter class - a simple colored logging format for python
    Supports - colored output
    Args:
        None
    '''
    #Color codes
    grey = "\x1b[34;20m" 
    blue = "\x1b[38;20m"
    yellow = "\x1b[36;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[41;1m"
    reset = "\x1b[0m"
    #general format string
    format = "%(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    #format dictionary
    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record : str) -> str:
        '''
        Format log message
        Args:
            record (str): log message
        Returns:
            formatted log message
        '''
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def test(logfile : str, loglevel : str) -> None:
    '''
    Test logger by printing all log levels
    Args:
        logfile (str): path to logfile
        loglevel (str): log level
    Returns:
        None
    '''
    log = kLogger("klogs", logfile, loglevel)
    log.debug("debug message")
    log.info("info message")
    log.warning("warning message")
    log.error("error message")
    log.critical("critical message")
    

if __name__ == "__main__":
    '''
    Examples:
        Test logger:
            python klogs.py -t -f test.log -l debug
        Add klogger to .py files in path:
            python klogs.py -p /home/user/project
        Help message:
            python klogs.py -h
    '''
    argparser = argparse.ArgumentParser(description='Klogs')
    argparser.add_argument('-t', '--test', help='Test Logger', action='store_true')
    argparser.add_argument('-f', '--file', help='Log file')
    argparser.add_argument('-l', '--level', help='Log level')
    argparser.add_argument('-p', '--path', help='Adds klogger to all .py files in path')
    args = argparser.parse_args()
    if args.test:
        test(args.file, args.level)
    elif args.path:
        add_to_directory(args.path)




