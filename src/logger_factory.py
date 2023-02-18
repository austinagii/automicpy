import autosklearn.classification
import category_encoders as ce
import featuretools as ft
import sys
import logging

class LoggerFactory_:
    
    LOG_LEVEL_BY_NAME = {
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }   
     
    def __init__(self):
        self.loggers = {}
        
    def get_logger(self, name, level='debug'):
        if level not in self.LOG_LEVEL_BY_NAME.keys():
            raise ValueError(f"Unsupported logging level '{level}'")
        name = name.upper()
            
        if self.loggers.get(name) is None:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter("{name}::{levelname} - {message}", style='{')
            handler.setFormatter(formatter)
            
            log = logging.getLogger(name)
            # just in case this logger already exists, remove the handlers
            log.handlers = []
            
            log.addHandler(handler)
            log.setLevel(self.LOG_LEVEL_BY_NAME[level])
            self.loggers[name] = log
            
        return self.loggers[name]