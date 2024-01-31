import logging


# Custom formatter
class hps_formatter(logging.Formatter):

    default_fmt  = "|{levelname:<7s}| {message}"
    info_fmt  = "{message}"
    debug_fmt  = "|{levelname:<7s}| {filename}:{funcName}:{lineno} {message}"

    def __init__(self):
        super().__init__(fmt=self.default_fmt, datefmt=None, style='{')  
    
    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt

        # Replace the original format with one customized by logging level
        self._style._fmt = hps_formatter.default_fmt
        if record.levelno == logging.DEBUG:
            self._style._fmt = hps_formatter.debug_fmt
        elif record.levelno == logging.INFO:
            self._style._fmt = hps_formatter.info_fmt


        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        #self._style._fmt = format_orig

        return result
