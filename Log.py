import sys
import contextlib


@contextlib.contextmanager
def smart_open(filename=None):
    if filename and filename != '-':
        fh = open(filename, 'a')
    else:
        fh = sys.stdout

    try:
        yield fh
    finally:
        if fh is not sys.stdout:
            fh.close()

class Log():
    def __init__(self, log_path):
        self.log_path = log_path

    def write(self, content):
        with smart_open(self.log_path) as file:
            file.write(content)
            if self.log_path != "-":
                print(content)

