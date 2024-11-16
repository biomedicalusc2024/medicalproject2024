import sys

def print_sys(s):
    """system print
    Args:
        s (str): the string to print
    """
    print(s, flush=True, file=sys.stderr)
