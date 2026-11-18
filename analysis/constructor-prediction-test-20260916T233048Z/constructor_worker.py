"""Private local fixture worker; no credential or remote model entry."""
import sys
from pipeline import _construct_child
if __name__=='__main__':
    _construct_child(*sys.argv[1:])
