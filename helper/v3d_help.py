'''
   Helper files for Vote3Deep. It should be eliminated or merged into other file
'''
from __future__ import print_function
import glob

def load_and_sort_paths(pathname):
    '''
      Performs Unix-style path expansion, returning a sorted list
      of paths to files that meet the pathname criteria.
      The argument pathname can be absolute or relative.

      e.g.
        pathname="/local/home/user/*.txt"
        load_and_sort_paths(pathname)

        # Returns all files with extension 'txt'
        # in the /local/home/user directory
    '''
    pathname_expansion = glob.glob(pathname)
    pathname_expansion.sort()
    return pathname_expansion
