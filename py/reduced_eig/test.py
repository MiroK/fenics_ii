import os
from os.path import splitext, join

module = '.'
ignore = ('test.py')
for f in os.listdir(module):
    if splitext(f)[1] == '.py' and f not in ignore:
        print '\t', f
        execfile(join('.', module, f))
