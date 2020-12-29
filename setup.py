# -*- coding: utf-8 -*-
"""
documentation
"""

from setuptools import setup, find_packages

from os.path import isfile, getmtime
from subprocess import call


def should_make(dst, src):
    src = src[0]
    return not isfile(dst) or getmtime(dst) < getmtime(src)


GPP_OPTIONS = ['-g', '-std=c++11', '-O3', '-Itracker', '-shared']
# GPP_OPTIONS += ['-D_GLIBCXX_DEBUG']
LINUX_OPTIONS = ['-fPIC']
WINDOWS_OPTIONS = ['-static-libgcc', '-static-libstdc++', '/usr/x86_64-w64-mingw32/lib/libwinpthread.a']

if should_make('suremco/_tracker.so', ['tracker/tracker.cpp']):
    call(['g++'] + GPP_OPTIONS + LINUX_OPTIONS +
         ['tracker/tracker.cpp', '-o', 'suremco/_tracker.so'])

if should_make('suremco/_tracker.dll', ['tracker/tracker.cpp']):
    call(['x86_64-w64-mingw32-g++'] + GPP_OPTIONS + WINDOWS_OPTIONS +
         ['tracker/tracker.cpp', '-o', 'suremco/_tracker.dll'])

if should_make('suremco/libwinpthread-1.dll', ['/usr/x86_64-w64-mingw32/lib/libwinpthread-1.dll']):
    call(['cp', '/usr/x86_64-w64-mingw32/lib/libwinpthread-1.dll', 'suremco/'])

setup(
    name='suremco',
    version='0.0.1+snapshot',
    description='Superresolution Emitter Counter',
    long_description='',
    author='Christian C. Sachs',
    author_email='c.sachs@fz-juelich.de',
    url='https://github.com/modsim/suremco',
    packages=find_packages(),
    requires=['yaval', 'PySide2', 'numpy', 'cv2', 'vispy', 'trackpy', 'scipy', 'pandas', 'numexpr'],
    license='BSD',
    py_modules=['suremco'],
    package_data={
        'suremco': [
            '_tracker.so',
            '_tracker.dll',
            'libwinpthread-1.dll'
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ]
)
