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
LINUX_OPTIONS = ['-fPIC']
WINDOWS_OPTIONS = ['-static-libgcc', '-static-libstdc++', '/usr/x86_64-w64-mingw32/lib/libwinpthread.a']

if should_make('superresolution/_tracker.so', ['tracker/tracker.cpp']):
    call(['g++'] + GPP_OPTIONS + LINUX_OPTIONS + ['tracker/tracker.cpp', '-o', 'superresolution/_tracker.so'])

if should_make('superresolution/_tracker.dll', ['tracker/tracker.cpp']):
    call(['x86_64-w64-mingw32-g++'] + GPP_OPTIONS + WINDOWS_OPTIONS + ['tracker/tracker.cpp', '-o', 'superresolution/_tracker.dll'])

if should_make('superresolution/libwinpthread-1.dll', ['/usr/x86_64-w64-mingw32/lib/libwinpthread-1.dll']):
    call(['cp', '/usr/x86_64-w64-mingw32/lib/libwinpthread-1.dll', 'superresolution/'])

setup(
    name='superresolution',
    version='0.0.1+snapshot',
    description='Superresolution Analysis Project',
    long_description='FOR INTERNAL USE ONLY',
    author='Christian C. Sachs',
    author_email='c.sachs@fz-juelich.de',
    url='https://github.com/modsim/superresolution-bacterial-analysis-tool',
    packages=['superresolution'],
    requires=['yaval', 'PySide', 'numpy', 'cv2', 'vispy', 'trackpy', 'scipy', 'pandas', 'numexpr'],
    license='BSD',
    py_modules=['superresolution'],
    package_data={
        'superresolution': [
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
        'Operating System :: MacOS :: MacOS X',   # lately no tests
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',    #
        'Programming Language :: Python :: 3.5',  # main focus
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ]
)
