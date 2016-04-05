# -*- coding: utf-8 -*-
"""
documentation
"""

from setuptools import setup, find_packages


setup(
    name='superresolution',
    version='0.0.1+snapshot',
    description='Superresolution Analysis Project',
    long_description='FOR INTERNAL USE ONLY',
    author='Christian C. Sachs',
    author_email='c.sachs@fz-juelich.de',
    url='https://github.com/modsim/superresolution-bacterial-analysis-tool',
    packages=['superresolution'],
    requires=['yaval'],
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
