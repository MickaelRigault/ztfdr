#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""
from distutils.core import setup
from setuptools import setup, find_packages


packages = find_packages()
print(f"packages to be installed: {packages}")


VERSION = '0.2.0'
        
setup(name='ztfdr',
      version=VERSION,
      description='Tools to read the ZTF Ia DataReleases',
      author='Mickael Rigault',
      author_email='m.rigault@ipnl.in2p3.fr',
      url='https://github.com/MickaelRigault/ztfdr',
      packages=packages,
#      package_data={'ztfdr': ['data/*']},
#      scripts=["bin/__.py",]
     )
# End of setupy.py ========================================================


