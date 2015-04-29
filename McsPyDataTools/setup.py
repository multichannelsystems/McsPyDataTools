try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import os
import shutil
#import McsPy

def copy_html_docs():
    """Copy previously rendered HTML docs to folder docs-html"""
    if os.path.exists('docs/_build'):
        if os.path.exists('docs-html'):
            shutil.rmtree('docs-html')
        try:
            shutil.copytree('docs/_build/html', 'docs-html', ignore=ignore_patterns('_sources'))
        except shutil.Error as e:
            print('Build-Html folder not copied. Error %s occured' % e)
        except shutil.Error as e:
            print('Build-Html folder not copied. Error %s occured' % e)
    else:
        print('No current rendered HTML-Documentation available!')

copy_html_docs()

setup(
    name='McsPyDataTools',
    #version=McsPy.__version__,
    version='0.2.0',
    description='Handling data recorded and provided by MCS systems', 
    long_description=open('README').read(),
    keywords = 'HDF5 data electrophysiology MCS',
    author='J. Dietzsch, Multi Channel Systems MCS GmbH',
    author_email='dietzsch@multichannelsystems.com',
    zip_safe=True,
    packages=['McsPy', 'McsPy.Test'],
    package_data ={
        'McsPy.Test': ['TestData/*.h5']
    },
    scripts=['bin/McsPyDataTools.py','bin/PlotExperimentData.py'],
    url='http://multichannelsystems.com',
    license='LICENSE.txt',
    install_requires=[
        "Pint >= 0.4.1",
        "numpy >= 1.8.0",
        "h5py >= 2.2.0"
    ],
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
    ]
)