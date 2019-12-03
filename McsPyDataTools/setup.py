try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import os
import shutil
#import McsPy

# Utility methods

def get_current_version():
    """Get current version from McsPy.__init__"""
    version = "0"
    try:
        f = open("./McsPy/__init__.py", "r")
        try:
            for line in f:
                if line.startswith("version"):
                    version = line.split()[2][1:-1]
                    break
        finally:
            f.close()
    except:
        version = "0"
    return version

def copy_html_docs():
    """Copy previously rendered HTML docs to folder docs-html"""
    if os.path.exists('docs/_build'):
        if os.path.exists('docs-html'):
            shutil.rmtree('docs-html')
        try:
            shutil.copytree('docs/_build/html', 'docs-html', ignore=shutil.ignore_patterns('_sources'))
        except shutil.Error as e:
            print('Build-Html folder not copied. Error %s occured' % e)
        except shutil.Error as e:
            print('Build-Html folder not copied. Error %s occured' % e)
    else:
        print('No current rendered HTML-Documentation available!')

def copy_pdf_docs():
    """Copy previously rendered PDF docs to folder docs-pdf"""
    if os.path.exists('docs-pdf'):
        shutil.rmtree('docs-pdf')
    if os.path.exists('docs/_build/latex/McsPyDataTools.pdf'):
        try:
            os.mkdir('docs-pdf')
            shutil.copy('docs/_build/latex/McsPyDataTools.pdf', 'docs-pdf/McsPyDataTools.pdf')
        except shutil.Error as e:
            print('PDF documentation not copied. Error %s occured' % e)
    else:
        print('No current PDF-Documentation available!')

copy_html_docs()
copy_pdf_docs()

setup(
    name='McsPyDataTools',
    #version=McsPy.__version__,
    version= get_current_version(),
    description='Handling data recorded and provided by MCS systems', 
    long_description=open('README.rst').read(),
    keywords = 'HDF5 data electrophysiology MCS',
    author='J. Dietzsch, Multi Channel Systems MCS GmbH',
    author_email='dietzsch@multichannelsystems.com',
    zip_safe=True,
    packages=['McsPy', 'McsPy.tests'],
    # Provide test data as an accompanied separate archive! -> just create the folder and show README.md
    #package_data ={
    #    'McsPy.tests': ['TestData/*.h5']
    #},
    package_data ={
        'McsPy.tests': ['TestData/README.md']
    },
    scripts=['bin/McsPyDataTools.py', 'bin/DataStreamInfo.py', 'bin/InertialAnalysis_6DoF-IMU.py'],
    url='http://multichannelsystems.com',
    license='LICENSE.txt',
    install_requires=[
        "Pint >= 0.8.1",
        "numpy >= 1.14.3",
        "h5py >= 2.8.0",
        "tabulate >= 0.8.2"
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Operating System :: Unix',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]
)
