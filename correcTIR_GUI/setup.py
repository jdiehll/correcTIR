from setuptools import setup

APP = ['main.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': True,
    'packages': ['tkinter']
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)


# Instructions to try
# pip install py2app
# python setup.py py2app
# this should create an app bundle. Actually creating a pkg file involves additional steps that will likely take more time to work out. 