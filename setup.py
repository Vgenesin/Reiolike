# To make te package installable via pip, we need to create a setup.py file in the root directory of the project. 
# This file will contain the necessary information about the package, such as its name, version, author, and dependencies.

from setuptools import setup, find_packages

setup(
    name="Reiolike", # Nome aggiornato del pacchetto
    version="0.1",
    packages=find_packages(), # cercherà automaticamente la cartella Reiolike
    install_requires=[
        "cobaya",
        "numpy",
        "corecon", # Aggiunto come dipendenza esterna
    ],
)