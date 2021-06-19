import os
from setuptools import setup, find_packages


with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def get_about():
    about = {}
    basedir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(basedir, "levenshtein_finder", "about.py")) as f:
        exec(f.read(), about)
    return about


def requirements():
    with open(
        os.path.join(os.path.dirname(__file__), "requirements.txt"), encoding="utf-8"
    ) as f:
        return f.read().splitlines()


about = get_about()
setup(
    name="levenshtein_finder",
    version=about["__version__"],
    author=about["__author__"],
    description=about["__description__"],
    url='https://github.com/lovit/levenshtein_finder',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements(),
    keywords=[],
    packages=find_packages(),
    entry_points={},
)
