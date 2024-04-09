from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fair_trees",
    version="2.4.2",
    packages=find_packages(),
    description="This package learns fair decision tree classifiers which can then be bagged into fair random forests, following the scikit-learn API standards.",
    author="Antonio Pereira Barata",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email="apbarata@gmail.com",
    url="https://github.com/pereirabarataap/fair_tree_classifier",
    install_requires=[
        "scipy",
        "numpy",
        "pandas",
        "joblib",
        "scikit-learn"
    ],
)