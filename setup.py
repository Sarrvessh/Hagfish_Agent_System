from setuptools import setup, find_packages

# Load long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hagfish-adaptive-trainer",
    version="0.1.1",
    packages=find_packages(),
    description="Adaptive resource optimizer for ML training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scikit-learn"
    ],
)
