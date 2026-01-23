from setuptools import setup, find_packages

# Load long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hagfish-adaptive-trainer",
    version="0.3.0",
    description="Adaptive training budget optimizer (bio-inspired Hagfish agent)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sarvesh PV",
    author_email="pvsarvesh29@gmail.com",
    url="https://github.com/Sarrvessh/Hagfish_Agent_System",
    project_urls={
        "Documentation": "https://github.com/Sarrvessh/Hagfish_Agent_System#readme",
        "Source": "https://github.com/Sarrvessh/Hagfish_Agent_System",
        "Tracker": "https://github.com/Sarrvessh/Hagfish_Agent_System/issues",
    },
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24,<2.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
