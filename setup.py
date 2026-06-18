from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="white-box-activation-steering",
    version="0.1.0",
    author="Muhammad Adam",
    author_email="madam2@andrew.cmu.edu",
    description=(
        "White-box activation steering to prevent deceptive alignment "
        "in cooperative multi-agent systems"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Emadamx/white-box-activation-steering",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
)
