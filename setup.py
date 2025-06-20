from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="centroid-model-blastospim",
    version="0.1.0",
    author="Dinesh Chhantyal",
    author_email="myagdichhantyal@gmail.com",
    description="A machine learning project for centroid detection in blastospim data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/centroid_model_blastospim",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "pre-commit>=2.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "centroid-train=scripts.train:main",
            "centroid-inference=scripts.inference:main",
            "centroid-preprocess=scripts.preprocess_data:main",
            "centroid-evaluate=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)
