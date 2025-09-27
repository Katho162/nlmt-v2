from setuptools import setup, find_packages

setup(
    name="nlmt-v2",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "torch",
        "scikit-learn",
        "fastapi",
        "uvicorn",
        "optuna",
    ],
    entry_points={
        "console_scripts": [
            "recommend=nlmt_v2.main:main",
        ],
    },
)
