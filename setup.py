from setuptools import setup, find_packages

setup(
    name="parallm",
    version="0.1.0",
    description="CLI tool for querying multiple models with prompts from a CSV",
    author="Rohit Krishnan",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "python-dotenv",
        "bodo",
        "llm",
    ],
    entry_points={
        "console_scripts": [
            "query-models=parallm.cli:main",
        ],
    },
)
