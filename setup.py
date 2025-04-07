# setup.py
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="parallm",
    version="0.1.0",
    author="Rohit Krishnan",
    author_email="rohit.krishnan@email.com",
    description="CLI tool for querying multiple models with prompts from a CSV",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/strangeloopcanon/parallm",
    packages=setuptools.find_packages(exclude=["data*", "tests*"]),
    install_requires=[
        "pandas",
        "python-dotenv",
        "bodo",
        "llm",
    ],
    entry_points={
        'console_scripts': [
            'query-models=parallm.cli:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    license="MIT",
)