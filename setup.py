import setuptools

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="parallm",
    version="0.1.0",
    author="Rohit Krishnan",
    author_email="rohit.krishnan@email.com",
    description="CLI tool for querying multiple models with prompts from a CSV",
    long_description=long_description,
    long_description_content_type="text/markdown", # Important for PyPI rendering
    url="https://github.com/strangeloopcanon/parallm",
    # Find packages automatically in the current directory, excluding tests/data
    packages=setuptools.find_packages(exclude=["data*", "tests*"]),
    # Specify dependencies
    install_requires=[
        "pandas",
        "python-dotenv",
        "bodo",
        "llm",
    ],
    # Define the command-line script
    entry_points={
        'console_scripts': [
            'query-models=parallm.cli:main',
        ],
    },
    # Classifiers help users find your project
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7', # Specify compatible Python versions
    license="MIT", # Short license name
    # Include the license file in source distributions (good practice)
    # 'license_files' is preferred for newer setuptools, but 'license' is essential.
    # Setuptools often includes LICENSE automatically if found, but being explicit is safer for sdists.
    # If you encounter issues with LICENSE not being included in sdist, add:
    # license_files = ('LICENSE',), # Tuple of license file names
)