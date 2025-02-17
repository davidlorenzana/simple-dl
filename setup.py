from setuptools import setup, find_packages

setup(
    name="simple_dl",
    version="0.1.0",
    description="simple_dl is a Python library for efficiently fetching and caching web content. It supports rate limiting, automatic retries, and response caching via SQLite with optional compression. Additionally, its `TorDownloader` subclass routes requests through the Tor network for enhanced anonymity, making it ideal for web scraping, API interactions, and media downloads.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="David Lorenzana Martínez",
    author_email="davlorenzana@gmail.com",
    url="https://github.com/davidlorenzana/simple-dl",
    packages=find_packages(),
    install_requires=[
        "requests",
        "tenacity",
        "pillow",
        "xxhash"
    ],
    extras_require={
        'tor': ['stem', 'requests[socks]'],
    },
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
)
