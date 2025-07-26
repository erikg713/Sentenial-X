from setuptools import setup, find_packages

setup(
    name="sentenialx",
    version="0.1.0",
    packages=find_packages(include=["sentenialx", "sentenialx.*"]),
    install_requires=[
        "typer[all]",
        "requests",
    ],
    entry_points={
        "console_scripts": [
            "sentenialx = sentenialx.cli:app",
        ],
    },
    author="Your Name",
    description="📡 Sentenial-X AI Agent CLI Interface",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
