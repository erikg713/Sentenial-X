from setuptools import setup, find_packages

setup(
    name="pentest_suite",
    version="0.1.0",
    packages=find_packages(where=".", exclude=["tests", "templates"]),
    install_requires=[
        "Flask",
        # add your actual requirements
    ],
    entry_points={
        "console_scripts": [
            "pentest-suite-cli=cli:main"
        ]
    },
    include_package_data=True,
    author="Your Name",
    description="Modular penetration testing suite with plugin support",
    url="https://github.com/yourorg/pentest_suite",
    license="MIT"
)
