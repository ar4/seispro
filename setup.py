import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seispro",
    version="0.0.4",
    author="Alan Richardson",
    author_email="alan@ausargeo.com",
    description="Seismic processing tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ar4/seispro",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['torch>=1.9.0',
                      'numpy'],
)
