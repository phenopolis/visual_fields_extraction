import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hvf_extraction_script_msaifee786", # Replace with your own username
    version="0.0.4",
    author="Murtaza Saifee",
    author_email="saifeeapps@gmail.com",
    description="Python extraction script for HVF report images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/msaifee786/hvf_extraction_script",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
