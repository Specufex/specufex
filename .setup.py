import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SpecUFEx",  # Replace with your own username
    version="0.1.1",
    author="Nate Groebner",
    author_email="ngroe0029@gmail.com",
    description="Python implementation of SpecUFEx",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/specufex/specufex/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
