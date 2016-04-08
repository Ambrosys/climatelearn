from setuptools import setup, find_packages

setup(
    name="climatelearn",
    version="0.1",
    description="A toolbox for machine learning for climate networks data",
    author="Ruggero Vasile et al.",
    author_email="ruleva1983@gmail.com",
    url="http://github.com/Ambrosys/climatelearn/",
    packages=['climatelearn'],
    install_requires=['pandas', 'numpy', 'scipy', 'pybrain']
)
