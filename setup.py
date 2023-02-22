from setuptools import Extension, setup

module = Extension("kmeanssp", sources=["kmeansmodule.c"])
setup(name="kmeanssp",
    version="1.0",
    description="Allows you to run a C implementation of the kmeans algorithm from Python.",
    ext_modules=[module])