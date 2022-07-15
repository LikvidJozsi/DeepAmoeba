import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    name='test app',
    ext_modules=cythonize(["test_2.pyx", "AmoebaPlayGround/GameEndChecker.pyx"], language_level=3, annotate=True),
    include_dirs=[numpy.get_include(), ],
    zip_safe=False
)
