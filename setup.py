import numpy
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='test app',
    ext_modules=cythonize(["AmoebaPlayGround/GameEndChecker.pyx"], language_level=3, annotate=True),
    include_dirs=[numpy.get_include(), ],
    zip_safe=False
)
