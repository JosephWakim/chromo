import sys

from setuptools import setup, find_packages
import versioneer


if __name__ == "__main__":
    if sys.version_info < (3, 5):
        raise ImportError("requires Python>=3.5")

    setup(name="chromo",
          author="Spakowitz Lab",
          version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass(),
          url="https://github.com/SpakowitzLab/chromo",
          license="",
          long_description=open("README.md").read(),
          classifiers=["Intended Audience :: Science/Research",
                       'Intended Audience :: Developers',
                       'Development Status :: 1 - Pre-Alpha',
                       'License :: OSI Approved :: MIT License',
                       "Natural Language :: English",
                       "Programming Language :: Python :: 3 :: Only",
                       "Programming Language :: Fortran",
                       "Topic :: Scientific/Engineering :: Chemistry",
                       "Topic :: Scientific/Engineering :: Physics",
                       ],
          packages=find_packages(include=["chromo"]),
          install_requires=["numpy", "pandas"],
          )
