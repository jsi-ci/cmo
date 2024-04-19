from setuptools import setup

setup(name='cmo',
      version='0.1',
      description='Constrained multiobjective benchmark suites and indicators',
      url='https://github.com/jsi-ci/cmo',
      author='Jordan Nicholas Cork',
      author_email='jordan_n_cork@hotmail.com',
      license='GPLv3',
      packages=['cmo'],
      install_requires=['numpy>=1.19.5', 'pymoo>=0.6.0', 'pygmo=>2.19.5'])
