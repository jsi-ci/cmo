from setuptools import setup, find_packages

setup(
    name='cmo',
    version='0.1.4',
    description='Constrained multiobjective benchmark suites and indicators',
    url='https://github.com/jsi-ci/cmo',
    author='Jordan Nicholas Cork',
    author_email='jordan_n_cork@hotmail.com',
    license='GPLv3',
    packages=find_packages(include=['cmo', 'cmo.*']),
    install_requires=[
        'numpy',
        'pymoo>=0.6.0'
    ],
    include_package_data=True
)
