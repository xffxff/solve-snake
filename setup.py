from setuptools import setup

setup(name='solve_snake',
      version='0.0.1',
      install_requires=[
        'opencv_python==3.4.5.20',
        'joblib==0.13.0',
        'psutil==5.4.8',
        'cloudpickle==0.6.1',
        'numpy==1.15.4',
        'tqdm==4.29.0',
        'mpi4py==3.0.0',
        'pandas==0.23.4',
        'gym==0.10.9',]  # And any other dependencies foo needs
)  