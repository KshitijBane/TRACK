from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='track',  
    version='2.0.0',  
    author='Kshitij Bane, Abhinav Narayan',
    description='TRAnsient cheCK',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),    
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'track = track.track:main'
        ]
    },
    include_package_data=True
)