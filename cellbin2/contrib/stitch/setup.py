from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requires = f.read().splitlines()

with open('mfws/__init__.py') as f:
    info = {}
    for line in f:
        if 'version' in line:
            exec(line, info)
            break

print(f"Version: {info['version']}")


setup(
    name='MFWS',
    version=info['version'],
    description='Multiple Fast Fourier Transform weighted stitching algorithm.',
    # long_description_content_type="text/markdown",
    # long_description=open('README.md').read(),
    packages=find_packages(),
    author='cell bin research group',
    author_email='bgi@genomics.cn',
    url='',
    install_requires=requires,
    python_requires='==3.8.*',
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
    ],
    entry_points={
        'console_scripts': [
            'mfws=mfws.main:main',
        ]
    },
  )
