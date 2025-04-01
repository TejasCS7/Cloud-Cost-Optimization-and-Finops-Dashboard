from setuptools import setup, find_packages

setup(
    name='cloud-finops-dashboard',
    version='1.0.0',
    description='Cloud FinOps Dashboard - GCP cost optimization and visualization tool',
    long_description=(
        'A Python-powered solution that transforms GCP billing data into actionable insights. '
        'Features advanced analytics, predictive modeling, and a Flask-based dashboard for '
        'optimizing cloud spending. Integrates with PostgreSQL for scalable cost management.'
    ),
    author='Tejas Gaikawad',
    author_email='tejasdgaikwad265@gmail.com',
    url='https://github.com/yourusername/cloud-finops-dashboard',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'psycopg2-binary>=2.9.0',
        'Flask>=3.0.0',
        'python-json-logger>=2.0.0',
    ],
    python_requires='>=3.8',
    license='MIT',  # SPDX license identifier
    entry_points={
        'console_scripts': [
            'finops-dashboard=src.main:main',
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: System Administrators',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: System :: Monitoring',
        'Topic :: Office/Business :: Financial',
    ],
    keywords=[
        'finops',
        'cloud-cost',
        'gcp',
        'postgresql',
        'dashboard'
    ],
    include_package_data=True,
    project_urls={
        'Source': 'https://github.com/yourusername/cloud-finops-dashboard',
        'Dataset': 'https://www.kaggle.com/datasets/sairamn19/gcp-cloud-billing-data',
    },
)