from setuptools import setup


setup(
    name="dpsdc",
    version="0.1",
    description="Workshop on Social Determinats of Care and Proxies of Disparities",
    author="Adrien Carrel, Giovanni Angelotti, Luca Leon Weishaupt, Lelia Marie Hampton, Nicole Dundas, Pierandrea Morandini",
    packages=["dpsdc"],
    include_package_data=True,
    install_requires=[
        "python-dotenv==1.0.0",
        "jupyter==1.0.0",
        "pandas==2.1.1",
        "scikit-learn==1.3.1",
        "seaborn==0.13.0",
        "pandas-gbq==0.19.2",
        "scipy==1.11.3",
        "tableone==0.8.0",
        "appdata==2.2.0",
        "tqdm",
    ],
)

