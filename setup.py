from setuptools import setup


setup(
    name="dpsdc",
    version="0.14.2",
    description="TBD",
    author="TBD A. Carrel, G. Angelotti, L.L. Weishaupt, L.M. Hampton, N. Dundas, P. Morandini, J. Byers, J. Gallifant, L.A. Celi",
    packages=["dpsdc"],
    include_package_data=True,
    python_requires="==3.10.13",
    install_requires=[
        "python-dotenv==1.0.0",
        "pandas==2.1.1",
        "scikit-learn==1.3.1",
        "seaborn==0.13.0",
        "pandas-gbq==0.19.2",
        "scipy==1.11.3",
        "tableone==0.8.0",
        "appdata==2.2.0",
        "tqdm==4.66.1",
        "shap==0.43.0",
        "lightgbm==4.1.0",
        "cross-learn @ git+https://github.com/jhn-nt/cross-learn.git",
        "pydata-google-auth==1.8.2"
    ],
)
