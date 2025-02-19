from setuptools import setup


setup(
    name="dpsdc",
    version="0.14.4",
    description="Code for Care Phenotypes In Critical Care",
    author="LL. Weishaupt, T. Wang,  J. Schamroth,  P. Morandini,  J. Matos, LM Hampton,  J. Gallifant,  A. Fiske,  N. Dundas,  K. David,  LA. Celi,  A. Carrel, J. Byers,  G. Angelotti",
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
