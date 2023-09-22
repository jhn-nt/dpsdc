from setuptools import setup


setup(
    name="dpsdc",
    version="0.1",
    description="Workshop on Social Determinats of Care and Proxies of Disparities",
    author="Giovanni Angelotti",
    packages=["dpsdc"],
    include_package_data=True,
    install_requires=[
        "pandas",
        "sklearn",
        "numpy",
        "seaborn",
        "shap",
        "tableone",
        "pandas-gbq",
        "cross-learn @ git+https://github.com/jhn-nt/cross-learn.git"
    ],
)

