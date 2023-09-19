from setuptools import setup


setup(
    name="dpsdc",
    version="0.1",
    description="Workshop on Social Determinats of Care and Proxies of Disparities",
    author="Giovanni Angelotti",
    packages=["dpsdc"],
    install_requires=[
        "jupyterlab",
        "pandas",
        "sklearn",
        "numpy",
        "seaborn",
        "shap",
        "tableone",
        "python-dotenv",
        "pandas-gbq",
        "ipywidgets==7.7.1",
    ],
)
