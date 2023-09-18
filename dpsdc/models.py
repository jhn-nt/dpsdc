from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from crlearn.evaluation import crossvalidate_classification

import numpy as np
import pandas as pd

from shap import LinearExplainer, TreeExplainer
from shap import summary_plot
from shap.plots import beeswarm
from itertools import product

import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt

from .etl import categorical_features, continuous_features, ordinal_features
from .etl import load_channels


class WorkshopModel:
    cv = RepeatedStratifiedKFold

    def __init__(self, outcome, features, covariates, n_repeats=10, random_state=0):
        self.outcome = outcome
        self.features = list(set(features).difference([self.outcome]))
        self.covariates = covariates
        self.random_state = random_state

        self.cv = self.cv(n_repeats=n_repeats)

        self.model = self.build(self.random_state, self.features, self.covariates)

    @staticmethod
    def build(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def tracing_func(*args, **kwargs):
        raise NotImplementedError()

    def plot(self):
        raise NotImplementedError()

    def fit(self, dataset):
        X = dataset[self.features + self.covariates]
        y = dataset[self.outcome].astype(int)
        performance, scores, _ = crossvalidate_classification(
            self.model, X, y, cv=self.cv, tracing_func=self.tracing_func, name=self.name
        )

        self.performance = performance["metrics"].query("side=='test'")
        self.scores = scores
        self.curves = performance["curves"]["roc"]
        self.fitted_model = self.model.fit(X, y)
        return self

    def plot_roc(self):
        roc = self.curves
        auc_mean = self.performance.groupby("class").mean().iloc[1].auc
        auc_std = self.performance.groupby("class").std().iloc[1].auc

        roc_mean = roc.query("side=='test'").groupby("fpr").mean()[1]
        roc_std = roc.query("side=='test'").groupby("fpr").std()[1]
        fig, ax = plt.subplots(figsize=(7, 7))

        _ = ax.plot(roc_mean.index, roc_mean, color=cm.tab10(0))
        _ = ax.fill_between(
            roc_mean.index, roc_mean, roc_mean + roc_std, color=cm.tab10(0), alpha=0.5
        )
        _ = ax.fill_between(
            roc_mean.index, roc_mean - roc_std, roc_mean, color=cm.tab10(0), alpha=0.5
        )

        _ = ax.plot([0, 1], [0, 1], color="k", alpha=0.3)
        _ = ax.set_xticks(np.arange(0, 1.1, 0.1))
        _ = ax.set_yticks(np.arange(0, 1.1, 0.1))
        _ = ax.grid(alpha=0.3)
        _ = ax.set_title(f"AUCROC - {self.name} - Outcome:{self.outcome}")
        _ = ax.text(0.7, 0.2, f"AUC:{auc_mean:.2f} (SD:{auc_std:.2f})")
        return fig


class LogisticRegressionModel(WorkshopModel):
    name = "Logistic Regression"

    @staticmethod
    def tracing_func(model):
        features = model[0].get_feature_names_out()
        features = [feature.split("__")[1] for feature in features]
        odds = np.exp(model[-1].coef_)
        return pd.DataFrame(odds, columns=features)

    @staticmethod
    def build(random_state, features, covariates):
        processor = ColumnTransformer(
            [
                (
                    "categorical",
                    OneHotEncoder(drop="if_binary"),
                    categorical_features(features),
                ),
                (
                    "continuous",
                    StandardScaler(),
                    continuous_features(features) + ordinal_features(features),
                ),
                (
                    "covariates",
                    make_pipeline(
                        SimpleImputer(),
                        StandardScaler(),
                        PCA(random_state=random_state, n_components=0.9),
                    ),
                    covariates,
                ),
            ]
        )

        classifier = LogisticRegression(
            random_state=random_state, class_weight="balanced"
        )
        return make_pipeline(processor, classifier)

    def plot(self):
        odds = pd.concat(self.scores.values())
        odds = odds[odds.mean().sort_values().index]

        features = [feature for feature in odds.columns if "pca" not in feature]
        odds = odds[features]
        fig, ax = plt.subplots(figsize=(10, 10))
        _ = odds.boxplot(ax=ax, vert=False, showfliers=False)
        _ = ax.plot([1, 1], ax.get_ylim(), color="k", alpha=0.3)
        _ = ax.set_xlabel("Odds")
        _ = ax.set_title(f"Logistic Regression - Outcome: {self.outcome}")
        fig.tight_layout()
        return fig

    def beeswarm(self, train_set, test_set):
        def transform(data):
            return self.fitted_model[:-1].transform(data).astype(float)

        def predict(data):
            return self.fitted_model[-1].predict_proba(data)

        train_set = train_set[self.features + self.covariates]
        test_set = test_set[self.features + self.covariates]
        features = self.fitted_model[0].get_feature_names_out()
        features = [feature.split("__")[1] for feature in features]

        features_ix = self.fitted_model[0].get_feature_names_out()
        features_ix = [
            i for (i, feature) in enumerate(features_ix) if "pca" not in feature
        ]

        explainer = LinearExplainer(
            self.fitted_model[-1],
            transform(train_set.sample(500)),
            feature_names=features,
        )
        shap_values = explainer(transform(test_set.sample(500)))[:, features_ix]

        plt.clf()
        beeswarm(shap_values, max_display=20, show=False)
        fig = plt.gcf()
        fig.tight_layout()
        fig.suptitle(f"SHAP - Logistic Regression - Outcome: {self.outcome}")
        return fig


class RandomForestModel(WorkshopModel):
    name = "Random Forest"

    @staticmethod
    def build(random_state, features, covariates):
        processor = ColumnTransformer(
            [
                (
                    "categorical",
                    OneHotEncoder(drop="if_binary"),
                    categorical_features(features),
                ),
                (
                    "continuous",
                    SimpleImputer(),
                    continuous_features(features + covariates)
                    + ordinal_features(features + covariates),
                ),
            ],
            remainder="passthrough",
        )

        classifier = RandomForestClassifier(
            random_state=random_state, class_weight="balanced"
        )
        return make_pipeline(processor, classifier)

    @staticmethod
    def tracing_func(model):
        features = model[0].get_feature_names_out()
        features = [feature.split("__")[1] for feature in features]
        fi = model[-1].feature_importances_
        return pd.Series(fi, index=features)

    def plot(self):
        fi = pd.concat(self.scores.values(), axis=1).T
        fi = fi[fi.mean().sort_values().index]
        features = [feature for feature in fi.columns if feature not in self.covariates]
        fi = fi[features]
        fig, ax = plt.subplots(figsize=(10, 10))
        _ = fi.boxplot(ax=ax, vert=False, showfliers=False)
        _ = ax.set_xlabel("Feature Importance")
        _ = ax.set_title(f"Random Forest - Outcome: {self.outcome}")
        fig.tight_layout()
        return fig

    def beeswarm(self, train_set, test_set):
        def transform(data):
            return self.fitted_model[:-1].transform(data).astype(float)

        train_set = train_set[self.features + self.covariates]
        test_set = test_set[self.features + self.covariates]

        features = self.fitted_model[0].get_feature_names_out()
        features = [feature.split("__")[1] for feature in features]

        features_ix = [
            i for (i, feature) in enumerate(features) if feature not in self.covariates
        ]

        explainer = TreeExplainer(
            self.fitted_model[-1],
            transform(train_set.sample(1000)),
            feature_names=features,
        )
        shap_values = explainer(
            transform(test_set.sample(500)), check_additivity=False
        )  # this needs to be fixed

        plt.clf()
        beeswarm(shap_values[:, features_ix, 1], max_display=20, show=False)
        fig = plt.gcf()
        fig.tight_layout()
        fig.suptitle(f"SHAP - Random Forest - Outcome: {self.outcome}")
        return fig
