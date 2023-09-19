from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
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
from .etl import load_channels, load_configs

from typing import List

CONFIGS = load_configs()
FEATURES = set(
    CONFIGS["TRADITIONAL_LABELS"]
    + CONFIGS["VARIABLES"]
    + CONFIGS["OUTCOMES"]
    + CONFIGS["PROXY"]
)
COVARIATES = set(load_channels(CONFIGS["COVARIATES_CHANNELS"]))
RANDOM_STATE = CONFIGS["RANDOM_STATE"]
N_COMPONENTS = CONFIGS["N_COMPONENTS"]


class WorkshopModel:
    cv = RepeatedStratifiedKFold(n_repeats=CONFIGS["N_REPEATS"])

    def __init__(self, outcome: str, exclude: List[str] = CONFIGS["OUTCOMES"]):
        self.outcome = outcome
        self.features = list(FEATURES.difference(list(outcome) + exclude))
        self.covariates = list(
            COVARIATES.difference(FEATURES).difference(list(outcome) + exclude)
        )
        self.model = self.build(RANDOM_STATE, self.features, self.covariates)

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
        y = dataset[self.outcome[0]].astype(int)
        performance, scores, _ = crossvalidate_classification(
            self.model, X, y, cv=self.cv, tracing_func=self.tracing_func, name=self.name
        )

        self.performance = performance["metrics"].query("side=='test'")
        self.scores = scores
        self.curves = performance["curves"]
        self.fitted_model = self.model.fit(X, y)
        return self

    def plot_curves(self):
        auc_mean = self.performance.groupby("class").mean().iloc[1].auc
        auc_std = self.performance.groupby("class").std().iloc[1].auc

        fig, ax = plt.subplots(1, 2, figsize=(15, 7))

        # AUC
        roc = self.curves["roc"]
        roc_mean = roc.query("side=='test'").groupby("fpr").mean()[1]
        roc_std = roc.query("side=='test'").groupby("fpr").std()[1]
        _ = ax[0].plot(roc_mean.index, roc_mean, color=cm.tab10(0))
        _ = ax[0].fill_between(
            roc_mean.index, roc_mean, roc_mean + roc_std, color=cm.tab10(0), alpha=0.5
        )
        _ = ax[0].fill_between(
            roc_mean.index, roc_mean - roc_std, roc_mean, color=cm.tab10(0), alpha=0.5
        )

        _ = ax[0].plot([0, 1], [0, 1], color="k", alpha=0.3)
        _ = ax[0].set_xticks(np.arange(0, 1.1, 0.1))
        _ = ax[0].set_yticks(np.arange(0, 1.1, 0.1))
        _ = ax[0].grid(alpha=0.3)
        _ = ax[0].set_title(f"AUCROC - {self.name} - Outcome:{self.outcome}")
        _ = ax[0].text(0.7, 0.2, f"AUC:{auc_mean:.2f} (SD:{auc_std:.2f})")
        _ = ax[0].set_xlabel("FPR")
        _ = ax[0].set_ylabel("TPR")

        # Precision-Recall
        pr = self.curves["pr"]
        pr_mean = pr.query("side=='test'").groupby("precision").mean()[1]
        pr_std = pr.query("side=='test'").groupby("precision").std()[1]
        _ = ax[1].plot(pr_mean.index, pr_mean, color=cm.tab10(0))
        _ = ax[1].fill_between(
            pr_mean.index, pr_mean, pr_mean + pr_std, color=cm.tab10(0), alpha=0.5
        )
        _ = ax[1].fill_between(
            pr_mean.index, pr_mean - pr_std, pr_mean, color=cm.tab10(0), alpha=0.5
        )
        _ = ax[1].set_xticks(np.arange(0, 1.1, 0.1))
        _ = ax[1].set_yticks(np.arange(0, 1.1, 0.1))
        _ = ax[1].grid(alpha=0.3)
        _ = ax[1].set_title(f"Precision Recall - {self.name} - Outcome:{self.outcome}")
        _ = ax[1].set_xlabel("Precision")
        _ = ax[1].set_ylabel("Recall")
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
                    make_pipeline(SimpleImputer(), StandardScaler()),
                    continuous_features(features) + ordinal_features(features),
                ),
                (
                    "covariates",
                    make_pipeline(
                        SimpleImputer(),
                        PowerTransformer(),
                        PCA(random_state=RANDOM_STATE, n_components=N_COMPONENTS),
                    ),
                    covariates,
                ),
            ]
        )

        classifier = LogisticRegression(
            random_state=RANDOM_STATE, class_weight="balanced"
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

        train_set = train_set[self.features + self.covariates]
        test_set = test_set[self.features + self.covariates]
        features = self.fitted_model[0].get_feature_names_out()
        features = [feature.split("__")[1] for feature in features]

        features_ix = self.fitted_model[0].get_feature_names_out()
        features_ix = [
            i for (i, feature) in enumerate(features_ix) if "pca" not in feature
        ]

        train_sample_size = 500 if train_set.shape[0] > 500 else train_set.shape[0]
        explainer = LinearExplainer(
            self.fitted_model[-1],
            transform(train_set.sample(train_sample_size)),
            feature_names=features,
        )

        test_sample_size = 500 if test_set.shape[0] > 500 else test_set.shape[0]
        shap_values = explainer(transform(test_set.sample(test_sample_size)))[
            :, features_ix
        ]

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

        train_sample_size = 500 if train_set.shape[0] > 500 else train_set.shape[0]
        explainer = TreeExplainer(
            self.fitted_model[-1],
            transform(train_set.sample(train_sample_size)),
            feature_names=features,
        )

        test_sample_size = 500 if test_set.shape[0] > 500 else test_set.shape[0]
        shap_values = explainer(
            transform(test_set.sample(test_sample_size)), check_additivity=False
        )  # this needs to be fixed

        plt.clf()
        beeswarm(shap_values[:, features_ix, 1], max_display=20, show=False)
        fig = plt.gcf()
        fig.tight_layout()
        fig.suptitle(f"SHAP - Random Forest - Outcome: {self.outcome}")
        return fig
