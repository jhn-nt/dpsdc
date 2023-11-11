from pathlib import Path
import json

from dataclasses import dataclass, field, fields
from itertools import product

from shap import LinearExplainer, TreeExplainer, plots

import numpy as np
from numpy.typing import ArrayLike
from typing import Any, Callable, Dict, List

from scipy.stats import ttest_1samp, ks_2samp, combine_pvalues

import pandas as pd


from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

from lightgbm import LGBMRegressor
from crlearn.evaluation import crossvalidate_regression

from .utils import categorical, continuous

from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
from matplotlib import cm

from joblib import Parallel, delayed


class Base(np.lib.mixins.NDArrayOperatorsMixin):
    """Base class implementing dunder math"""

    def conjugate(self):
        return np.conjugate(self)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        def getter(attribute):
            def f(x):
                return getattr(x, attribute) if hasattr(x, attribute) else x

            return f

        output = {}
        for attr in fields(self):
            attr_value = getattr(self, attr.name)
            attr_inputs = map(getter(attr.name), inputs)
            if hasattr(attr_value, "__array_ufunc__"):
                output[attr.name] = attr_value.__array_ufunc__(
                    ufunc, method, *attr_inputs, **kwargs
                )
            else:
                output[attr.name] = ufunc(*attr_inputs)

        return self.__class__(**output)


@dataclass(frozen=True, slots=True)
class ECDF(Base):
    """Dataclass implementing Empirical Cumulative Distribution Function, ECDF, estimation
    from a sample of a continuous random variable.
    """

    thresholds: ArrayLike = field(repr=False)
    densities: ArrayLike = field(repr=False)

    @classmethod
    def empty(cls, n_thresholds: int = 100):
        return cls(thresholds=np.empty(n_thresholds), densities=np.empty(n_thresholds))

    @classmethod
    def from_sklearn(
        cls,
        x: ArrayLike,
        quantiles: List[float] = np.arange(0, 1.01, 0.01),
        random_state: int = 0,
    ):
        transformer = QuantileTransformer(random_state=random_state).fit(
            np.asarray(x).reshape(-1, 1)
        )
        return cls(
            densities=np.squeeze(quantiles),
            thresholds=np.squeeze(
                transformer.inverse_transform(quantiles.reshape(-1, 1))
            ),
        )

    @classmethod
    def from_sample(
        cls,
        x: ArrayLike,
        lower: float = 0.01,
        upper: float = 0.99,
        n_thresholds: int = 100,
    ) -> "ECDF":
        sample_size = x.size
        qmin, qmax = np.quantile(x, [lower, upper])
        thresholds = np.linspace(qmin, qmax, n_thresholds)

        @np.vectorize
        def p_t_greater_than_x(threshold):
            return np.sum(x < threshold) / sample_size

        densities = p_t_greater_than_x(thresholds)
        return cls(thresholds=thresholds, densities=densities)

    @classmethod
    def interpolate_from_sample(cls, *args, n_points: int = 100, **kwargs) -> "ECDF":
        thresholds, raw_densities = cls.from_sample(*args, **kwargs).compute()
        eps = 1 / len(thresholds)
        interpol_densities = np.linspace(eps, 1 - eps, n_points)
        interpol_thresholds = np.interp(interpol_densities, raw_densities, thresholds)
        return cls(thresholds=interpol_thresholds, densities=interpol_densities)

    def compute(self):
        return self.thresholds, self.densities

    @staticmethod
    def plot_with_confidence_intervals(ax, ecdf_list, color, label):
        ecdf_mean = np.mean(ecdf_list)
        ecdf_ci = 1.96 * np.std(ecdf_list)

        x = ecdf_mean.thresholds
        sd = ecdf_ci.thresholds
        y = ecdf_mean.densities

        _ = ax.plot(x, y, color=color, label=label)
        _ = ax.fill_betweenx(y, x - sd, x + sd, color=color, alpha=0.3)
        return ax


@dataclass(frozen=True, slots=True)
class UnivariateAnalysis:
    proxy_name: str = field(repr=True)
    disparities_axis_name: str = field(repr=True)
    disparities_axis_uom: str = field(repr=True)
    protocol__hours: float = field(repr=True)
    n_variances: int = field(default=10, repr=True)
    max_timestamp_variance__minutes: float = field(default=10, repr=True)
    min_timestamp_variance__minutes: float = field(default=0, repr=True)
    n_quantiles: int = 100
    order: int = 1
    n_splits: float = field(default=5, repr=True)
    n_repeats: float = field(default=10, repr=True)
    random_state: float = field(default=0, repr=True)

    @staticmethod
    def add_variance(x, variance):
        return x + np.random.normal(0.0, variance**0.5, x.shape)

    def get_protocol(self, x, variance):
        return np.random.normal(self.protocol__hours, variance**0.5, x.shape)

    @staticmethod
    def crossvalidate_experiment(x, y, cv, order, aggfunc, random_state, name):
        xmap = QuantileTransformer(random_state=random_state)
        ymap = QuantileTransformer(random_state=random_state)

        def aggregate(x, y):
            x_t, y_t = (
                pd.DataFrame({"x": np.round(x, order), "y": y})
                .groupby("x")
                .agg(aggfunc)
                .reset_index()
                .values.T
            )
            return x_t, y_t

        score_f = lambda y, y_pred, fold, coef: pd.Series(
            {
                "r2": r2_score(np.squeeze(y), y_pred),
                # "mse":mean_squared_error(np.squeeze(y),y_pred),
                "fold": fold,
                "name": name,
                "slope": coef[0],
                "bias": coef[1],
            }
        )

        scores = []
        traces = []
        for fold, (train, _) in enumerate(cv.split(x, y)):
            x_q = np.squeeze(xmap.fit_transform(x[train].reshape(-1, 1)))
            y_q = np.squeeze(ymap.fit_transform(y[train].reshape(-1, 1)))
            x_q_agg, y_q_agg = aggregate(x_q, y_q)

            coef = np.polyfit(x_q_agg, y_q_agg, deg=1)
            y_q_agg_pred = np.polyval(coef, x_q_agg)

            scores.append(score_f(y_q_agg, y_q_agg_pred, fold, coef))
            traces.append(
                pd.DataFrame(
                    {
                        "x": x_q_agg,
                        "y_true": y_q_agg,
                        "y_pred": y_q_agg_pred,
                        "name": name,
                        "fold": fold,
                    }
                )
            )

        return pd.concat(scores, axis=1).T, pd.concat(traces, axis=0)

    @staticmethod
    def crossvalidate_ecdf(x, protocol, observed, cv, name, n_quantiles, random_state):
        ecdfs = []
        pvals = []
        pfunc = lambda x, y, fold: pd.Series(
            {
                "less": ks_2samp(x, y, alternative="less").pvalue,
                "greater": ks_2samp(x, y, alternative="greater").pvalue,
                "two-sided": ks_2samp(x, y, alternative="two-sided").pvalue,
                "name": name,
                "fold": fold,
            }
        )

        for fold, (_, train) in enumerate(cv.split(x, observed, protocol)):
            quantiles = ECDF.interpolate_from_sample(
                x[train], n_points=n_quantiles
            ).densities
            q_x = ECDF.interpolate_from_sample(
                x[train], n_points=n_quantiles
            ).thresholds
            q_observed = ECDF.interpolate_from_sample(
                observed[train], n_points=n_quantiles
            ).thresholds
            q_protocol = ECDF.interpolate_from_sample(
                protocol[train], n_points=n_quantiles
            ).thresholds
            ecdfs.append(
                pd.DataFrame(
                    {
                        "quantile": quantiles,
                        "x": q_x,
                        "observed": q_observed,
                        "protocol": q_protocol,
                        "name": name,
                        "fold": fold,
                    }
                )
            )

            pvals.append(pfunc(observed[train], protocol[train], fold))

        return pd.concat(ecdfs), pd.concat(pvals, axis=1).T

    def run(self, disparity: ArrayLike, proxy: ArrayLike):
        # Schedule to test for different additive variances
        schedule__minutes = np.linspace(
            self.min_timestamp_variance__minutes,
            self.max_timestamp_variance__minutes,
            self.n_variances + 1,
        )

        # Each variance is tested through repeated cross-validation
        cv = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )

        x = np.asarray(disparity).reshape(-1, 1)
        results = []
        ecdfs = []
        for variance__minutes in schedule__minutes:
            name = f"{variance__minutes}"
            y = self.add_variance(np.asarray(proxy), variance__minutes / 60)
            y_baseline = self.get_protocol(np.asarray(proxy), variance__minutes / 60)

            results.append(
                self.crossvalidate_experiment(
                    x,
                    y,
                    cv=cv,
                    aggfunc=np.mean,
                    random_state=self.random_state,
                    order=self.order,
                    name=name,
                )
            )
            ecdfs.append(
                self.crossvalidate_ecdf(
                    x,
                    y_baseline,
                    y,
                    cv=cv,
                    n_quantiles=self.n_quantiles,
                    random_state=self.random_state,
                    name=name,
                )
            )

        return (
            pd.concat(map(lambda x: x[0], results), axis=0),
            pd.concat(map(lambda x: x[1], results), axis=0),
            pd.concat(map(lambda x: x[0], ecdfs), axis=0),
            pd.concat(map(lambda x: x[1], ecdfs), axis=0),
        )

    def plot_regression(self, results, variance, fold=0):
        # getting data
        x = results[1].query(f"fold=={fold}").query(f"name=='{variance}'").x
        y_true = results[1].query(f"fold=={fold}").query(f"name=='{variance}'").y_true
        y_pred = results[1].query(f"fold=={fold}").query(f"name=='{variance}'").y_pred

        # preparing plot legend string
        crossval_results = self.format_regression_scores(
            results[0].query(f"name=='{variance}'")
        )[0]
        vals = np.squeeze(crossval_results.astype(str).values).tolist()
        names = crossval_results.columns.to_list()
        legend = "\n".join([f"{t}: {v}" for (t, v) in zip(names, vals)])

        fig, ax = plt.subplots()
        _ = ax.scatter(x, y_true)
        _ = ax.plot(x, y_pred, color="k", label=legend)
        _ = ax.grid(alpha=0.3)

        _ = ax.set_xticks(np.arange(0, 1.1, 0.1))
        _ = ax.set_ylabel(f"Average {self.proxy_name} Interval Quantile [AU]")
        _ = ax.set_xlabel(f"{self.disparities_axis_name} Quantile [AU]")
        _ = ax.set_title(f"{variance} Minutes")
        _ = ax.legend()
        return fig

    def plot_regression_by_variance(self, results, fold=0):
        variances = results[1].name.unique()
        return {
            variance: self.plot_regression(results, variance) for variance in variances
        }

    @staticmethod
    def format_regression_scores(df):
        df.name = df.name.astype(float)
        table = UnivariateAnalysis.format_table(df)
        table["p"] = df.groupby("name").slope.agg(
            lambda x: ttest_1samp(x.astype(float), 0.0).pvalue
        )
        return table, combine_pvalues(table.p).pvalue

    @staticmethod
    def format_table(df):
        df = df.drop("fold", axis=1)
        aggregated_df = df.groupby("name").agg(UnivariateAnalysis.print_mean_std)
        return aggregated_df

    @staticmethod
    def print_mean_std(series):
        return f"{series.mean():.4f} (SD={series.std():.4f})"

    def plot_ecdf_by_variance(self, results):
        def plot_ecdf(ax, density, mn, sd, color="k", **kwargs):
            _ = ax.plot(mn, density, color=color, **kwargs)
            _ = ax.fill_betweenx(
                density, mn - 1.96 * sd, mn + 1.96 * sd, color=color, alpha=0.3
            )
            _ = ax.set_ylabel("F [AU]")
            _ = ax.set_xlabel(f"Average {self.proxy_name} Interval [Hoour(s)]")
            return ax

        def print_pval(ix):
            ps = results[3].query(ix)["two-sided"].astype(float).values
            pval = combine_pvalues(ps).pvalue
            return f"p:{pval:.4f}"

        output = {}
        for ix, group in results[2].groupby("name")[
            ["quantile", "x", "observed", "protocol"]
        ]:
            data = group.groupby("quantile").agg(["mean", "std"]).swaplevel(1, 0, 1)

            fig, ax = plt.subplots()
            ax = plot_ecdf(
                ax,
                data.index,
                data["mean"].observed,
                data["std"].observed,
                color=cm.tab10(0),
                label="observed",
            )
            ax = plot_ecdf(
                ax,
                data.index,
                data["mean"].protocol,
                data["std"].protocol,
                color="k",
                label="protocol",
            )

            _ = ax.set_yticks(np.arange(0, 1.1, 0.1))
            _ = ax.grid(alpha=0.3)
            _ = ax.text(sum(ax.get_xlim()) / 2, sum(ax.get_ylim()) / 2, print_pval(ix))
            _ = ax.set_title(f"{ix} Minutes")
            _ = ax.legend()
            output[ix] = fig

        return output

    def to_df(self, results):
        regression_df, regression_fisher = self.format_regression_scores(results[0])
        ecdf_tests = (
            results[3]
            .drop(["name", "fold"], axis=1)
            .apply(lambda x: combine_pvalues(x.astype(float)).pvalue)
        )
        ecdf_tests["regression_fisher"] = regression_fisher
        return regression_df, ecdf_tests


@dataclass(frozen=True)
class Regressor:
    """Base class implementing the minimum functionality of a regressor.
    This class is specifically tailored ot work with dpsdc data.
    """

    name: str = field(repr=True)
    estimator: BaseEstimator = field(repr=False)
    tracing_func: Callable = field(repr=False)

    @staticmethod
    def make_processor(X, y, **kwargs):
        categorical_processor = make_pipeline(
            SimpleImputer(strategy="most_frequent"), OneHotEncoder(drop="if_binary")
        )
        continuous_processor = make_pipeline(
            SimpleImputer(), QuantileTransformer(output_distribution="normal")
        )
        processor = ColumnTransformer(
            [
                ("continuous", continuous_processor, continuous(X)),
                ("categotical", categorical_processor, categorical(X)),
            ]
        )
        return processor

    @staticmethod
    def make_regressor(X, y, **kwargs):
        raise NotImplementedError("Subclass this method")

    @staticmethod
    def tracing_func(model):
        raise NotImplementedError("Subclass this method")

    @staticmethod
    def shap_explainer(model, X):
        raise NotImplementedError("Subclass this method")

    def get_shap_values(self, X, y, **kwargs):
        model = self.estimator.fit(X, y, **kwargs)

        def feature_names():
            features = model[0].get_feature_names_out()
            return [feat.split("__", 1)[1] for feat in features]

        observations = model[0].transform(X)
        shap_values = self.shap_explainer(model[1], observations)
        if shap_values is not None:
            shap_values.feature_names = feature_names()
        return shap_values

    @classmethod
    def build(cls, X, y, random_state=0, **kwargs):
        processor = cls.make_processor(X, y, **kwargs)
        regressor = cls.make_regressor(X, y, **kwargs)
        return cls(
            name=cls.name,
            estimator=Pipeline([("processor", processor), ("regressor", regressor)]),
            tracing_func=cls.tracing_func,
        )

    def run(self, X, y, cv):
        results = crossvalidate_regression(
            self.estimator,
            X,
            y,
            cv=cv,
            tracing_func=self.tracing_func,
            name=self.name,
            progress_bar=False,
        )
        return (*results, self.get_shap_values(X, y))


class RidgeReg(Regressor):
    """Implementation of a Ridge Regressor."""

    name = "Ridge"

    @staticmethod
    def make_regressor(X, y, random_state=0, cv=KFold(), **kwargs):
        path = Path(__file__).parent / "params" / "ridge.json"
        param_grid = json.load(open(path, "r"))
        estimator = Ridge(random_state=random_state)
        return GridSearchCV(estimator, param_grid, cv=cv)

    @staticmethod
    def tracing_func(model):
        features = model[0].get_feature_names_out()
        features = [feat.split("__", 1)[1] for feat in features]
        fi = model[-1].best_estimator_.coef_
        return pd.Series(fi, index=features, name="Ridge")

    @staticmethod
    def shap_explainer(model, X):
        return LinearExplainer(model.best_estimator_, X)(X)


class QuantileRidgeReg(Regressor):
    """Implementation of a Quantile Ridge Regressor."""

    name = "Quantile Ridge"

    @staticmethod
    def make_regressor(X, y, random_state=0, cv=KFold(), **kwargs):
        path = Path(__file__).parent / "params" / "ridge.json"
        param_grid = json.load(open(path, "r"))
        param_grid = {f"regressor__{key}": item for key, item in param_grid.items()}

        regressor_ = Ridge(random_state=random_state)
        estimator = TransformedTargetRegressor(
            regressor=regressor_,
            transformer=QuantileTransformer(output_distribution="normal"),
        )
        return GridSearchCV(estimator, param_grid, cv=cv)

    @staticmethod
    def tracing_func(model):
        features = model[0].get_feature_names_out()
        features = [feat.split("__", 1)[1] for feat in features]
        fi = model[-1].best_estimator_.regressor_.coef_
        return pd.Series(fi, index=features, name="Quantile Ridge")

    @staticmethod
    def shap_explainer(model, X):
        regressor_ = model.best_estimator_.regressor_
        return LinearExplainer(regressor_, X)(X)


class LGBMReg(Regressor):
    """Implementation of Gradient Boosting."""

    name = "LGBM"

    @staticmethod
    def make_regressor(X, y, random_state=0, cv=KFold(), **kwargs):
        path = Path(__file__).parent / "params" / "lgbm.json"
        param_grid = json.load(open(path, "r"))
        estimator = LGBMRegressor(random_state=random_state, verbose=-1)
        return GridSearchCV(estimator, param_grid, cv=cv)

    @staticmethod
    def tracing_func(model):
        features = model[0].get_feature_names_out()
        features = [feat.split("__", 1)[1] for feat in features]
        fi = model[-1].best_estimator_.feature_importances_
        return pd.Series(fi, index=features, name="LGBM")

    @staticmethod
    def shap_explainer(model, X):
        return TreeExplainer(model.best_estimator_, X)(X)


class QuantileLGBMReg(Regressor):
    """Implementation of Quantile-loss Gradient Boosting."""

    name = "Quantile LGBM"

    @staticmethod
    def make_regressor(X, y, random_state=0, cv=KFold(), **kwargs):
        path = Path(__file__).parent / "params" / "lgbm.json"
        param_grid = json.load(open(path, "r"))
        param_grid = {f"regressor__{key}": item for key, item in param_grid.items()}

        regressor = LGBMRegressor(random_state=random_state, verbose=-1)
        estimator = TransformedTargetRegressor(
            regressor=regressor, transformer=QuantileTransformer()
        )

        return GridSearchCV(estimator, param_grid, cv=cv)

    @staticmethod
    def tracing_func(model):
        features = model[0].get_feature_names_out()
        features = [feat.split("__", 1)[1] for feat in features]
        fi = model[-1].best_estimator_.regressor_.feature_importances_
        return pd.Series(fi, index=features, name="Quantile LGBM")

    @staticmethod
    def shap_explainer(model, X):
        return TreeExplainer(model.best_estimator_.regressor_, X)(X)


class MedianReg(Regressor):
    """Implementation of a Median Baseline."""

    name = "Median"

    @staticmethod
    def make_regressor(X, y, random_state=0, **kwargs):
        return DummyRegressor(strategy="median")

    @staticmethod
    def tracing_func(model):
        pass

    @staticmethod
    def shap_explainer(model, X):
        pass


@dataclass(frozen=True)
class MultivariateAnalysis:
    proxy_name: str = field(repr=True)
    disparities_axis_name: str = field(repr=True)
    disparities_axis_uom: str = field(repr=True)
    n_variances: int = field(default=10, repr=True)
    max_timestamp_variance__minutes: float = field(default=10, repr=True)
    min_timestamp_variance__minutes: float = field(default=0, repr=True)
    n_splits: float = field(default=5, repr=True)
    n_repeats: float = field(default=1, repr=True)
    random_state: float = field(default=0, repr=True)
    IQR: List[float] = field(default_factory=lambda: [0.05, 0.95], repr=True)
    n_jobs: int = field(default=1, repr=True)
    dry: bool = field(default=False, repr=False)

    def remove_outliers_from_proxy(self, X_y: pd.DataFrame):
        q = np.quantile(X_y.proxy, self.IQR)
        return X_y.query(f"(proxy>={q[0]})&(proxy<{q[1]})")

    @staticmethod
    def add_variance_to_proxy(X_y: pd.DataFrame, timestamp_variance__min: float):
        X_y["proxy"] += np.random.normal(
            loc=0, scale=(timestamp_variance__min / 60) ** 0.5, size=X_y.shape[0]
        )
        return X_y

    @staticmethod
    def split_X_and_y(X_y):
        return X_y.drop("proxy", axis=1), X_y.proxy

    @staticmethod
    def plot_observed_predicted_quantiles(results):
        qq_plots = results[1]

        fig, ax = plt.subplots(figsize=(7, 7))
        res = qq_plots.groupby(["name", "q"]).agg(["mean", "std"]).swaplevel(1, 0, 1)
        for i, name in enumerate(res.index.get_level_values(0).unique()):
            mn = _ = res.loc[name]["mean"]
            sd = _ = res.loc[name]["std"]

            _ = ax.plot(mn.q_true, mn.q_pred, label=name, color=cm.tab10(i))
            _ = ax.fill_between(
                mn.q_true,
                mn.q_pred - sd.q_pred,
                mn.q_pred + sd.q_pred,
                color=cm.tab10(i),
                alpha=0.3,
            )

        _ = ax.plot(mn.q_true, mn.q_true, color="k", alpha=0.7, zorder=-1)
        _ = ax.legend()
        _ = ax.grid(alpha=0.3)
        _ = ax.set_ylabel("Predicted Quantile Value")
        _ = ax.set_xlabel("True Quantile Value")
        return fig

    @staticmethod
    def plot_fi_boxplots(results):
        def plot_fi(fi):
            fi = fi.T
            fig, ax = plt.subplots(figsize=(10, 10))
            _ = fi[fi.mean().sort_values().index].boxplot(
                showfliers=False, vert=False, ax=ax
            )
            _ = ax.set_xlabel("[AU]")
            fig.tight_layout()
            return fig

        traces = results[2]
        figures = {}
        for key, item in traces.items():
            figures[key] = plot_fi(item)

        return figures

    @staticmethod
    def plot_shapvalues(results, timestamp_variance=5):
        def plot_shap(shapval):
            fig, ax = plt.subplots()
            plots.beeswarm(shapval[timestamp_variance], max_display=35)
            fig.tight_layout()
            return fig

        shapvals = results[3]
        figures = {}
        for key, item in shapvals.items():
            try:
                figures[key] = plot_shap(item)
            except:
                pass

        return figures

    def get_models(self):
        if self.dry:
            model_list = [
                RidgeReg,
            ]
        else:
            model_list = [
                LGBMReg,
                RidgeReg,
                QuantileRidgeReg,
                QuantileLGBMReg,
                MedianReg,
            ]
        return model_list

    @staticmethod
    def to_df(results):
        def format_scores(results, side):
            table = (
                results[0]
                .query(f"side=='{side}'")
                .groupby("name")
                .agg(["mean", "std"])
                .sort_values(by=("mse", "mean"))
            )
            mean_table = table.swaplevel(1, 0, 1)["mean"].applymap(lambda x: f"{x:.3f}")
            std_table = table.swaplevel(1, 0, 1)["std"].applymap(
                lambda x: f"  ({x:.3f})"
            )
            return mean_table + std_table

        def format_traces(model_trace, name):
            pval = lambda x: ttest_1samp(x, 0).pvalue
            table = model_trace.agg(["mean", "std"], axis=1).sort_values(
                by="mean", ascending=False
            )
            pvals = model_trace.apply(pval, axis=1).rename("p")
            mean_table = table["mean"].apply(lambda x: f"{x:.3f}")
            std_table = table["std"].apply(lambda x: f"  ({x:.3f})")
            mean_and_std = (mean_table + std_table).rename(name)
            return pd.concat([mean_and_std, pvals], axis=1)

        fi_tables = {}
        for key, items in results[2].items():
            fi_tables[key] = format_traces(items, key)

        return (
            format_scores(results, "test"),
            format_scores(results, "train"),
            fi_tables,
        )

    def run(self, X_y):
        get_traces = lambda fi: pd.concat(
            map(lambda x: pd.concat(x[1].values(), axis=1), fi), axis=1
        )
        get_scores = lambda score: pd.concat(
            map(lambda x: x[0]["statistics"], score), axis=0
        )
        get_curves = lambda curve: pd.concat(
            map(lambda x: x[0]["curves"]["quantiles"], curve), axis=0
        )
        get_shapvals = lambda shap: list(map(lambda x: x[-1], shap))

        X_y = self.remove_outliers_from_proxy(X_y)

        timestamp_variances_iterable = np.linspace(
            self.min_timestamp_variance__minutes,
            self.max_timestamp_variance__minutes,
            self.n_variances,
        )
        model_classes = self.get_models()
        cv = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )

        results = {}
        schedule = product(model_classes, timestamp_variances_iterable)

        def crossval(model_class, timestamp_var):
            data = self.add_variance_to_proxy(X_y, timestamp_var)
            X, y = self.split_X_and_y(data)
            model = model_class.build(X, y, random_state=self.random_state, cv=cv)
            return (model_class.name, model.run(X, y, cv))

        raw_results = Parallel(n_jobs=self.n_jobs)(
            delayed(crossval)(*inputs) for inputs in schedule
        )
        for key, item in raw_results:
            if key in results:
                results[key] += [item]
            else:
                results[key] = [item]

        traces = []
        scores = []
        curves = []
        shapvals = []

        for key, items in results.items():
            curves.append(get_curves(items))
            scores.append(get_scores(items))
            shapvals.append((key, get_shapvals(items)))

            try:
                traces.append((key, get_traces(items)))
            except:
                pass

        curves = pd.concat(curves, axis=0)
        scores = pd.concat(scores, axis=0)

        return scores, curves, dict(traces), dict(shapvals)


@dataclass(frozen=True)
class ExploatoryAnalysis:
    proxy_name: str = field(repr=True)
    disparities_axis_name: str = field(repr=True)
    disparities_axis_uom: str = field(repr=True)
    protocol__hours: float = field(repr=True)
    n_variances: int = field(default=5, repr=True)
    max_timestamp_variance__minutes: float = field(default=10, repr=True)
    min_timestamp_variance__minutes: float = field(default=1e-3, repr=True)

    @staticmethod
    def discretize_dataframe(df: pd.DataFrame, quantiles: List[int]):
        temp = df.copy()

        for feature in continuous(temp):
            temp[feature] = pd.qcut(temp[feature], quantiles)

        return temp

    @staticmethod
    def boxplot_by_features(
        value: pd.Series,
        features: pd.DataFrame,
        quantiles: List[int] = [0.0, 0.25, 0.5, 0.75, 1.0],
        y_label="",
    ):
        temp = ExploatoryAnalysis.discretize_dataframe(features, quantiles)
        df = pd.concat([temp, value], axis=1)

        output = {}
        for column in temp.columns:
            fig, ax = plt.subplots()
            df.boxplot(value.name, by=column, ax=ax, showfliers=False, rot=45)
            _ = ax.set_title(column.capitalize())
            _ = ax.set_xlabel("")
            _ = ax.set_ylabel(y_label)
            fig.suptitle("")
            output[column] = fig

        return output

    def build_boxplots(self, proxy, disparities):
        return self.boxplot_by_features(
            proxy, disparities, y_label=f"{self.proxy_name} [Hour(s)]"
        )

    @staticmethod
    def histplot(ax, data: pd.Series, n_bins: int = 100, iqr=(0.01, 0.99), **kwargs):
        q = data.quantile(iqr).values
        resolution = np.diff(q) / n_bins
        _ = ax.hist(data, bins=np.arange(*q, resolution), **kwargs)
        return ax

    @staticmethod
    def plot_trend_new(dataframe: pd.DataFrame, dis_col: str, proxy_name: str):
        main_trend_data = (
            dataframe.groupby(["day", dis_col])["average_item_interval"]
            .mean()
            .reset_index()
        )

        std_dev = (
            dataframe.groupby(["day", dis_col])["average_item_interval"]
            .apply(lambda x: np.std(x))
            .reset_index()
        )

        fig, ax = plt.subplots()

        for category in main_trend_data[dis_col].unique():
            subset = main_trend_data[main_trend_data[dis_col] == category]

            ax.plot(
                subset["day"],
                subset["average_item_interval"],
                label=f"{dis_col}: {category}",
                marker="o",
            )

            std_subset = std_dev[std_dev[dis_col] == category]
            confidence_intervals = std_subset["average_item_interval"]
            ax.fill_between(
                np.array(subset["day"], dtype=float),
                (subset["average_item_interval"] - confidence_intervals),
                (subset["average_item_interval"] + confidence_intervals),
                alpha=0.2,
            )

        ax.set_xlabel("Day")
        ax.set_ylabel(f"Average {proxy_name} Interval [Hour(s)]")
        ax.set_title(dis_col.capitalize())

        ax.set_xlim([0, 10])
        ax.set_ylim([0, 5])
        ax.legend()

        return fig

    @staticmethod
    def plot_by_general_population(dataframe: pd.DataFrame, proxy_name: str):
        main_trend_data = (
            dataframe.groupby("day")["average_item_interval"].mean().reset_index()
        )

        std_dev = dataframe.groupby("day")["average_item_interval"].std().reset_index()

        fig, ax = plt.subplots()
        ax.plot(
            main_trend_data["day"],
            main_trend_data["average_item_interval"],
            label="Main Trend",
            color="b",
            marker="o",
        )  # Giovanni: added markers for improved readability

        ax.fill_between(
            np.array(main_trend_data["day"], dtype=float),
            main_trend_data["average_item_interval"] - std_dev["average_item_interval"],
            main_trend_data["average_item_interval"] + std_dev["average_item_interval"],
            alpha=0.2,
            color="b",
            label="Confidence Interval (Std Dev)",
        )

        ax.set_xlabel("Day")
        ax.set_ylabel(f"Average {proxy_name} Interval [Hour(s)]")
        ax.legend()

        ax.set_ylim([0, 5])
        ax.set_xlim([0, 10])

        return fig

    def plot_daily_trends(
        self,
        daily_proxy: pd.DataFrame,
        features: pd.DataFrame,
        quantiles: List[int] = [0.0, 0.25, 0.5, 0.75, 1.0],
    ):
        assert "day" in daily_proxy.columns
        assert "average_item_interval" in daily_proxy.columns

        temp = ExploatoryAnalysis.discretize_dataframe(features, quantiles)
        axis_features = temp.columns.to_list()

        df = daily_proxy.merge(temp, on="stay_id")

        output = {}
        for feature in axis_features:
            output[feature] = self.plot_trend_new(df, feature, self.proxy_name)

        output["general"] = self.plot_by_general_population(df, self.proxy_name)

        return output

    def plot_timestamp_variance_effect(self, proxy: pd.Series):
        fig, ax = plt.subplots(
            1, self.n_variances, figsize=(5 * self.n_variances, 5), sharex=True
        )

        variances = np.logspace(
            np.log10((self.min_timestamp_variance__minutes / 60) ** 0.5),
            np.log10((self.max_timestamp_variance__minutes / 60) ** 0.5),
            self.n_variances,
        )

        for cax, variance in zip(ax, variances):
            data = proxy + np.random.normal(0, variance, (proxy.shape[0],))
            self.histplot(cax, data)
            _ = cax.set_title(f"{(variance**2)*60:.2f} [Min]")
            _ = cax.grid(alpha=0.3)
            _ = cax.set_xlabel(f"Average {self.proxy_name} Interval [Hour(s)]")
            _ = cax.set_ylabel("Frequency [#]")
        return fig
