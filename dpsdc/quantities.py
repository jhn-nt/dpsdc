from pathlib import Path
import json

from dataclasses import dataclass, field, fields
from itertools import product

from shap import LinearExplainer, TreeExplainer, plots

import numpy as np
from numpy.typing import ArrayLike
from typing import Any, Callable, Dict, List

import pandas as pd


from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

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
        _ = ax.fill_betweenx(y, x, x + sd, color=color, alpha=0.3)
        _ = ax.fill_betweenx(y, x - sd, x, color=color, alpha=0.3)
        return ax


@dataclass(frozen=True, slots=True)
class QuantilePair(Base):
    """Dataclass implementing Quantile-Quantile Pair analysis.
    This represent an auxiliary class that stores two ECDF from continuous random variables x and y.
    It then computes the quantile-quantile slope and bias of the resulting Qunatile-Quantile Relation.
    """

    x: ECDF = field(repr=False)
    y: ECDF = field(repr=False)
    slope: float
    bias: float

    @classmethod
    def from_ecdfs(cls, x, y):
        slope, bias = np.polyfit(x.thresholds, y.thresholds, 1)
        return QuantilePair(x=x, y=y, slope=slope, bias=bias)

    def interpolate(self):
        return self.x.thresholds * self.slope + self.bias

    @staticmethod
    def plot_with_confidence_intervals(ax, qq_list, color, label):
        qq_mean = np.mean(qq_list)
        qq_ci = 1.96 * np.std(qq_list)

        x = qq_mean.x.thresholds
        y = qq_mean.y.thresholds
        sd = qq_ci.y.thresholds

        _ = ax.plot(x, y, color=color, label=label)
        _ = ax.fill_between(x, y, y + sd, color=color, alpha=0.3)
        _ = ax.fill_between(x, y - sd, y, color=color, alpha=0.3)
        return ax


@dataclass(frozen=True, slots=True)
class UnivariateAnalysis:
    proxy_name: str = field(repr=True)
    disparities_axis_name: str = field(repr=True)
    disparities_axis_uom: str = field(repr=True)
    protocol__hours: float = field(repr=True)
    n_points: int = field(repr=True)
    n_variances: int = field(default=10, repr=True)
    max_timestamp_variance__minutes: float = field(default=10 / 60, repr=True)
    min_timestamp_variance__minutes: float = field(default=0, repr=True)
    n_splits: float = field(default=5, repr=True)
    n_repeats: float = field(default=10, repr=True)
    random_state: float = field(default=0, repr=True)

    def estimate_quantile_mappings_between_proxy_and_disparity_axis(
        self, proxy: pd.Series, disparity_axis: pd.Series
    ):
        proxy = proxy.values
        disparity_axis = disparity_axis.values

        cv = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )
        timestamp_variances = np.linspace(
            self.min_timestamp_variance__minutes,
            self.max_timestamp_variance__minutes,
            self.n_variances,
        )

        trace = []
        baseline = []

        for timestamp_variance, (_, test) in product(
            timestamp_variances, cv.split(disparity_axis, proxy)
        ):
            x = ECDF.interpolate_from_sample(
                disparity_axis[test], n_points=self.n_points
            )

            observed = ECDF.interpolate_from_sample(
                proxy[test] + np.random.normal(0, timestamp_variance, test.size),
                n_points=self.n_points,
            )
            protocol = ECDF.interpolate_from_sample(
                np.random.normal(self.protocol__hours, timestamp_variance, test.size),
                n_points=self.n_points,
            )

            trace.append(QuantilePair.from_ecdfs(x, observed))
            baseline.append(QuantilePair.from_ecdfs(x, protocol))

        return trace, baseline

    def test_null_hypothesis_that_observed_quantile_mapping_adheres_to_protocol(
        self, trace, baseline
    ):
        trace_slopes = np.asarray([*map(lambda x: x.slope, trace)])
        baseline_slopes = np.asarray([*map(lambda x: x.slope, baseline)])
        pvalue = ttest_ind(trace_slopes, baseline_slopes).pvalue
        return trace_slopes, baseline_slopes, pvalue

    def to_df(self, trace, baseline):
        x = np.stack([*map(lambda qq: qq.x.thresholds, trace)])
        obs = np.stack([*map(lambda qq: qq.y.thresholds, trace)])
        pro = np.stack([*map(lambda qq: qq.y.thresholds, baseline)])
        pct = (obs - pro) / pro
        data = np.stack([x, pro, obs, pct])

        pval_f = lambda pop, a: ttest_ind(*pop, axis=a).pvalue
        fmt_f = lambda x: np.apply_along_axis(
            lambda x: "{0:.2f} [{1:.2f} , {2:.2f}]".format(*x),
            0,
            np.quantile(x, [0.5, 0.05, 0.95], axis=1),
        )

        header = pd.MultiIndex.from_tuples(
            [
                (self.disparities_axis_name, self.disparities_axis_uom),
                ("Protocol", "Hour(s)"),
                ("Observed", "Hour(s)"),
                ("Change", "%"),
            ]
        )
        table_df = pd.DataFrame(
            fmt_f(data).T,
            columns=header,
        )
        table_df[("p", "")] = pval_f(data[[1, 2]], 0)
        return table_df

    def plot(self, results, slopes):
        trace, baseline = results
        observed_slopes, protocol_slopes, p = slopes

        # 1. Plotting QQ PLots
        fig1, ax = plt.subplots()
        _ = QuantilePair.plot_with_confidence_intervals(ax, baseline, "k", "Protocol")
        _ = QuantilePair.plot_with_confidence_intervals(
            ax, trace, cm.tab10(0), "Observed"
        )
        _ = ax.set_ylabel(f"Average {self.proxy_name} Interval Quantile [Hour(s)]")
        _ = ax.set_xlabel(
            f"{self.disparities_axis_name} Quantile [{self.disparities_axis_uom}]"
        )
        _ = ax.grid(alpha=0.3)
        _ = ax.legend()

        # 2. Plotting Distributions of slopes
        fig2, ax = plt.subplots()
        _ = ax.hist(
            protocol_slopes,
            label=f"Protocol: {np.mean(protocol_slopes):.4f}  (SD={np.std(protocol_slopes):.4f})",
            color="k",
        )
        _ = ax.hist(
            observed_slopes,
            label=f"Observed: {np.mean(observed_slopes):.4f}  (SD={np.std(observed_slopes):.4f})",
            color=cm.tab10(0),
        )
        _ = ax.grid(alpha=0.3)
        _ = ax.set_xlabel(f"Hour(s) Quantile/ {self.disparities_axis_uom} Quantile")
        _ = ax.set_ylabel("#")
        _ = ax.legend()
        _ = ax.text(sum(ax.get_xlim()) / 2, sum(ax.get_ylim()) / 2, f"p={p:.2f}")

        # 3. Plotting ECDFs for observed and protocollar proxy.
        observed = [*map(lambda qq: qq.y, trace)]
        protocol = [*map(lambda qq: qq.y, baseline)]

        fig3, ax = plt.subplots()
        ECDF.plot_with_confidence_intervals(ax, observed, cm.tab10(0), "Observed")
        ECDF.plot_with_confidence_intervals(ax, protocol, "k", "Protocol")
        _ = ax.set_xlabel(f"Average {self.proxy_name} Interval [Hour(s)]")
        _ = ax.set_ylabel("Density [AU]")
        _ = ax.legend()
        _ = ax.grid(alpha=0.3)

        # 4. Plotting ECDF for the disparity axis
        disparity_axis_ecdfs = [*map(lambda qq: qq.x, trace)]
        fig4, ax = plt.subplots()
        ECDF.plot_with_confidence_intervals(ax, disparity_axis_ecdfs, cm.tab10(0), "")
        _ = ax.set_xlabel(f"{self.disparities_axis_name} [{self.disparities_axis_uom}]")
        _ = ax.set_ylabel("Density [AU]")
        _ = ax.grid(alpha=0.3)
        return fig1, fig2, fig3, fig4


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
            regressor=regressor_, transformer=QuantileTransformer()
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
        estimator = LGBMRegressor(
            random_state=random_state, verbose=-1, objective="quantile"
        )
        return GridSearchCV(estimator, param_grid, cv=cv)

    @staticmethod
    def tracing_func(model):
        features = model[0].get_feature_names_out()
        features = [feat.split("__", 1)[1] for feat in features]
        fi = model[-1].best_estimator_.feature_importances_
        return pd.Series(fi, index=features, name="Quantile LGBM")

    @staticmethod
    def shap_explainer(model, X):
        return TreeExplainer(model.best_estimator_, X)(X)


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
    max_timestamp_variance__minutes: float = field(default=10 / 60, repr=True)
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
            loc=0, scale=timestamp_variance__min, size=X_y.shape[0]
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
                mn.q_pred,
                mn.q_pred + sd.q_pred,
                color=cm.tab10(i),
                alpha=0.3,
            )
            _ = ax.fill_between(
                mn.q_true,
                mn.q_pred - sd.q_pred,
                mn.q_pred,
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
                QuantileRidgeReg,
                MedianReg,
            ]
        else:
            model_list = [
                LGBMReg,
                RidgeReg,
                # QuantileRidgeReg,
                # QuantileLGBMReg,
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
            table = model_trace.agg(["mean", "std"], axis=1).sort_values(
                by="mean", ascending=False
            )
            mean_table = table["mean"].apply(lambda x: f"{x:.3f}")
            std_table = table["std"].apply(lambda x: f"  ({x:.3f})")
            return (mean_table + std_table).rename(name)

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
