import unittest
from .quantities import UnivariateAnalysis, MultivariateAnalysis, ExploatoryAnalysis
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')


class SetupTest(unittest.TestCase):
    def test_univariate_analysis(self):
        proxy = pd.Series(np.random.uniform(1, 10, 1000))
        disparity_axis = pd.Series(np.random.uniform(1, 10, 1000))

        experiment = UnivariateAnalysis(
            proxy_name="proxy_name",
            disparities_axis_name="disparity_name",
            disparities_axis_uom="disparity_uom",
            protocol__hours=1,
            n_quantiles=100,
        )
        scores = experiment.run(disparity_axis, proxy)
        regression_plots = experiment.plot_regression_by_variance(scores)
        ecdf_plots = experiment.plot_ecdf_by_variance(scores)
        regression_table, fisher_tests = experiment.to_df(scores)

    def test_multivariate_analysis(self):
        X_y = pd.DataFrame(np.random.uniform(0, 1, (1000, 100)))
        X_y["proxy"] = np.random.uniform(0, 1, (1000,))

        experiment = MultivariateAnalysis(
            proxy_name="proxy_name",
            disparities_axis_name="disparity_name",
            disparities_axis_uom="disparity_uom",
            dry=True,
        )

        results = experiment.run(X_y)
        qq_plots_per_model = experiment.plot_observed_predicted_quantiles(results)
        fi_plots_per_model = experiment.plot_fi_boxplots(results)
        shap_plots_per_model = experiment.plot_shapvalues(results)
        test_scores, train_scores, fi_per_model = experiment.to_df(results)

    def test_exploratory_analysis(self):
        proxy = pd.Series(np.random.uniform(0, 1, (100,)))
        experiment = ExploatoryAnalysis(
            proxy_name="proxy_name",
            disparities_axis_name="disparity_name",
            disparities_axis_uom="disparity_uom",
            protocol__hours=2,
        )
        # TODO: missing class testing
        variance_effect_fig = experiment.plot_timestamp_variance_effect(proxy)
