import unittest
from .quantities import UnivariateAnalysis, MultivariateAnalysis
import numpy as np
import pandas as pd


class SetupTest(unittest.TestCase):
    def test_univariate_analysis(self):
        proxy = pd.Series(np.random.uniform(1, 10, 1000))
        disparity_axis = pd.Series(np.random.uniform(1, 10, 1000))

        experiment = UnivariateAnalysis(
            proxy_name="proxy_name",
            disparities_axis_name="disparity_name",
            disparities_axis_uom="disparity_uom",
            protocol__hours=1,
            n_points=100,
        )
        results = (
            experiment.estimate_quantile_mappings_between_proxy_and_disparity_axis(
                proxy, disparity_axis
            )
        )
        slopes = experiment.test_null_hypothesis_that_observed_quantile_mapping_adheres_to_protocol(
            *results
        )

        _ = experiment.plot(results, slopes)

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
        observed_predicted_quantiles_plot = (
            experiment.plot_observed_predicted_quantiles(results)
        )
        fi_plots_per_model = experiment.plot_fi_boxplots(results)
        shap_plots_per_model = experiment.plot_shapvalues(results)
        test_scores, train_scores, fi_per_model = experiment.to_df(results)
