import ipywidgets as widgets
from IPython.display import display
import numpy as np
import seaborn as sns
from typing import Any, List, Tuple, Callable
from .etl import (
    load_channels,
    load_view,
    categorical_features,
    continuous_features,
    ordinal_features,
)


def select_feature(name: str, pool: List[str], default: int = 0):
    channels = load_channels()
    title = widgets.HTML(f"<b>{name}</b>")
    defaultc = list(sorted(channels.keys()))[default]
    parent = widgets.Dropdown(
        options=sorted(channels.keys()), description="Channel", value=defaultc
    )
    child = widgets.Dropdown(description="Feature")

    def dep(*args):
        child.options = list(sorted(set(pool).intersection(channels[parent.value])))

    parent.observe(dep)
    return widgets.VBox([title, parent, child]), {f"{name.lower()}": child}


def select_options(**kwargs) -> Any:
    options = {}
    children = [widgets.HTML(value=f"<b>Options</b>")]
    for key, item in kwargs.items():
        wdgt = widgets.Dropdown(options=item, description=key, value=item[0])
        options[key] = wdgt
        children.append(wdgt)

    box = widgets.VBox(children)
    return box, options


def chain_selections(items: List[Tuple[Any, dict]], box: Any, f: callable):
    out = {}
    ui = []
    for __ui__, __out__ in items:
        ui.append(__ui__)
        out.update(**__out__)
    return box(ui), widgets.interactive_output(f, out)


def histogram_with_hue():
    def quantile_filter(series, minq, maxq):
        q = np.squeeze(series.quantile([minq, maxq]).values)
        return series.between(*q)

    def histogram(histogram, hue, stat, kde, norm):
        try:
            data = load_view([histogram, hue], None)
            if data[histogram].dtype == float:
                data = data[quantile_filter(data[histogram], 0.01, 0.99)]
                discrete = False
            else:
                discrete = True
                kde = False
            return sns.histplot(
                data,
                x=histogram,
                hue=hue,
                stat=stat,
                kde=kde,
                common_norm=norm,
                discrete=discrete,
            )
        except:
            return None

    ui, out = chain_selections(
        [
            select_feature("Histogram", continuous_features() + ordinal_features()),
            select_feature("Hue", categorical_features()),
            select_options(
                stat=["probability", "count", "frequency", "density", "percent"],
                kde=[True, False],
                norm=[False, True],
            ),
        ],
        widgets.HBox,
        histogram,
    )

    display(ui, out)


def scatterplot_with_hue():
    def scatterplot(x, y, hue, stratify, sample_size):
        try:
            data = load_view([x, y, hue], None)
            if stratify:
                min_sample_size = round(
                    sample_size * data.groupby(hue).count().min(axis=None)
                )
                data = data.groupby(hue).sample(min_sample_size)
            return sns.scatterplot(data=data, x=x, y=y, hue=hue, alpha=0.5)
        except:
            pass

    ui, out = chain_selections(
        [
            select_feature("x", continuous_features()),
            select_feature("y", continuous_features(), default=1),
            select_feature("Hue", categorical_features()),
            select_options(
                stratify=[False, True], sample_size=np.arange(0.1, 1.1, 0.1)
            ),
        ],
        widgets.HBox,
        scatterplot,
    )

    display(ui, out)
