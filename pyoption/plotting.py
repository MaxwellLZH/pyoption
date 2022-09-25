"""
Plotting related functionalities, including:
- P&L (profit and loss)
"""
import numpy as np
from typing import Iterable, Optional, Union, Tuple, Dict
from numbers import Number
import matplotlib.pyplot as plt

from .classes import Option, OptionPortfolio


def calculate_profit_loss(
    p: Union[OptionPortfolio, Option],
    price_range: Tuple[float] = None,
    interval=0.1,
    return_detail=True,
) -> Tuple[Iterable[float], Iterable[float], Optional[Dict]]:
    """ Return (array of prices, array of the portfolio P&L, mapping from each option to its P&L)

    :param return_detail: bool, if set to True, will additionally return the PL array for each Option inside OptionPorfolio,
        in the form of a dictionary
    """
    # determine the price range for plotting
    if price_range is None:
        exercise_price = p.exercise_price
        if isinstance(exercise_price, Number):
            exercise_price = [exercise_price]
        lower, upper = min(exercise_price) - 20, max(exercise_price) + 20
    else:
        lower, upper = price_range

    arr_price = np.arange(lower, upper, interval)
    arr_pl = [p.value(price) for price in arr_price]

    if return_detail and isinstance(p, OptionPortfolio):
        detail = {}
        for opt, cnt in p.options.items():
            detail[opt] = [opt.value(price) * cnt for price in arr_price]
        return arr_price, arr_pl, detail
    else:
        return arr_price, arr_pl


def plot_profit_and_loss(
    p: Union[OptionPortfolio, Option],
    price_range: Tuple[float] = None,
    interval=0.1,
    show_each_option=True,
    figsize=(12, 8),
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if show_each_option:
        arr_price, arr_pl, detail = calculate_profit_loss(
            p=p, price_range=price_range, interval=interval, return_detail=True
        )
    else:
        arr_price, arr_pl = calculate_profit_loss(
            p=p, price_range=price_range, interval=interval, return_detail=False
        )

    ax.plot(arr_price, arr_pl, label='Portfolio', lw=2.6)
    if show_each_option:
        for opt, pl in detail.items():
            ax.plot(arr_price, pl, label=str(opt), alpha=0.3, linestyle='--')
    ax.legend()
    return fig
