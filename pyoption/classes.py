from cmath import isfinite
from enum import Enum
from multiprocessing.sharedctypes import Value
from typing import Tuple, List, Dict, Union, Type


class OptionType(Enum):
    CALL = 1
    PUT = 2


class ExerciseType(Enum):
    EUROPEAN = 1
    AMERICAN = 2


class Option:
    option_type = None

    def __init__(self, exercise_price, time_to_expiration=None, exercise_type=ExerciseType.EUROPEAN):
        self.exercise_price = exercise_price
        self.exercise_type = exercise_type
        self.time_to_expiration = time_to_expiration

    def __repr__(self):
        return f"{self.option_type.name.capitalize()}Option(exercise_price={self.exercise_price}, exercise_type={self.exercise_type.name})"

    def __hash__(self):
        return hash((self.option_type, self.exercise_price, self.exercise_type))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __mul__(self, n: int):
        return OptionPortfolio([(self, n)])

    __rmul__ = __mul__

    def __neg__(self):
        return OptionPortfolio([(self, -1)])

    def __add__(self, other):
        if isinstance(other, Option):
            return OptionPortfolio([(self, 1), (other, 1)])
        elif isinstance(other, OptionPortfolio):
            return other + self

    def __sub__(self, other):
        if isinstance(other, Option):
            return OptionPortfolio([(self, 1), (other, -1)])
        elif isinstance(other, OptionPortfolio):
            return -(other - self)

    def value(self, spot_price):
        if self.option_type == OptionType.CALL:
            return max(spot_price - self.exercise_price, 0)
        elif self.option_type == OptionType.PUT:
            return max(self.exercise_price - spot_price, 0)
        else:
            raise ValueError(f"Unrecognized option type: {self.option_type}")

    def plot_pl(self, price_range=None, figsize=(12, 8), ax=None):
        from .plotting import plot_profit_and_loss

        return plot_profit_and_loss(
            self,
            price_range=price_range,
            show_each_option=False,
            figsize=figsize,
            ax=ax,
        )


class CallOption(Option):
    option_type = OptionType.CALL


class PutOption(Option):
    option_type = OptionType.PUT


class OptionPortfolio:
    """ An OptionPortfolio is a combination of multiple options """

    def __init__(self, options: List[Tuple[Option, int]]):
        # a mapping from option to its quantity
        self.options = {opt: cnt for opt, cnt in options}

    def __repr__(self):
        s = ", ".join(str(opt) + ":" + str(cnt) for opt, cnt in self.options.items())
        return "OptionPortfolio({})".format(s)

    def __mul__(self, n: int):
        for opt in self.options.keys():
            self.options[opt] = self.options[opt] * n
        return self

    def __neg__(self):
        for opt in self.options.keys():
            self.options[opt] = self.options[opt] * -1
        return self

    def __add__(self, other):
        if isinstance(other, Option):
            self.options[other] = self.options.get(other, 0) + 1
        elif isinstance(other, OptionPortfolio):
            for opt in set(self.options.keys()) | set(other.options.keys()):
                self.options[opt] = self.options.get(opt, 0) + other.options.get(opt, 0)
        else:
            raise ValueError(
                "Expecting Option or OptionPortfolio type, get: {}".format(type(other))
            )
        return self

    def __sub__(self, other):
        if isinstance(other, Option):
            self.options[other] = self.options.get(other, 0) - 1
        elif isinstance(other, OptionPortfolio):
            for opt in set(self.options.keys()) | set(other.options.keys()):
                self.options[opt] = self.options.get(opt, 0) - other.options.get(opt, 0)
        else:
            raise ValueError(
                "Expecting Option or OptionPortfolio type, get: {}".format(type(other))
            )
        return self

    def value(self, spot_price):
        return sum(
            opt.value(spot_price=spot_price) * cnt for opt, cnt in self.options.items()
        )

    @property
    def exercise_price(self) -> List[float]:
        return [opt.exercise_price for opt in self.options.keys()]

    def plot_pl(
        self, price_range=None, show_each_option=True, figsize=(12, 8), ax=None
    ):
        from .plotting import plot_profit_and_loss

        return plot_profit_and_loss(
            self,
            price_range=price_range,
            show_each_option=show_each_option,
            figsize=figsize,
            ax=ax,
        )


if __name__ == "__main__":
    c1 = CallOption(100)
    c2 = CallOption(110)
    c3 = CallOption(105)

    # p = c1 + c2 * 3
    # print(p)
    # p = p - c3
    # print(p.value(120))

    print(2 * c1)
