import datetime as dt
import logging
from math import exp, sqrt
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from abides_core import NanosecondTime

from .mean_reverting_oracle import MeanRevertingOracle


logger = logging.getLogger(__name__)


class SparseMeanRevertingOracle(MeanRevertingOracle):
    """The SparseMeanRevertingOracle produces a fundamental value time series for
    each requested symbol, and provides noisy observations of the fundamental
    value upon agent request.  This "sparse discrete" fundamental uses a
    combination of two processes to produce relatively realistic synthetic
    "values": a continuous mean-reverting Ornstein-Uhlenbeck process plus
    periodic "megashocks" which arrive following a Poisson process and have
    magnitude drawn from a bimodal normal distribution (overall mean zero,
    but with modes well away from zero).  This is necessary because OU itself
    is a single noisy return to the mean (from a perturbed initial state)
    that does not then depart the mean except in terms of minor "noise".

    Historical dates are effectively meaningless to this oracle.  It is driven by
    the numpy random number seed contained within the experimental config file.
    This oracle uses the nanoseconds portion of the current simulation time as
    discrete "time steps".

    This version of the MeanRevertingOracle expects agent activity to be spread
    across a large amount of time, with relatively sparse activity.  That is,
    agents each acting at realistic "retail" intervals, on the order of seconds
    or minutes, spread out across the day.
    """

    def __init__(
        self,
        mkt_open: NanosecondTime,
        mkt_close: NanosecondTime,
        symbols: Dict[str, Dict[str, Any]],
    ) -> None:
        # Symbols must be a dictionary of dictionaries with outer keys as symbol names and
        # inner keys: r_bar, kappa, sigma_s.
        self.mkt_open: NanosecondTime = mkt_open
        self.mkt_close: NanosecondTime = mkt_close

        self.symbols: Dict[str, Dict[str, Any]] = symbols

        self.f_log: Dict[str, List[Dict[str, Any]]] = {}

        # The dictionary r holds the most recent fundamental values for each symbol.
        self.r: Dict[str, pd.Series] = {}

        # The dictionary megashocks holds the time series of megashocks for each symbol.
        # The last one will always be in the future (relative to the current simulation time).
        #
        # Without these, the OU process just makes a noisy return to the mean and then stays there
        # with relatively minor noise.  Here we want them to follow a Poisson process, so we sample
        # from an exponential distribution for the separation intervals.
        self.megashocks: Dict[str, List[Dict[str, Any]]] = {}

        then = dt.datetime.now()

        # Note that each value in the self.r dictionary is a 2-tuple of the timestamp at
        # which the series was computed and the true fundamental value at that time.
        for symbol in symbols:
            s = symbols[symbol]
            logger.debug(
                "SparseMeanRevertingOracle computing initial fundamental value for {}".format(
                    symbol
                )
            )
            self.r[symbol] = (mkt_open, s["r_bar"])
            self.f_log[symbol] = [
                {"FundamentalTime": mkt_open, "FundamentalValue": s["r_bar"]}
            ]

            # Compute the time and value of the first megashock.  Note that while the values are
            # mean-zero, they are intentionally bimodal (i.e. we always want to push the stock
            # some, but we will tend to cancel out via pushes in opposite directions).
            ms_time_delta = np.random.exponential(scale=1.0 / s["megashock_lambda_a"])
            mst = self.mkt_open + ms_time_delta
            msv = s["random_state"].normal(
                loc=s["megashock_mean"], scale=sqrt(s["megashock_var"])
            )
            msv = msv if s["random_state"].randint(2) == 0 else -msv

            self.megashocks[symbol] = [{"MegashockTime": mst, "MegashockValue": msv}]

        now = dt.datetime.now()

        logger.debug(
            "SparseMeanRevertingOracle initialized for symbols {}".format(symbols)
        )
        logger.debug(
            "SparseMeanRevertingOracle initialization took {}".format(now - then)
        )

    def compute_fundamental_at_timestamp(
        self, ts: NanosecondTime, v_adj, symbol: str, pt: NanosecondTime, pv
    ) -> int:
        """
        Arguments:
          ts: A requested timestamp to which we should advance the fundamental.
          v_adj: A value adjustment to apply after advancing time (must pass zero if none).
          symbol: A symbol for which to advance time.
          pt: A previous timestamp.
          pv: A previous fundamental.

        Returns:
          The new value.

        The last two parameters should relate to the most recent time this method was invoked.

        As a side effect, it updates the log of computed fundamental values.
        """

        s = self.symbols[symbol]

        # This oracle uses the Ornstein-Uhlenbeck Process.  It is quite close to being a
        # continuous version of the discrete mean reverting process used in the regular
        # (dense) MeanRevertingOracle.

        # Compute the time delta from the previous time to the requested time.
        d = ts - pt

        # Extract the parameters for the OU process update.
        mu = s["r_bar"]
        gamma = s["kappa"]
        theta = s["fund_vol"]  # the volatility value for the generated time-series.

        # The OU process is able to skip any amount of time and sample the next desired value
        # from the appropriate distribution of possible values.
        v = s["random_state"].normal(
            loc=mu + (pv - mu) * (exp(-gamma * d)),
            scale=sqrt(((theta**2) / (2 * gamma)) * (1 - exp(-2 * gamma * d))),
        )

        # Apply the value adjustment that was passed in.
        v += v_adj

        # The process is not permitted to become negative.
        v = max(0, v)

        # For our purposes, the value must be rounded and converted to integer cents.
        v = int(round(v))

        # Cache the new time and value as the "previous" fundamental values.
        self.r[symbol] = (ts, v)

        # Append the change to the permanent log of fundamental values for this symbol.
        self.f_log[symbol].append({"FundamentalTime": ts, "FundamentalValue": v})

        # Return the new value for the requested timestamp.
        return v

    def advance_fundamental_value_series(
        self, current_time: NanosecondTime, symbol: str
    ) -> int:
        """This method advances the fundamental value series for a single stock symbol,
        using the OU process.  It may proceed in several steps due to our periodic
        application of "megashocks" to push the stock price around, simulating
        exogenous forces."""

        # Generation of the fundamental value series uses a separate random state object
        # per symbol, which is part of the dictionary we maintain for each symbol.
        # Agent observations using the oracle will use an agent's random state object.
        s = self.symbols[symbol]

        # This is the previous fundamental time and value.
        pt, pv = self.r[symbol]

        # If time hasn't changed since the last advance, just use the current value.
        if current_time <= pt:
            return pv

        # Otherwise, we have some work to do, advancing time and computing the fundamental.

        # We may not jump straight to the requested time, because we periodically apply
        # megashocks to push the series around (not always away from the mean) and we need
        # to compute OU at each of those times, so the aftereffects of the megashocks
        # properly affect the remaining OU interval.

        mst = self.megashocks[symbol][-1]["MegashockTime"]
        msv = self.megashocks[symbol][-1]["MegashockValue"]

        while mst < current_time:
            # A megashock is scheduled to occur before the new time to which we are advancing.  Handle it.

            # Advance time from the previous time to the time of the megashock using the OU process and
            # then applying the next megashock value.
            v = self.compute_fundamental_at_timestamp(mst, msv, symbol, pt, pv)

            # Update our "previous" values for the next computation.
            pt, pv = mst, v

            # Since we just surpassed the last megashock time, compute the next one, which we might or
            # might not immediately consume.  This works just like the first time (in __init__()).

            mst = pt + int(np.random.exponential(scale=1.0 / s["megashock_lambda_a"]))
            msv = s["random_state"].normal(
                loc=s["megashock_mean"], scale=sqrt(s["megashock_var"])
            )
            msv = msv if s["random_state"].randint(2) == 0 else -msv

            self.megashocks[symbol].append(
                {"MegashockTime": mst, "MegashockValue": msv}
            )

            # The loop will continue until there are no more megashocks before the time requested
            # by the calling method.

        # Once there are no more megashocks to apply (i.e. the next megashock is in the future, after
        # current_time), then finally advance using the OU process to the requested time.
        v = self.compute_fundamental_at_timestamp(current_time, 0, symbol, pt, pv)

        return v

    def get_daily_open_price(
        self, symbol: str, mkt_open: NanosecondTime, cents: bool = True
    ) -> int:
        """Return the daily open price for the symbol given.

        In the case of the MeanRevertingOracle, this will simply be the first
        fundamental value, which is also the fundamental mean. We will use the
        mkt_open time as given, however, even if it disagrees with this.
        """

        # The sparse oracle doesn't maintain full fundamental value history, but rather
        # advances on demand keeping only the most recent price, except for the opening
        # price.  Thus we cannot honor a mkt_open that isn't what we already expected.

        logger.debug(
            "Oracle: client requested {} at market open: {}".format(
                symbol, self.mkt_open
            )
        )

        open_price = self.symbols[symbol]["r_bar"]
        logger.debug("Oracle: market open price was was {}".format(open_price))

        return open_price

    def observe_price(
        self,
        symbol: str,
        current_time: NanosecondTime,
        random_state: np.random.RandomState,
        sigma_n: int = 1000,
    ) -> int:
        """Return a noisy observation of the current fundamental value.

        While the fundamental value for a given equity at a given time step does
        not change, multiple agents observing that value will receive different
        observations.

        Only the Exchange or other privileged agents should use sigma_n==0.

        sigma_n is experimental observation variance.  NOTE: NOT STANDARD DEVIATION.

        Each agent must pass its RandomState object to observe_price.  This ensures that
        each agent will receive the same answers across multiple same-seed simulations
        even if a new agent has been added to the experiment.
        """

        # If the request is made after market close, return the close price.
        if current_time >= self.mkt_close:
            r_t = self.advance_fundamental_value_series(self.mkt_close - 1, symbol)
        else:
            r_t = self.advance_fundamental_value_series(current_time, symbol)

        # Generate a noisy observation of fundamental value at the current time.
        if sigma_n == 0:
            obs = r_t
        else:
            obs = int(round(random_state.normal(loc=r_t, scale=sqrt(sigma_n))))

        logger.debug(
            "Oracle: current fundamental value is {} at {}".format(r_t, current_time)
        )
        logger.debug("Oracle: giving client value observation {}".format(obs))

        # Reminder: all simulator prices are specified in integer cents.
        return obs
