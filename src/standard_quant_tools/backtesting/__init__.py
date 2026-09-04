"""
What a backtest MEANS once it has been run — as distinct from `backtest`,
which runs one.

TWO PACKAGES ONE LETTER APART, and the split is real rather than
accidental. `backtest` holds the engine, the strategies, the sizing and the
cost models: everything that turns prices and a rule into an equity curve.
This one holds what you do to that curve before believing it — the
deflated Sharpe and the probability of backtest overfitting in
`overfitting`, and the per-trade decomposition in `trade_analysis`.

The names are close enough to mistype, so the rule is: if it produces a
result, it is `backtest`; if it judges one, it is `backtesting`.

THIS FILE EXISTS BECAUSE ITS ABSENCE WAS NOT A DECISION. Every other
subpackage here declares itself and this one did not, so it resolved as a
PEP 420 namespace package — which imports correctly, and is why nothing
ever failed. The costs are quiet ones: a namespace package silently merges
with any same-named directory elsewhere on `sys.path`, and packaging tools
treat it differently from a regular package. Neither had bitten, and both
are the kind of thing that bites once, far from here.

Deliberately not re-exporting the module contents. Both callers import the
modules (`from standard_quant_tools.backtesting import overfitting as lib`)
rather than names, and a convenience layer nobody asked for is a second
place for the export list to drift out of date.
"""
