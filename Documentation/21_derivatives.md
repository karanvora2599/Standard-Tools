# Derivatives

Twelve tools for what an option is worth and what holding it does to you.
Pricing under four models, the second-order greeks that explain why a
delta-hedged book still loses money, the internal consistency of a quoted
surface, and what a hedge actually costs to run.

`12_options.md` covers the pricing models themselves — Black-Scholes,
Black-76, Bachelier, the binomial lattice — and is the right place to start
if the question is "what does this option cost". This document is about
everything that happens after that number exists.

## There is no options data provider, and that is deliberate

Nothing here fetches a chain. Every tool takes quotes as arguments: a smile
arrives as parallel lists of strikes and implied vols, a term structure as a
mapping of expiry to volatility, an execution as a list of fills.

The library has no options data source, and a tool that pretended otherwise
would fetch equity prices and compute a "chain" that does not exist. Passing
the quotes in has a second benefit that turns out to matter more: the same
tools work on a hypothetical surface, which is most of what they are used
for. Asking "what would this structure be worth if the skew steepened" is a
more common question than "what is it worth right now", and it is one a
chain-fetching tool could not answer at all.

## The three misreadings these tools exist to prevent

Each of these is a number that looks self-explanatory and is not. They are
called out in the tool descriptions themselves, because the description is
what a model reads before choosing.

### Volatility means different things to different models

The lognormal models — Black-Scholes and Black-76 — take a **relative**
volatility, a fraction of the underlying per year. Bachelier takes an
**absolute** one, in the underlying's own units.

Passing `0.30` to Bachelier on an $80 future means thirty cents of annual
volatility, not 30%, and the resulting price is wrong by two orders of
magnitude. No type system catches this: both are positive floats in the same
range. `get_option_pricing` says so in its description and `12_options.md`
says so at length, and it remains the easiest way to get a badly wrong
number out of this library.

### The expected move is not a bound

`get_expected_move` returns a **one standard deviation** move. It gets
quoted as "the expected move" and then read as a ceiling — and under the
model's own assumptions it is exceeded about 32% of the time, roughly one
earnings print in three.

Both conventions are returned, because two circulate and they differ by 20%:

| Convention | Formula | What it is |
|---|---|---|
| One standard deviation | `S · σ · √T` | The distributional statement |
| Straddle approximation | `0.8 · S · σ · √T` | What the at-the-money straddle costs |

A strategy that sells the straddle because "the move is priced at 5%" is
short exactly the third of the distribution that exceeds it. Pass
`realized_moves` — past absolute moves over the same horizon — to get the
**historical** exceedance rate rather than the lognormal one, which is the
number that would actually have decided the trade.

### A calendar spread prices the forward vol, not the difference

30-day implied at 25 and 60-day at 28 does not offer you 28 for the second
month. Total variance adds across time, so the second month is offered at
whatever makes the arithmetic work:

```
√((0.28² × 60 − 0.25² × 30) / 30) = 0.307
```

`analyze_vol_term_structure` returns that number. Trading off the quoted
levels rather than the forward can reverse the sign of the position. When
the forward *variance* comes out negative the quotes admit a calendar
arbitrage — which in practice means one of them is stale, and the near
expiry is usually the illiquid one.

## Greeks beyond the first order

`get_option_pricing` returns delta, gamma, vega, theta and rho — today's
risk. `get_option_greeks` returns how that risk **changes**, which is what
actually gets a hedged position into trouble.

| Greek | What it measures | Why it matters |
|---|---|---|
| **vanna** | ∂delta/∂vol | A short-vol book that is delta-flat today is not delta-flat after a vol spike |
| **volga** | ∂vega/∂vol | Vega peaks at the money, so a wing's vega *grows* as vol rises — why short wings lose more than the vega number suggested |
| **charm** | ∂delta/∂time | Why a Friday delta-flat book opens Monday short with no move in the underlying |
| **speed** | ∂gamma/∂spot | The size of the rehedge on a large gap |

**Units are stated per greek in the result**, because there is no convention
and the mismatch is a real source of error. Vega and volga are per one
volatility *point* (0.01); theta and charm are per calendar day.

Every one of these is validated against a central finite difference of the
first-order greek it differentiates — the analytic form and the numerical
derivative share no code, so agreement to 1e-9 means the algebra is right
rather than merely self-consistent. See
`tests/analysis/test_derivatives.py`.

**The honest limit:** every greek here is the derivative of one model at one
volatility. A real book has a smile, and the market does not shift every
strike's vol by the same amount — so summing vega across strikes and
multiplying by an expected vol move overstates the P&L.

## Surface consistency

### `check_put_call_parity`

`C − P = S·e^(−qT) − K·e^(−rT)` is a **model-free** identity. It follows
from the payoffs alone and holds under any distribution, whatever the
volatility and whatever the model — which is what makes it the right first
check on a quoted chain. A violation is a data problem or an opportunity,
never a modelling disagreement.

**The usual cause is not arbitrage.** In order of likelihood: the two quotes
carry different timestamps, one leg is a last-traded price standing in for a
mid, the dividend assumption is wrong, or the underlying is hard to borrow —
in which case the apparent violation is exactly the borrow cost. The result
returns the **implied dividend yield** and the **implied forward** so the
cause is identifiable rather than merely flagged.

### `fit_volatility_smile`

Fits the smile as a quadratic in **log-moneyness**, not in strike. The smile
is approximately symmetric in `log(K/F)` and emphatically not in `K`: a
parabola in strike puts its vertex at a fixed price, so the same shape refit
after a 10% rally reports a different skew.

The three coefficients are the three things a trader quotes — `atm_vol` is
the level, `skew` is the slope at the money (per unit of log-moneyness, so
comparable across expiries and underlyings), and `curvature` is what a
butterfly prices.

**Arbitrage is checked, and it is concavity that breaks the density.** This
is the opposite of the intuition that a "violent" smile is the dangerous
one. Durrleman's condition carries a `+w''/2` term, so a strongly *convex*
smile pushes it further above zero; a butterfly arbitrage is literally a
concave price in strike. Measured: a smile with a curvature of +25 passes
the check and one with −4 fails it.

The fit **does not extrapolate**. A quadratic continued into the wings
reaches negative variance at an ordinary distance from the money, so a
strike outside the fitted range returns a refusal rather than a
polynomial's opinion.

## What the hedge costs

### `simulate_delta_hedge`

A continuously hedged option earns `(σ_realized² − σ_implied²)` times the
dollar gamma, integrated over its life. That expectation is known in closed
form. What the closed form does not give you is the **dispersion**, and the
dispersion is what decides whether the trade is sized correctly.

**Discrete hedging error scales as 1/√n, not 1/n.** Going from daily to
twice-daily rehedging cuts the standard deviation of the P&L by 29%, not by
half — while doubling the transaction cost. That tradeoff is the reason to
simulate rather than compute, and the test suite measures the exponent
rather than asserting it.

Path dependence is the other half. The same realized volatility earns
different amounts depending on *where* the underlying spent its time: gamma
is concentrated at the money, so a path that oscillates around the strike
collects far more than one that trends away and realizes the same vol.

The sign convention is stated in every result — **short the option,
delta-hedged long** — because "the hedged P&L" is ambiguous and the sign
flips with it.

### `get_option_risk_scenarios`

A full **revaluation** grid over spot and volatility, not a delta-gamma
approximation of one. Measured error of the Taylor estimate on a long call:

| Spot move | Delta-gamma error |
|---:|---:|
| 10% | 1.2% |
| 20% | 5.0% |
| 30% | 11.1% |

The error grows with the cube of the move, which is why a stress test built
on greeks understates a real gap — exactly the scenario a stress test exists
for.

**The two axes are shocked independently and the market does not move that
way.** Equity vol rises when spot falls, reliably enough that "spot −20%,
vol unchanged" is a cell in the table and not a state of the world. Read the
down-spot/up-vol diagonal rather than a row.

## The tools

| Tool | Answers |
|---|---|
| `get_option_pricing` | What is this option worth, under four models |
| `get_implied_volatility` | What volatility reproduces this price |
| `get_option_greeks` | How does the risk change — vanna, volga, charm, speed |
| `analyze_option_strategy` | Payoff, breakevens and aggregate greeks of an arbitrary multi-leg position |
| `fit_volatility_smile` | Level, skew and curvature — and whether the quotes admit a butterfly arbitrage |
| `get_volatility_cone` | Where today's implied sits in this name's own realized history |
| `analyze_vol_term_structure` | Contango or backwardation, and the forward vols a calendar spread prices |
| `check_put_call_parity` | Are these two quotes mutually consistent, and if not, why |
| `get_implied_forward` | The carry forward, with financing, dividend and borrow broken out |
| `get_expected_move` | What move is priced — as one standard deviation, not a bound |
| `simulate_delta_hedge` | What the hedge earns, and how widely that varies |
| `get_option_risk_scenarios` | Full revaluation over spot × vol, not a Taylor estimate |

Full argument lists: [20_tool_index.md](20_tool_index.md#derivatives--derivatives).

## Unbounded losses are reported as unbounded

`analyze_option_strategy` finds breakevens numerically, by scanning the
payoff for sign changes. A short call has no worst case, and a numerical
scan cannot find one that does not exist — so when the extreme sits at the
edge of the scanned range the result says `max_loss_unbounded: true` rather
than returning the edge as though it were a bound.

A finite number standing in for an infinite risk is the single most
dangerous thing a payoff calculator can return, and it is what returning
`profit[0]` without the flag would do.

## Related

- [12_options.md](12_options.md) — the pricing models themselves
- [22_microstructure.md](22_microstructure.md) — what it costs to trade the hedge
- [19_runtimes.md](19_runtimes.md) — why derivatives is its own execution boundary
