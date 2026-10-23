# Finance

Nine tools for what holding a set of positions costs and what a stream of
cash is worth: the financing, borrow and cash drag on the positions, the
margin a broker must hold against them and how long a stress path leaves
before the first call, their currency exposure and carry and the FX forward
that hedges it, and the time value of dated cash flows -- NPV, IRR,
duration, a cost of capital and a discounted valuation.

The runtime takes POSITIONS or CASH FLOWS. That is what separates it from
`portfolio`, which builds a book out of tickers and an asset covariance,
and from `allocation`, which takes return streams. Every rate, spread, FX
rate and day count is the caller's: this library has no rates or FX
provider, and a tool never substitutes a number it was not given.

---

## 1. The tools

| Tool | What it answers |
|---|---|
| `estimate_financing_cost` | What a set of positions costs to finance over a horizon under the account's terms: debit interest on borrowed cash, credit on idle cash, each short's borrow, the rebate on segregated proceeds, the net cost, the cash drag, and the net cost as an annual rate on gross exposure and on equity |
| `compute_margin_requirement` | The Reg T initial and FINRA 4210 maintenance margin each position needs and the account's total, each line naming the paragraph it applies, with the excess over each when the equity is given |
| `analyze_margin_runway` | Along a stress path you supply, the day of the first maintenance call and its amount, the day forced deleveraging starts if calls go unmet, and the least cash today that avoids every call |
| `get_currency_exposure` | How much of a multi-currency set of positions is exposed to each currency once its FX forwards and spot hedges are counted, how much of each is hedged, and how much of each position's return was local, FX and cross |
| `analyze_book_carry` | What each position and the whole set earn in a year if prices do not move: a future's (S - F)/(F T), a funded line's income less financing and borrow |
| `price_fx_forward_hedge` | The covered-interest-parity forward for a currency pair with each leg on its own day count and any cross-currency basis, its points, what hedging into the base currency picks up or costs a year, and a foreign asset's hedged against unhedged return |
| `analyze_cash_flows` | What a stream of periodic or dated cash flows is worth at a rate (NPV), what it returns (every real IRR, the IRR only when unique; MIRR), and how its value moves with the yield (Macaulay and modified duration, convexity, DV01) |
| `estimate_cost_of_capital` | What capital costs: CAPM cost of equity, after-tax cost of debt, WACC on market values, Hamada's unlevered beta and the beta and WACC relevered at a target D/E, and the cost of equity and WACC one standard error of beta either side |
| `value_cash_flows` | What a forecast is worth by DCF with a Gordon or exit-multiple terminal value and how much of it is the terminal value; what discount rate, terminal growth or exit multiple a target value implies (reverse DCF); the value across a grid of rates and growths or multiples |

The five position tools take the same `positions`, `base_currency` and
`fx_rates` -- `analyze_margin_runway` inside its `holdings` -- with a
`Position` per line (`standard_quant_tools.agent.runtimes._positions`),
each line carrying the extra terms its tool reads. A position names its
`instrument`, never a ticker to fetch, so every tool here is a pure
function of its arguments and replays exactly.

---

## 2. Financing and margin

### What a set of positions costs to finance

`estimate_financing_cost` applies the account's financing terms --
`financing`, the same `FinancingSpec` input the portfolio simulation and
the pair backtest take: a benchmark rate (scalar or `{ISO date: rate}`), a
required `day_count` (ACT/360 or ACT/365F), credit and debit spreads,
`borrow_bps` per instrument with a `default_borrow_bps`, and short proceeds
held as cash or segregated at a rebate -- to `positions` and the account's
own `equity` over a horizon of `days` or `start_date` to `end_date`. Every
line is a funded holding, so cash is equity less the lines' net; positive
cash earns the benchmark less the credit spread and negative cash pays it
plus the debit spread, every short pays its borrow on its market value (its
own `borrow_fee_bps`, else the terms' entry for it, else the default -- the
result says which), and segregated proceeds earn the benchmark less the
rebate spread.

Interest is simple, on today's balances at today's prices and FX rates:
$10m long on $5m at 5.30% + 50 bp for 30 days ACT/360 pays 24,166.67. A
dated benchmark is read on the dates it gives, each rate in force until the
next (Friday's covers the weekend), and a run of more than four days
without a fixing is reported. `cash_drag` is what idle cash and segregated
proceeds earn below the benchmark -- the spreads the broker keeps. A short
with no borrow rate is refused, as the spec refuses it. Futures and swaps
are not funded this way and are left out of the lines; cash held in a
second currency at its own rate is a second call.

### The margin it needs

`compute_margin_requirement` gives US Reg T initial margin -- 50% of a
long, 150% of a short with its proceeds (12 CFR 220.12(a), (c)) -- and FINRA
Rule 4210 maintenance: 25% of a long (4210(c)(1)); a short under $5, the
greater of $2.50 a share and 100% of its value ((c)(2)); at $5 or more, the
greater of $5.00 a share and 30% ((c)(3)); a leveraged or inverse ETF at
those rates times its leverage factor, at most 100% (4210(f)(8)(A),
Regulatory Notice 09-53); an uncovered short equity option at its value
plus 20% of the underlying less the out-of-the-money amount, at least its
value plus 10% of the underlying (call) or strike (put) (4210(f)(2)(E)(i));
a long option paid in full.

Each position carries `kind` ('equity', 'etf', 'call', 'put'), an ETF's
`leverage`, and an option's `strike` and `underlying_price`, with the
premium as its price and the contract size as its multiplier. Every amount
is the equity the account must hold, in USD, so the lines add up: a short's
150% is its proceeds plus the 50% reported. The account's initial
requirement is the greater of the Reg T amount, the maintenance amount and
the $2,000 minimum equity of 4210(b). `house_rates` raises a line's
maintenance rate and is refused below the rule's, naming the minimum.

Lines are margined without offsets -- a covered call, a spread or a hedge
earns no reduction -- so a hedged book's requirement is an upper bound.
These are regulatory minimums, not what a broker charges.

### How long a stress path leaves

`analyze_margin_runway` walks a stress path one day at a time, day 0 being
today: daily `equity_changes` against an `initial_margin` and
`maintenance_margin` held for the path (given, or computed from `holdings`
today), or daily `returns` per instrument that revalue the holdings' stocks
and ETFs and recompute their requirements on each day's prices, tiers
included. The first day equity is below maintenance is a call for the
amount that restores the initial requirement (`restore_to`, default
"initial") or maintenance: initial 10k, maintenance 8k and equity 12k
losing 1k a day call on day 5 for 3k.

A call the market does not cure within `cure_days` -- default 15, the most
FINRA 4210(f)(6) allows; a broker may liquidate sooner -- starts forced
deleveraging on its last day, and one still open when the path ends is
reported as outstanding. `buffer` is the least cash added today that leaves
no day below maintenance (one unit less calls on `tightest_day`). Nothing
is posted, deposited or sold: it measures.

---

## 3. Currency, carry and the FX forward

### Currency exposure, after the hedges

`get_currency_exposure` takes the positions plus `hedges`: a map from a
currency pair to a notional in units of its first currency, positive bought
and negative sold. `{"EURUSD": -10000000}` is a forward (or spot trade)
selling ten million EUR against USD. The second leg of a hedge is the equal
and opposite value in the base currency at the caller's rates, so a hedge
has no value of its own here: it moves exposure from one currency to
another, and its legs sum to nothing in the base currency, bit for bit. A
cross pair ("EURGBP" in a USD book) needs a rate for both legs.

For each currency the result gives the positions' net exposure, the
hedges', the sum, the gross and the net in the currency's own units, and
the hedge ratio -hedge / position: 1 is fully hedged, above 1 over-hedged,
and negative a "hedge" that adds to the exposure. The base currency has no
ratio, since it carries no currency risk, and neither does a currency the
positions net to nothing in; both are null with the reason.

Give every position a `local_return` (its price return in its own currency
over the period) and `fx_returns` (each currency's change in its rate,
units of base per one) and each position's base-currency return is split
into its local, FX and cross terms, r_L + r_FX + r_L x r_FX =
(1 + r_L)(1 + r_FX) - 1. The three add up to the base return exactly. The
hedges' FX P&L is reported beside the positions', per currency and in
total: a hedge sized on the starting value cancels the FX term and leaves
the cross term, the gain on the gain.

### Carry, one annual rate per line

`analyze_book_carry` reads two kinds of line, told apart by the terms each
carries (Koijen, Moskowitz, Pedersen & Vrugt, "Carry", 2018). A future or
forward carries `spot` and `years_to_expiry`, its `price` being the
future's; its carry is (S - F)/(F T), what a long earns if the spot stays
put and the future converges to it. For a future priced off the carry
identity F = S exp((r - q) T) that is q - r to first order in T, with an
error of about (r - q)^2 T / 2, and `carry_log`, ln(S / F) / T, is q - r
exactly. A funded holding -- a share, a bond, a currency deposit -- carries
`income_rate` and `financing_rate`: a long earns income less financing, a
short earns financing less income less its `borrow_fee_bps`, which a funded
short must state (0 if it borrows free).

An FX forward is a forward-priced line; a foreign deposit funded at home is
a funded one, and carries the interest differential exactly. A short earns
the opposite of a long. The set's carry is the gross-notional-weighted
carry of its lines, and `annual_carry` the amount in the base currency.
Carry is the return if nothing moves, not a forecast.

### The FX forward that hedges it

`price_fx_forward_hedge` prices a pair quoted as units of its second
currency per one of its first ("EURUSD" 1.10 is 1.10 USD per EUR) by
covered interest parity, F = S (1 + r_second t_second) / (1 + r_first
t_first), each leg's year fraction under its own day count -- USD and EUR
money markets count ACT/360, GBP ACT/365F, and the caller states which.
ACT/ACT and 30/360 count from dates and need `start_date`. `base_currency`
is the hedger's, one of the pair's two; the other is foreign, and
`basis_bps`, the cross-currency basis, is added to the foreign rate, so a
negative basis (as EUR and JPY have traded against USD) raises a
base-currency hedger's pickup.

Forward points are in pips of 0.0001, or 0.01 when the quote currency is
JPY, unless `pip_size` says otherwise. `hedge_carry` is the forward premium
over the base leg's year fraction: positive is a pickup, negative a cost.
With a foreign asset's `asset_return` and the pair's `spot_at_maturity`,
the result gives the unhedged return and the return hedged with
`hedge_ratio` of the starting value sold forward. EURUSD 1.1000, USD 5.30%
and EUR 3.90% ACT/360 for 90 days: 1.103813, +38.13 pips, a pickup of
1.3865% a year.

Carry is a rate on the forward's notional, the hedge's pickup a rate on
spot value; the two differ by F/S, and a short forward's carry times F/S is
the hedge's `hedge_carry` exactly.

---

## 4. Cash flows, the cost of capital and a valuation

### NPV, IRR and duration in one call

`analyze_cash_flows` is one tool for the time value of a stream of cash,
rather than an `npv`, an `irr` and a `duration` that would each re-read the
same flows. The flows are either a list -- flow k at k / `periods_per_year`
years, so flow 0 is now and is NOT discounted (numpy-financial's
convention; Excel's NPV discounts its first value a whole period) -- or
`dated_flows`, {ISO date: amount}, timed from the first date under a
`day_count` the caller must give (`ACT/365F`, `ACT/360`, `30/360` or
`ACT/ACT`). A rate compounds `periodic`ally (at the flow frequency, as a
bond yield is quoted), `annual`ly (XIRR's effective annual rate, the
default for dated flows) or `continuous`ly; every rate in and out is in
that convention. On Excel's documented example the XIRR is 0.373362535 and
the XNPV at 9% 2086.65; a 10-year 5% semiannual bond at par has Macaulay
duration 7.989446, modified 7.794581 and DV01 0.0779458. A bond's duration
takes its coupons and principal only: with the price among the flows the
present value at the yield is zero, and the durations are null with that
reason.

The IRR is null unless the flows have exactly one. Every real root is
found, not the one nearest a starting guess: the number of roots is at most
the sign changes of the flows (Descartes' rule, which holds for an
exponential sum), every root lies inside bounds read from the amounts, and
Rolle's theorem isolates them -- the derivative of the sum, with one sign
change fewer, splits the range into pieces each holding at most one root.
Lorie and Savage's pump (-1,600 / +10,000 / -10,000) gives `irr_status`
'multiple' with `irr_roots` 25% and 400%; an all-positive stream gives
'none'. `norstrom_holds` reports Norstrom's sufficient condition: when the
cumulative flows change sign exactly once, exactly one IRR is above 0%.
Past 20 sign changes the roots are bracketed on a grid and the status is
'uncertified', since two roots closer than the grid can be missed in pairs.
MIRR (`finance_rate` and `reinvest_rate`, both or neither) is Excel's for
annual flows: 12.6094% on its documented example.

### What capital costs

`estimate_cost_of_capital` computes CAPM and WACC from the caller's inputs,
every one of them: the risk-free rate, the beta, the equity risk premium,
the market values of equity and debt, the pre-tax cost of debt (required
with any debt) and the tax rate (required, in [0, 1)). Nothing is fetched
or defaulted but a debt value of zero, and with no debt the WACC is the
cost of equity to the bit. Hamada's unlevered beta, beta / (1 + (1 - t)
D/E), is always reported; with `target_debt_to_equity` the `at_target`
block gives the beta, cost of equity and WACC relevered there (1.3 ->
0.931900 -> 1.3 at D/E 0.5 and t = 21%), holding the cost of debt as given
-- more leverage usually costs more, and the result says so. With
`beta_standard_error` the `beta_sensitivity` block gives the cost of equity
and WACC at beta -/+ one error: ERP x SE each way, which on a regression
beta is routinely 100 bp of cost of equity.

### A discounted valuation, and the one a price implies

`value_cash_flows` is a DCF of forecast free cash flows (years 1..N, at
year end, an effective annual rate) with a Gordon terminal value
FCF_N (1 + g) / (r - g) or an exit multiple times a year-N metric (FCF_N by
default, EBITDA if given as `terminal_metric`). The terminal value's share
of the value is always reported, with a warning above 75%: then the
valuation is a statement about g and r. Each method's value also comes back
as the other's assumption -- the multiple a Gordon value implies and the
perpetual growth an exit multiple implies. Constant growth reproduces the
perpetuity: FCF_1 = 10 at 9% and 3% is 166.667 whatever N is. Gordon growth
at or above the rate is refused with the reason.

With `target_value` and `solve_for` it is a reverse DCF: the discount rate,
terminal growth or exit multiple the target implies. The rate and the
growth are found by scanning their whole admissible range for every sign
change and polishing each bracket with Brent's method; the bracket and the
range searched are reported. A target reached at two rates -- negative
near-term flows can do that -- is null with both listed, and an unreachable
one is null with the range. The `sensitivity` grid is the value at each
discount rate (rows) against each growth or multiple (columns), by default
+/-2 points of rate and +/-1 point of growth (or the multiple x 0.8 ...
1.2) around the input or the solved value; a Gordon cell with g >= r is
null.

---

## 5. Nothing here fetches

There is no rates, FX, margin-parameter or fundamentals provider behind
this runtime, and none of its tools pretends to one. A benchmark rate, a
spread, a borrow fee, an FX rate, a discount rate, a beta and a day count
all arrive as arguments, and a tool missing one refuses with the field to
supply rather than assuming a market. That is also why every number here
replays: the same positions and terms give the same bits on any machine.

## 6. What is not here

- Exchange or broker house margin (SPAN parameter files, house tables):
  the regulatory minimum is computed, and `house_rates` raises it.
- Strategy-based option offsets (spreads, covered calls, straddles), LEAPS
  at 75%, broad-based index options and options on leveraged ETFs: refused
  or stated in the result.
- Cash in a second currency financed at its own rate within one call.
- A term structure of rates or a full fixed-income stack: a cash-flow
  stream is discounted at one rate in one convention.
- Point-in-time fundamentals: a valuation's cash flows are the caller's.

## Related

- [05_portfolio.md](05_portfolio.md) -- building a book from tickers and a
  covariance, which this runtime then finances and margins.
- [31_allocation.md](31_allocation.md) -- capital across return streams,
  and a book's time- and money-weighted return with its flows.
- [28_delta_one.md](28_delta_one.md) -- a quoted future's basis and the
  carry component it implies.
- [04_backtesting.md](04_backtesting.md) -- the same `FinancingSpec` inside
  the engines.
