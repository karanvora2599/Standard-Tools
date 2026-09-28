"""Option-chain throughput: one call per contract against one call per chain.

Two workloads, sized to one equity surface:

- an implied-volatility solve over a 476-contract chain (17 expiries x 14
  strikes x calls and puts), quoted off a smile with 2% noise so a share of
  the quotes sit outside the no-arbitrage bounds, as real ones do;
- the greeks of those 476 contracts on a 61-spot grid -- 29,036 valuations,
  the input to a gamma profile -- and the zero-gamma search on top of it.

Each is timed as the scalar loop (`implied_volatility` / `black_scholes_
greeks` / `option_greeks` per contract), the batch on the compiled kernel,
and the batch on its numpy fallback. Min of N, warm.

    python tests/bench/bench_options.py
    SQT_NUM_THREADS=1 python tests/bench/bench_options.py   # serial kernels
"""

import gc
import os
import time

import numpy as np

from standard_quant_tools.analysis import options_batch as ob
from standard_quant_tools.analysis.derivatives import option_greeks
from standard_quant_tools.analysis.options import (
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
)
from standard_quant_tools.error import ValidationError

REPS = int(os.environ.get("BENCH_REPS", "7"))
SPOT = 230.0


def bench(fn, reps=REPS):
    fn()  # warm
    best = float("inf")
    gc.disable()
    try:
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
    finally:
        gc.enable()
    return best * 1e3  # ms


def chain():
    rng = np.random.default_rng(7)
    expiries = (
        np.array(
            [
                7,
                14,
                21,
                30,
                45,
                60,
                90,
                120,
                150,
                180,
                240,
                300,
                365,
                450,
                540,
                640,
                730,
            ]
        )
        / 365.0
    )
    moneyness = np.linspace(0.7, 1.35, 14)
    t, k, call = np.meshgrid(expiries, SPOT * moneyness, [True, False], indexing="ij")
    t, k, call = t.ravel(), k.ravel(), call.ravel()
    x = np.log(k / SPOT)
    vol = 0.24 - 0.35 * x + 0.9 * x * x  # a skewed equity smile
    rate, q = 0.043, 0.005
    fair = np.array(
        [
            black_scholes_price(SPOT, kk, tt, rate, vv, "call" if c else "put", q)
            for kk, tt, vv, c in zip(k, t, vol, call)
        ]
    )
    price = fair * np.exp(rng.normal(0.0, 0.02, fair.size))
    return dict(price=price, k=k, t=t, vol=vol, call=call, rate=rate, q=q)


def main():
    c = chain()
    n = c["price"].size
    has_native = ob._native("implied_volatility_batch") is not None
    threads = os.environ.get("SQT_NUM_THREADS", "OpenMP default")
    print(f"contracts={n}  native={has_native}  SQT_NUM_THREADS={threads}  reps={REPS}")

    def iv_scalar():
        for i in range(n):
            try:
                implied_volatility(
                    c["price"][i],
                    SPOT,
                    c["k"][i],
                    c["t"][i],
                    c["rate"],
                    "call" if c["call"][i] else "put",
                    c["q"],
                )
            except ValidationError:
                pass

    def iv_batch():
        return ob.implied_volatility_batch(
            c["price"], SPOT, c["k"], c["t"], c["rate"], c["q"], c["call"]
        )

    spots = SPOT * np.linspace(0.7, 1.3, 61)

    def greeks_scalar_raw():
        for s in spots:
            for i in range(n):
                black_scholes_greeks(
                    s,
                    c["k"][i],
                    c["t"][i],
                    c["rate"],
                    c["vol"][i],
                    "call" if c["call"][i] else "put",
                    c["q"],
                )

    def greeks_scalar_full():
        for s in spots:
            for i in range(n):
                option_greeks(
                    spot=s,
                    strike=c["k"][i],
                    time_to_expiry=c["t"][i],
                    volatility=c["vol"][i],
                    risk_free_rate=c["rate"],
                    option_type="call" if c["call"][i] else "put",
                    dividend_yield=c["q"],
                )

    def greeks_batch():
        return ob.black_scholes_greeks_batch(
            spots, c["k"], c["t"], c["vol"], c["rate"], c["q"], c["call"], grid=True
        )

    # Long the strikes below spot, short those above, sized toward the
    # money: a book whose net gamma crosses zero inside the grid. (Signing
    # by call/put would cancel exactly -- a call and a put at one strike and
    # expiry have the same gamma.)
    qty = np.where(c["k"] < SPOT, 1.0, -1.0) * np.exp(
        -((np.log(c["k"] / SPOT) / 0.15) ** 2)
    )

    def zero_gamma():
        return ob.zero_gamma_spot(
            c["k"],
            c["t"],
            c["vol"],
            qty,
            spot_low=spots[0],
            spot_high=spots[-1],
            risk_free_rate=c["rate"],
            dividend_yield=c["q"],
            n_grid=61,
        )

    refused = iv_batch()["refusals"]
    print(f"refused quotes: {refused} ({sum(refused.values())}/{n})")
    rows = []
    t_iv = bench(iv_scalar, reps=max(3, REPS // 2))
    rows.append(("implied vol, 476 contracts", "scalar loop", t_iv))
    t_grid_raw = bench(greeks_scalar_raw, reps=3)
    rows.append(
        ("greeks 61 x 476 = 29,036", "scalar loop, black_scholes_greeks", t_grid_raw)
    )
    t_grid_full = bench(greeks_scalar_full, reps=3)
    rows.append(("greeks 61 x 476 = 29,036", "scalar loop, option_greeks", t_grid_full))
    for label, flag in (("batch, C++", True), ("batch, numpy fallback", False)):
        if flag and not has_native:
            continue
        saved = ob.HAS_CPP
        ob.HAS_CPP = flag and saved
        try:
            rows.append(("implied vol, 476 contracts", label, bench(iv_batch)))
            rows.append(("greeks 61 x 476 = 29,036", label, bench(greeks_batch)))
            z = zero_gamma()
            rows.append(
                (
                    f"zero-gamma spot ({z['refinement_evaluations']} refinement steps)",
                    label,
                    bench(zero_gamma),
                )
            )
        finally:
            ob.HAS_CPP = saved

    width = max(len(r[0]) for r in rows)
    how = max(len(r[1]) for r in rows)
    for what, label, ms in rows:
        base = t_iv if what.startswith("implied") else t_grid_raw
        ratio = (
            f"{base / ms:8.1f}x"
            if "batch" in label and not what.startswith("zero")
            else ""
        )
        print(f"{what:<{width}}  {label:<{how}}  {ms:10.3f} ms  {ratio}")
    print(
        "ratios are against the scalar loop of the same workload "
        "(black_scholes_greeks for the grid, the lighter of the two scalars)"
    )


if __name__ == "__main__":
    main()
