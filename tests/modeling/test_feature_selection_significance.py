"""
`select_features` selects (the CHANGELOG entry of 2026-10-04).

Its IC floor defaults to 0.0, so a call with no arguments kept every
feature that was not a duplicate: on the live panel, all eight, while the
significance screen called one of them significant. The floor the screen
recommended for it, `honest_floor`, kept none of the eight and was measured
on every date, holdout included.

The selection now tests each cluster representative against a permutation
null on its selection window, before the holdout is read. The default null,
`entity_shuffle`, hands each entity's whole feature series to another
entity -- one permutation for every date -- so each series keeps its serial
correlation and the feature-entity link, a static tilt included, is broken.
The circular shift the screen uses keeps that tilt in its null.

What these tests hold:

- the fast entity-shuffle draw is the brute-force reassignment, draw for
  draw, on complete, ragged and tied panels, for both correlations;
- it has power on a static tilt where the circular shift has none, and
  stays near its nominal size on a true null that carries per-entity levels;
- the gate keeps a planted signal and drops noise as 'insignificant' with
  its p-value, reads the selection window only, tests representatives that
  cleared the floor and nothing else, and caps after it;
- `significance='none'` returns what the function returned before the gate,
  to the bit, against a verbatim copy of that code;
- the screen's default numbers do not move: the dates are now factorized
  once per test rather than once per draw, which is the same arithmetic;
- the descriptions no longer offer `honest_floor` as a selection floor and
  name the null each tool actually draws.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as SchemaError

import standard_quant_tools.modeling.analysis.feature_selection as selection_module
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.feature_models import (
    PermutationTestInput,
    ScreenFeatureSignificanceInput,
    SelectFeaturesInput,
)
from standard_quant_tools.modeling.agent.feature_tools import (
    FEATURE_TOOL_DEFS,
    run_feature_permutation_test,
    screen_feature_significance,
    select_features,
)
from standard_quant_tools.modeling.agent.models import RegisterExternalPanelInput
from standard_quant_tools.modeling.agent.tools import register_external_panel
from standard_quant_tools.modeling.analysis.feature_report import (
    feature_predictive_stats,
    redundancy_report,
)
from standard_quant_tools.modeling.analysis.feature_selection import (
    _selection_cutoff,
    _signed_rank_ic,
    _window,
)
from standard_quant_tools.modeling.analysis.feature_selection import (
    select_features as select_features_on,
)
from standard_quant_tools.modeling.analysis.feature_stability import (
    _circular_shift_null,
    _entity_shuffle_null,
    _null_distribution,
    estimate_draw_seconds,
    permutation_test_ic,
)
from standard_quant_tools.modeling.validation.metrics import cross_sectional_ic

# ── planted panels ───────────────────────────────────────────────────────


def _panel(
    *,
    n_dates: int = 120,
    n_entities: int = 12,
    seed: int = 0,
    ragged: bool = False,
    ties: bool = False,
    signal: float = 0.1,
) -> pd.DataFrame:
    """A feature `x` with a per-entity level plus noise, and a target that
    loads on it by `signal`. `ragged` drops one row in ten; `ties` rounds
    the feature to whole numbers, so ranks tie."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    level = rng.normal(size=n_entities)
    rows = []
    for date in dates:
        for j in range(n_entities):
            if ragged and rng.random() < 0.1:
                continue
            x = level[j] + rng.normal()
            if ties:
                x = float(np.round(x))
            rows.append(
                {
                    "date": date,
                    "entity": f"E{j:02d}",
                    "x": x,
                    "target": signal * x + rng.normal(),
                }
            )
    return pd.DataFrame(rows)


def _brute_force_entity_null(frame, feature, n_permutations, method, seed):
    """The definition, written out: pivot to dates x entities (entities by
    name), hand entity e the series of entity order[e], and take the mean
    per-date IC over whatever rows survive."""
    frame = frame.dropna(subset=["date", feature, "target"])
    entities = sorted(frame["entity"].unique())
    dates = sorted(frame["date"].unique())
    grid = frame.pivot(index="date", columns="entity", values=feature)
    feature_grid = grid.reindex(index=dates, columns=entities).to_numpy()
    target_grid = (
        frame.pivot(index="date", columns="entity", values="target")
        .reindex(index=dates, columns=entities)
        .to_numpy()
    )
    rows = np.broadcast_to(np.arange(len(dates))[:, None], feature_grid.shape)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_permutations):
        order = rng.permutation(len(entities))
        shuffled = feature_grid[:, order]
        usable = ~np.isnan(shuffled) & ~np.isnan(target_grid)
        series = cross_sectional_ic(
            target_grid[usable], shuffled[usable], rows[usable], method=method
        )
        draws.append(series.mean() if len(series) else np.nan)
    return np.array(draws)


def _fast_entity_null(frame, feature, n_permutations, method, seed):
    frame = frame.dropna(subset=["date", feature, "target"])
    return _entity_shuffle_null(
        frame["target"].to_numpy(dtype=float),
        frame[feature].to_numpy(dtype=float),
        frame["date"].to_numpy(),
        frame["entity"].to_numpy(),
        n_permutations,
        method,
        seed,
    )[0]


def _static_tilt(seed: int, *, signal: float, n_dates: int = 150, n_entities: int = 20):
    """Feature and target both carry a per-entity level; the feature is a
    persistent AR(1) around its level. With `signal` > 0 the target's level
    IS the feature's level, which is a static cross-sectional signal; at 0
    the two levels are independent, which is a true null whose observed IC
    still carries the chance alignment of the levels."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    feature_level = rng.normal(size=n_entities)
    target_level = (
        signal * feature_level if signal else 0.3 * rng.normal(size=n_entities)
    )
    frames = []
    for j in range(n_entities):
        noise = np.empty(n_dates)
        noise[0] = rng.normal()
        for t in range(1, n_dates):
            noise[t] = 0.95 * noise[t - 1] + np.sqrt(1 - 0.95**2) * rng.normal()
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "entity": f"E{j:02d}",
                    "x": feature_level[j] + 0.5 * noise,
                    "target": target_level[j] + rng.normal(size=n_dates),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _signal_among_noise(
    *, n_noise: int = 6, n_dates: int = 200, n_entities: int = 20, seed: int = 3
):
    """`signal` is 0.3 * target + noise; the `noise_*` columns are
    independent of the target everywhere."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    n = n_dates * n_entities
    target = rng.normal(size=n)
    data = {
        "date": np.repeat(dates.to_numpy(), n_entities),
        "entity": np.tile([f"E{j:02d}" for j in range(n_entities)], n_dates),
        "target": target,
        "signal": 0.3 * target + rng.normal(size=n),
    }
    for k in range(n_noise):
        data[f"noise_{k}"] = rng.normal(size=n)
    return pd.DataFrame(data), ["signal"] + [f"noise_{k}" for k in range(n_noise)]


def _register(frame: pd.DataFrame, tmp_path, name: str = "panel") -> str:
    path = tmp_path / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return register_external_panel(
        RegisterExternalPanelInput(path=str(path), horizon=5)
    ).dataset_id


# ── the entity-shuffle null ──────────────────────────────────────────────


class TestTheEntityShuffleDrawIsTheReassignment:
    """The fast draw sums one entities-by-entities matrix under the
    permutation on complete dates and recomputes only the dates where an
    entity is missing. These pin it to the definition, draw for draw."""

    @pytest.mark.parametrize("method", ["spearman", "pearson"])
    @pytest.mark.parametrize(
        "shape",
        [{}, {"ragged": True}, {"ties": True}, {"ties": True, "ragged": True}],
        ids=["complete", "ragged", "ties", "ties-ragged"],
    )
    def test_it_matches_brute_force(self, method, shape):
        frame = _panel(**shape)
        fast = _fast_entity_null(frame, "x", 100, method, 7)
        slow = _brute_force_entity_null(frame, "x", 100, method, 7)
        np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-12)

    def test_a_constant_cross_section_counts_as_zero(self):
        """`cross_sectional_ic` reports a date with no spread as 0.0 and
        counts it; so must the null, or its mean would be over a different
        set of dates than the observed IC's."""
        frame = _panel()
        first = frame["date"] == frame["date"].min()
        frame.loc[first, "x"] = 1.0
        fast = _fast_entity_null(frame, "x", 50, "spearman", 1)
        slow = _brute_force_entity_null(frame, "x", 50, "spearman", 1)
        np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-12)

    def test_the_row_order_does_not_change_the_draws(self):
        """Entities are ordered by name before the draw, so the same seed
        hands the same series to the same entity whatever order the panel's
        rows arrive in."""
        frame = _panel(ragged=True)
        shuffled = frame.sample(frac=1.0, random_state=4).reset_index(drop=True)
        a = permutation_test_ic(frame, "x", n_permutations=60, null="entity_shuffle")
        b = permutation_test_ic(shuffled, "x", n_permutations=60, null="entity_shuffle")
        assert a["p_value"] == b["p_value"]
        assert a["null_mean"] == pytest.approx(b["null_mean"], abs=1e-15)

    def test_the_same_seed_gives_the_same_p_value(self):
        frame = _panel()
        runs = [
            permutation_test_ic(
                frame, "x", n_permutations=80, null="entity_shuffle", random_seed=11
            )
            for _ in range(2)
        ]
        assert runs[0] == runs[1]

    def test_a_repeated_date_entity_pair_is_refused_by_name(self):
        frame = _panel()
        doubled = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        with pytest.raises(ValidationError, match="one row per \\(date, entity\\)"):
            permutation_test_ic(doubled, "x", n_permutations=20, null="entity_shuffle")

    def test_the_observed_assignment_ties_itself_exactly(self):
        """Three entities have six assignments, so about one draw in six is
        the observed one. Each must count as at least as extreme as the
        observed IC: compared with the IC `cross_sectional_ic` reports, they
        fell 1e-16 short and p came out at 0.005 instead of about 1/6."""
        frame = _panel(n_entities=3, n_dates=200, signal=0.5)
        result = permutation_test_ic(
            frame, "x", n_permutations=200, null="entity_shuffle"
        )
        assert result["p_value"] > 1 / 7
        # The reported IC is still the one every other tool reports.
        codes = pd.factorize(frame["date"], sort=True)[0]
        assert result["observed_ic"] == float(
            cross_sectional_ic(
                frame["target"].to_numpy(dtype=float),
                frame["x"].to_numpy(dtype=float),
                codes,
                method="spearman",
            ).mean()
        )

    def test_an_unknown_null_names_all_three(self):
        with pytest.raises(ValidationError) as excinfo:
            permutation_test_ic(_panel(), "x", n_permutations=20, null="bootstrap")
        message = str(excinfo.value)
        for name in ("circular_shift", "within_date", "entity_shuffle"):
            assert name in message


class TestWhatTheNullsKeep:
    """A static tilt -- the feature ranks the entities the same way on
    every date, and the target's entity levels follow it -- is a real
    cross-sectional signal. Rolling a series in time leaves the entity's
    level where it was, so the circular-shift null carries the signal in
    every draw and cannot see it; reassigning series between entities
    breaks it."""

    def test_the_entity_shuffle_sees_a_static_signal_the_circular_shift_cannot(self):
        frame = _static_tilt(0, signal=0.3)
        entity = permutation_test_ic(
            frame, "x", n_permutations=200, null="entity_shuffle"
        )
        circular = permutation_test_ic(
            frame, "x", n_permutations=200, null="circular_shift"
        )
        assert entity["observed_ic"] > 0.05
        assert entity["p_value"] < 0.05
        assert abs(entity["null_mean"]) < 0.25 * entity["observed_ic"]
        # The circular-shift null is centred near the observed IC itself:
        # the tilt survives every roll.
        assert circular["null_mean"] > 0.5 * circular["observed_ic"]
        assert circular["p_value"] > 0.05

    def test_the_entity_shuffle_holds_its_size_on_a_true_null_with_levels(self):
        """Independent per-entity levels in feature and target: the
        observed IC carries their chance alignment, and the entity shuffle
        draws exactly that variation. 20 true nulls at alpha 0.05: about
        one rejection expected."""
        rejected = sum(
            permutation_test_ic(
                _static_tilt(seed, signal=0.0),
                "x",
                n_permutations=100,
                null="entity_shuffle",
                random_seed=seed,
            )["significant_at_05"]
            for seed in range(20)
        )
        assert rejected <= 4, f"{rejected}/20 true nulls rejected"


class TestTheScreensNumbersDoNotMove:
    """`permutation_test_ic` now hands the dates to every IC pass as their
    sorted codes, factorized once. A zoned date column arrives as an object
    array of Timestamps, and factorizing it again on every draw was most of
    the cost of a draw (17 of 18 ms on the live panel). Codes in sorted
    order factorize to themselves, so the draws are the same numbers; these
    compare against the null functions called on the raw dates, which is
    what every draw used to do."""

    @staticmethod
    def _zoned():
        frame = _panel(ragged=True)
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize("UTC")
        return frame

    def test_circular_shift_draws_are_unchanged(self):
        frame = self._zoned()
        result = permutation_test_ic(frame, "x", n_permutations=50, random_seed=3)
        usable = frame.dropna(subset=["date", "x", "target"])
        target = usable["target"].to_numpy(dtype=float)
        values = usable["x"].to_numpy(dtype=float)
        raw_dates = usable["date"].to_numpy()
        draws = _circular_shift_null(
            target,
            values,
            raw_dates,
            usable["entity"].to_numpy(),
            50,
            "spearman",
            3,
        )
        observed = float(
            cross_sectional_ic(target, values, raw_dates, method="spearman").mean()
        )
        assert result["observed_ic"] == observed
        assert result["null_mean"] == float(np.mean(draws))
        expected_p = (np.sum(np.abs(draws) >= abs(observed)) + 1) / (draws.size + 1)
        assert result["p_value"] == float(expected_p)

    def test_within_date_draws_are_unchanged(self):
        frame = self._zoned()
        result = permutation_test_ic(
            frame, "x", n_permutations=50, random_seed=3, null="within_date"
        )
        usable = frame.dropna(subset=["date", "x", "target"])
        draws = _null_distribution(
            usable["target"].to_numpy(dtype=float),
            usable["x"].to_numpy(dtype=float),
            usable["date"].to_numpy(),
            50,
            "spearman",
            3,
        )
        assert result["null_mean"] == float(np.mean(draws))
        assert result["null_std"] == float(np.std(draws))


# ── the gate ─────────────────────────────────────────────────────────────


class TestTheGateSelects:
    def test_it_keeps_the_signal_and_drops_the_noise_with_its_p_value(self):
        panel, features = _signal_among_noise()
        result = select_features_on(panel, features)

        assert "signal" in result["selected"]
        assert result["significance"] == {
            "null": "entity_shuffle",
            "alpha": 0.05,
            "n_permutations": 200,
            "random_seed": 0,
            "n_tested": len(features),
            "n_passed": len(result["selected"]),
            "correction": "none",
            "n_passed_uncorrected": len(result["selected"]),
        }
        assert set(result["selection_p_value"]) == set(features)
        assert result["selection_p_value"]["signal"] < 0.05
        insignificant = [d for d in result["dropped"] if d["reason"] == "insignificant"]
        assert len(insignificant) == len(features) - len(result["selected"])
        for drop in insignificant:
            assert drop["p_value"] == result["selection_p_value"][drop["feature"]]
            assert drop["p_value"] >= 0.05
            assert drop["duplicate_of"] is None
            assert "against the entity-shuffle null at alpha 0.05" in drop["detail"]
            assert f"p={drop['p_value']:.3f}" in drop["detail"]
        # The holdout is measured for what was kept, and only that.
        assert set(result["holdout_ic"]) == set(result["selected"])

    def test_the_warning_counts_what_noise_alone_would_pass(self):
        panel, features = _signal_among_noise()
        result = select_features_on(panel, features)
        sentence = next(w for w in result["warnings"] if "cleared p < 0.05" in w)
        n = len(features)
        assert sentence.startswith(f"{len(result['selected'])} of {n} features")
        assert f"about {0.05 * n:.1f} of {n} clear from noise alone" in sentence
        assert "not corrected" in sentence

    def test_it_reads_the_selection_window_only(self):
        """The test is fixed before the holdout is read. Replacing every
        holdout target with garbage changes `holdout_ic` and nothing the
        test decided."""
        panel, features = _signal_among_noise()
        _dates, cutoff = _selection_cutoff(panel, None, 0.3)
        tampered = panel.copy()
        later = pd.to_datetime(tampered["date"]) > cutoff
        tampered.loc[later, "target"] = np.random.default_rng(99).normal(
            size=int(later.sum())
        )
        clean = select_features_on(panel, features)
        dirty = select_features_on(tampered, features)
        assert clean["selection_p_value"] == dirty["selection_p_value"]
        assert clean["selected"] == dirty["selected"]
        assert clean["holdout_ic"] != dirty["holdout_ic"]

    def test_only_representatives_that_cleared_the_floor_are_tested(self):
        """Redundancy, then the floor, then the test: a duplicate is
        dropped as redundant and a feature under the floor as weak, and
        neither is tested."""
        panel, features = _signal_among_noise()
        panel["signal_copy"] = panel["signal"]
        result = select_features_on(
            panel, features + ["signal_copy"], min_abs_rank_ic=0.02
        )
        reasons = {d["feature"]: d["reason"] for d in result["dropped"]}
        weak = [f for f, r in reasons.items() if r == "weak"]
        redundant = [f for f, r in reasons.items() if r == "redundant"]
        assert len({"signal", "signal_copy"} & set(redundant)) == 1
        assert weak, "a 0.02 floor drops some of the noise before the test"
        assert not set(result["selection_p_value"]) & (set(weak) | set(redundant))
        assert result["significance"]["n_tested"] == len(result["selection_p_value"])

    def test_the_cap_comes_after_the_test(self):
        panel, features = _signal_among_noise()
        panel["signal_2"] = panel["target"] * 0.3 + np.random.default_rng(5).normal(
            size=len(panel)
        )
        result = select_features_on(panel, features + ["signal_2"], max_features=1)
        assert len(result["selected"]) == 1
        capped = [d for d in result["dropped"] if d["reason"] == "capped"]
        # Both signals passed the test; one of them is capped, not tested out.
        assert len(capped) == 1
        assert result["selection_p_value"][capped[0]["feature"]] < 0.05

    def test_nothing_passing_is_said_in_so_many_words(self):
        panel, features = _signal_among_noise()
        noise = [f for f in features if f.startswith("noise")]
        result = select_features_on(panel, noise, alpha=0.001)
        assert result["selected"] == []
        assert result["holdout_ic"] == {}
        sentence = next(w for w in result["warnings"] if "No feature cleared" in w)
        assert "p < 0.001 against the entity-shuffle null" in sentence
        assert "the same series assigned to random entities" in sentence

    def test_an_untestable_feature_is_dropped_with_no_p_value(self):
        """An infinity is refused by the permutation test, which cannot put
        it on either side of the null. The selection drops that feature as
        insignificant with the refusal as its detail, and still answers for
        the rest."""
        panel, features = _signal_among_noise(n_noise=2)
        panel.loc[0, "noise_0"] = np.inf
        result = select_features_on(panel, features)
        drop = next(d for d in result["dropped"] if d["feature"] == "noise_0")
        assert drop["reason"] == "insignificant"
        assert drop["p_value"] is None
        assert drop["detail"].startswith("not testable on the selection window")
        assert result["selection_p_value"]["noise_0"] is None
        assert "signal" in result["selected"]

    def test_three_entities_cannot_reach_alpha_and_the_result_says_so(self):
        """Three entities have six assignments: one draw in six is the
        observed one, so no p-value falls much below 1/6."""
        panel = _panel(n_entities=3, n_dates=200, signal=0.5)
        result = select_features_on(panel, ["x"])
        assert result["selected"] == []
        assert result["selection_p_value"]["x"] > 0.1
        assert any("only 6 ways" in w for w in result["warnings"])

    def test_a_panel_with_enough_entities_carries_no_such_warning(self):
        panel, features = _signal_among_noise()
        result = select_features_on(panel, features)
        assert not any("ways to assign" in w for w in result["warnings"])

    def test_the_circular_shift_can_be_asked_for(self):
        panel, features = _signal_among_noise(n_noise=2)
        result = select_features_on(panel, features, significance="circular_shift")
        assert result["significance"]["null"] == "circular_shift"
        assert "signal" in result["selected"]
        assert any("circular-shift null" in w for w in result["warnings"])

    def test_the_budget_is_refused_before_the_first_draw(self, monkeypatch):
        panel, features = _signal_among_noise()

        def _never(*args, **kwargs):  # pragma: no cover - the point is it is not called
            raise AssertionError("a draw was made before the budget was checked")

        monkeypatch.setattr(selection_module, "permutation_test_ic", _never)
        with pytest.raises(ValidationError) as excinfo:
            select_features_on(panel, features, n_permutations=1000, max_draws=5000)
        message = str(excinfo.value)
        assert f"{len(features) * 1000:,} permutation draws" in message
        assert f"{len(features)} features x 1000 permutations" in message
        assert "entity-shuffle null" in message
        assert "significance='none'" in message
        assert f"max_draws={len(features) * 1000}" in message

    @pytest.mark.parametrize(
        "kwargs, field",
        [
            ({"significance": "bootstrap"}, "significance"),
            ({"alpha": 0.0}, "alpha"),
            ({"alpha": 1.0}, "alpha"),
            ({"n_permutations": 10}, "n_permutations"),
        ],
    )
    def test_a_direct_caller_is_refused_by_name(self, kwargs, field):
        panel, features = _signal_among_noise(n_noise=1)
        with pytest.raises(ValidationError, match=field):
            select_features_on(panel, features, **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"significance": "bootstrap"},
            {"alpha": 0.0},
            {"alpha": 1.0},
            {"n_permutations": 10},
            {"n_permutations": 6000},
            {"random_seed": -1},
            {"max_draws": 0},
        ],
    )
    def test_the_schema_refuses_the_same(self, kwargs):
        with pytest.raises(SchemaError):
            SelectFeaturesInput(dataset_id="ds_x", **kwargs)


# ── significance='none' is the old function ──────────────────────────────


def _select_features_before_the_gate(
    panel,
    feature_ids,
    *,
    cluster_threshold=0.9,
    min_abs_rank_ic=0.0,
    max_features=0,
    selection_end=None,
    holdout_fraction=0.3,
):
    """`select_features` as it was before the significance test, verbatim
    in what it computed (redundancy, floor, cap, holdout), with the one
    change made separately on the same day: a single-member cluster's
    `max_abs_correlation` is None rather than 1.0."""

    def _abs(stats, feature):
        value = (stats.get(feature) or {}).get("rank_ic_mean")
        return abs(float(value)) if value is not None and np.isfinite(value) else 0.0

    feature_ids = list(feature_ids)
    dates, cutoff = _selection_cutoff(panel, selection_end, holdout_fraction)
    if cutoff is None:
        selection_panel, holdout_panel = panel, panel.iloc[0:0]
    else:
        date_values = pd.to_datetime(panel["date"])
        selection_panel = panel[date_values <= cutoff]
        holdout_panel = panel[date_values > cutoff]
    predictive = feature_predictive_stats(selection_panel, feature_ids)
    redundancy = redundancy_report(
        selection_panel, feature_ids, cluster_threshold=cluster_threshold
    )
    correlation = redundancy["correlation"]
    clusters = []
    for members in redundancy["clusters"]:
        members = sorted(members)
        keeper = sorted(members, key=lambda f: (-_abs(predictive, f), f))[0]
        pairs = [
            abs(correlation.get(a, {}).get(b, 0.0))
            for a in members
            for b in members
            if a != b
        ]
        clusters.append(
            {
                "members": members,
                "representative": keeper,
                "max_abs_correlation": max(pairs) if pairs else None,
                "size": len(members),
            }
        )
    clusters.sort(key=lambda record: (-record["size"], record["representative"]))
    dropped, survivors = [], []
    for cluster in clusters:
        keeper = cluster["representative"]
        survivors.append(keeper)
        for member in cluster["members"]:
            if member != keeper:
                dropped.append(
                    {
                        "feature": member,
                        "reason": "redundant",
                        "duplicate_of": keeper,
                        "detail": (
                            f"same signal as {keeper!r} at "
                            f"|rho| >= {cluster_threshold:.2f}"
                        ),
                    }
                )
    kept = []
    for feature in survivors:
        strength = _abs(predictive, feature)
        if strength < min_abs_rank_ic:
            dropped.append(
                {
                    "feature": feature,
                    "reason": "weak",
                    "duplicate_of": None,
                    "detail": (
                        f"|rank IC| {strength:.4f} below the "
                        f"{min_abs_rank_ic:.4f} floor"
                    ),
                }
            )
        else:
            kept.append(feature)
    kept.sort(key=lambda f: (-_abs(predictive, f), f))
    if max_features and len(kept) > max_features:
        for feature in kept[max_features:]:
            dropped.append(
                {
                    "feature": feature,
                    "reason": "capped",
                    "duplicate_of": None,
                    "detail": (
                        f"ranked {kept.index(feature) + 1} by |rank IC|, past "
                        f"the max_features={max_features} cap"
                    ),
                }
            )
        kept = kept[:max_features]
    holdout_ic, holdout_window = {}, None
    if cutoff is not None:
        holdout_window = _window(dates[dates > cutoff])
        if kept:
            holdout_stats = feature_predictive_stats(holdout_panel, kept)
            holdout_ic = {f: _signed_rank_ic(holdout_stats, f) for f in kept}
    return {
        "selected": kept,
        "dropped": sorted(dropped, key=lambda d: d["feature"]),
        "n_clusters": len(clusters),
        "clusters": clusters,
        "holdout_window": holdout_window,
        "selection_ic": {f: _signed_rank_ic(predictive, f) for f in feature_ids},
        "holdout_ic": holdout_ic,
        "vif": redundancy["vif"],
        "condition_number": redundancy["condition_number"],
        "correlation": redundancy["correlation"],
    }


class TestNoneIsTheOldSelection:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"holdout_fraction": 0.0},
            {"min_abs_rank_ic": 0.02},
            {"max_features": 3},
            {"cluster_threshold": 0.0},
        ],
        ids=["defaults", "no-holdout", "floor", "cap", "one-cluster"],
    )
    def test_it_is_the_old_answer_to_the_bit(self, kwargs):
        panel, features = _signal_among_noise()
        panel["signal_copy"] = panel["signal"]
        features = features + ["signal_copy"]
        expected = _select_features_before_the_gate(panel, features, **kwargs)
        actual = select_features_on(panel, features, significance="none", **kwargs)
        for key, value in expected.items():
            assert actual[key] == value, key
        assert actual["selection_p_value"] == {}
        assert actual["significance"] is None

    def test_it_says_no_test_was_applied(self):
        panel, features = _signal_among_noise(n_noise=2)
        result = select_features_on(
            panel, features, significance="none", min_abs_rank_ic=0.01
        )
        assert (
            "No significance test was applied: every feature that was not "
            "redundant and cleared min_abs_rank_ic=0.01 was kept."
        ) in result["warnings"]

    def test_the_default_does_not_carry_that_sentence(self):
        panel, features = _signal_among_noise(n_noise=2)
        result = select_features_on(panel, features)
        assert not any("No significance test" in w for w in result["warnings"])


# ── the tools ────────────────────────────────────────────────────────────


class TestTheToolsCarryIt:
    def test_select_features_returns_the_test(self, tmp_path):
        panel, _features = _signal_among_noise(n_noise=3)
        dataset_id = _register(panel, tmp_path)
        result = select_features(SelectFeaturesInput(dataset_id=dataset_id))
        assert result.significance is not None
        assert result.significance.null == "entity_shuffle"
        assert result.significance.n_passed == len(result.selected)
        assert "signal" in result.selected
        assert set(result.selection_p_value) == {"signal"} | {
            d.feature for d in result.dropped if d.reason == "insignificant"
        }
        for drop in result.dropped:
            if drop.reason == "insignificant":
                assert drop.p_value == result.selection_p_value[drop.feature]

    def test_none_through_the_tool(self, tmp_path):
        panel, features = _signal_among_noise(n_noise=3)
        dataset_id = _register(panel, tmp_path)
        result = select_features(
            SelectFeaturesInput(dataset_id=dataset_id, significance="none")
        )
        assert result.significance is None
        assert result.selection_p_value == {}
        assert sorted(result.selected) == sorted(features)
        assert all(d.p_value is None for d in result.dropped)

    def test_the_permutation_test_takes_the_entity_shuffle(self, tmp_path):
        panel, _features = _signal_among_noise(n_noise=1)
        dataset_id = _register(panel, tmp_path)
        result = run_feature_permutation_test(
            PermutationTestInput(
                dataset_id=dataset_id, feature="signal", null="entity_shuffle"
            )
        )
        assert result.null == "entity_shuffle"
        assert result.p_value < 0.05
        assert result.null_mean is not None

    def test_the_screen_takes_it_and_reports_each_nulls_centre(self, tmp_path):
        panel, features = _signal_among_noise(n_noise=2)
        dataset_id = _register(panel, tmp_path)
        for null in ("circular_shift", "entity_shuffle"):
            result = screen_feature_significance(
                ScreenFeatureSignificanceInput(
                    dataset_id=dataset_id, n_permutations=50, null=null
                )
            )
            assert result.null == null
            assert all(row.null_mean is not None for row in result.features)
            sentence = next(w for w in result.warnings if "honest_floor=" in w)
            assert "not a min_abs_rank_ic for select_features" in sentence

    def test_the_screen_refusal_prices_the_null_it_was_asked_for(self, tmp_path):
        """It quoted one 1.6 ms for every null and every panel; the circular
        shift measured 8-11x that on the live panel's zoned dates."""
        import re

        panel, features = _signal_among_noise(n_noise=2)
        dataset_id = _register(panel, tmp_path)
        for null in ("circular_shift", "entity_shuffle", "within_date"):
            with pytest.raises(ValidationError) as excinfo:
                screen_feature_significance(
                    ScreenFeatureSignificanceInput(
                        dataset_id=dataset_id,
                        n_permutations=1000,
                        max_draws=100,
                        null=null,
                    )
                )
            message = str(excinfo.value)
            stated = re.search(
                r"At about ([0-9.]+) ms a draw under the '(\w+)' null on this "
                r"panel's ([0-9,]+) rows",
                message,
            )
            assert stated, message
            assert stated.group(2) == null
            rows = int(stated.group(3).replace(",", ""))
            assert stated.group(1) == f"{estimate_draw_seconds(null, rows) * 1e3:.1f}"

    def test_the_draw_estimates_rank_the_nulls_as_measured(self):
        rows = 31_680
        circular = estimate_draw_seconds("circular_shift", rows)
        entity = estimate_draw_seconds("entity_shuffle", rows)
        within = estimate_draw_seconds("within_date", rows)
        assert within < entity < circular
        # Measured 1.3-1.4 ms a draw on a panel this size.
        assert 0.0010 < circular < 0.0025


class TestTheDescriptions:
    @staticmethod
    def _by_name():
        return {name: text for name, text, _ in FEATURE_TOOL_DEFS}

    def test_no_description_offers_honest_floor_as_a_selection_floor(self):
        for name, text in self._by_name().items():
            assert "defensible floor for select_features" not in text, name
            assert "the IC floor this panel supports" not in text.lower(), name
        screen = self._by_name()["screen_feature_significance"]
        assert "not a min_abs_rank_ic for select_features" in screen

    def test_the_permutation_test_names_the_null_it_draws(self):
        text = self._by_name()["run_feature_permutation_test"]
        assert "Shuffles the feature within each date" not in text
        assert "null='circular_shift'" in text
        assert "null='entity_shuffle'" in text
        assert "null_mean" in text

    def test_select_features_promises_what_it_does(self):
        text = self._by_name()["select_features"]
        assert "drop what falls below an IC floor" not in text
        assert "significance='entity_shuffle'" in text
        assert "'insignificant'" in text
        assert "'none'" in text

    def test_the_single_feature_tools_point_at_their_screens(self):
        names = self._by_name()
        assert "screen_feature_stability" in names["get_feature_drift"]
        assert "screen_feature_stability" in names["get_feature_regime_stability"]
        assert "screen_feature_significance" in names["run_feature_permutation_test"]
        assert "run_feature_permutation_test" in names["screen_feature_significance"]
        assert "analyze_features" in names["get_feature_ic_decay"]
        assert "report.leakage" in names["get_feature_ic_decay"]
