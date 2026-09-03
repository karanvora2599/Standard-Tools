"""
Bounds that belong to the SPEC, not to the code that applies them.

These live in a leaf module with no imports of their own because both ends
need them: `specs.py` puts them in the JSON schema, and `dataset/lags.py`
enforces them with an explanatory message. `specs.py` cannot import from
`dataset.lags` directly -- `dataset/__init__` imports `builder`, which
imports `specs` -- so a shared leaf is what breaks the cycle without either
side owning the other.

WHY THE SCHEMA MATTERS AS MUCH AS THE CHECK. A bound enforced only in a
`field_validator` is invisible in `model_json_schema()`, which is the
document an LLM actually reads before calling a tool. The adversarial
suite found exactly that: it synthesized `lags=[61]` because nothing in the
schema said 60 was the ceiling, and a caller reading the same schema would
have made the same mistake. The value is declared here, applied as an item
constraint there, and re-checked with a real explanation in the validator.
"""

#: Deepest single lag. A quarter of daily bars: past this the row is
#: describing a different regime rather than the recent path, and the
#: warm-up it costs is charged to every entity in the panel.
MAX_LAG = 60

#: How many lags one feature may request. The panel grows by this per
#: feature, so it is bounded per-feature rather than only in total.
MAX_LAGS_PER_FEATURE = 20

#: Ceiling on the whole expanded panel. An agent composing 30 features at
#: 20 lags each would otherwise ask for a 630-column panel and discover the
#: cost as a memory error rather than as a refusal.
MAX_EXPANDED_COLUMNS = 400

__all__ = ["MAX_EXPANDED_COLUMNS", "MAX_LAG", "MAX_LAGS_PER_FEATURE"]
