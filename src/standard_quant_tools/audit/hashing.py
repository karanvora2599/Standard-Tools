"""Content-fingerprint hashing shared by every other module in this package:
`hash_payload` for JSON-serializable objects (decision records, chain-index
entries), `hash_dataframe` for OHLCV data provenance, `canonical_frame_hash`
for a frame whose fingerprint must not depend on the pandas version, and
`round_floats`, which an output passes through before its rounded hash is
taken."""

import hashlib
import json
import math
from typing import Any, List, Tuple

#: Significant digits an output's rounded hash keeps. Twelve because that is
#: what the reproducibility contract promises across native builds and
#: instruction-set paths: the AVX2+FMA and scalar reductions differ by a few
#: units in the last place (measured worst case 6e-15 relative on
#: rolling_beta), three orders of magnitude inside the twelfth digit.
ROUNDED_SIGNIFICANT_DIGITS = 12


def hash_dataframe(df: Any) -> str:
    """
    Content fingerprint of a DataFrame (columns + values + index), stable
    across runs.

    Covers the COLUMN NAMES and dtypes as well as the values. Hashing values
    alone (pd.util.hash_pandas_object is a per-row digest that never sees the
    column labels) meant two frames holding identical numbers under entirely
    different column names produced the same fingerprint -- e.g. a
    Close/Open frame and a Volume/Adj frame collided, which defeats the point
    of a provenance hash whose whole job is to tell different data apart.

    NOTE (format change): fingerprints produced here differ from those written
    by versions before this fix. Replaying a decision record captured by an
    older version will report a data_source mismatch even when the underlying
    data is unchanged. Only the `content_hash` values inside `data_sources`
    are affected -- the tamper-evident record chain is built by `hash_payload`
    (below), which is unchanged for all normal records, so existing audit
    trails still verify.
    """
    import numpy as np
    import pandas as pd

    hashed = pd.util.hash_pandas_object(df, index=True)
    values_digest = hashlib.sha256(np.asarray(hashed.values).tobytes()).hexdigest()
    # Column identity, in the frame's own column order (a reordering is a
    # genuinely different frame for provenance purposes).
    schema = json.dumps(
        (
            [[str(c), str(dtype)] for c, dtype in zip(df.columns, df.dtypes)]
            if hasattr(df, "columns")
            else []
        ),
        sort_keys=False,
    )
    combined = f"{schema}|{values_digest}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


#: The byte layout `canonical_frame_hash` feeds to SHA-256. It is part of
#: the hashed header, so a later layout can never reproduce a value this one
#: produced for different content.
CANONICAL_FRAME_HASH_LAYOUT = 1


def _tz_name(tz: Any) -> str:
    """A time zone's IANA name however the installed pandas represents it:
    zoneinfo carries `.key`, pytz `.zone`, and `datetime.timezone.utc`
    prints as "UTC"."""
    return str(getattr(tz, "key", None) or getattr(tz, "zone", None) or tz)


def _is_text(values: Any) -> bool:
    """True for a column of strings (missing values allowed) whatever its
    dtype: `object` under pandas 2, `str` under pandas 3, `string`, or an
    Arrow string type."""
    import pandas as pd
    from pandas.api import types as pdt

    dtype = values.dtype
    if isinstance(dtype, pd.StringDtype):
        return True
    if isinstance(dtype, pd.ArrowDtype):
        import pyarrow as pa

        arrow = dtype.pyarrow_dtype
        return bool(pa.types.is_string(arrow) or pa.types.is_large_string(arrow))
    if dtype == object:
        return pdt.infer_dtype(values, skipna=True) in ("string", "empty")
    return False


def _labelled(labels: List[str], codes: Any) -> List[bytes]:
    """Distinct labels as UTF-8 JSON in first-appearance order, then each
    row's position in that list (-1 for missing) as little-endian int64."""
    import numpy as np

    text = json.dumps(labels, ensure_ascii=False).encode("utf-8", "surrogatepass")
    return [text, np.ascontiguousarray(codes, dtype="<i8").tobytes()]


def _canonical_column(values: Any) -> Tuple[str, List[bytes]]:
    """(logical kind, canonical bytes) for one column.

    The kind names what the data is -- float64, int, bool, timestamp[ns,
    UTC], string, category<...> -- and never how the installed pandas
    spells the dtype, which is what `hash_dataframe` records and what
    pandas 3 changed for every text column. The bytes depend only on the
    values: one NaN bit pattern, datetimes as UTC nanoseconds, text as its
    distinct values plus a code per row, so `None` and `NaN` in a text
    column and `object`, `str` and `string` storage hash the same.
    """
    import numpy as np
    import pandas as pd
    from pandas.api import types as pdt

    dtype = values.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        kind, chunks = _canonical_column(pd.Series(values.cat.categories))
        ordered = ", ordered" if dtype.ordered else ""
        codes = np.ascontiguousarray(values.cat.codes.to_numpy(), dtype="<i8")
        return f"category<{kind}{ordered}>", chunks + [codes.tobytes()]
    if pdt.is_datetime64_any_dtype(dtype):
        tz = getattr(dtype, "tz", None)
        naive = values.dt.tz_convert(None) if tz is not None else values
        try:
            naive = naive.dt.as_unit("ns")
            unit = "ns"
        except (OverflowError, ValueError):
            # A value outside the nanosecond range (pandas 3 parses one
            # where pandas 2 could not): the column's own resolution.
            unit = str(np.datetime_data(naive.dtype)[0])
        raw = np.ascontiguousarray(naive.to_numpy().view("i8"), dtype="<i8")
        zone = f", {_tz_name(tz)}" if tz is not None else ""
        return f"timestamp[{unit}{zone}]", [raw.tobytes()]
    if pdt.is_timedelta64_dtype(dtype):
        raw = values.dt.as_unit("ns").to_numpy().view("i8")
        return "duration[ns]", [np.ascontiguousarray(raw, dtype="<i8").tobytes()]
    if pdt.is_bool_dtype(dtype):
        mask = values.isna().to_numpy(dtype=bool)
        filled = values.fillna(False) if mask.any() else values
        data = np.ascontiguousarray(filled.to_numpy(dtype=bool)).view("u1")
        return "bool", [data.tobytes(), np.packbits(mask).tobytes()]
    if pdt.is_integer_dtype(dtype):
        mask = values.isna().to_numpy(dtype=bool)
        filled = values.fillna(0) if mask.any() else values
        unsigned = pdt.is_unsigned_integer_dtype(dtype)
        data = np.ascontiguousarray(
            filled.to_numpy(dtype="uint64" if unsigned else "int64"),
            dtype="<u8" if unsigned else "<i8",
        )
        kind = "uint" if unsigned else "int"
        return kind, [data.tobytes(), np.packbits(mask).tobytes()]
    if pdt.is_float_dtype(dtype):
        width = np.dtype(getattr(dtype, "numpy_dtype", dtype)).itemsize
        bits = 32 if width == 4 else 64
        data = np.array(
            values.to_numpy(dtype=f"float{bits}", na_value=np.nan),
            dtype=f"<f{bits // 8}",
        )
        missing = np.isnan(data)
        if missing.any():
            # One positive quiet NaN, whatever sign or payload was stored.
            # Written as bits: the sign of `np.nan` itself is not promised.
            if bits == 64:
                data.view("<u8")[missing] = 0x7FF8000000000000
            else:
                data.view("<u4")[missing] = 0x7FC00000
        return f"float{bits}", [data.tobytes()]
    if _is_text(values):
        codes, uniques = pd.factorize(values, use_na_sentinel=True)
        return "string", _labelled([str(u) for u in uniques], codes)
    # Anything else (mixed objects, periods, intervals): each value's repr,
    # which for the scalars that reach here is the same under pandas 2 and
    # pandas 3. The dtype's own name is the kind only where it is not
    # `object`, whose spelling never changed.
    codes, uniques = pd.factorize(
        values.map(lambda value: None if value is None else repr(value))
    )
    kind = "object" if dtype == object else f"other<{dtype}>"
    return kind, _labelled([str(u) for u in uniques], codes)


def _column_digest(chunks: List[bytes]) -> bytes:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(len(chunk).to_bytes(8, "little"))
        digest.update(chunk)
    return digest.digest()


def canonical_frame_hash(df: Any) -> str:
    """
    Content fingerprint of a DataFrame that does not depend on the pandas,
    numpy or pyarrow version: 16 hex characters of SHA-256.

    `hash_dataframe` covers each column's `str(dtype)`, and pandas 3 prints
    a text column as `str` where pandas 2 prints `object`, and parses a
    date string to `datetime64[s]` where pandas 2 gives `datetime64[ns]`.
    The same file therefore fingerprints differently under the two, and a
    check built on it refuses unchanged data. This hash covers the same
    things -- row count, index, column names in order, each column's type
    and every value -- through a representation chosen to be the same under
    every pandas:

    - the header: the layout version, the row count, the index ("range"
      for the default 0..n-1, otherwise each level's name and kind) and
      each column's name and logical kind (`float64`, `float32`, `int`,
      `uint`, `bool`, `timestamp[ns]`, `timestamp[ns, <zone>]`,
      `duration[ns]`, `string`, `category<...>`, `object`);
    - per column, a SHA-256 of canonical bytes: floats as little-endian
      IEEE with one NaN pattern; datetimes converted to UTC nanoseconds;
      integers as int64 and booleans as bytes, each with a missing-value
      mask; text as its distinct values in UTF-8 plus an int64 code per
      row, so `None` and `NaN` hash alike and `object`, `str` and
      `string` storage hash alike;
    - the final SHA-256 over the header and the column digests in order.

    What it treats as the same frame: a text column stored as `object`,
    `str` or `string`; a datetime at `[ns]`, `[us]`, `[ms]` or `[s]` that
    holds the same instants; a nullable and a numpy integer or boolean
    column with the same values; an integer index equal to 0..n-1 and a
    RangeIndex. What it tells apart: any value (one ulp included), a
    missing value, a column name or position, a row order, float32 against
    float64, and a time zone.
    """
    import numpy as np
    import pandas as pd

    rows = int(len(df))
    digests: List[bytes] = []
    header: dict = {"layout": CANONICAL_FRAME_HASH_LAYOUT, "rows": rows}

    index = df.index
    default_index = index.nlevels == 1 and index.name is None
    if default_index and isinstance(index, pd.RangeIndex):
        default_index = index.start == 0 and index.step == 1
    elif default_index:
        default_index = pd.api.types.is_integer_dtype(index.dtype) and bool(
            np.array_equal(index.to_numpy(), np.arange(rows))
        )
    if default_index:
        header["index"] = "range"
    else:
        levels = []
        for position in range(index.nlevels):
            level = pd.Series(index.get_level_values(position))
            kind, chunks = _canonical_column(level)
            name = index.names[position]
            levels.append([None if name is None else str(name), kind])
            digests.append(_column_digest(chunks))
        header["index"] = levels

    columns = []
    for position, name in enumerate(df.columns):
        kind, chunks = _canonical_column(df.iloc[:, position])
        columns.append([str(name), kind])
        digests.append(_column_digest(chunks))
    header["columns"] = columns

    encoded = json.dumps(header, sort_keys=True, ensure_ascii=False).encode(
        "utf-8", "surrogatepass"
    )
    final = hashlib.sha256(len(encoded).to_bytes(8, "little"))
    final.update(encoded)
    for digest in digests:
        final.update(digest)
    return final.hexdigest()[:16]


def _canonical_default(obj: Any) -> Any:
    """
    Fallback encoder for objects `json` can't serialize natively.

    Plain `default=str` silently routed numpy arrays (and anything else with
    an abbreviating __str__) through a LOSSY repr: numpy truncates with '...'
    past ~1000 elements, so two different large arrays produced byte-identical
    canonical forms and therefore the same hash. Anything array-like is
    converted to its full element list here instead; genuinely opaque objects
    still fall back to str(), which is fine for the scalars (datetime, Path,
    Decimal) that actually reach this path in practice.
    """
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):  # numpy ndarray / scalar, pandas Series/Index
        return tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    if isinstance(obj, (bytes, bytearray)):
        return obj.hex()
    return str(obj)


def round_floats(obj: Any, digits: int = ROUNDED_SIGNIFICANT_DIGITS) -> Any:
    """
    `obj` with every float rounded to `digits` significant decimal digits,
    for a hash that survives a change in the last bits and nothing more.

    Exactly:

    - a finite, non-zero float becomes the double nearest its value
      correctly rounded to `digits` significant digits, ties to even --
      what `format(x, '.11e')` prints for twelve, read back. The rounding
      is decimal and relative, so 1234.56789012345 and 1.23456789012345e-9
      both keep twelve digits;
    - 0.0 and -0.0 both become 0.0: the sign of a zero is a last-bit
      difference, and JSON spells the two apart;
    - NaN, inf and -inf are kept. Their JSON tokens carry no low bits, so
      there is nothing to round, and an infinity keeps its sign;
    - int and bool are untouched, at any size: an integer is a count or an
      index, and rounding one would hide a real difference;
    - str, None and dict KEYS are untouched; dict values, list and tuple
      items are rounded at any depth (a tuple comes back as a list, which
      JSON spells the same way);
    - a numpy array or scalar is converted with `tolist()` first, as the
      exact hash's encoder converts it, then rounded;
    - anything else is returned as it is and hashed as before.

    What it cannot do: a value that is zero in exact arithmetic but comes
    out as rounding noise (1e-17 on one path, -3e-18 on another) agrees to
    no number of significant digits, so the rounded hash differs there. And
    two values a last bit apart that straddle a twelfth-digit rounding
    boundary round apart; the chance is about the size of their difference
    relative to the twelfth digit, measured at none in 7,231 rolling betas.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return obj
        if obj == 0.0:
            return 0.0
        return float(format(obj, f".{digits - 1}e"))
    if isinstance(obj, dict):
        return {key: round_floats(value, digits) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(value, digits) for value in obj]
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):  # numpy ndarray / scalar, pandas Series/Index
        return round_floats(tolist(), digits)
    return obj


def hash_payload(obj: Any) -> str:
    """
    Content fingerprint of a JSON-serializable object (dict/list/scalar).

    Output is unchanged for objects made only of native JSON types (which is
    every DecisionRecord / chain-index entry), so the tamper-evident record
    chain built on this function stays valid across this change -- only the
    previously-lossy non-JSON fallback path behaves differently.
    """
    canonical = json.dumps(obj, sort_keys=True, default=_canonical_default)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
