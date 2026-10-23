"""
ISO 4217 currency codes, read strictly enough that a wrong one is refused
with the code that was meant.

A position's currency is a label for the numbers beside it: nothing here
fetches a rate, and the arithmetic never reads the code except to look up
the rate the caller supplied for it. What the label CAN do wrong is name the
wrong unit -- and the commonest way is not a typo. London prices arrive in
pence ('GBp', 'GBX'), Johannesburg's in cents ('ZAc') and Tel Aviv's in
agorot ('ILA'); upper-cased, 'GBp' becomes 'GBP' and a holding is booked at
a hundred times its value. Those spellings are refused before anything is
upper-cased, with the division that fixes them.

The list is static and kept here, so a code is valid or not regardless of
what is installed: ISO 4217's List One as it stood on 2026-10-22, less the
two codes that name no unit a price can be quoted in (XXX, "no currency",
and XTS, reserved for testing). Fund codes (USN, CLF, MXV, ...) and the
precious metals are kept: inflation-indexed bonds are quoted in CLF and
MXV units, and an account can be kept in ounces of gold. A withdrawn code
is refused naming its successor -- 'HRK' became 'EUR' on 2023-01-01,
'BGN' on 2026-01-01, 'ANG' became 'XCG' in 2025 and 'ZWL' became 'ZWG' in
2024 -- and any other unknown code with the nearest valid one.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Tuple

from standard_quant_tools.error import ValidationError

__all__ = [
    "ACTIVE_CURRENCIES",
    "WITHDRAWN_CURRENCIES",
    "currency_code",
    "nearest_currency",
]

#: ISO 4217 List One alphabetic codes in use on 2026-10-22, without XXX
#: (no currency) and XTS (testing).
ACTIVE_CURRENCIES: FrozenSet[str] = frozenset("""
    AED AFN ALL AMD AOA ARS AUD AWG AZN BAM BBD BDT BHD BIF BMD BND BOB BOV
    BRL BSD BTN BWP BYN BZD CAD CDF CHE CHF CHW CLF CLP CNY COP COU CRC CUC
    CUP CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD
    GNF GTQ GYD HKD HNL HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY KES KGS
    KHR KMF KPW KRW KWD KYD KZT LAK LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK
    MNT MOP MRU MUR MVR MWK MXN MXV MYR MZN NAD NGN NIO NOK NPR NZD OMR PAB
    PEN PGK PHP PKR PLN PYG QAR RON RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP
    SLE SLL SOS SRD SSP STN SVC SYP SZL THB TJS TMT TND TOP TRY TTD TWD TZS
    UAH UGX USD USN UYI UYU UYW UZS VED VES VND VUV WST XAF XAG XAU XBA XBB
    XBC XBD XCD XCG XDR XOF XPD XPF XPT XSU XUA YER ZAR ZMW ZWG
    """.split())

#: Withdrawn codes a caller may still hold in a file, each with the code
#: that replaced it.
WITHDRAWN_CURRENCIES: Dict[str, str] = {
    **{
        code: "EUR"
        for code in (
            "ATS BEF BGN CYP DEM EEK ESP FIM FRF GRD HRK IEP ITL LTL LUF LVL "
            "MTL NLG PTE SIT SKK XEU"
        ).split()
    },
    "ANG": "XCG",
    "AZM": "AZN",
    "BYB": "BYN",
    "BYR": "BYN",
    "CSD": "RSD",
    "GHC": "GHS",
    "MGF": "MGA",
    "MRO": "MRU",
    "MZM": "MZN",
    "ROL": "RON",
    "RUR": "RUB",
    "SDD": "SDG",
    "SRG": "SRD",
    "STD": "STN",
    "TMM": "TMT",
    "TRL": "TRY",
    "VEB": "VES",
    "VEF": "VES",
    "ZMK": "ZMW",
    "ZWD": "ZWG",
    "ZWL": "ZWG",
    "ZWN": "ZWG",
    "ZWR": "ZWG",
}

#: Market quotes in a hundredth of a currency: (unit, the ISO code). 'GBp'
#: is matched as written, BEFORE upper-casing turns it into 'GBP'; the
#: others in any case ('ZAc' upper-cases to 'ZAC').
_HUNDREDTHS_EXACT: Dict[str, Tuple[str, str]] = {"GBp": ("pence", "GBP")}
_HUNDREDTHS: Dict[str, Tuple[str, str]] = {
    "GBX": ("pence", "GBP"),
    "ZAC": ("cents", "ZAR"),
    "ILA": ("agorot", "ILS"),
}

#: Market names that are not ISO codes, with the code they mean.
_MARKET_NAMES: Dict[str, str] = {
    "CNH": "CNY",
    "RMB": "CNY",
    "NIS": "ILS",
    "NTD": "TWD",
}

#: A currency's name or sign where its code belongs. 'YEN' is here because
#: the nearest code by spelling is 'YER', Yemen's rial.
_NAMES: Dict[str, str] = {
    "$": "USD",
    "US$": "USD",
    "DOLLAR": "USD",
    "DOLLARS": "USD",
    "€": "EUR",
    "EURO": "EUR",
    "EUROS": "EUR",
    "£": "GBP",
    "POUND": "GBP",
    "POUNDS": "GBP",
    "STERLING": "GBP",
    "¥": "JPY",
    "YEN": "JPY",
    "YUAN": "CNY",
    "RENMINBI": "CNY",
    "FRANC": "CHF",
    "RUPEE": "INR",
    "WON": "KRW",
}

#: Crypto assets are instruments, not ISO currencies.
_CRYPTO = frozenset({"BTC", "XBT", "ETH", "USDT", "USDC", "SOL", "XRP"})

#: Codes ISO assigns but no price is quoted in.
_NOT_A_UNIT = {"XXX": "no currency", "XTS": "testing"}

#: The order a tie between equally near codes is broken in: the most traded
#: currencies first, then every other national currency, then the fund
#: codes, metals and units of account.
_PREFERRED = (
    "USD EUR JPY GBP CHF CAD AUD NZD CNY HKD SGD SEK NOK DKK KRW INR BRL MXN " "ZAR"
).split()
_SPECIAL = frozenset(
    "BOV CHE CHW CLF COU MXV USN UYI UYW XAG XAU XPD XPT XBA XBB XBC XBD XDR "
    "XSU XUA".split()
)


def _rank(code: str) -> Tuple[int, str]:
    if code in _PREFERRED:
        return _PREFERRED.index(code), code
    return (len(_PREFERRED) + (1 if code in _SPECIAL else 0)), code


def _distance(a: str, b: str) -> int:
    """Edits between two strings, a swap of neighbours counting as one."""
    rows = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        rows[i][0] = i
    for j in range(len(b) + 1):
        rows[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            rows[i][j] = min(
                rows[i - 1][j] + 1, rows[i][j - 1] + 1, rows[i - 1][j - 1] + cost
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                rows[i][j] = min(rows[i][j], rows[i - 2][j - 2] + 1)
    return rows[-1][-1]


def nearest_currency(text: str) -> str:
    """The active code nearest `text` by spelling; ties go to the more
    widely traded currency, so 'USS' suggests 'USD' rather than 'USN'."""
    probe = str(text).strip().upper()
    if probe in _NAMES:
        return _NAMES[probe]
    return min(ACTIVE_CURRENCIES, key=lambda c: (_distance(probe, c), _rank(c)))


def currency_code(value: Any, *, argument: str = "currency") -> str:
    """
    `value` as an active ISO 4217 code, upper-cased, or a refusal that names
    `argument`, the value and the code to write instead.
    """
    if not isinstance(value, str):
        raise ValidationError(
            f"{argument}={value!r} is not a currency code. Write the "
            "three-letter ISO 4217 code, e.g. 'USD', 'EUR' or 'JPY'."
        )
    text = value.strip()
    if not text:
        raise ValidationError(
            f"{argument} is empty. Write the three-letter ISO 4217 code the "
            "amount is in, e.g. 'USD', 'EUR' or 'JPY'."
        )
    hundredth = _HUNDREDTHS_EXACT.get(text) or _HUNDREDTHS.get(text.upper())
    if hundredth is not None:
        unit, code = hundredth
        raise ValidationError(
            f"{argument}={value!r} is {unit}, a hundredth of {code}, and not an "
            f"ISO 4217 code. Write {argument}='{code}' and give the amount in "
            f"{code}: divide a price quoted in {unit} by 100."
        )
    code = text.upper()
    if code in ACTIVE_CURRENCIES:
        return code
    if code in WITHDRAWN_CURRENCIES:
        successor = WITHDRAWN_CURRENCIES[code]
        raise ValidationError(
            f"{argument}={value!r} is a withdrawn ISO 4217 code, replaced by "
            f"'{successor}'. Write {argument}='{successor}' and give the "
            f"amount in {successor}."
        )
    if code in _MARKET_NAMES:
        raise ValidationError(
            f"{argument}={value!r} is a market name, not an ISO 4217 code; the "
            f"code is '{_MARKET_NAMES[code]}'. Write "
            f"{argument}='{_MARKET_NAMES[code]}'."
        )
    if code in _CRYPTO:
        raise ValidationError(
            f"{argument}={value!r} is a crypto asset, not an ISO 4217 "
            f"currency. Hold it as an instrument priced in a currency, e.g. "
            f"instrument={text!r} with {argument}='USD'."
        )
    if code in _NOT_A_UNIT:
        raise ValidationError(
            f"{argument}={value!r} is ISO 4217's code for {_NOT_A_UNIT[code]}, "
            "which no amount is quoted in. Write the code of the currency the "
            "amount is in, e.g. 'USD'."
        )
    raise ValidationError(
        f"{argument}={value!r} is not an active ISO 4217 currency code. Did "
        f"you mean '{nearest_currency(code)}'? Codes are three letters, e.g. "
        "'USD', 'EUR' or 'JPY'."
    )
