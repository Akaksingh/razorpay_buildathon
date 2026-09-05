"""Indian PIN-code reference: zone, state, city tier, serviceability priors.

Indian PINs are hierarchical: digit 1 = postal zone, digits 1-2 = state
sub-zone, digits 1-3 = sorting district. We exploit that so an unseen PIN
still resolves to a state and a tier instead of an "unknown" bucket.
Tier priors below are cold-start values only; the live engine overrides them
with observed delivery outcomes per PIN from the feature store.
"""
from __future__ import annotations

from dataclasses import dataclass

# digits 1-2 -> state / UT (India Post allocation)
_STATE_BY_PREFIX2 = {
    "11": "Delhi", "12": "Haryana", "13": "Haryana", "14": "Punjab", "15": "Punjab", "16": "Punjab",
    "17": "Himachal Pradesh", "18": "Jammu & Kashmir", "19": "Jammu & Kashmir",
    "20": "Uttar Pradesh", "21": "Uttar Pradesh", "22": "Uttar Pradesh", "23": "Uttar Pradesh",
    "24": "Uttar Pradesh", "25": "Uttar Pradesh", "26": "Uttar Pradesh", "27": "Uttar Pradesh",
    "28": "Uttar Pradesh", "30": "Rajasthan", "31": "Rajasthan", "32": "Rajasthan", "33": "Rajasthan",
    "34": "Rajasthan", "36": "Gujarat", "37": "Gujarat", "38": "Gujarat", "39": "Gujarat",
    "40": "Maharashtra", "41": "Maharashtra", "42": "Maharashtra", "43": "Maharashtra", "44": "Maharashtra",
    "45": "Madhya Pradesh", "46": "Madhya Pradesh", "47": "Madhya Pradesh", "48": "Madhya Pradesh",
    "49": "Chhattisgarh", "50": "Telangana", "51": "Andhra Pradesh", "52": "Andhra Pradesh",
    "53": "Andhra Pradesh", "56": "Karnataka", "57": "Karnataka", "58": "Karnataka", "59": "Karnataka",
    "60": "Tamil Nadu", "61": "Tamil Nadu", "62": "Tamil Nadu", "63": "Tamil Nadu", "64": "Tamil Nadu",
    "67": "Kerala", "68": "Kerala", "69": "Kerala", "70": "West Bengal", "71": "West Bengal",
    "72": "West Bengal", "73": "West Bengal", "74": "West Bengal", "75": "Odisha", "76": "Odisha",
    "77": "Odisha", "78": "Assam", "79": "North East", "80": "Bihar", "81": "Bihar", "82": "Bihar",
    "83": "Jharkhand", "84": "Bihar", "85": "Bihar",
}

# digits 1-3 -> (city, tier). Tier 1 = metro, 2 = large city, 3 = district town.
_CITY_BY_PREFIX3 = {
    "110": ("New Delhi", 1), "400": ("Mumbai", 1), "401": ("Thane", 1), "560": ("Bengaluru", 1),
    "600": ("Chennai", 1), "500": ("Hyderabad", 1), "700": ("Kolkata", 1), "411": ("Pune", 1),
    "380": ("Ahmedabad", 1), "122": ("Gurugram", 1), "201": ("Noida / Ghaziabad", 1),
    "302": ("Jaipur", 2), "226": ("Lucknow", 2), "160": ("Chandigarh", 2), "641": ("Coimbatore", 2),
    "682": ("Kochi", 2), "440": ("Nagpur", 2), "452": ("Indore", 2), "462": ("Bhopal", 2),
    "395": ("Surat", 2), "390": ("Vadodara", 2), "530": ("Visakhapatnam", 2), "751": ("Bhubaneswar", 2),
    "800": ("Patna", 2), "141": ("Ludhiana", 2), "143": ("Amritsar", 2), "282": ("Agra", 2),
    "208": ("Kanpur", 2), "221": ("Varanasi", 2), "492": ("Raipur", 2), "834": ("Ranchi", 2),
    "781": ("Guwahati", 2), "695": ("Thiruvananthapuram", 2), "620": ("Tiruchirappalli", 2),
    "625": ("Madurai", 2), "575": ("Mangaluru", 2), "580": ("Hubballi", 2), "413": ("Solapur", 3),
    "431": ("Aurangabad", 3), "845": ("Motihari", 4), "847": ("Darbhanga", 4), "851": ("Begusarai", 4),
    "271": ("Gonda", 4), "272": ("Basti", 4), "276": ("Azamgarh", 4), "233": ("Ghazipur", 4),
    "813": ("Bhagalpur", 3), "822": ("Daltonganj", 4), "742": ("Murshidabad", 4), "733": ("Raiganj", 4),
    "786": ("Dibrugarh", 3), "797": ("Kohima", 4), "331": ("Churu", 4), "345": ("Jaisalmer", 4),
    "364": ("Bhavnagar", 3), "370": ("Kutch", 4), "509": ("Mahabubnagar", 4), "515": ("Anantapur", 3),
    "535": ("Vizianagaram", 4), "577": ("Shivamogga", 3), "591": ("Belagavi", 3), "629": ("Nagercoil", 3),
    "673": ("Kozhikode", 2),
}

# Cold-start priors by tier: (delivery success rate, COD RTO rate)
_TIER_PRIORS = {1: (0.965, 0.14), 2: (0.94, 0.19), 3: (0.905, 0.26), 4: (0.86, 0.34)}


@dataclass(frozen=True)
class PinInfo:
    pin: str
    valid: bool
    state: str
    city: str
    tier: int
    serviceability_prior: float
    rto_prior: float


def lookup(pin: str | int | None) -> PinInfo:
    p = "".join(ch for ch in str(pin or "") if ch.isdigit())
    if len(p) != 6 or p[0] == "0":
        return PinInfo(p, False, "Unknown", "Unknown", 4, 0.80, 0.40)
    state = _STATE_BY_PREFIX2.get(p[:2], "Unknown")
    city, tier = _CITY_BY_PREFIX3.get(p[:3], (None, None))
    if city is None:
        # Unlisted sorting district: tier 3 if the state has a listed metro in
        # the same zone, else tier 4. Cheap, monotone, and never "unknown".
        tier = 3 if any(k.startswith(p[0]) and v[1] == 1 for k, v in _CITY_BY_PREFIX3.items()) else 4
        city = f"{state} district ({p[:3]}xxx)"
    serv, rto = _TIER_PRIORS[tier]
    return PinInfo(p, state != "Unknown", state, city, tier, serv, rto)


# Curated sampling pools for the synthetic generator (weights ~ order share)
SAMPLE_PINS: dict[int, list[str]] = {
    1: ["110001", "110017", "110092", "400001", "400053", "400701", "560001", "560034", "560103",
        "600001", "600040", "500001", "500081", "700001", "700091", "411001", "411045", "380001",
        "122001", "122018", "201301", "201010"],
    2: ["302001", "302017", "226001", "226010", "160017", "641001", "682001", "440001", "452001",
        "462001", "395001", "390001", "530001", "751001", "800001", "141001", "143001", "282001",
        "208001", "221001", "492001", "834001", "781001", "695001", "620001", "625001", "575001",
        "580001", "673001"],
    3: ["413001", "431001", "813001", "786001", "364001", "515001", "577001", "591001", "629001"],
    4: ["845401", "847201", "851101", "271001", "272001", "276001", "233001", "822101", "742101",
        "733101", "797001", "331001", "345001", "370001", "509001", "535001"],
}
