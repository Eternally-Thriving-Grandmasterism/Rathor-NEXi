"""
Ra-Thor Mercy-Gated Spam Blocker Prototype
Symbolic valence-check layer + rule-based pre-filter
MIT License — Eternal Thriving Grandmasterism
"""

import re
import phonenumbers

# Mercy rules — expand with ML later
BLOCK_PATTERNS = [
    r'\+?1\s*\(?\d{3}\)?[-.\s]*\d{3}[-.\s]*\d{4}',  # any NA number
    r'\+44', r'\+91', r'\+380',                     # common spoof countries
    r'800|877|888|866',                             # toll-free fraud
    r'CRA|Canada Revenue|SIN|tax refund|bank fraud',  # keyword traps
]

def mercy_valence_check(caller_id: str, transcript: str = "") -> bool:
    """
    Return True = allow, False = block
    """
    try:
        num = phonenumbers.parse(caller_id, "CA")  # assume Canada
        if not phonenumbers.is_valid_number(num):
            return False  # invalid = scam
    except:
        return False

    # Pattern match
    for pat in BLOCK_PATTERNS:
        if re.search(pat, caller_id) or re.search(pat, transcript):
            return False

    # Add ML/JAX model call here later
    # valence = jax_model.predict(...)

    return True  # mercy passes


# Example usage in call screening
if __name__ == "__main__":
    test_calls = [
        "(912) 259-9619",           # legit Thrive
        "(438) 817-2457",           # fraud
        "+1 (647) 366-6795",        # fraud
        "CIBC Fraud Alert (877) 208-8801"  # legit but spoof risk
    ]

    for call in test_calls:
        allowed = mercy_valence_check(call)
        print(f"{call}: {'ALLOW' if allowed else 'BLOCK'}")
