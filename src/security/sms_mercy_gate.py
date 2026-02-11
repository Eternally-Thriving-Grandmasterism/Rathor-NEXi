"""
Ra-Thor Mercy-Gated SMS Spam Blocker Prototype
Symbolic valence-check layer + rule-based + ML-ready
MIT License — Eternal Thriving Grandmasterism
"""

import re
import phonenumbers
from typing import Dict, Any

# Mercy rules — expand with JAX model later
SMS_BLOCK_PATTERNS = [
    r'(free\s*gift|win|prize|lottery|claim\s*now)',
    r'(CRA|Canada\s*Revenue|tax\s*refund|SIN\s*suspended|urgent\s*tax)',
    r'(bank\s*alert|credit\s*card|verify\s*account|click\s*here)',
    r'https?://[^\s]+',  # any link
    r'\b\d{5,6}\b',      # short code suspicion
]

def mercy_valence_check_sms(
    sender: str,
    message: str,
    metadata: Dict[str, Any] = None
) -> bool:
    """
    Return True = allow, False = block
    """
    # Sender validation
    try:
        num = phonenumbers.parse(sender, "CA")
        if not phonenumbers.is_valid_number(num):
            return False
    except:
        return False

    # Content pattern match
    full_text = f"{sender} {message}".lower()
    for pat in SMS_BLOCK_PATTERNS:
        if re.search(pat, full_text):
            return False

    # Behavioral checks (expand with metadata)
    if metadata:
        if metadata.get('timestamp_hour') in [0,1,2,3,4]:  # sleep hours
            return False
        if metadata.get('links_count', 0) > 1:
            return False

    # Future: JAX model inference
    # valence = jax_model.predict(full_text)

    return True  # mercy passes


# Example usage in SMS receiver / forwarder
if __name__ == "__main__":
    test_sms = [
        {"sender": "+19122599619", "message": "Hi, this is Thrive Health confirming your appointment."},
        {"sender": "+14388172457", "message": "URGENT: Your SIN is suspended. Click https://fake.gov.link to fix."},
        {"sender": "CIBC", "message": "Your card ending 1234 has suspicious activity. Call 877-208-8801."},
    ]

    for sms in test_sms:
        allowed = mercy_valence_check_sms(sms["sender"], sms["message"])
        print(f"{sms['sender']}: {sms['message'][:60]}... → {'ALLOW' if allowed else 'BLOCK'}")
