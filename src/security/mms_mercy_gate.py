"""
Ra-Thor Mercy-Gated MMS Media Scanner Prototype
Image / QR / text / stego valence-check layer
MIT License — Eternal Thriving Grandmasterism
"""

import re
import numpy as np
import cv2
import pytesseract
from PIL import Image
import pyzbar.pyzbar as pyzbar
from typing import Dict, Any

# Mercy rules — expand with JAX model later
MMS_BLOCK_PATTERNS = [
    r'(free\s*gift|win|prize|lottery|claim\s*now)',
    r'(CRA|tax\s*refund|SIN\s*suspended|urgent\s*tax)',
    r'(bank\s*alert|credit\s*card|verify\s*account|click\s*here)',
    r'https?://[^\s]+',  # any link
]

def mercy_valence_check_mms(
    sender: str,
    message_text: str,
    media_path: str = None,   # path to image/audio/video
    metadata: Dict[str, Any] = None
) -> bool:
    """
    Return True = allow, False = block
    """
    full_text = f"{sender} {message_text}".lower()

    # Text pattern match
    for pat in MMS_BLOCK_PATTERNS:
        if re.search(pat, full_text):
            return False

    if media_path:
        try:
            img = cv2.imread(media_path)
            if img is None:
                return False  # corrupt media = block

            # OCR text extraction
            ocr_text = pytesseract.image_to_string(Image.open(media_path)).lower()
            if any(re.search(pat, ocr_text) for pat in MMS_BLOCK_PATTERNS):
                return False

            # QR code detection
            decoded = pyzbar.decode(img)
            for obj in decoded:
                qr_data = obj.data.decode('utf-8').lower()
                if any(pat in qr_data for pat in MMS_BLOCK_PATTERNS):
                    return False

            # Simple stego suspicion (entropy check — placeholder)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            entropy = -np.sum([p * np.log2(p) for p in np.histogram(gray, bins=256, density=True)[0] if p > 0])
            if entropy > 7.9:  # high entropy = potential hidden data
                return False

        except Exception:
            return False  # media processing failure = block

    # Behavioral checks
    if metadata:
        if metadata.get('short_code', False):
            return False
        if metadata.get('links_count', 0) > 1:
            return False

    return True  # mercy passes


# Example usage
if __name__ == "__main__":
    test_mms = [
        {"sender": "+19122599619", "text": "Appointment confirmation", "media": None},
        {"sender": "+14388172457", "text": "Click to claim prize https://fake.link", "media": "qr_scam.jpg"},
    ]

    for mms in test_mms:
        allowed = mercy_valence_check_mms(mms["sender"], mms["text"], mms.get("media"))
        print(f"{mms['sender']}: {mms['text'][:60]}... → {'ALLOW' if allowed else 'BLOCK'}")
