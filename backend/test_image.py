# backend/test_image.py — quick local validation
# Run: cd backend && python test_image.py
import sys
sys.path.insert(0, '.')
print("TEST FILE STARTED")
from detectors.image_detector import analyze_image
from PIL import Image
import numpy as np
import tempfile, os

# Create a synthetic test image (solid color, no face)
test_img = Image.fromarray(np.random.randint(0, 255,
    (256, 256, 3), dtype=np.uint8))

with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
    test_img.save(tmp.name)
    result = analyze_image(tmp.name)
    os.unlink(tmp.name)

print("Result:", result)
assert 'result' in result, "Missing 'result' key"
assert 'confidence' in result, "Missing 'confidence' key"
assert 'flags' in result, "Missing 'flags' key"
assert result['result'] in ['FAKE', 'REAL', 'ERROR'], "Invalid result value"
print("✓ image_detector.py is working correctly")