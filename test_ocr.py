# test_ocr.py
import time, traceback, os
print("TEST OCR: start")
print("cwd:", os.getcwd())

try:
    t0 = time.time()
    print("1) Importing PIL...")
    from PIL import Image
    print("  PIL ok (ptime {:.2f}s)".format(time.time()-t0))
except Exception as e:
    print("  PIL import failed:", e)
    traceback.print_exc()
    raise SystemExit(1)

try:
    t0 = time.time()
    print("2) Importing pytesseract...")
    import pytesseract
    print("  pytesseract ok (ptime {:.2f}s)".format(time.time()-t0))
    try:
        print("  pytesseract.tesseract_cmd:", getattr(pytesseract.pytesseract, "tesseract_cmd", None))
    except Exception:
        pass
except Exception as e:
    print("  pytesseract import failed:", e)
    traceback.print_exc()
    raise SystemExit(1)

IMAGE = "data/vault/sample.jpg"
if not os.path.exists(IMAGE):
    print("Image not found:", IMAGE)
    raise SystemExit(1)

# 3) image_to_string test
try:
    print("3) Running pytesseract.image_to_string (start)")
    t0 = time.time()
    img = Image.open(IMAGE).convert("RGB")
    txt = pytesseract.image_to_string(img)
    print("  image_to_string finished (ptime {:.2f}s)".format(time.time()-t0))
    print("  snippet:", repr(txt[:200]))
except Exception as e:
    print("  image_to_string error:", e)
    traceback.print_exc()

# 4) image_to_data test (bounding boxes)
try:
    print("4) Running pytesseract.image_to_data (start)")
    t0 = time.time()
    from pytesseract import Output
    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    print("  image_to_data finished (ptime {:.2f}s)".format(time.time()-t0))
    print("  keys:", list(data.keys()))
    print("  first_texts:", data.get("text", [])[:10])
except Exception as e:
    print("  image_to_data error:", e)
    traceback.print_exc()

print("TEST OCR: done")
