import base64
import sys
from io import BytesIO
from PIL import Image

def get_image_size(base64_image):
    try:
        # Base64-String dekodieren
        image_data = base64.b64decode(base64_image)
        # Bild aus Bytes laden
        image = Image.open(BytesIO(image_data))
        # Bildgröße ermitteln
        width, height = image.size
        return {"width": width, "height": height}
    except Exception as e:
        return {"error": f"Fehler bei der Verarbeitung des Bildes: {str(e)}"}

if __name__ == "__main__":
    # Base64-String vom Standard-Input lesen
    base64_image = sys.stdin.read().strip()
    # Bildgröße ermitteln
    result = get_image_size(base64_image)
    # Ergebnis als JSON zurückgeben
    import json
    print(json.dumps(result))