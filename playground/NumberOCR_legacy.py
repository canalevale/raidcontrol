import numpy as np
import cv2
import joblib
from typing import List, Tuple, Optional

class DigitReaderSVM:
    """
    Given a cropped image of a 3-digit plate:
      1) segments the three digits,
      2) normalizes each digit (binary, resize),
      3) flattens raw pixels,
      4) predicts each digit with a pre-trained sklearn SVM (.pkl),
      5) returns the integer.
    """

    def __init__(
        self,
        svm_model_path: str,
        scaler_path: Optional[str] = None,
        target_size: Tuple[int, int] = (28, 28),
        binarize_inverted: Optional[bool] = None,
    ):
        """
        svm_model_path: path to joblib dump of sklearn SVM trained on single digits.
        scaler_path: optional joblib dump of StandardScaler (used during training).
        target_size: size to which each digit is resized before flattening.
        binarize_inverted: None = auto, True/False = force inversion.
        """
        self.model = joblib.load(svm_model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        self.target_size = target_size
        self.binarize_inverted = binarize_inverted

    # ------------------------ Public API ------------------------

    def read_number(self, crop_bgr_or_gray: np.ndarray) -> Tuple[Optional[int], List[int]]:
        """
        Returns:
          number_int: e.g., 123 or None on failure,
          digits: list of predicted digits (ideally length 3).
        """
        print("[INFO] Starting digit reading pipeline.")
        gray = self._to_gray(crop_bgr_or_gray)
        bw = self._binarize(gray)
        clean = self._clear_border(bw)
        boxes = self._find_digit_boxes(clean)
        #cv2.imwrite("./clean.png",clean)
        if len(boxes) == 0:
            print("[WARN] No digit candidates found.")
            return None, []

        # pick top-3 by area, then sort left->right
        boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)[:3]
        boxes = sorted(boxes, key=lambda b: b[0])

        feats = []
        for b in boxes:
            digit_img = self._extract_and_normalize(gray, b)
            vec = self._make_features(digit_img)
            feats.append(vec)

        X = np.stack(feats, axis=0).astype(np.float32)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        preds = self.model.predict(X).astype(int)
        try:
            number_int = int("".join(str(int(d)) for d in preds))
        except Exception:
            number_int = None

        print(f"[INFO] Predicted digits: {preds} -> number: {number_int}")
        return number_int, preds.tolist()

    # ------------------------ Internal helpers ------------------------

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
        if self.binarize_inverted is None:
            inv = cv2.bitwise_not(th)
            choice = "inv" if np.count_nonzero(inv) > np.count_nonzero(th) else "th"
            bw = inv if choice == "inv" else th
        elif self.binarize_inverted:
            bw = cv2.bitwise_not(th)
        else:
            bw = th
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        #cv2.imwrite("./th.png",bw)

        return bw

    def _find_digit_boxes(self, bw: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = bw.shape
        area = h * w
        cnts = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        candidates = []
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            a = cw * ch
            if a < 0.03*area or a > 0.8*area:
                continue
            aspect = cw / (ch + 1e-6)
            if not (0.2 <= aspect <= 1.2):
                continue
            candidates.append((x, y, cw, ch))

        print(f"[DEBUG] Found {len(candidates)} digit candidates.")
        return candidates
    
    def _clear_border(self, img_bin: np.ndarray) -> np.ndarray:
        """
        Remove connected components touching the image border.
        img_bin: binary image (0 and 255)
        """
        # Copia para no modificar el original
        cleared = img_bin.copy()

        # Altura y ancho
        h, w = img_bin.shape

        # Crear máscara más grande (floodFill necesita +2 pixeles)
        mask = np.zeros((h+2, w+2), np.uint8)

        # Revisar bordes: izquierda, derecha, arriba, abajo
        for y in range(h):
            for x in [0, w-1]:
                if cleared[y, x] == 255:
                    cv2.floodFill(cleared, mask, (x, y), 0)

        for x in range(w):
            for y in [0, h-1]:
                if cleared[y, x] == 255:
                    cv2.floodFill(cleared, mask, (x, y), 0)

        return cleared

    def _extract_and_normalize(self, gray: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = box
        digit = gray[y:y+h, x:x+w]
        _, digit = cv2.threshold(digit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pad = int(0.2 * max(w, h))
        digit = cv2.copyMakeBorder(digit, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        digit = cv2.resize(digit, self.target_size, interpolation=cv2.INTER_AREA)
        _, dth = cv2.threshold(digit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(dth) > 127:
            dth = cv2.bitwise_not(dth)
        #cv2.imwrite(f"./{x}.png",dth)
        return dth

    def _make_features(self, digit_img: np.ndarray) -> np.ndarray:
        feat = digit_img.astype(np.float32) / 255.0
        return feat.ravel()

# ------------------------ Main ------------------------
if __name__ == "__main__":

    BUNDLE_PATH = r"models/OCR/svm_ocr_arialv2.pkl"
    reader = DigitReaderSVM(svm_model_path=BUNDLE_PATH,binarize_inverted=False)


    # Example:
    crop = cv2.imread(r"examples/number_50.png")
    #crop = cv2.imread(r".\Source\Notebooks\sample_plate_896.png")
    number, digits = reader.read_number(crop)
    print("Number:", number, "Digits:", digits)
    pass