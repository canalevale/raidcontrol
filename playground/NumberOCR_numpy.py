import numpy as np
import cv2
import joblib
from typing import List, Tuple, Optional


# ------------------------ NumPy SVC (RBF, OvO) ------------------------

def rbf_kernel(X: np.ndarray, SV: np.ndarray, gamma: float) -> np.ndarray:
    """
    X:  (n, d)
    SV: (m, d)
    Returns K: (n, m) where K[i,j] = exp(-gamma * ||X_i - SV_j||^2)
    """
    X = X.astype(np.float64, copy=False)
    SV = SV.astype(np.float64, copy=False)

    X_norm = np.sum(X * X, axis=1, keepdims=True)          # (n, 1)
    SV_norm = np.sum(SV * SV, axis=1, keepdims=True).T     # (1, m)
    dist2 = X_norm + SV_norm - 2.0 * (X @ SV.T)            # (n, m)
    return np.exp(-gamma * dist2)


def ovo_decision_function(X: np.ndarray, model: dict) -> np.ndarray:
    """
    Compute OvO decision values for sklearn/libsvm-style multiclass SVC.
    Returns:
      dec: (n_samples, n_pairs)
    """
    gamma = float(np.asarray(model["gamma"]).reshape(-1)[0]) #float(model["gamma"][0])
    SV = model["support_vectors"]
    dual = model["dual_coef"]         # (k-1, n_SV_total)
    intercept = model["intercept"]    # (n_pairs,)
    n_support = model["n_support"]    # (k,)
    classes = model["classes"]        # (k,)

    k = classes.shape[0]
    K = rbf_kernel(X, SV, gamma=gamma)  # (n, n_SV_total)

    # SV are grouped by class in order of classes_
    starts = np.cumsum(np.r_[0, n_support[:-1]])
    ends = starts + n_support

    dec_list = []
    pair_idx = 0

    # Pair order: (0,1),(0,2)...(0,k-1),(1,2)...(k-2,k-1)
    for i in range(k):
        for j in range(i + 1, k):
            si, ei = starts[i], ends[i]
            sj, ej = starts[j], ends[j]

            # Mapping consistent with standard libsvm packing used by sklearn.
            row_i = j - 1
            row_j = i

            contrib_i = K[:, si:ei] @ dual[row_i, si:ei].T   # (n,)
            contrib_j = K[:, sj:ej] @ dual[row_j, sj:ej].T   # (n,)

            dec = contrib_i + contrib_j + intercept[pair_idx]
            dec_list.append(dec)
            pair_idx += 1

    return np.stack(dec_list, axis=1)


def ovo_predict_from_dec(dec: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """
    Vote-based OvO prediction: if dec(pair)>0 => i wins else j wins.
    """
    n = dec.shape[0]
    k = classes.shape[0]
    votes = np.zeros((n, k), dtype=np.int32)

    pair_idx = 0
    for i in range(k):
        for j in range(i + 1, k):
            i_wins = dec[:, pair_idx] > 0
            votes[i_wins, i] += 1
            votes[~i_wins, j] += 1
            pair_idx += 1

    pred_idx = np.argmax(votes, axis=1)
    return classes[pred_idx]


class NumpySVC:
    """
    NumPy-only inference for an exported sklearn SVC (RBF, OvO).
    """
    def __init__(self, npz_path: str):
        self.model = dict(np.load(npz_path, allow_pickle=True))

    def predict(self, X: np.ndarray) -> np.ndarray:
        dec = ovo_decision_function(X, self.model)
        return ovo_predict_from_dec(dec, self.model["classes"])


# ------------------------ Digit Reader ------------------------

class DigitReaderSVMNumpy:
    """
    Same pipeline as your DigitReaderSVM, but the classifier is a NumPy SVC loaded from .npz
    instead of sklearn/joblib.
    """

    def __init__(
        self,
        svm_npz_path: str,
        scaler_path: Optional[str] = None,
        target_size: Tuple[int, int] = (28, 28),
        binarize_inverted: Optional[bool] = None,
    ):
        """
        svm_npz_path: path to .npz exported bundle from sklearn SVC.
        scaler_path: optional joblib dump of StandardScaler (if used during training).
        target_size: size to which each digit is resized before flattening.
        binarize_inverted: None = auto, True/False = force inversion.
        """
        self.model = NumpySVC(svm_npz_path)

        # Optional: keep scaler exactly as in training
        # If you want zero joblib, export mean_/scale_ and apply with NumPy.
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
        print("[INFO] Starting digit reading pipeline (NumPy SVM).")
        gray = self._to_gray(crop_bgr_or_gray)
        bw = self._binarize(gray)
        clean = self._clear_border(bw)
        boxes = self._find_digit_boxes(clean)
        cv2.imwrite("clean.png", bw)
        print(len(boxes))
        if len(boxes) == 0:
            print("[WARN] No digit candidates found.")
            return None, [], "unknown"

        # pick top-3 by area, then sort left->right
        boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)[:3]
        boxes = sorted(boxes, key=lambda b: b[0])

        feats = []
        for b in range(len(boxes)):
            digit_img = self._extract_and_normalize(gray, boxes[b])
            cv2.imwrite("digit_" + str(b) + ".png", digit_img)
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
        
        plate_color = self._detect_plate_color_from_channels(crop_bgr_or_gray)
        print(f"[INFO] Predicted digits: {preds} -> number: {number_int} (plate color: {plate_color})")
        return number_int, preds.tolist(), plate_color

    # ------------------------ Internal helpers ------------------------

    def _detect_plate_color_from_channels(self, bgr: np.ndarray) -> str:
        """
        Detect dominant plate color using RGB channel dominance.
        Assumes white digits over colored background.
        """
        b, g, r = cv2.split(bgr)

        # Ignore near-white pixels (digits)
        white_mask = (b > 200) & (g > 200) & (r > 200)
        color_mask = ~white_mask

        if np.count_nonzero(color_mask) < 50:
            return "unknown"

        b_mean = b[color_mask].mean()
        g_mean = g[color_mask].mean()
        r_mean = r[color_mask].mean()

        if r_mean > g_mean * 1.2 and r_mean > b_mean * 1.2:
            return "red"
        if g_mean > r_mean * 1.2 and g_mean > b_mean * 1.2:
            return "green"

        return "unknown"


    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        #return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        """
        Returns an 'intensity' image optimized for white digits on colored background.
        If input is BGR, uses LAB L-channel; if already gray, returns copy.
        """
        if img.ndim != 3:
            return img.copy()

        # OpenCV uses BGR
        b, g, r = cv2.split(img)
        cv2.imwrite("b.png", b)
        cv2.imwrite("g.png", g)
        cv2.imwrite("r.png", r)

        # Pick channel with highest contrast (std)
        chans = [b, g, r]
        stds = [c.std() for c in chans]
        best = chans[int(np.argmax(stds))]

        return best


    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        #blur = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(2.0, (8, 8))
        gray = clahe.apply(gray)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if self.binarize_inverted is None:
            inv = cv2.bitwise_not(th)
            choice = "inv" if np.count_nonzero(inv) > np.count_nonzero(th) else "th"
            bw = inv if choice == "inv" else th
        elif self.binarize_inverted:
            bw = cv2.bitwise_not(th)
        else:
            bw = th

        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8), iterations=1)
        cv2.imwrite("bw.png", bw)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        cv2.imwrite("bw_open.png", bw)
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
            if a < 0.05 * area or a > 0.8 * area:
                continue
            aspect = cw / (ch + 1e-6)
            if not (0.2 <= aspect <= 1.5):
                continue
            candidates.append((x, y, cw, ch))

        print(f"[DEBUG] Found {len(candidates)} digit candidates.")
        return candidates

    def _clear_border(self, img_bin: np.ndarray) -> np.ndarray:
        """
        Remove connected components touching the image border.
        img_bin: binary image (0 and 255)
        """
        cleared = img_bin.copy()
        h, w = img_bin.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)

        for y in range(h):
            for x in [0, w - 1]:
                if cleared[y, x] == 255:
                    cv2.floodFill(cleared, mask, (x, y), 0)

        for x in range(w):
            for y in [0, h - 1]:
                if cleared[y, x] == 255:
                    cv2.floodFill(cleared, mask, (x, y), 0)
        cv2.imwrite("cleared.png", cleared)
        return cleared
    def _deskew_binary(self, digit_bin: np.ndarray) -> np.ndarray:
        """
        Deskew a binary digit image using image moments.
        Expects foreground white (255) on black (0).
        """
        ys, xs = np.where(digit_bin > 0)
        if len(xs) < 20:
            return digit_bin  # too little signal, skip

        m = cv2.moments(digit_bin, binaryImage=True)
        if abs(m["mu02"]) < 1e-6:
            return digit_bin

        # Skew estimate
        skew = m["mu11"] / m["mu02"]

        h, w = digit_bin.shape
        M = np.float32([[1, skew, -0.5 * w * skew],
                        [0, 1, 0]])

        deskewed = cv2.warpAffine(
            digit_bin, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return deskewed

    def _extract_and_normalize(self, gray: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = box
        digit = gray[y:y + h, x:x + w]
        _, digit = cv2.threshold(digit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(digit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        mask = np.zeros_like(digit, dtype=np.uint8)
        # Tune this: minimum contour area to keep
        min_area = 15
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
        digit_cropped = cv2.bitwise_and(digit, mask)
        pad = int(0.2 * max(w, h))
        digit_cropped = cv2.copyMakeBorder(digit_cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        digit_cropped = cv2.resize(digit_cropped, self.target_size, interpolation=cv2.INTER_AREA)

        _, dth = cv2.threshold(digit_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(dth) > 127:
            dth = cv2.bitwise_not(dth)

        return dth

    def _make_features(self, digit_img: np.ndarray) -> np.ndarray:
        feat = digit_img.astype(np.float32) / 255.0
        return feat.ravel()


# ------------------------ Main ------------------------
if __name__ == "__main__":

    # This is the NumPy-exported SVC bundle (NOT .pkl)
    BUNDLE_NPZ_PATH = r"models/OCR/svc_digit_rbf.npz"

    reader = DigitReaderSVMNumpy(
        svm_npz_path=BUNDLE_NPZ_PATH,
        scaler_path=None,               # keep None if you trained with passthrough
        binarize_inverted=False
    )

    crop = cv2.imread(r"examples/number_50.png")
    number, digits,     plate_color = reader.read_number(crop)
    print("Number:", number, "Digits:", digits, "Plate color:", plate_color)
