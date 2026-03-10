import cv2
import numpy as np
import yaml
import onnxruntime as ort
from typing import List, Tuple, Optional


# ===================== ONNX CNN =====================

class ONNXDigitCNN:
    """
    Expects input: (N,1,64,64) float32 in [0,1]
    """
    def __init__(self, onnx_path: str, num_threads: int = 2):
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(num_threads)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess = ort.InferenceSession(
            onnx_path,
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        logits = self.sess.run([self.output_name], {self.input_name: X})[0]
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        pred = probs.argmax(axis=1)
        conf = probs[np.arange(len(pred)), pred]
        return pred.astype(int), conf.astype(float)


# ===================== DIGIT READER =====================

class DigitReaderCNNONNX:
    """
    YOLO crop -> color mask -> deskew -> plate ROI
    -> binarize -> projection segmentation
    -> normalize 64x64 -> CNN
    """

    def __init__(
        self,
        onnx_path: str,
        yaml_path: str,
        num_threads: int = 2,
        debug: bool = False
    ):
        self.model = ONNXDigitCNN(onnx_path, num_threads)
        self.debug = debug
        self.target_size = (56, 56) #(64,64)

        with open(yaml_path, "r") as f:
            self.cfg = yaml.safe_load(f)

    # ---------- Public ----------

    def read_number(self, crop: np.ndarray, bgr: bool = True) -> Tuple[Optional[int], List[int], List[float], str]:
        """
        Lee un número de 3 dígitos desde un crop de imagen.
        
        Args:
            crop: Imagen (BGR o RGB según parámetro bgr)
            bgr: Si True, crop está en BGR; si False, en RGB
            
        Returns:
            (number, digits, confidences, plate_color)
        """
        # Validación de entrada
        if crop is None or crop.size == 0:
            return "", [], [], "unknown"
        if len(crop.shape) != 3 or crop.shape[2] != 3:
            return "", [], [], "unknown"
            
        try:
            if not bgr:
                crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            crop = self._rescale(crop)
            color = self._detect_plate_color(crop)
            mask = self._plate_color_mask_2(crop, color)
            
            if self.debug:
                cv2.imwrite("debug_mask.png", mask)
            if mask.sum() == 0:
                return None, [], [], color

            crop = cv2.bitwise_and(crop, crop, mask=mask)
            filled_mask = cv2.bitwise_not(mask)
            crop = cv2.inpaint(crop, filled_mask, 3, cv2.INPAINT_TELEA)
            if self.debug:
                cv2.imwrite("debug_crop.png", crop)

            # rot = self._deskew(crop_bgr, mask)
            # if self.debug:
            #     cv2.imwrite("debug_rot.png", rot)
            #plate = self._crop_plate(crop_bgr, color)
            #if self.debug:
            #    cv2.imwrite("debug_plate.png", plate)

            gray = self._anti_color_gray(crop, color)
            if self.debug:
                cv2.imwrite("debug_gray.png", gray)
            
            bw = self._binarize_2(gray, False)
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
            bw = self._clear_border(bw) 
            #bw = self._reconnect(bw)
            if self.debug:
                cv2.imwrite("debug_bw.png", bw)

            boxes = self._find_digit_boxes(bw)
            if len(boxes) == 0:
                return "", [], [], color
            boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)[:3]
            boxes = sorted(boxes, key=lambda b: b[0])
            X = []
            for i, box in enumerate(boxes):
                d64 = self._extract_and_normalize(gray, box)
                if self.debug:
                    cv2.imwrite(f"digit_{i}.png", d64)
                X.append(d64)
            
            X = np.stack(X, axis=0).astype(np.float32) / 255.0
            X = X[:, None, :, :]  # (N,1,64,64)

            preds, confs = self.model.predict(X)

            try:
                number = int("".join(str(p) for p in preds))
            except Exception:
                number = ""
            return number, preds.tolist(), confs.tolist(), color
        except cv2.error as e:
            print(f"[ERROR] OpenCV error in read_number: {e}")
            return "", [], [], "unknown"
        except Exception as e:
            print(f"[ERROR] Unexpected error in read_number: {e}")
            import traceback
            traceback.print_exc()
            return "", [], [], "unknown"
        

    # ---------- Geometry ----------
    def _rescale(self, img_bgr: np.ndarray) -> np.ndarray:
        h, w, _ = img_bgr.shape
        if h > w:
            new_w = 200
            new_h = int(h * new_w / w)
        else:
            new_h = 150
            new_w = int(w * new_h / h)
        return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

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
    
        return cleared

    def _detect_plate_color(self, bgr):
        b, g, r = cv2.split(bgr)
        if self.debug:
            cv2.imwrite("debug_b.png", b)
            cv2.imwrite("debug_g.png", g)
            cv2.imwrite("debug_r.png", r)
        m = self._binarize_2(b)
        mask = cv2.bitwise_not(m)
        mask = self._clear_border(mask)
        if self.debug:
            cv2.imwrite("debug_color_mask.png", mask)
        if mask.sum() < 50:
            return "unknown"
        if r[mask>0].mean() > g[mask>0].mean():
            return "red"
        if g[mask>0].mean() > r[mask>0].mean():
            return "green"
        return "unknown"

    def _plate_color_mask_2(self, bgr, color):
        try:
            b, g, r = cv2.split(bgr)
            if color == "red":
                m = b
            elif color == "green":
                m = b
            else:
                m = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            m = self._binarize_2(m, False)
            inv = cv2.bitwise_not(m)
            if self.cfg["geometry"]["enable"]:
                if self.debug:
                    cv2.imwrite("debug_inv.png", inv)
                k = self.cfg["geometry"]["close_kernel"]
                it = self.cfg["geometry"]["close_iter"]
                inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, np.ones(tuple(k), np.uint8), it)
                if self.debug:
                    cv2.imwrite("debug_inv_close.png", inv)

            inv = self._clear_border(inv)

            if self.debug:
                cv2.imwrite("debug_inv_clear.png", inv)

            if self.cfg["geometry"]["enable"]:
                k = self.cfg["geometry"]["dilate_kernel"]
                it = self.cfg["geometry"]["dilate_iter"]
                inv = cv2.morphologyEx(inv, cv2.MORPH_DILATE, np.ones(tuple(k), np.uint8), it)
                if self.debug:
                    cv2.imwrite("debug_inv_dilate.png", inv)
                    
            cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnts, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)
            filled_mask = np.zeros(inv.shape, np.uint8)
            cv2.drawContours(filled_mask, [box], 0, 255, -1)
            return filled_mask
        except Exception as e:
            print(f"[ERROR] in _plate_color_mask_2: {e}")
            return np.zeros(bgr.shape[:2], np.uint8)

    # Las siguientes funciones se mantienen por si se necesitan en el futuro
    # pero actualmente no se usan en el pipeline principal
    
    def _deskew(self, bgr, mask):
        """Rota la imagen para corregir el ángulo de la placa."""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        (cx, cy), (w, h), angle = cv2.minAreaRect(c)
        if h < w:
            angle += 90
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        return cv2.warpAffine(bgr, M, bgr.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)

    # ---------- OCR ----------

    def _anti_color_gray(self, plate, color):
        b, g, r = cv2.split(plate)
        return g if color == "red" else r if color == "green" else cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    def _binarize_2(self, gray: np.ndarray, dilate: bool = True) -> np.ndarray:
        """Binariza imagen con CLAHE y Otsu."""
        clahe = cv2.createCLAHE(2.0, (8, 8))
        gray = clahe.apply(gray)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if dilate:
            bw = cv2.morphologyEx(th, cv2.MORPH_DILATE, np.ones((3,3), np.uint8), iterations=1)
        else:
            bw = th
        if self.debug:
            cv2.imwrite("debug_bw_otsu.png", bw)
        return bw
    
    def _reconnect(self, bw):
        """Reconecta componentes cercanos (no usado actualmente)."""
        k = self.cfg["reconnect"]["kernel"]
        it = self.cfg["reconnect"]["iterations"]
        return cv2.morphologyEx(
            bw,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, tuple(k)),
            it
        )
    
    def _find_digit_boxes(self, bw: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Encuentra bounding boxes de posibles dígitos."""
        h, w = bw.shape
        area = h * w
        cnts = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        candidates = []

        # Leer umbrales desde config
        dd = self.cfg.get("digit_detection", {})
        MIN_AREA_RATIO = float(dd.get("min_area_ratio", 0.02))
        MAX_AREA_RATIO = float(dd.get("max_area_ratio", 0.8))
        MIN_ASPECT = float(dd.get("min_aspect", 0.1))
        MAX_ASPECT = float(dd.get("max_aspect", 1.6))

        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            a = cw * ch
            if a < MIN_AREA_RATIO * area or a > MAX_AREA_RATIO * area:
                continue
            aspect = cw / (ch + 1e-6)
            if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
                continue
            candidates.append((x, y, cw, ch))

        if self.debug:
            print(f"[DEBUG] Found {len(candidates)} digit candidates.")
        return candidates
    
    # ---------- Segmentation ----------
    
    def _extract_and_normalize(self, gray: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        """Extrae y normaliza un dígito individual a 64x64."""
        # Leer umbral desde config
        dd = self.cfg.get("digit_detection", {})
        MIN_CONTOUR_AREA = int(dd.get("min_contour_area", 15))
        
        x, y, w, h = box
        digit = gray[y:y + h, x:x + w]
        _, digit = cv2.threshold(digit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(digit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        mask = np.zeros_like(digit, dtype=np.uint8)
        # Filtrar contornos pequeños (ruido)
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
        digit_cropped = cv2.bitwise_and(digit, mask)
        
        # Padding y cierre morfológico
        pad = int(0.2 * max(w, h))
        digit_cropped = cv2.copyMakeBorder(digit_cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        digit_cropped = cv2.morphologyEx(digit_cropped, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

        # Redimensionar manteniendo aspect ratio
        h_crop, w_crop = digit_cropped.shape
        target_h, target_w = self.target_size[1], self.target_size[0]
        scale = min(target_w / w_crop, target_h / h_crop)
        new_w = int(w_crop * scale)
        new_h = int(h_crop * scale)
        digit_cropped = cv2.resize(digit_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Centrar en canvas
        canvas = np.zeros(self.target_size[::-1], dtype=digit_cropped.dtype)
        y0 = (target_h - new_h) // 2
        x0 = (target_w - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = digit_cropped
        _, dth = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invertir si es necesario (fondo debe ser negro)
        if np.mean(dth) > 127:
            dth = cv2.bitwise_not(dth)
        return dth



# ------------------------ Main ------------------------
if __name__ == "__main__":

    ONNX_PATH = "models/OCR/digit_cnn_64.onnx"
    CFG_PATH = "config.yaml"   # <-- put your thresholds/kernels here

    reader = DigitReaderCNNONNX(
        onnx_path=ONNX_PATH,
        yaml_path=CFG_PATH,            # set None to run with defaults
        num_threads=2,
        debug=True,
    )

    crop = cv2.imread("examples/number_50.png")
    number, digits, confs, plate_color = reader.read_number(crop)
    print("Number:", number, "Digits:", digits, "Plate color:", plate_color, "Confs:", confs)
