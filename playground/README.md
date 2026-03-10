# Playground - Scripts de Testing y Desarrollo

Este directorio contiene herramientas de testing, calibración y experimentación utilizadas durante el desarrollo del sistema de detección.

## 🔧 Herramientas de Calibración

### `focus_calibration.py`
Herramienta interactiva para calibrar la posición del lente de la cámara IMX519. Muestra vista en vivo y permite ajustar el enfoque manualmente para encontrar la distancia focal óptima.

**Uso:** Calibración inicial de la cámara antes de poner el sistema en producción.

### `hailo_test.py`
Script standalone para probar la inferencia con el acelerador Hailo8L. Carga el modelo YOLOv8 en formato HEF y muestra las detecciones en tiempo real con bounding boxes.

**Uso:** Verificar que el modelo compilado funciona correctamente con Hailo antes de integrarlo al sistema principal.

## 📊 Evaluación y Testing

### `ocr_evaluation.py`
Evalúa la precisión del sistema OCR procesando un dataset de imágenes anotadas. Genera métricas de accuracy por dígito y reportes CSV con los resultados.

**Uso:** Validar la efectividad del modelo CNN de OCR con ground truth conocido.

### `onnx_test.py`
Testing de modelos ONNX exportados. Verifica que la conversión desde PyTorch/Ultralytics a ONNX mantenga la precisión esperada.

**Uso:** Validación de modelos exportados antes de deployment.

## 💾 Recolección de Datos

### `dataset_collector.py`
Guarda crops de las detecciones (ciclistas y números) en disco para crear datasets de entrenamiento. Organiza las imágenes automáticamente por tipo y timestamp.

**Uso:** Recolectar data real durante eventos para mejorar modelos futuros.

## 🧪 Implementaciones Antiguas (Histórico)

### `NumberOCR_legacy.py`
Primera implementación del sistema OCR. Usaba técnicas tradicionales de visión (thresholding, contornos) sin deep learning.

**Uso:** Referencia histórica. Fue reemplazado por implementaciones más robustas.

### `NumberOCR_numpy.py`
Segunda iteración del OCR usando SVM (Support Vector Machine) entrenado con NumPy. Mejoraba sobre la versión legacy pero seguía siendo menos preciso que CNN.

**Uso:** Referencia histórica. Fue reemplazado por `NumberOCR_CNN.py` que está en producción.

## 🔄 Utilidades de Conversión

### `export_ncnn.py`
Script para exportar modelos a formato NCNN (framework optimizado para dispositivos móviles/embebidos).

**Uso:** Experimentación con backends alternativos a Hailo para deployment en hardware diferente.

---

**Nota:** Los scripts en este directorio NO se usan en producción. El código de producción está en `raidcontrol/`.
