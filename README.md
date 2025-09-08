# Raidcontrol
Este proyecto implementa un sistema de detección de bicicletas y cruce de meta usando una Raspberry Pi 5 con cámara IMX519, un modelo YOLOv8 y un OCR basado en SVM (Support Vector Machine)

## Scripts
### BikeDetector:
El script procesa el video en tiempo real, dibuja una línea virtual de meta y, cuando un ciclista la cruza, guarda un recorte (crop) con:

- Fecha y hora exacta del cruce.

- Distancia en píxeles respecto a la línea.

Todos los parámetros principales (clase a detectar, tamaño de imagen, posición de la línea, umbrales de confianza, etc.) son totalmente configurables mediante un archivo config.yaml.

Los crops generados se almacenan automáticamente en la carpeta de salida definida en la configuración.

###  NumberOCR
Módulo de OCR clásico basado en SVM (scikit-learn) para leer placas de tres dígitos a partir de un recorte (crop) de imagen.