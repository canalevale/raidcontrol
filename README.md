# Raidcontrol

Sistema de detección automática de ciclistas y lectura de numeros para eventos deportivos.

**Proyecto Final de Ingeniería Electrónica - FCEIA, Universidad Nacional de Rosario**

---

## Hardware

- Raspberry Pi 5
- Hailo8L AI Accelerator
- Cámara IMX519

## Instalación

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar variables de entorno

Copiar el archivo de ejemplo y editarlo con tus valores:

```bash
cp .env.example .env
nano .env
```

Variables requeridas:
- `API_BASE_URL` - URL del servidor backend
- `DEVICE_API_KEY` - API key de autenticación

**Importante:** El archivo `.env` no se sube al repositorio por seguridad.

### 3. Configurar parámetros del sistema

Editar `config.yaml` para ajustar:
- Resolución de cámara
- Posición de la línea de meta
- Thresholds de detección
- Parámetros de OCR

## Uso

### Ejecutar en modo producción

```bash
python raidcontrol/bikedetector.py
```

### Ejecutar con visualización (debug)

```bash
python raidcontrol/bikedetector_debug.py
```

### Calibrar el enfoque de la cámara

```bash
cd playground
python focus_calibration.py
```

## Estructura

```
raidcontrol/              # Scripts de producción
├── bikedetector.py       # Detección principal
├── NumberOCR_CNN.py      # OCR de dorsales
└── uploader.py           # Upload a backend

playground/               # Scripts de testing y desarrollo
├── hailo_test.py
├── focus_calibration.py
├── dataset_collector.py
└── ocr_evaluation.py

models/                   # Modelos entrenados (YOLOv8 + OCR)
config.yaml               # Configuración principal
.env                      # Variables sensibles (no en repo)
```

## Funcionalidades

- Detección de ciclistas y dorsales cruzando línea de meta
- OCR automático de números de 3 dígitos
- Tracking de dorsales sin ciclista visible
- Sistema de cola local con upload asíncrono
- Reintentos automáticos si falla la red
- Almacenamiento local de eventos e imágenes

## Notas

- Los modelos YOLOv8 están compilados a formato HEF para Hailo
- El OCR funciona con dorsales de colores: rojo, verde o blanco
- Las imágenes se guardan en `local_db/images/YYYY-MM-DD/`
- Los eventos se almacenan en SQLite hasta que se suben al backend

