#!/bin/bash

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Iniciando servicios de RaidControl...${NC}"

# Función para manejar la salida al presionar Ctrl+C
cleanup() {
    echo -e "${YELLOW}\nDeteniendo servicios...${NC}"
    kill $BIKEDETECTOR_PID $UPLOADER_PID $UVICORN_PID 2>/dev/null
    wait $BIKEDETECTOR_PID $UPLOADER_PID $UVICORN_PID 2>/dev/null
    echo -e "${GREEN}Servicios detenidos.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Iniciar bikedetector
echo -e "${YELLOW}Lanzando bikedetector...${NC}"
python raidcontrol/bikedetector.py  config.yaml &
BIKEDETECTOR_PID=$!
echo -e "${GREEN}bikedetector iniciado (PID: $BIKEDETECTOR_PID)${NC}"

# # Iniciar uploader
echo -e "${YELLOW}Lanzando uploader...${NC}"
python raidcontrol/uploader.py  config.yaml &
UPLOADER_PID=$!
echo -e "${GREEN}uploader iniciado (PID: $UPLOADER_PID)${NC}"

# Iniciar uvicorn con monitor.py
echo -e "${YELLOW}Lanzando uvicorn...${NC}"
uvicorn monitor:app --host 0.0.0.0 --port 8080 --reload &
UVICORN_PID=$!
echo -e "${GREEN}uvicorn iniciado (PID: $UVICORN_PID)${NC}"

echo -e "${GREEN}Todos los servicios están activos.${NC}"
echo -e "${YELLOW}Presiona Ctrl+C para detener todos los servicios.${NC}"

# Mantener el script activo
wait