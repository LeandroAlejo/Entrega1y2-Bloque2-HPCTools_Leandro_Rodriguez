#!/bin/bash
set -e

# Detectar la carpeta donde reside este script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/entornoTarea2"

echo "=== Configurando entorno virtual en: $VENV_DIR ==="

# 1. Cargar módulos del clúster compatibles con tu sistema
module --force purge
module load cesga/2020
module load python/3.10.8

# 2. Crear entorno si no existe
if [ ! -d "$VENV_DIR" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv "$VENV_DIR"
else
    echo "El entorno ya existe en $VENV_DIR."
fi

# 3. Activar e instalar paquetes desde requirements.txt
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

# 4. Liberar espacio en disco
pip cache purge

echo "=== Entorno virtual preparado y listo ==="
