#!/bin/bash
#SBATCH --job-name=bert_squad_ddp
#SBATCH --partition=short
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=bert_ddp_results_%j.log

# 1. Limpiar y cargar módulos del CESGA
module --force purge
module load cesga/2020 python/3.10.8

# 2. Activar exclusivamente el entorno local de la tarea
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="$SCRIPT_DIR/entornoTarea2"

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: No se encontró el entorno virtual en $VENV_DIR."
    echo "Es necesario ejecutar primero: bash setup_env.sh"
    exit 1
fi

# 3. Configuración de red para torchrun multinodo
NODELIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MASTER_ADDR=${NODELIST[0]}
MASTER_PORT=29500

echo "=========================================================="
echo " JOB ID: $SLURM_JOB_ID"
echo " Nodos asignados: ${NODELIST[*]}"
echo " Nodo Maestro (Rank 0): $MASTER_ADDR:$MASTER_PORT"
echo "=========================================================="

# 4. Lanzamiento distribuido con srun + torchrun (1 tarea srun por nodo -> torchrun lanza 2 procesos GPU)
# Al disponer de 64 CPUs por nodo, cada GPU cuenta con 32 cores disponibles -> --num_workers 8 o 16
srun torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    "$SCRIPT_DIR/train_distributed.py" \
    --batch_size 32 \
    --num_workers 0 \
    --use_amp \
    --compile \
    --steps 500 \
    --warmup 10
