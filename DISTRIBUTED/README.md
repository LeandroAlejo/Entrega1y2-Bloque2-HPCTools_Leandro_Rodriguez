# Explicaciones Entrega 2: DISTRIBUTED

- **Autor:** Leandro Alejo Rodriguez Alvarez
- **Fecha de entrega:** 31/10/2026
- **Asignatura:** HPCTools

---

## Archivos entregados

- `setup_env.sh`: Script para crear el entorno virtual `entornoTarea2` e instalar las dependencias necesarias.
- `requirements.txt`: Lista de dependencias del proyecto (`torch`, `transformers`, `datasets`).
- `train_distributed.py`: Código en Python adaptado para entrenamiento distribuido con PyTorch nativo (DDP).
- `run_distributed.sh`: Script de Slurm para lanzar el trabajo multi-nodo en Finisterrae III con `torchrun`.
- `bert_ddp_results_11128103.log`: Log sgenerado de la ejecución en 2 nodos con 4 GPUs A100.
- `README.md`: Explicación de las decisiones de paralelización, configuración del clúster y análisis de escalado.

---

## Guía rápida de pasos para ejecutar

1. **Crear o preparar el entorno virtual:**  
   Ejecutar en la terminal:
   ```bash
   bash setup_env.sh
   ```
2. **Revisar los parámetros de ejecución:**  
   En `run_distributed.sh` se han configurado **500 pasos** evaluados (más 10 de calentamiento) con un tamaño de lote local de 32 por GPU, usando precisión mixta (`bfloat16`) y compilación con `torch.compile`.
3. **Lanzar la prueba distribuida en el cluster:**  
   Enviar el trabajo a Slurm:
   ```bash
   sbatch run_distributed.sh
   ```

---

## Explicación de cada archivo entregado

## 0. (`bert_ddp_results_11128103.log`) Log de ejecución obtenido

Archivo con la traza completa de la ejecución en 2 nodos físicos (`a100-17` y `a100-18`), utilizado para extraer las métricas de rendimiento y las conclusiones del apartado 5. Muestra una finalización limpia y sincronizada de los 4 procesos sin errores de cierre de socket.

## 1. (`setup_env.sh`) Configuración del entorno

Crea el entorno virtual local `entornoTarea2` utilizando los módulos del CESGA (`cesga/2020` y `python/3.10.8`). 

Al igual que en la primera entrega, se instala PyTorch con soporte para CUDA 12.6. Para esta entrega distribuida no se requiere instalar librerías externas adicionales, ya que PyTorch incorpora de forma nativa todo el soporte necesario: el backend NCCL para comunicar las GPUs, el módulo DDP y la herramienta de orquestación `torchrun`.

## 2. (`requirements.txt`) Dependencias del proyecto

Contiene las mismas librerías que la entrega anterior:
```text
--extra-index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)
torch
transformers
datasets
```

## 3. (`run_distributed.sh`) Orquestación multi-nodo con Slurm y torchrun

Este script solicita los recursos fijados en el enunciado (2 nodos con 2 GPUs NVIDIA A100 cada uno, sumando 4 GPUs en total) y gestiona la ejecución combinando Slurm y `torchrun`:

- **Política de CPUs en CESGA:** En Finisterrae III, cada GPU A100 exige reservar 32 núcleos de CPU. Al pedir 2 GPUs por nodo, con `#SBATCH --cpus-per-task=64` asignamos los 64 cores correspondientes a cada nodo físico (128 cores en total entre los dos nodos).
- **Lanzamiento con `srun` + `torchrun`:** Se usa `#SBATCH --ntasks-per-node=1` para que Slurm cree un único proceso por nodo, y desde ahí `torchrun --nproc_per_node=2` se encarga de crear un proceso de Python por cada GPU local.
- **Punto de encuentro:** Obtenemos la IP/nombre del primer nodo (`a100-17`) con `scontrol show hostnames` y lo usamos como `MASTER_ADDR` en el puerto 29500 para que los procesos de ambos nodos se conecten y sincronicen al arrancar.

## 4. (`train_distributed.py`) Implementación de DDP y decisiones de diseño

Para entrenar en las 4 GPUs se ha utilizado **DDP (DistributedDataParallel)** nativo de PyTorch.

Los puntos clave que componen el script son:

1. **Inicialización del grupo de procesos:** Se inicializa NCCL (`dist.init_process_group`) y cada proceso toma su GPU correspondiente según la variable `LOCAL_RANK` que le pasa `torchrun`.
2. **Descarga protegida:** Para evitar que las 4 GPUs intenten descargar o escribir los archivos del modelo y del dataset a la vez en el disco compartido, el `rank 0` lo hace primero y el resto espera mediante una barrera de sincronización (`dist.barrier()`).
3. **Reparto de datos con `DistributedSampler`:** Se encarga de dividir el dataset SQuAD en 4 partes disjuntas para que ninguna GPU repita muestras en una misma época.
4. **Escalado del Learning Rate:** Al usar 4 GPUs con batch size local de 32, el batch global por paso pasa a ser de 128 secuencias (el cuádruple). Por regla general, al multiplicar el batch por 4, multiplicamos también la tasa de aprendizaje por 4 (de $3\cdot 10^{-5}$ a $1.2\cdot 10^{-4}$) para que el entrenamiento siga aprendiendo al mismo ritmo y no se desestabilice.
5. **Cálculo de métricas globales:** Al terminar los 500 pasos, usamos operaciones de reducción colectiva (`dist.reduce`) para sumar las muestras procesadas de todas las GPUs y hacer la media de la función de pérdida (`loss`), imprimiendo los resultados consolidados únicamente desde el proceso principal (`rank 0`).

### Decisión sobre los trabajadores de CPU (`num_workers=0`)
En las pruebas iniciales con workers en el DataLoader, la ejecución arrojó un error de bus (`SIGBUS`) provocado por el desbordamiento de la memoria compartida del sistema (`/dev/shm`). Con 2 procesos GPU por nodo y múltiples workers por proceso, se creaban demasiados subprocesos compitiendo por memoria IPC en Linux.

Al fijar `num_workers=0`, cada proceso de Python lee directamente de la memoria RAM del nodo sin generar subprocesos ni usar `/dev/shm`. Tal y como demostramos en el Baseline (donde pasar de 0 a 16 workers solo supuso una aceleración de 1.01x), el dataset SQuAD ya queda completamente preprocesado en memoria RAM al arrancar el script, por lo que la entrega de batches es inmediata y la GPU nunca sufre de inanición de datos.

### Decisión sobre el Profiler (no usar --profile, a diferencia de como se hizo en el BASELINE)
Se optó por medir el estado estacionario sin activar `torch.profiler` para evitar que la instrumentación de eventos y llamadas de CUDA introdujese sobrecargas o asimetrías temporales entre procesos.

---

## 5. Resultados y Conclusiones

A continuación se comparan los resultados obtenidos entre la versión más rápida del Baseline (Etapa 5: 1 GPU A100 con compilación y BF16) y la versión distribuida en 2 nodos con 4 GPUs A100:

```text
=============================================================================================================================================
TABLA COMPARATIVA DE RENDIMIENTO: BASELINE (1 GPU) vs DISTRIBUTED DDP (4 GPUs en 2 Nodos)
=============================================================================================================================================
Configuración                      Nodos  GPUs   Batch Global  Throughput     Throughput/GPU   Speedup   Eficiencia   Peak VRAM    Final Loss
                                                               (samples/s)      (samples/s)                             (MB)       (500 steps)
---------------------------------------------------------------------------------------------------------------------------------------------
[BASELINE] 1x A100 (Compilado)       1      1         32          290.61           290.61        1.00x     100.0%      5713.08 MB   1.023299
[DISTRIBUTED] 4x A100 DDP (2 nodos)  2      4        128         1192.05           298.01        4.10x     102.5%      6220.00 MB   0.990099
=============================================================================================================================================
```

```text
=============================================================================================================================================
TIEMPOS Y POTENCIA DE CÓMPUTO MEDIDOS (500 PASOS DE ENTRENAMIENTO)
=============================================================================================================================================
Configuración                      Pasos Medidos   Muestras Totales   Tiempo Medido     Achieved TFLOP/s   MFU Real %   Techo Teórico
---------------------------------------------------------------------------------------------------------------------------------------------
[BASELINE] 1x A100 (Compilado)          500             16,000           55.06 s             62.4000         20.00%      312.0 TFLOP/s
[DISTRIBUTED] 4x A100 DDP (2 nodos)     500             64,000           53.69 s            255.9551         20.51%     1248.0 TFLOP/s
=============================================================================================================================================
```

---

### Conclusiones del análisis

1. **Escalado superlineal (Speedup de 4.10x y Eficiencia del 102.5%):**  
   Al pasar de 1 a 4 GPUs repartidas en dos nodos físicos distintos, el throughput global aumentó de 290.61 a **1,192.05 muestras por segundo**. Se supera ligeramente el escalado lineal perfecto ($4\times$), lo cual es habitual en PyTorch 2.x al compilar con TorchInductor: al operar con un batch global más amplio (128 secuencias simultáneas en lugar de 32), la ocupación de los Tensor Cores en las matrices densas es más eficiente y amortiza aún mejor el coste fijo del framework.

2. **Impacto de la red e interconexión InfiniBand:**  
   En DDP, cada paso realiza un *Ring All-Reduce* para sincronizar los gradientes entre las GPUs locales y a través de la red entre ambos nodos. Al mantener un tamaño de lote local de 32 y precisión BF16, el tiempo de cálculo por paso es lo bastante grande como para que la comunicación quede oculta tras el pase backward. La prueba es que el rendimiento individual por tarjeta no decayó: cada GPU procesó **298.01 muestras/s**, igualando e incluso superando ligeramente los 290.61 samples/s que conseguía la GPU en solitario en el Baseline.

3. **Volumen de datos procesado y tiempo:**  
   En prácticamente el mismo tiempo de ejecución (~53.7 segundos frente a los 55.0 segundos del baseline), la versión distribuida procesó **64.000 muestras frente a las 16.000 de la prueba con una sola GPU**, multiplicando por 4 el trabajo útil realizado. La potencia alcanzada ascendió a **255.96 TFLOP/s sostenidos**, manteniendo un MFU del **20.51%** sobre el techo agregado de las 4 tarjetas (1,248 TFLOP/s).

4. **Convergencia del entrenamiento (`loss`):**  
   A pesar de cuadruplicar el tamaño del lote efectivo por iteración, la pérdida final tras 500 pasos fue de **0.9901**, mejorando ligeramente el valor obtenido en el Baseline mononodo (1.0232). Esto confirma que la aplicación de la regla de escalado lineal en la tasa de aprendizaje ($1.2\cdot 10^{-4}$) preservó la dinámica de aprendizaje del optimizador y mantuvo una convergencia completamente estable.

5. **Consumo de memoria VRAM:**  
   La memoria máxima asignada por GPU subió de 5,713 MB a 6,220 MB (~500 MB adicionales). Este incremento moderado es completamente normal y corresponde a los búferes de comunicación internos que reserva la biblioteca NCCL en la memoria de vídeo para gestionar las transferencias de gradientes entre dispositivos.