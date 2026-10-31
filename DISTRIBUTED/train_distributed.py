import os
import argparse
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.utils.flop_counter import FlopCounterMode
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, default_data_collator

# ==============================================================================
# 1. INICIALIZACIÓN DISTRIBUIDA (DDP) Y PARÁMETROS DE EJECUCIÓN
# Inicializamos el grupo de procesos mediante el backend NCCL. Cada proceso
# consulta las variables de entorno inyectadas por torchrun para determinar
# su rango global y fijar el dispositivo físico local en el nodo.
# ==============================================================================
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

parser = argparse.ArgumentParser(description="PyTorch DDP BERT SQuAD")
parser.add_argument('--batch_size', type=int, default=32, 
                    help="Número de secuencias procesadas por GPU individual.")
parser.add_argument('--num_workers', type=int, default=0, 
                    help="Procesos paralelos de CPU. Con el dataset en RAM, 0 evita contención IPC.")
parser.add_argument('--pin_memory', action='store_true', 
                    help="Asigna páginas bloqueadas en RAM para permitir transferencias PCIe asíncronas.")
parser.add_argument('--use_amp', action='store_true', 
                    help="Habilita torch.autocast en bfloat16 para derivar GEMMs a los Tensor Cores.")
parser.add_argument('--compile', action='store_true', 
                    help="Ejecuta torch.compile para capturar el grafo y fusionar kernels con Inductor.")
parser.add_argument('--steps', type=int, default=500, 
                    help="Iteraciones de entrenamiento medidas en estado estacionario.")
parser.add_argument('--warmup', type=int, default=10, 
                    help="Pasos previos descartados para evitar penalizaciones por compilación JIT.")
parser.add_argument('--lr', type=float, default=3e-5,
                    help="Tasa de aprendizaje base de referencia mononodo.")
args = parser.parse_args()

# ==============================================================================
# 2. CARGA DE MODELO Y TOKENIZADOR
# Usamos BERT-base para Question Answering (SQuAD).
# Rank 0 descarga y genera la caché local en disco; dist.barrier() bloquea
# al resto para evitar condiciones de carrera durante la lectura del checkpoint.
# ==============================================================================
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 384
DOC_STRIDE = 128

if global_rank == 0:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
dist.barrier()

if global_rank != 0:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

model = model.to(device)

# ==============================================================================
# 3. CÁLCULO DINÁMICO DE FLOPs POR SECUENCIA
# FlopCounterMode rastrea a nivel de operador C++ (ATen) la cantidad exacta de
# operaciones de coma flotante que demanda un forward + backward completo para
# una longitud de 384 tokens, evitando aproximaciones manuales.
# ==============================================================================
def measure_model_flops(test_model):
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, MAX_LENGTH), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((1, MAX_LENGTH), dtype=torch.long, device=device)
    dummy_start_positions = torch.tensor([0], dtype=torch.long, device=device)
    dummy_end_positions = torch.tensor([0], dtype=torch.long, device=device)
    
    dummy_batch = {
        "input_ids": dummy_input_ids,
        "attention_mask": dummy_attention_mask,
        "start_positions": dummy_start_positions,
        "end_positions": dummy_end_positions
    }

    with FlopCounterMode(display=False) as fwd_counter:
        _ = test_model(**dummy_batch)
    forward_flops = fwd_counter.get_total_flops()

    with FlopCounterMode(display=False) as step_counter:
        out = test_model(**dummy_batch)
        loss = out.loss
        loss.backward()
    train_step_flops = step_counter.get_total_flops()

    test_model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    macs_forward = forward_flops // 2
    tflops_per_seq = train_step_flops / 1e12

    return macs_forward, forward_flops, train_step_flops, tflops_per_seq

macs_fwd, fwd_flops, total_flops_per_seq, tflops_per_sequence = measure_model_flops(model)

if global_rank == 0:
    print("\n" + "="*60)
    print(" CÓMPUTO CALCULADO AUTOMÁTICAMENTE (POR SECUENCIA)")
    print(f" MACs Forward (1 seq)       : {macs_fwd:,}")
    print(f" FLOPs Forward (1 seq)      : {fwd_flops:,}")
    print(f" FLOPs Train (Fwd + Bwd)    : {total_flops_per_seq:,}")
    print(f" Constante por secuencia    : {tflops_per_sequence:.8e} TFLOPs")
    print("="*60)

# ==============================================================================
# 4. PREPARACIÓN DEL DATASET Y PIPELINE DE ENTRADA DISTRIBUIDO
# DistributedSampler reparte el dataset de forma disjunta entre los 4 rangos.
# drop_last=True mantiene constante la dimensión de las matrices en todos los
# procesos, evitando recompilaciones dinámicas al utilizar torch.compile.
# ==============================================================================
if global_rank == 0:
    raw_datasets = load_dataset("rajpurkar/squad", split="train")
dist.barrier()

if global_rank != 0:
    raw_datasets = load_dataset("rajpurkar/squad", split="train")

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions, examples["context"],
        max_length=MAX_LENGTH, truncation="only_second", stride=DOC_STRIDE,
        return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length",
    )
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions, end_positions = [], []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1: idx += 1
        context_start = idx
        while sequence_ids[idx] == 1: idx += 1
        context_end = idx - 1

        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            token_idx = context_start
            while token_idx <= context_end and offset[token_idx][0] <= start_char: token_idx += 1
            start_positions.append(token_idx - 1)
            
            token_idx = context_end
            while token_idx >= context_start and offset[token_idx][1] >= end_char: token_idx -= 1
            end_positions.append(token_idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

train_dataset = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets.column_names)
train_dataset.set_format("torch")

train_sampler = DistributedSampler(
    train_dataset,
    num_replicas=world_size,
    rank=global_rank,
    shuffle=True,
    drop_last=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=train_sampler,
    num_workers=args.num_workers,
    pin_memory=args.pin_memory,
    collate_fn=default_data_collator,
    drop_last=True
)

if args.compile:
    model = torch.compile(model)

# Envoltorio DDP: registra hooks para sincronizar gradientes vía Ring All-Reduce en backward
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# Escalado lineal del Learning Rate en base al tamaño de lote global efectivo
effective_lr = args.lr * world_size
optimizer = AdamW(model.parameters(), lr=effective_lr)
model.train()

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

step = 0
start_time = None
total_samples = 0
last_loss_tensor = None
epoch = 0

# ==============================================================================
# 5. FUNCIÓN DE PASO DE ENTRENAMIENTO ASÍNCRONO
# non_blocking=True solapa la transferencia de host a device con el cómputo.
# zero_grad(set_to_none=True) libera memoria en lugar de sobreescribir ceros.
# ==============================================================================
def train_one_step(batch):
    batch = {k: v.to(device, non_blocking=args.pin_memory) for k, v in batch.items()}
    optimizer.zero_grad(set_to_none=True)
    
    if args.use_amp:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(**batch)
            loss = output.loss
    else:
        output = model(**batch)
        loss = output.loss
        
    loss.backward()
    optimizer.step()
    return loss

if global_rank == 0:
    print(f"\n--- [DDP] GPUs Totales={world_size} | BS/GPU={args.batch_size} (Global={args.batch_size * world_size}) | Workers={args.num_workers} | PinMem={args.pin_memory} | AMP={args.use_amp} | Compile={args.compile} ---")

# ==============================================================================
# 6. BUCLE DE ENTRENAMIENTO Y COORDINACIÓN MULTINODO
# set_epoch() actualiza la permutación pseudoaleatoria del sampler por época.
# dist.barrier() garantiza que la medición temporal se inicie de forma unificada.
# ==============================================================================
while step < (args.warmup + args.steps):
    train_sampler.set_epoch(epoch)
    
    for batch in train_loader:
        if step == args.warmup:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dist.barrier()
            start_time = time.perf_counter()
            total_samples = 0

        last_loss_tensor = train_one_step(batch)
        step += 1

        if step > args.warmup:
            total_samples += args.batch_size

        if step >= (args.warmup + args.steps):
            break
    epoch += 1

# ==============================================================================
# 7. REDUCCIÓN DE MÉTRICAS GLOBALES (ALL-REDUCE) Y EVALUACIÓN DE RENDIMIENTO
# ==============================================================================
if torch.cuda.is_available():
    torch.cuda.synchronize()
dist.barrier()
elapsed = time.perf_counter() - start_time if start_time else 1.0

# Reducción de muestras totales procesadas por todos los procesos
global_samples = torch.tensor(total_samples, device=device)
dist.reduce(global_samples, dst=0, op=dist.ReduceOp.SUM)

# Memoria máxima registrada entre todos los dispositivos GPU del clúster
local_peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
peak_mem_tensor = torch.tensor(local_peak_mem, device=device)
dist.reduce(peak_mem_tensor, dst=0, op=dist.ReduceOp.MAX)

# Promedio de la función de pérdida final
dist.reduce(last_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
final_loss = (last_loss_tensor / world_size).item()

if global_rank == 0:
    total_cluster_samples = global_samples.item()
    samples_per_sec = total_cluster_samples / elapsed
    throughput_per_gpu = samples_per_sec / world_size
    achieved_tflops = samples_per_sec * tflops_per_sequence

    hardware_peak_per_gpu = 312.0 if args.use_amp else 19.5
    cluster_hardware_peak = hardware_peak_per_gpu * world_size
    cluster_tc_peak = 312.0 * world_size

    real_hardware_mfu = (achieved_tflops / cluster_hardware_peak) * 100
    normalized_tc_mfu = (achieved_tflops / cluster_tc_peak) * 100

    print(f"\nResultados tras {args.steps} pasos (descartando {args.warmup} de warmup):")
    print(f" training time medido      : {elapsed:.2f} s ({elapsed/60:.2f} min)")
    print(f" samples / s (Global)      : {samples_per_sec:.2f}")
    print(f" samples / s / GPU         : {throughput_per_gpu:.2f}")
    print(f" peak GPU memory (Max GPU) : {peak_mem_tensor.item():.2f} MB")
    print(f" achieved TFLOP/s          : {achieved_tflops:.4f}")
    print(f" MFU % (Hardware real)     : {real_hardware_mfu:.2f}% (Techo: {cluster_hardware_peak} TFLOP/s)")
    print(f" MFU % (vs Tensor Cores)   : {normalized_tc_mfu:.2f}% (Techo: {cluster_tc_peak} TFLOP/s)")
    print(f" final loss                : {final_loss:.6f}")

dist.barrier()
dist.destroy_process_group()