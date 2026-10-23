# Using LUMI for SimLAP

Practical notes for this repo on LUMI-G. The chapter map follows the
[LUMI AI Guide](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main).
ImageNet layout and the read benchmark live in [in1k-loading.md](in1k-loading.md).

Account `project_465003400`. Paths are in `cluster/lumi.env`.

| What | Where |
|------|--------|
| Code | `/project/project_465003400/SSLHydra` |
| Container | `/appl/local/laifs/containers/lumi-multitorch-latest.sif` |
| Overlay venv | `/scratch/project_465003400/cache/SSLHydra-venv` |
| ImageNet | `/scratch/project_465003400/datasets/` |
| Checkpoints, Slurm logs | `/scratch/project_465003400/SSLHydra/outputs` |
| Caches (`HF_HOME`, `TORCH_HOME`, pip, MIOpen) | `/scratch/project_465003400/cache/` |

`$HOME` is 20 GB and is for dotfiles. Do not put the repo, the venv, or
framework caches there.

## What a LUMI-G node is

One LUMI-G node has four MI250X GPUs. Each MI250X is two GCDs, and PyTorch
sees **8 devices**. A full-node job in this repo is:

```bash
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
```

`--mem=0` gives the job the whole node (512 GB). `/tmp` is tmpfs counted
against that, which is what makes a ~150 GB staged ImageNet tree possible.
On `small-g`, `/tmp` is only about 64 GB unless the job is exclusive.

Slurm quickstart: <https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/slurm-quickstart/>.
Your account id: `lumi-workspaces`.

## 1. First job (Guide 01)

Login node. Build the overlay env once, then submit a 30-minute smoke run.

```bash
cd /project/project_465003400/SSLHydra
bash cluster/setup_lumi_env.sh
./cluster/submit_simclr.sh smoke
```

`setup_lumi_env.sh` must print `ok lumi overlay venv` and
`torch.cuda.is_available()` true. The venv is only valid **inside** the
container (`--system-site-packages` on top of the image's PyTorch). Do not
reinstall torch into it.

Logs:

```text
/scratch/project_465003400/SSLHydra/outputs/slurm/<jobname>-<jobid>.{out,err}
```

`module purge` in the sbatch script warns about modules that could not be
unloaded. That warning is expected.

Longer runs:

```bash
./cluster/submit_simclr.sh            # SimCLR, 100 epoch, 24 h
./cluster/submit_simclr.sh 1000       # 1000 epoch, 48 h
METHOD=supcon ./cluster/submit_simclr.sh 100
METHOD=simlap ./cluster/submit_simclr.sh 100
./cluster/submit_xclr.sh              # X-CLR, 8×128, 100 epoch
./cluster/submit_xclr.sh smoke
```

`METHOD` is `simclr`, `supcon`, `simlap`, or `xclr`.

## 2. Container and extra packages (Guide 02)

LUMI AI jobs run in a Singularity image from the
[LUMI AI Factory stack](https://docs.lumi-supercomputer.eu/laif/software/ai-environment/).
The image is built for this machine's ROCm and Slingshot stack. It will not
run on Isambard or Eureka.

Images on the system live under `/appl/local/laifs/containers/`. For PyTorch,
`lumi-multitorch-full-*` (what `lumi-multitorch-latest.sif` points at) is the
right base. Layers, from the guide:

1. `lumi-multitorch-rocm-*` — Ubuntu + ROCm
2. `lumi-multitorch-libfabric-*` — Slingshot libfabric
3. `lumi-multitorch-mpich-*` — GPU-aware MPICH
4. `lumi-multitorch-torch-*` — PyTorch
5. `lumi-multitorch-full-*` — DeepSpeed, Flash Attention, vLLM, and the rest

Every GPU sbatch script loads the bindings module before `singularity exec`.
That module attaches the Slingshot network and the work directory. Setting
those binds by hand is unnecessary.

```bash
module purge
module load Local-LAIF lumi-aif-singularity-bindings
```

Extra Python packages go into the overlay venv, not a rebuilt image.
`cluster/setup_lumi_env.sh` installs `tensorboard`, `gin-config`, and `lmdb`.
Add a package by repeating that pattern inside the container:

```bash
singularity exec --bind /scratch/project_465003400 "${SIF}" \
  bash -lc 'source /scratch/project_465003400/cache/SSLHydra-venv/bin/activate && pip install <pkg>'
```

A venv is thousands of small files. If the scratch inode quota gets tight,
the guide's fix is to pack it once:

```bash
mksquashfs simlap-venv simlap-venv.sqsh -processors 1 -no-xattrs
```

and bind it with `-B simlap-venv.sqsh:/user-software:image-src=/` plus
`SINGULARITYENV_PREPEND_PATH=/user-software/bin`. This repo currently keeps
the unpacked venv.

### Environment variables that matter here

| Variable | Why | Where we set it |
|----------|-----|-----------------|
| `HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME` | Default caches land in `$HOME` and blow the file quota | `cluster/lumi.env` |
| `MIOPEN_USER_DB_PATH`, `MIOPEN_CUSTOM_CACHE_DIR` | MIOpen locks its Find DB. Lustre has no usable file locking, and a shared `/tmp/.cache/miopen` is owned by whoever created it | node-local `/tmp/$USER-miopen-$SLURM_JOB_ID` in `cluster/simclr.sbatch`; `cluster/jobs/with_miopen.sh` copies the persistent DB in `MIOPEN_CACHE` onto that path and back at exit |
| `TMPDIR` for DataLoader shm | `file_system` sharing strategy must sit on node tmpfs | `train_cl.sh` sets `/tmp/SSLHydra-$SLURM_JOB_ID` |
| `FI_MR_CACHE_MONITOR=userfaultfd` | Default `memhooks` monitor hangs multi-node jobs inside these containers | set this yourself before a multi-node `sbatch` until the job script exports it. Later images (after `lumi-multitorch-u24r70f21m50t210-20260807`) switch the default to `kdreg2` |
| `ROCR_VISIBLE_DEVICES` | Slurm's GPU binding. If it is set, `CUDA_VISIBLE_DEVICES` must be unset or PyTorch sees the wrong devices | `simclr.sbatch` |

Inspect the image from a login node (no GPU):

```bash
module purge
module load Local-LAIF lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
singularity run "$SIF" pip list
```

A one-GPU import check (burns a short `small-g` allocation):

```bash
srun -A project_465003400 -p small-g -n 1 --gpus-per-task=1 --time=00:10:00 \
  singularity run "$SIF" python -c "import torch; print(torch.cuda.device_count())"
```

## 3. Training data format (Guide 03)

Lustre metadata servers stall when a job stats a million JPEGs. Convert
before training, or read an archive that is already one file. The guide's
Tiny-ImageNet DataLoader times (100k images, batch 32, 7 workers):

| Format | Mean (s) | Notes |
|--------|----------|--------|
| LMDB | 31.7 | No shape restriction. Custom dataset. |
| HDF5 | 37.9 | Needs rectangular arrays. Awkward for variable-size images. |
| squashfs | 49.2 | Mount and use `ImageFolder`. Slowest of the three. |

On ImageNet-1k (200k of 1.2M images) LMDB was about 28% faster than squashfs
in the guide (1547 s vs 1982 s). HDF5 was dropped there because image sizes
differ.

This repo's order inside `cluster/simclr.sbatch`:

1. **LMDB** if `/scratch/project_465003400/datasets/IN1K-lmdb/train/data.mdb`
   exists. Two files, random read, no mount.
2. Else **stage** the Hugging Face parquet cache to `/tmp/IN1K-jpeg` and use
   `ImageFolder`. Needs `--mem=0` so tmpfs can hold ~150 GB.
3. A plain `train/` tree on Lustre is the fallback. Avoid it for real runs.

Do not loop-mount `IN1K-jpeg.sqsh` for training. The kernel squashfs cache
serializes readers. Pack and bench details, including the numbers from jobs
`22191062`, `22191637`, and `22191801`, are in [in1k-loading.md](in1k-loading.md).

Build the LMDB once:

```bash
sbatch cluster/pack_imagenet_lmdb.sbatch
```

Official conversion scripts (HDF5 / LMDB / squashfs) are under
[03-file-formats](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/03-file-formats).

## 4. Where to put bytes (Guide 04)

`/project` and `/scratch` on this machine are both Lustre (LUMI-P). LUMI-F
(`/flash/<project>`) is the fast tier: about 10× the bandwidth of LUMI-P and
about 3× the storage billing. We do not use it. Billing rules:
<https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/#storage-billing>.

| Tier | Path here | Use |
|------|-----------|-----|
| LUMI-P project | `/project/project_465003400/SSLHydra` | Git checkout, sbatch scripts |
| LUMI-P scratch | `/scratch/project_465003400/` | Datasets, venv, checkpoints, caches. Files with no access for 90 days can be removed. |
| Node tmpfs | `/tmp` on the compute node | MIOpen DB, DataLoader shm, staged JPEG tree. Wiped when the job ends. `with_miopen.sh` copies the Find DB back to scratch. |
| LUMI-F | `/flash/project_465003400` | Not used. Move data in only for the run, then out. |
| Object store | LUMI-O | Poor throughput for training reads. Not used. |

LUMI-P has 32 OSTs, ~161 MB/s each from a compute node. A large file striped
across 32 OSTs can do about 16 × 161 MB/s with 16 concurrent readers. Set
the stripe **before** the file has data:

```bash
lfs setstripe --stripe-count 32 --stripe-size 4m /path/to/empty-dir-or-file
```

LUMI-F has 58 OSTs at ~1710 MB/s each; stripe count 58 or a multiple.
`/tmp` on `standard-g` reads at ~7800 MB/s and writes at ~2900 MB/s, and
that does not scale with more threads. It is per-node: a multi-node job
must stage on every node.

## 5. Multi-GPU and multi-node (Guide 05)

SimLAP is data parallel. The model fits on one GCD, so the job is PyTorch
DDP via `torchrun`, not DeepSpeed.

`cluster/jobs/train_cl.sh` runs, inside the container:

```bash
python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 \
  main_cl.py --method "$METHOD" --data "$IMAGENET_ROOT" ...
```

One Slurm task per node (`--ntasks-per-node=1`, 56 CPUs). `torchrun` forks
the 8 processes. That matches the guide's `run_ddp_torchrun.sh`.

Same setup for an arbitrary program (`cluster/lumi_ddp.sh`): environment from
`cluster/lumi.env`, container, ImageNet (LMDB, else parquet staged to
`/tmp`, else `ImageFolder`), MIOpen / torch / HF caches, then `torchrun`.
A literal `$IMAGENET_ROOT` argument is replaced inside the job.

```bash
./cluster/lumi_ddp.sh src/main_cl.py --data '$IMAGENET_ROOT' --epochs 100
NNODES=2 TIME=48:00:00 ./cluster/lumi_ddp.sh src/main_cl.py --data '$IMAGENET_ROOT'
```

The code side of DDP, if you add a new entry point:

- `dist.init_process_group(backend="nccl")` (on ROCm, NCCL is RCCL)
- `local_rank = int(os.environ["LOCAL_RANK"])` and `torch.cuda.set_device(local_rank)`
- wrap the model in `DistributedDataParallel`
- `DistributedSampler` on the loader; call `set_epoch` each epoch

Batch sizes already wired in `train_cl.sh`:

| Method | Global batch | Notes |
|--------|--------------|--------|
| SimCLR | 8 × 512 = 4096 | fp32 + gradient checkpointing |
| SupCon, SimLAP | 4096 | fp16 AMP. Two nodes → 16 × 256. |
| X-CLR | 8 × 128 = 1024 | `cluster/xclr.sbatch` |

Multi-node: `NNODES=2 ./cluster/submit_simclr.sh 100`. The sbatch script
takes rank 0's hostname as `RDZV_HOST` and launches one `srun` task per node
with `--cpu-bind=none`. `train_cl.sh` then uses `c10d` rendezvous. For a
multi-node job also export `FI_MR_CACHE_MONITOR=userfaultfd` (see section 2).
RCCL over Slingshot: <https://github.com/HewlettPackard/shs-ccl-docs/blob/main/rccl/rccl_tuning_guide.md>.

CPU–GCD binding, from the guide. `torchrun` can do it with
`--numa-binding=exclusive`. The `srun`-per-rank style (8 tasks per node,
7 CPUs each) uses this mask:

```bash
CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"
srun --cpu-bind=v,mask_cpu=$CPU_BIND_MASKS ...
```

Our scripts do not pass that mask. `RANK` and `LOCAL_RANK` exist only inside
the `srun` step, so they are exported in the command string, not in the
batch script.

Not all of a GCD's 64 GB shows up in `torch.cuda.memory_allocated()`. RCCL
buffers sit outside that counter and grow with the number of communicators.
Leave headroom. DeepSpeed ZeRO is in the guide
([chapter 05](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/05-multi-gpu-and-node))
and in the `full` image. This repo does not use it.

## 6. Watching a job (Guide 06)

```bash
squeue --me
squeue -j <jobid>
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,NodeList -P
tail -n 80 /scratch/project_465003400/SSLHydra/outputs/slurm/<jobname>-<jobid>.err
```

GPU occupancy from the node that is already running the job:

```bash
srun --jobid <jobid> --overlap --pty rocm-smi
```

`rocm-smi` utilization stays at 100% while a kernel waits on data. Power is
the useful signal. One busy GCD draws on the order of 300 W. A busy MI250X
(two GCDs, one power reading) draws about 500 W, which is why only four of
the eight devices show a power number.

PyTorch profiler, short window only (a full-epoch Chrome trace exceeds the
~2 GB tab limit). Open the file at <https://ui.perfetto.dev/>.

```python
from torch.profiler import profile, ProfilerActivity
# around the step you care about, not the whole run:
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    ...
prof.export_chrome_trace("trace.json")
```

Hardware counters (`rocprof`, Omnitrace, Omniperf) are covered in the
[LUMI AI workshop](https://lumi-supercomputer.github.io/LUMI-training-materials/ai-20241126/extra_04_CheckingGPU/)
and the [performance workshop](https://lumi-supercomputer.github.io/LUMI-training-materials/paow-20240611/).

## 7. Metrics (Guide 07–09)

Training logs to Weights & Biases when `WANDB_MODE=online` (the default in
`cluster/lumi.env`: project `hydra`, entity `erow`). Rank 0 only. Offline:

```bash
WANDB_MODE=offline ./cluster/submit_simclr.sh smoke
```

`train_cl.sh` passes `--no-wandb` unless the mode is `online`. Put the API
key in the environment or `~/.netrc` on LUMI. Do not commit it.

TensorBoard is installed in the overlay venv. The guide's pattern is a
`SummaryWriter` on rank 0 only. The LUMI web UI
(<https://docs.lumi-supercomputer.eu/runjobs/webui/>) has a TensorBoard app:
point it at the log directory and give it RAM rather than more CPUs.
MLflow is [chapter 08](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/08-MLflow-visualization)
of the guide. We do not run a tracking server here.

## 8. Chapter we do not use

[Chapter 10](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/10-LLM-inference)
is vLLM inference on the same `lumi-multitorch-full` image, with
`VLLM_CACHE_ROOT` on scratch. Nothing in SimLAP serves a language model.

## Guide index

| Chapter | What we took from it |
|---------|----------------------|
| [01 QuickStart](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/01-quickstart) | `sbatch` on LUMI-G, account, log file next to the job |
| [02 Environment](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/02-setting-up-environment) | `lumi-aif-singularity-bindings`, overlay venv, MIOpen and cache paths |
| [03 File formats](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/03-file-formats) | LMDB over squashfs over a million JPEGs |
| [04 Storage](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/04-data-storage) | LUMI-P vs `/tmp`, striping |
| [05 Multi-GPU](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/05-multi-gpu-and-node) | One task per node, `torchrun`, 8 GCDs, rendezvous |
| [06 Profiling](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/06-monitoring-and-profiling) | `rocm-smi` power, short PyTorch traces |
| [07 TensorBoard](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/07-TensorBoard-visualization) | Rank-0 writer, web UI |
| [09 Wandb](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/09-Wandb-visualization) | `WANDB_MODE` / project / entity in `lumi.env` |

Further reading from the guide: [LUMI docs](https://docs.lumi-supercomputer.eu/),
[AI Factory services](https://docs.lumi-supercomputer.eu/software/local/lumi-aif/),
[PyTorch on LUMI](https://docs.lumi-supercomputer.eu/laif/software/ai-environment/).
