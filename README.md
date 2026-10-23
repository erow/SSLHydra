# SSLHydra

Self-supervised learning with Hydra, a pair-conditioned gated contrastive objective, and iBOT. This repository includes the training implementation, ImageNet kNN evaluation, LUMI launch scripts, and ablation records.

The training code builds on [iBOT](https://github.com/bytedance/ibot). Its original documentation and installation instructions are in [src/ibot/README.md](src/ibot/README.md) and [src/ibot/INSTALL.md](src/ibot/INSTALL.md).

## Layout

- [src/ibot](src/ibot): training, models, Hydra loss, and regression checks.
- [scripts](scripts): weighted kNN evaluation, descriptor extraction, and experiment submission.
- [cluster](cluster): LUMI Slurm/container launcher and environment configuration.
- [docs/hydra.md](docs/hydra.md): objective and implementation.
- [docs/ablation.md](docs/ablation.md): experiment settings, results, and run records.
- [docs/runs.md](docs/runs.md): tuning notes.

## Training and evaluation

The launch scripts contain paths and account settings for the existing LUMI deployment. Configure `cluster/lumi.env` and the paths in the submission scripts for your installation. They require a compatible PyTorch environment, FFCV ImageNet files, and, for Hydra, the precomputed reference descriptor table. See the iBOT installation guide above for the base dependencies.

Preview a 100-epoch ablation command:

```bash
STAGE=2 ROWS=B2i-ibot bash scripts/submit_ablation.sh
```

Submit training and a dependent kNN evaluation with the locally configured environment:

```bash
STAGE=2 ROWS=B2i-ibot SUBMIT=1 USE_FP16=false WANDB_TAGS=abl-b2i \
STUDY_SRC=/path/to/this/repository/src/ibot \
bash scripts/submit_ablation.sh
```

Run weighted kNN on teacher features:

```bash
bash cluster/lumi_ddp.sh python scripts/knn_eval_ffcv.py \
  --arch vit_small --nb_knn 10 20 50 \
  --checkpoints model=/path/to/checkpoint.pth \
  --output /path/to/knn.json
```

The current implementation includes full-precision checkpoint resumption and a patch projection that avoids the fp32 MIOpen convolution kernel failure observed on LUMI. Experiment results are single-seed unless stated otherwise; consult the ablation document for training budgets and comparison limits.

## License

[Apache License 2.0](LICENSE). Original iBOT attribution is retained in the source files and upstream documentation.
