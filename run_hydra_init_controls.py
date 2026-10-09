"""Strict four-way Hydra initialization control on a stabilized schedule."""

import argparse
import os
import sys
from pathlib import Path


PRETRAINED = (
    '/scratch/project_465003400/SSLHydra/outputs/'
    'hydra-k8-fneg-t02-ibot-frozen/checkpoint0009.pth'
)
OUTPUT_ROOT = Path('/scratch/project_465003400/SSLHydra/outputs/hydra-init-controls-10ep')
DATA = '/scratch/project_465003400/datasets/ffcv/IN1K_train_smart500_q90.ffcv'
CONFIGS = [
    ('random_all', None),
    ('pretrained_gate_only', 'gate'),
    ('pretrained_projectors_only', 'projectors'),
    ('pretrained_full_hydra', 'all'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=int,
                        choices=range(len(CONFIGS)))
    args = parser.parse_args()
    name, parts = CONFIGS[args.config]
    output = OUTPUT_ROOT / name
    output.mkdir(parents=True, exist_ok=True)
    os.environ['WANDB_NAME'] = f'hydra-init-control-{name}'

    command = [
        sys.executable, str(Path(__file__).with_name('main_ibot.py')),
        '--arch', 'vit_base',
        '--epochs', '10',
        '--warmup_epochs', '2',
        '--warmup_teacher_temp_epochs', '1',
        '--lr', '0.00005',
        '--weight_decay', '0.04',
        '--weight_decay_end', '0.04',
        '--momentum_teacher', '0.9995',
        '--loss', 'hydra',
        '--hydra_pair', 'sim',
        '--hydra_k', '8',
        '--hydra_out_dim', '2048',
        '--hydra_temperature', '0.2',
        '--hydra_tau', '0.1',
        '--hydra_filter_negatives', 'false',
        '--hydra_lr_scale', '0.1',
        '--saveckp_freq', '1',
        '--num_workers', '10',
        '--seed', '0',
        '--data_path', DATA,
        '--output_dir', str(output),
    ]
    if parts is not None:
        command.extend([
            '--init_hydra', PRETRAINED,
            '--init_hydra_parts', parts,
        ])
    print(
        f'INIT_CONTROL config={args.config} name={name} parts={parts or "none"}',
        flush=True,
    )
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
