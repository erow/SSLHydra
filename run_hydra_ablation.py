"""Ten-epoch Hydra ablation matrix for a global batch size of 1024."""

import argparse
import os
import sys
from pathlib import Path


PRETRAINED = (
    '/scratch/project_465003400/SSLHydra/outputs/'
    'hydra-k8-fneg-t02-ibot-frozen/checkpoint0009.pth'
)
OUTPUT_ROOT = Path('/scratch/project_465003400/SSLHydra/outputs/hydra-ablation-10ep')
DATA = '/scratch/project_465003400/datasets/ffcv/IN1K_train_smart500_q90.ffcv'

# One-factor-at-a-time around config 0.  The two out-dimension variants are
# necessarily random-head controls because the available pretrained head is
# 2048-dimensional.
CONFIGS = [
    ('base_pre_m9995_d2048_k8_t20_tau10_hlr10', {}),
    ('random_head', {'pretrained': False}),
    ('momentum_996', {'momentum_teacher': 0.996}),
    ('momentum_999', {'momentum_teacher': 0.999}),
    ('outdim_768_random', {'pretrained': False, 'hydra_out_dim': 768}),
    ('outdim_4096_random', {'pretrained': False, 'hydra_out_dim': 4096}),
    ('k_1', {'hydra_k': 1}),
    ('k_4', {'hydra_k': 4}),
    ('temperature_010', {'hydra_temperature': 0.1}),
    ('temperature_040', {'hydra_temperature': 0.4}),
    ('tau_005', {'hydra_tau': 0.05}),
    ('tau_020', {'hydra_tau': 0.2}),
    ('hydra_lr_003', {'hydra_lr_scale': 0.03}),
    ('hydra_lr_030', {'hydra_lr_scale': 0.3}),
    ('filter_negatives', {'hydra_filter_negatives': True}),
    ('force_ones_gate', {'hydra_force_ones': True}),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=int,
                        choices=range(len(CONFIGS)))
    args = parser.parse_args()
    name, override = CONFIGS[args.config]
    cfg = {
        'momentum_teacher': 0.9995,
        'hydra_out_dim': 2048,
        'hydra_k': 8,
        'hydra_temperature': 0.2,
        'hydra_tau': 0.1,
        'hydra_lr_scale': 0.1,
        'hydra_filter_negatives': False,
        'hydra_force_ones': False,
        'pretrained': True,
    }
    cfg.update(override)
    output = OUTPUT_ROOT / name
    output.mkdir(parents=True, exist_ok=True)
    os.environ['WANDB_NAME'] = f'hydra-ab10-{name}'

    command = [
        sys.executable, str(Path(__file__).with_name('main_ibot.py')),
        '--arch', 'vit_base',
        '--epochs', '10',
        '--warmup_epochs', '1',
        '--loss', 'hydra',
        '--hydra_pair', 'sim',
        '--hydra_k', str(cfg['hydra_k']),
        '--hydra_out_dim', str(cfg['hydra_out_dim']),
        '--hydra_temperature', str(cfg['hydra_temperature']),
        '--hydra_tau', str(cfg['hydra_tau']),
        '--hydra_filter_negatives', str(cfg['hydra_filter_negatives']).lower(),
        '--hydra_force_ones', str(cfg['hydra_force_ones']).lower(),
        '--hydra_lr_scale', str(cfg['hydra_lr_scale']),
        '--momentum_teacher', str(cfg['momentum_teacher']),
        '--saveckp_freq', '1',
        '--num_workers', '10',
        '--data_path', DATA,
        '--output_dir', str(output),
    ]
    if cfg['pretrained']:
        command.extend(['--init_hydra', PRETRAINED])
    print(f'ABLATION config={args.config} name={name} values={cfg}', flush=True)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
