"""Prepare data, train forecasting models and evaluate frozen winners."""

import argparse
from src.runtime import configure_runtime


def main():
    """Parse the requested stage and dispatch it with runtime settings applied."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage')
    prepare = commands.add_parser(
        'prepare', help='Check data and construct the four feature spaces; no training'
    )
    prepare.add_argument(
        '--config', default='config/config.yaml', help='YAML configuration path'
    )
    prepare.add_argument(
        '--smoke',
        action='store_true',
        help='Use the small diagnostic config for this entire study',
    )

    stage_help = {
        'train': 'Search on TRAIN/VAL and freeze all space winners',
        'evaluate': 'Evaluate frozen winners on TEST once',
    }
    for stage, description in stage_help.items():
        command = commands.add_parser(stage, help=description)
        command.add_argument(
            '--resume', required=True, help='manifest.json printed by prepare'
        )
        if stage == 'train':
            command.add_argument(
                '--model', choices=('XGBoost', 'CNN_LSTM'),
                help='Required on first train; saved model is used on resume',
            )
    args = parser.parse_args()
    if args.stage is None:
        parser.print_help()
        return
    configure_runtime()
    from src.config import load_config, smoke_config
    from src.experiments import run

    if args.stage == 'prepare':
        config = load_config(args.config)
        run('prepare', config=smoke_config(config) if args.smoke else config)
    else:
        run(args.stage, manifest=args.resume, model=getattr(args, 'model', None))


if __name__ == '__main__':
    main()
