"""Project CLI; run python main.py --help for options."""
import argparse
from src.runtime import configure_runtime


def main():
    parser = argparse.ArgumentParser(description='PV forecasting with CNN-LSTM and XGBoost')
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--model', choices=['CNN_LSTM', 'XGBoost', 'all'])
    parser.add_argument('--physics', choices=['on', 'off'], help='Override the physical-feature switch for both model families')
    parser.add_argument('--smoke', action='store_true', help='At most 240 rows, PRE=2, H=1, one epoch / five trees')
    args = parser.parse_args()
    configure_runtime()
    from src.config import load_config
    from src.train import run
    config = load_config(args.config)
    if args.physics is not None:
        config['data']['physics'] = args.physics == 'on'
    names = ['CNN_LSTM', 'XGBoost'] if args.model == 'all' else [args.model or config['model']]
    for name in names:
        run(config, name, smoke=args.smoke)


if __name__ == '__main__':
    main()
