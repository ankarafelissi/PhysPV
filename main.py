"""Project CLI; run python main.py --help for options."""
import argparse
from src.runtime import configure_runtime


def main():
    parser = argparse.ArgumentParser(description='PV forecasting with CNN-LSTM and XGBoost')
    parser.add_argument('--config', default='config/config.yaml', help='YAML configuration path')
    parser.add_argument('--model', choices=['CNN_LSTM', 'XGBoost', 'all'])
    parser.add_argument('--physics', choices=['on', 'off'], help='Override the physical-feature switch for both model families')
    parser.add_argument('--smoke', action='store_true', help='At most 240 rows, PRE=2, H=1, one epoch / five trees')
    parser.add_argument('--feature-study', '--optimize', dest='feature_study', action='store_true',
                        help='Run joint TPE across Original/Pearson/Physics/Intrinsic')
    parser.add_argument('--evaluate-test', action='store_true', help='Evaluate TEST after the feature study selection')
    args = parser.parse_args()
    configure_runtime()
    from src.config import load_config
    from src.train import run
    config = load_config(args.config)
    if args.feature_study or not (args.model or args.physics or args.smoke):
        from src.experiments import run as experiment
        if args.model or args.physics:
            parser.error('--feature-study controls both models and all feature arms; omit --model/--physics')
        if args.smoke:
            parser.error('Use python -m src.experiments --smoke for the workflow diagnostic')
        experiment(config, evaluate_test=args.evaluate_test)
        return
    if args.evaluate_test:
        parser.error('--evaluate-test requires --feature-study')
    if args.physics is not None:
        from src.data import CANDIDATE_FEATURES
        enabled = args.physics == 'on'
        config['data']['physics'] = enabled
        config['data']['candidate_subset'] = list(CANDIDATE_FEATURES) if enabled else []
    names = ['CNN_LSTM', 'XGBoost'] if args.model == 'all' else [args.model or config['model']]
    for name in names:
        run(config, name, smoke=args.smoke)


if __name__ == '__main__':
    main()
