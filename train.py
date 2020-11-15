import json
import argparse
from src.model_handler.TrainHandler import start_training


def parse_args():
    parser = argparse.ArgumentParser(description="Adipocyte Fluorescence Predictor CLI Tool")
    parser.add_argument("-s", "--setting_file", type=str,
                        help="JSON filepath that contains settings.")
    args = parser.parse_args()
    print(args)
    return args


def get_settings(json_path):
    with open(json_path, "r") as json_file:
        settings = json.load(json_file)
    print(settings)
    return settings


def main():
    args = parse_args()
    settings = get_settings(args.setting_file)
    start_training(**settings)


if __name__ == "__main__":
    main()