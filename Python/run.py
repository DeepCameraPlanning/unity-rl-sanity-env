import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    from src.train import train
    from src.utils import utils

    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    if config.run_type == "train":
        train(config)


if __name__ == "__main__":
    main()
