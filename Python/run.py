import hydra
from omegaconf import DictConfig

def test():
    from src.utils.utils import testViz
    testViz()

@hydra.main(config_path="configs/", config_name="config_og.yaml")
def main(config: DictConfig):
    from src.infer import infer
    from src.train import train
    from src.utils import utils
    from src.models.modules.modelUtils import versionControl
  
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # setting version 
    
    versionControl.set_version_str(config.versionCam,
                                   config.versionObs)
    
    if config.run_type == "train":
        train(config)

    if config.run_type == "infer":
        infer(config)


if __name__ == "__main__":
    # test()
    #
    main()
