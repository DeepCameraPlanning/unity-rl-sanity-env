from typing import Sequence

from omegaconf import DictConfig, OmegaConf
import rich.tree
import rich.syntax


def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        # "compnode",
        "model",
        "env",
        "datamodule",
        "xp_name",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """
    Adapted from: https://github.com/ashleve/lightning-hydra-template.
    Prints content of DictConfig using Rich library and its tree structure.
    :param config: configuration composed by Hydra.
    :param fields: determines which main fields from config will be printed and
        in what order.
    :param resolve: whether to resolve reference fields of DictConfig.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)
