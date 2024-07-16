import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from lib.datasets import ThingsMEGDatasetWithImages
from lib.utils import set_seed
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="config-baseline")
def run(args: DictConfig) -> None:
    set_seed(args.seed)
    logdir = HydraConfig.get().runtime.output_dir

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    train_set = ThingsMEGDatasetWithImages("train", args.data_dir)
    val_set = ThingsMEGDatasetWithImages("val", args.data_dir)

    # ------------------
    #       Model
    # ------------------
    image_module = torch.hub.load(args.image_module.repo, args.image_module.model)

    train_set.save_embedded_images(
        image_module, args.image_module.repo + "-" + args.image_module.model
    )
    val_set.save_embedded_images(
        image_module, args.image_module.repo + "-" + args.image_module.model
    )


if __name__ == "__main__":
    run()
