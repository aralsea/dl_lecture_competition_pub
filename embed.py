import hydra
import torch
from omegaconf import DictConfig

from src.library.datasets import ThingsMEGDatasetWithImages
from src.library.utils import get_model_id, set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config-baseline")
def run(args: DictConfig) -> None:
    set_seed(args.seed)
    # ------------------
    #    Dataloader
    # ------------------

    train_set = ThingsMEGDatasetWithImages("train", args.data_dir)
    val_set = ThingsMEGDatasetWithImages("val", args.data_dir)

    # ------------------
    #       Model
    # ------------------

    image_module = torch.hub.load(args.image_module.repo, args.image_module.model)
    model_id = get_model_id(args)
    train_set.save_embedded_images(image_module, model_id=model_id)
    val_set.save_embedded_images(image_module, model_id=model_id)


if __name__ == "__main__":
    run()
