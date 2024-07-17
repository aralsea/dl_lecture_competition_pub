import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.library.datasets import ThingsMEGDatasetWithImages
from src.library.image_module import get_image_module
from src.library.utils import get_model_id, set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config-resnet152")
def run(args: DictConfig) -> None:
    model_id = get_model_id(args)
    set_seed(args.seed)
    logdir = HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------

    train_set = ThingsMEGDatasetWithImages(
        "train",
        args.data_dir,
        # embedding_model_id=args.image_module.repo + "-" + args.image_module.model,
    )

    val_set = ThingsMEGDatasetWithImages(
        "val",
        args.data_dir,
        # embedding_model_id=args.image_module.repo + "-" + args.image_module.model,
    )

    # ------------------
    #       Model
    # ------------------
    image_module = get_image_module(args)
    for param in image_module.parameters():
        param.requires_grad = False

    train_set.save_embedded_images(image_module, model_id)
    val_set.save_embedded_images(image_module, model_id)


if __name__ == "__main__":
    run()
