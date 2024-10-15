import wandb
import torch.distributed as dist
import einops

class WandbWriter:
    def __init__(self, *, config, entity, project_name, run_name, tags=None):
        if dist.get_rank() != 0:
            self.run = None
            return

        self.run = wandb.init(project=project_name, name=run_name, tags=tags, config=config, entity=entity)
        wandb.define_metric("*", step_metric="train_step")

    def add_scalar(self, name, value, step):
        if self.run is not None:
            self.run.log({name: value, "train_step": step})

    def add_image(self, name, image, step):
        if self.run is not None:
            self.run.log({name: [wandb.Image(einops.rearrange(image, 'c h w -> h w c'))], "train_step": step})

    def watch_model(self, model):
        if self.run is not None:
            wandb.watch(model)