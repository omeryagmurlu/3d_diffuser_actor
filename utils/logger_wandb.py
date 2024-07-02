import wandb


class WandBLogger():
    def __init__(self, args):
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            # dir=args.wandb_dir,
        )

    def log(self, metrics, **kwargs):
        wandb.log(metrics, **kwargs)

    def finish(self):
        wandb.finish()

    def log_summary(self, summary_key, summary_value):
        wandb.run.summary[summary_key] = summary_value
