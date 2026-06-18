import os

import lightning as L
import torch
import wandb
from hydra.utils import get_original_cwd, to_absolute_path
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ThroughputMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import TorchElasticEnvironment
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf
from transformers import get_inverse_sqrt_schedule, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from uniflowmatch.datasets import *
from uniflowmatch.loss import get_loss
from uniflowmatch.models import (
    LoRAUniFlowMatch,
    UniFlowMatch,
    UniFlowMatchClassificationRefinement,
    UniFlowMatchConfidence,
)
from uniflowmatch.models.base import UniFlowMatchModelsBase
from uniflowmatch.models.ufm import is_symmetrized
from uniflowmatch.utils.misc import hash_to_port
from uniflowmatch.utils.viz import get_visualizer


class LightningModel(L.LightningModule):
    def __init__(self, model: UniFlowMatchModelsBase, config):
        super().__init__()
        self.model = model
        self.config = config

        self.train_supervisions = {
            name: get_loss(value["class"], **value["kwargs"]) for name, value in config["loss"]["train_loss"].items()
        }

        self.test_supervisions = {
            name: get_loss(value["class"], **value["kwargs"]) for name, value in config["loss"]["test_loss"].items()
        }

        print("Supervisions used(training): ", self.train_supervisions)
        print("Supervisions used(test): ", self.test_supervisions)

        self.visualizers = (
            {name: get_visualizer(viz_config) for name, viz_config in config["visualizer"]["visualizers"].items()}
            if "visualizer" in config
            else {}
        )

        self.train_viz_step = 0
        self.val_viz_step = 0

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        results = self.model(*batch)
        with torch.autocast("cuda", enabled=False):
            loss_dict = {}
            log_dict = {}
            for name, supervision in self.train_supervisions.items():
                enabled, supervision_loss, supervision_info = supervision.compute_loss(batch, results)
                loss_dict[name] = supervision_loss
                log_dict.update(supervision_info)
                assert enabled, f"Loss {name} not enabled"
            loss = sum(loss_dict.values())

        # Hack, remove me
        viz_dict = {k: v for k, v in log_dict.items() if "viz" in k}
        other_dict = {k: v for k, v in log_dict.items() if "viz" not in k}

        self.log_dict(other_dict, on_step=True)
        self.visualizer_step(batch, results, training=True, viz_dict=viz_dict)

        print("Training step loss: ", loss_dict)

        if loss.isnan():
            print("Training loss is nan: ", loss_dict)
            return torch.tensor(0.0, device=loss.device, requires_grad=True)

        del batch
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        results = self.model(*batch)
        with torch.autocast("cuda", enabled=False):
            loss_dict = {}
            log_dict = {}
            for name, supervision in self.test_supervisions.items():
                # import pdb; pdb.set_trace()
                enabled, supervision_loss, supervision_info = supervision.compute_loss(batch, results)
                loss_dict[name] = supervision_loss
                log_dict.update(supervision_info)
                assert enabled, f"Loss {supervision.name} not enabled"
            loss = sum(loss_dict.values())

        # append _val to all keys in the log_dict
        log_dict = {f"{key}_val": value for key, value in log_dict.items()}

        # Hack, remove me
        viz_dict = {k: v for k, v in log_dict.items() if "viz" in k}
        other_dict = {k: v for k, v in log_dict.items() if "viz" not in k}

        dataset_name = list(self.trainer.datamodule.val_datasets.keys())[dataloader_idx]
        updated_log_dict = {f"{dataset_name}_{key}": value for key, value in other_dict.items()}

        self.log_dict(updated_log_dict, on_step=True, on_epoch=True, add_dataloader_idx=False, sync_dist=True)
        self.visualizer_step(batch, results, training=False, viz_dict=viz_dict)

        if loss.isnan():
            print("Validation loss is nan: ", loss_dict)

        torch.cuda.empty_cache()
        return loss

    def visualizer_step(self, batch, model_result, training=True, viz_dict=None):
        # decide if we should visualize
        mode = self.trainer.state.fn
        interval = self.config["visualizer"][mode]["interval"]
        num_samples = self.config["visualizer"][mode]["num_samples"]

        if training:
            self.train_viz_step += 1
        else:
            self.val_viz_step += 1

        cur_step = self.train_viz_step if training else self.val_viz_step

        if cur_step % interval == 0 and (self.trainer.global_rank == 0):
            if num_samples == "all":
                num_samples = len(batch[0]["img"])

            append_name = "train" if training else "val"

            if self.config["disable_wandb"]:
                return

            for batch_idx in range(num_samples):
                for name, visualizer in self.visualizers.items():
                    image = visualizer.visualize(batch_idx, model_result, batch)
                    wandb.log({f"{name}_{append_name}": wandb.Image(image)}, commit=True)

            if viz_dict is not None:
                for name, value in viz_dict.items():
                    wandb.log({f"{name}_{append_name}": value}, commit=True)

    def on_train_epoch_start(self):
        train_dataset = self.trainer.datamodule.training_dataset
        if hasattr(train_dataset, "set_epoch"):
            if self.current_epoch > 0:
                # we have already set the epoch to 0 in the datamodule setup
                # setting it repeatedly may reset the seed when the loaders
                # are working.
                train_dataset.set_epoch(self.current_epoch)
                self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)
                print("Setting train dataset & sampler epoch to ", self.current_epoch)

    def on_validation_epoch_start(self):
        val_datasets = self.trainer.datamodule.val_datasets
        for name, val_dataset in val_datasets.items():
            if hasattr(val_dataset, "set_epoch"):
                if self.current_epoch > 0:
                    val_dataset.set_epoch(self.current_epoch)

        for name, dataloader in self.trainer.val_dataloaders.items():
            dataloader.sampler.set_epoch(self.current_epoch)
            print("Setting val dataset & sampler epoch to ", self.current_epoch)

    def configure_optimizers(self):
        training_scheme = self.config["training_scheme"]

        # obtain the parameter groups from the model
        param_groups = self.model.get_parameter_groups()

        # add learning rate configs to it
        param_groups = {k: (v, self.config["training_scheme"]["learning_rate"][k]) for k, v in param_groups.items()}

        # check that all the model's parameter are covered by the param_groups
        all_params = []
        for name, (params, lr) in param_groups.items():
            all_params.extend(params)

        assert set(all_params) == set(
            self.model.parameters()
        ), "The parameter group have missed some parameters, please check your partitioning"

        # Skip computing gradients for parameters with lr=0 to save memory
        for name, (params, lr) in param_groups.items():
            if lr == 0:
                print(f"Freezing parameters in {name}")
                for param in params:
                    param.requires_grad = False

        # create the optimizer and add weights with lr > 0
        optimizer = torch.optim.AdamW(
            [
                {"params": param, "lr": lr, "name": group_name}
                for group_name, (param, lr) in param_groups.items()
                if lr > 0
            ],
            betas=training_scheme["betas"],
            weight_decay=training_scheme["weight_decay"],
        )

        # create the learning rate scheduler uniformly for each component
        scheduler_args = training_scheme["lr_scheduler"]

        if scheduler_args["scheduler"] is None:
            return optimizer
        elif scheduler_args["scheduler"] == "cosine":
            scheduler_kwargs = scheduler_args["scheduler_kwargs"]
            iteration_per_epoch = (
                len(self.trainer.datamodule.train_dataloader()) // self.trainer.accumulate_grad_batches
            )

            scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                optimizer,
                num_warmup_steps=scheduler_kwargs["num_warpup_epochs"] * iteration_per_epoch,
                num_training_steps=scheduler_kwargs["num_epochs"] * iteration_per_epoch,
                min_lr_rate=scheduler_kwargs["min_lr_rate"],
            )

            # configure scheduler to be called every epoch in PL syntax
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        elif scheduler_args["scheduler"] == "inverse_sqrt":
            scheduler_kwargs = scheduler_args["scheduler_kwargs"]
            iteration_per_epoch = (
                len(self.trainer.datamodule.train_dataloader()) // self.trainer.accumulate_grad_batches
            )

            scheduler = get_inverse_sqrt_schedule(
                optimizer,
                num_warmup_steps=scheduler_kwargs["num_warpup_epochs"] * iteration_per_epoch,
                timescale=scheduler_kwargs["timescale_epochs"] * iteration_per_epoch,
            )

            # configure scheduler to be called every epoch in PL syntax
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        elif scheduler_args["scheduler"] == "linear":
            scheduler_kwargs = scheduler_args["scheduler_kwargs"]
            iteration_per_epoch = (
                len(self.trainer.datamodule.train_dataloader()) // self.trainer.accumulate_grad_batches
            )

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=scheduler_kwargs["num_warpup_epochs"] * iteration_per_epoch,
                num_training_steps=scheduler_kwargs["num_epochs"] * iteration_per_epoch,
            )

            # configure scheduler to be called every epoch in PL syntax
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

    # doc: (configure_model is a) Hook to create modules in a strategy and precision aware context.
    def configure_model(self):
        # According to Pytorch Lightning doc, calling torch.compile here can work with DDP
        # https://lightning.ai/docs/pytorch/stable/advanced/compile.html#apply-torch-compile-in-configure-model

        if self.config["training_scheme"]["torch_compile"]["enable_torch_copile"]:

            print("Compiling model with torch.jit")

            self.model.compile(
                dynamic=self.config["training_scheme"]["torch_compile"]["dynamic"],
                fullgraph=self.config["training_scheme"]["torch_compile"]["fullgraph"],
                options=self.config["training_scheme"]["torch_compile"]["compile_options"],
            )

    def on_after_backward(self):
        # Debug for gradient mismatch
        # target_size = (256, 256, 1, 1)
        # target_stride = (256, 1, 256, 256)

        # for name, param in self.model.named_parameters():
        #     grad = param.grad
        #     if grad is None:
        #         continue
        #     if grad.size() == target_size and grad.stride() == target_stride:
        #         print(f"Found parameter with incorrect gradient: '{name}'")
        #         print(f"Gradient shape: {grad.size()}, strides: {grad.stride()}")

        # debug for missing gradients
        # missing_grad_params = []
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad and param.grad is None:
        #         missing_grad_params.append(name)

        # if missing_grad_params:
        #     print("Parameters requiring gradients but missing gradients:")
        #     for name in missing_grad_params:
        #         print(f"  - {name}")
        # else:
        #     print("All parameters requiring gradients received gradients!")
        pass


class LightningDataModule(L.LightningDataModule):
    """
    For us, this class is a wrapper around the UFM dataset and sampler
    because we need pytorch lightning to handle initializing the DDP. (If not,
    we need to do it before constructing the sampler and pass the dataloader
    to lightning, and it will initialize the DDP again, causing errors)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage: str):
        if stage == "fit":
            self.training_dataset = eval(self.config["dataset"]["train_dataset"])
            self.training_dataset.set_epoch(0)

        val_dataset_str = self.config["dataset"]["test_dataset"]
        self.val_datasets = {
            dataset.split("(")[0]: eval(dataset) for dataset in val_dataset_str.split("+") if "(" in dataset
        }

        for k, v in self.val_datasets.items():
            v.set_epoch(0)
            print(f"Setting epoch for {k} to 0")

    def _interleave_imgs(self, img1, img2):
        res = {}
        for key, value1 in img1.items():

            value2 = img2[key]
            if isinstance(value1, torch.Tensor):
                value = torch.stack((value1, value2), dim=1).flatten(0, 1)
            else:
                value = [x for pair in zip(value1, value2) for x in pair]
            res[key] = value
        return res

    def make_batch_symmetric(self, batch):
        view1, view2 = batch
        view1, view2 = (self._interleave_imgs(view1, view2), self._interleave_imgs(view2, view1))
        return view1, view2

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        # compute flow & occlusion supervision from depthmap on GPU for speed
        batch = apply_flow_postprocessing_and_merge_batch(batch)

        # handle make symmetrized and collapse "data_norm_type" to a single value
        if self.config["training_scheme"]["symmetrize_inputs"]:
            batch = self.make_batch_symmetric(batch)

        # compute symmetrized
        is_symmetrized_ = is_symmetrized(*batch)

        for d in batch:
            d["symmetrized"] = [is_symmetrized_] * batch[0]["img"].shape[0]

        for i in range(len(batch)):
            assert (
                len(set(batch[i]["data_norm_type"])) == 1
            ), "Data normalization type should be the same for all images"
            batch[i]["data_norm_type"] = set(batch[i]["data_norm_type"]).pop()

            assert len(set(batch[i]["symmetrized"])) == 1, "Being symmetric should be the same for all images"
            batch[i]["symmetrized"] = set(batch[i]["symmetrized"]).pop()

        return batch

    def build_dataset(self, dataset, batch_size, num_workers, test=False, persistent_workers=False):
        split = ["Train", "Test"][test]
        print(f"Building {split} Data loader for dataset: ", dataset)
        loader = get_data_loader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_mem=True,
            shuffle=not (test),
            drop_last=not (test),
            persistent_workers=persistent_workers,
            collate_fn=collate_fn_with_delayed_flow_postprocessing,
        )

        print(f"{split} dataset length: ", len(loader))
        return loader

    def train_dataloader(self):

        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))

        batch_size_each_gpu = self.config["training_scheme"]["effective_batch_size"] // (
            num_nodes * torch.cuda.device_count()
        )
        dataloader = get_data_loader(
            self.training_dataset,
            batch_size_each_gpu,
            num_workers=self.config["dataset"]["num_workers"],
            shuffle=self.config["dataset"]["shuffle"],
            drop_last=self.config["dataset"]["drop_last"],
            pin_mem=self.config["dataset"]["pin_memory"],
            persistent_workers=True,  # keep the worker alive at the cost of memory
            collate_fn=collate_fn_with_delayed_flow_postprocessing,
        )

        dataloader.sampler.set_epoch(0)  # set epoch of the sampler
        return dataloader

    def val_dataloader(self):
        batch_size_each_gpu = 1  # self.config["training_scheme"]["effective_batch_size"] // torch.cuda.device_count()

        data_loader_val = {
            k: self.build_dataset(v, batch_size_each_gpu, self.config["dataset"]["num_workers"], test=True)
            for k, v in self.val_datasets.items()
        }

        for k, v in data_loader_val.items():
            v.sampler.set_epoch(0)  # set epoch of the sampler

        return data_loader_val


def train_pl_main(args):
    # torch.autograd.set_detect_anomaly(True) turn on if you face CUDA errors
    torch.backends.cudnn.benchmark = False
    # torch._dynamo.config.optimize_ddp = False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.set_float32_matmul_precision(
        "high"
    )  # highest=float32, high=tf32, medium=fp16. TF32 is approximately fp16 with larger range

    # set a random port between 20000 and 50000
    os.environ["MASTER_PORT"] = str(hash_to_port(os.path.basename(os.getcwd())) + 1)
    print("Using master port: ", os.environ["MASTER_PORT"], "hashed from ", os.getcwd())

    if "MASTER_ADDR" not in os.environ:
        # MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
        os.environ["MASTER_ADDR"] = os.environ.get("SLURM_NODELIST", "localhost").split(",")[0]
        print("Setting MASTER_ADDR to: ", os.environ["MASTER_ADDR"])

    # dump all args into a yaml file for inspection
    with open("config.yaml", "w") as f:
        OmegaConf.save(args, f)

    num_gpus = torch.cuda.device_count()
    os.environ["SLURM_NTASKS_PER_NODE"] = str(num_gpus)  # resolve PSC and lightning discripency on n_tasks

    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    os.environ["SLURM_NTASKS"] = str(num_nodes * num_gpus)  # resolve PSC and lightning discripency on n_tasks

    print("Number of nodes: ", num_nodes)
    print("N_TASKS_PER_NODE: ", os.environ["SLURM_NTASKS_PER_NODE"])
    print("N_TASKS: ", os.environ["SLURM_NTASKS"])
    print("N_GPUS: ", num_gpus)

    if num_nodes == 1:
        os.environ["SLURM_JOB_NAME"] = "bash"

    L.seed_everything(args["seed"])
    experiment_folder = os.getcwd()

    MODEL_CLASSES = {
        "UniFlowMatch": UniFlowMatch,
        "UniFlowMatchConfidence": UniFlowMatchConfidence,
        "UniFlowMatchClassificationRefinement": UniFlowMatchClassificationRefinement,
        "LoRAUniFlowMatch": LoRAUniFlowMatch,
    }

    raw_model = MODEL_CLASSES[args["model"]["model_class"]](**args["model"]["model_args"])

    model = LightningModel(
        raw_model,
        args,  # resolve args to get the actual values
    )

    datamodule = LightningDataModule(args)

    # optionally resume only model weight from a checkpoint
    if args["resume_model"] is not None:

        # special case if this is a HF repo
        if "infinity1096/UFM" in args["resume_model"]:
            if not "Refine" in args["resume_model"]:
                model_ref = UniFlowMatchConfidence.from_pretrained(args["resume_model"])
            else:
                model_ref = UniFlowMatchClassificationRefinement.from_pretrained(args["resume_model"])
            ckpt = {"state_dict": {"model." + k: v for k, v in model_ref.state_dict().items()}}
            print("Resumed model from HF repo: ", args["resume_model"])
        else:
            print("Resuming model from checkpoint: ", args["resume_model"])
            ckpt = torch.load(args["resume_model"], map_location="cpu")

        # remove the "._orig__mod" from the state dict (caused by torch compile in old code version)
        state_dict = {k.replace("._orig_mod", ""): v for k, v in ckpt["state_dict"].items()}

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            print("Failed to load state dict strictly, trying best-effort...")
            model.load_state_dict(state_dict, strict=False)
            print("Loaded state dict non-strictly.")

    if args["disable_wandb"]:
        logger = None
    else:
        logger = WandbLogger(
            name=args["wandb_name"] if "wandb_name" in args else None,
            project="uniflowmatch",
            config=args,
            tags=args["wandb_tags"] if "wandb_tags" in args else None,
        )
        # logger.watch(model, log="all") # uncomment to log gradients and parameters

    trainer = L.Trainer(
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True) if args["find_unused_parameters"] else "ddp",
        num_nodes=num_nodes,
        devices=torch.cuda.device_count(),
        default_root_dir=experiment_folder,
        check_val_every_n_epoch=(
            args["training_scheme"]["validate_every_epochs"]
            if args["training_scheme"]["validate_steps"] is None
            else None
        ),
        val_check_interval=args["training_scheme"]["validate_steps"],
        accumulate_grad_batches=args["training_scheme"]["accumulate_grad_batches"],
        enable_model_summary=True,
        use_distributed_sampler=False,  # avoid lightning to replace our sampler
        max_epochs=args["training_scheme"]["num_epochs"],
        limit_train_batches=args["training_scheme"].get("limit_train_batches", 1.0),  # smoke/overfit knob
        limit_val_batches=args["training_scheme"].get("limit_val_batches", 1.0),
        num_sanity_val_steps=args["training_scheme"].get("num_sanity_val_steps", 2),
        log_every_n_steps=1,
        logger=logger,
        precision=args["training_scheme"]["precision"],  # typically "bf16-mixed"
        gradient_clip_algorithm=args["training_scheme"]["gradient_clip_algorithm"],
        gradient_clip_val=args["training_scheme"]["gradient_clip_val"],
        # profiler=PyTorchProfiler(filename="profiler_output.txt"),  # profile the training
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                save_top_k=args["training_scheme"]["keep_n_epoch_ckpts"],  # save all checkpoint history
                monitor="epoch",  # save based on step
                every_n_epochs=args["training_scheme"]["save_ckpt_every_epochs"],  # every certain epochs
                mode="max",
                dirpath=os.path.join(experiment_folder, "checkpoints_epoch"),
                filename="checkpoint-{epoch:03d}",
                save_last=True,
            ),
            ModelCheckpoint(
                save_top_k=-1,  # save all checkpoint history
                monitor="step",  # save based on step
                every_n_train_steps=args["training_scheme"]["save_ckpt_every_steps"],  # every certain steps
                mode="max",
                dirpath=os.path.join(experiment_folder, "checkpoints_step"),
                filename="checkpoint-{step:06d}",
                save_last=True,
                verbose=1,
            ),
        ],
        plugins=TorchElasticEnvironment() if num_nodes > 1 else None,  # to use torchrun for multi-node training
    )

    # start training or validation. resume all is handled here
    # (resume all = resume model, optimizers, dataloaders as if continuing a interrupted training)
    if args["launch_mode"] == "train":
        trainer.fit(model, datamodule, ckpt_path=args["resume_all"])
    elif args["launch_mode"] == "validate":
        trainer.validate(model, datamodule, ckpt_path=args["resume_all"])
