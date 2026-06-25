import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, NativeScaler, get_state_dict

import dna_models
import other_models
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from dna_models import DenoisingAttention, DenoisingAttentionSharedValues
from samplers import RASampler
import utils


def get_args_parser():
    parser = argparse.ArgumentParser("ViT w/ DnA training and evaluation script", add_help=False)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--grad_accum_steps", default=1, type=int)

    parser.add_argument("--model", default="dna_vit_base_patch16_224",type=str)
    parser.add_argument("--input-size", default=224, type=int)
    parser.add_argument("--drop", default=0.0, type=float)
    parser.add_argument("--drop-path", default=0.1, type=float)

    parser.add_argument("--model-ema", action="store_true")
    parser.add_argument("--no-model-ema", action="store_false", dest="model_ema")
    parser.set_defaults(model_ema=True)
    parser.add_argument("--model-ema-decay", default=0.99996, type=float)
    parser.add_argument("--model-ema-force-cpu", action="store_true", default=False)

    parser.add_argument("--opt", default="adamw", type=str)
    parser.add_argument("--opt-eps", default=1e-8, type=float)
    parser.add_argument("--opt-betas", default=None, type=float, nargs="+")
    parser.add_argument("--clip-grad", default=None, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=0.05, type=float)

    parser.add_argument("--sched", default="cosine", type=str)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--lr-noise", nargs="+", default=None, type=float)
    parser.add_argument("--lr-noise-pct", default=0.67, type=float)
    parser.add_argument("--lr-noise-std", default=1.0, type=float)
    parser.add_argument("--warmup-lr", default=1e-6, type=float)
    parser.add_argument("--min-lr", default=1e-5, type=float)
    parser.add_argument("--decay-epochs", default=30, type=float)
    parser.add_argument("--warmup-epochs", default=5, type=int)
    parser.add_argument("--cooldown-epochs", default=10, type=int)
    parser.add_argument("--patience-epochs", default=10, type=int)
    parser.add_argument("--decay-rate", "--dr", default=0.1, type=float)

    parser.add_argument("--color-jitter", default=0.3, type=float)
    parser.add_argument("--aa", default="rand-m9-mstd0.5-inc1", type=str)
    parser.add_argument("--smoothing", default=0.1, type=float)
    parser.add_argument("--train-interpolation", default="bicubic", type=str)
    parser.add_argument("--repeated-aug", action="store_true")
    parser.add_argument("--no-repeated-aug", action="store_false", dest="repeated_aug")
    parser.set_defaults(repeated_aug=True)
    parser.add_argument("--train-mode", action="store_true")
    parser.add_argument("--no-train-mode", action="store_false", dest="train_mode")
    parser.set_defaults(train_mode=True)
    parser.add_argument("--reprob", default=0.25, type=float)
    parser.add_argument("--remode", default="pixel", type=str)
    parser.add_argument("--recount", default=1, type=int)

    parser.add_argument("--mixup", default=0.0, type=float)
    parser.add_argument("--cutmix", default=0.0, type=float)
    parser.add_argument("--cutmix-minmax", default=None, nargs="+", type=float)
    parser.add_argument("--mixup-prob", default=1.0, type=float)
    parser.add_argument("--mixup-switch-prob", default=0.5, type=float)
    parser.add_argument("--mixup-mode", default="batch", type=str)

    parser.add_argument("--data-path", default="./data", type=str)
    parser.add_argument("--data-set", default="IMNET", choices=["IMNET"], type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-crop-ratio", default=0.875, type=float)
    parser.add_argument("--dist-eval", action="store_true", default=False)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin-mem", action="store_true")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://", type=str)

    parser.add_argument("--run_name", default=None, type=str)
    parser.add_argument("--wandb_entity", default=None, type=str)
    parser.add_argument("--wandb_project", default=None, type=str)
    return parser


def build_data_loaders(args):
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True,
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True,
            )

        if args.dist_eval:
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=False,
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return dataset_train, dataset_val, data_loader_train, data_loader_val


def create_criterion(args, mixup_active):
    if mixup_active:
        return SoftTargetCrossEntropy()
    if args.smoothing > 0.0:
        return LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    return torch.nn.CrossEntropyLoss()


def load_checkpoint(path):
    if path.startswith("https"):
        return torch.hub.load_state_dict_from_url(path, map_location="cpu", check_hash=True)
    return torch.load(path, map_location="cpu", weights_only=False)


def log_neg_scalers_to_wandb(model, epoch):
    if (epoch + 1) % 25 != 0:
        return

    log_dict = {}
    for module in model.modules():
        if not isinstance(module, (DenoisingAttention, DenoisingAttentionSharedValues)):
            continue

        scalers = module.neg_scalers.detach().cpu()
        layer_idx = module.layer_idx
        for head_idx, value in enumerate(scalers.tolist()):
            log_dict[f"neg_scalers/layer_{layer_idx}/head_{head_idx}"] = float(value)

        log_dict[f"neg_scalers/layer_{layer_idx}/mean"] = float(scalers.mean().item())
        log_dict[f"neg_scalers/layer_{layer_idx}/std"] = float(scalers.std().item())
        log_dict[f"neg_scalers/layer_{layer_idx}/min"] = float(scalers.min().item())
        log_dict[f"neg_scalers/layer_{layer_idx}/max"] = float(scalers.max().item())

    if log_dict:
        wandb.log(log_dict, step=epoch)


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    args.world = utils.get_world_size()
    args.effective_batchsize = int(args.batch_size * args.world * args.grad_accum_steps)

    wandb_run = None
    if utils.is_main_process() and args.run_name is not None:
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train, dataset_val, data_loader_train, data_loader_val = build_data_loaders(args)

    mixup_active = args.mixup > 0 or args.cutmix > 0 or args.cutmix_minmax is not None
    mixup_fn = None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        img_size=args.input_size,
    )
    model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    args.lr = args.lr * args.effective_batchsize / 512.0
    print(f"Linear scaled LR: {args.lr}, based on effective batch size of {args.effective_batchsize}")

    opt_name = args.opt.lower().split("_")[-1]
    filter_bias_and_bn = opt_name != "adamw"
    if opt_name == "adamw":
        print("AdamW baseline: applying weight decay to all parameters to match ViT_Lin base runs.")
    optimizer = create_optimizer(args, model_without_ddp, filter_bias_and_bn=filter_bias_and_bn)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    loss_scaler = NativeScaler()
    criterion = create_criterion(args, mixup_active)

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if args.model_ema and checkpoint.get("model_ema") is not None:
            utils._load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
        if "scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["scaler"])
        if not args.eval and {"optimizer", "lr_scheduler", "epoch"} <= checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = int(checkpoint["epoch"]) + 1

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} validation images: {test_stats['acc1']:.1f}%")
        if wandb_run is not None:
            wandb.finish()
        return

    output_dir = Path(args.output_dir) if args.output_dir else None
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
            model_ema=model_ema,
            mixup_fn=mixup_fn,
            set_training_mode=args.train_mode,
            args=args,
        )

        lr_scheduler.step(epoch)

        if output_dir is not None:
            utils.save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "model_ema": get_state_dict(model_ema) if args.model_ema else None,
                    "scaler": loss_scaler.state_dict(),
                    "args": args,
                },
                output_dir / "checkpoint.pth",
            )

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} validation images: {test_stats['acc1']:.1f}%")

        if test_stats["acc1"] > max_accuracy:
            max_accuracy = test_stats["acc1"]
            if output_dir is not None:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "model_ema": get_state_dict(model_ema) if args.model_ema else None,
                        "scaler": loss_scaler.state_dict(),
                        "args": args,
                    },
                    output_dir / "best_checkpoint.pth",
                )
        print(f"Max accuracy: {max_accuracy:.2f}%")

        log_stats = {
            **{f"train_{key}": value for key, value in train_stats.items()},
            **{f"test_{key}": value for key, value in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if utils.is_main_process() and wandb_run is not None:
            wandb.log(log_stats, step=epoch)
            log_neg_scalers_to_wandb(model_without_ddp, epoch)

        if output_dir is not None and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as handle:
                handle.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print("Training time", str(datetime.timedelta(seconds=int(total_time))))
    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ViT w/ DnA training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
