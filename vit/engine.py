import math
from typing import Iterable, Optional

import torch
from timm.data import Mixup
from timm.utils import ModelEma, accuracy

import utils


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0.0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    set_training_mode: bool = True,
    args=None,
):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 10

    accum_iter = max(1, int(getattr(args, "grad_accum_steps", 1)))
    num_training_steps = len(data_loader)
    last_window_size = num_training_steps % accum_iter or accum_iter
    optimizer.zero_grad(set_to_none=True)

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        is_last_step = (data_iter_step + 1) == num_training_steps
        update_grad = ((data_iter_step + 1) % accum_iter == 0) or is_last_step
        is_in_last_window = data_iter_step >= (num_training_steps - last_window_size)
        accum_denominator = last_window_size if is_in_last_window else accum_iter
        loss = loss / accum_denominator

        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}, stopping training")

        is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        clip_grad = max_norm if max_norm is not None and max_norm > 0 else None
        loss_scaler(
            loss,
            optimizer,
            clip_grad=clip_grad,
            parameters=model.parameters(),
            create_graph=is_second_order,
            need_update=update_grad,
        )

        if update_grad:
            optimizer.zero_grad(set_to_none=True)
            if model_ema is not None:
                model_ema.update(model)

        if device.type == "cuda":
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if mixup_fn is None:
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = samples.shape[0]
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1,
            top5=metric_logger.acc5,
            losses=metric_logger.loss,
        )
    )
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}
