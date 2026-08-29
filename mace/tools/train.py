###########################################################################################
# Training script
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.optim import LBFGS
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_ema import ExponentialMovingAverage
from torchmetrics import Metric

from mace.cli.visualise_train import TrainingPlotter

from . import torch_geometric
from .checkpoint import CheckpointHandler, CheckpointState
from .torch_tools import to_numpy
from .utils import (
    MetricsLogger,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
    filter_nonzero_weight,
)


@dataclasses.dataclass
class SWAContainer:
    model: AveragedModel
    scheduler: SWALR
    start: int
    loss_fn: torch.nn.Module


def valid_err_log(
    valid_loss,
    eval_metrics,
    logger,
    log_errors,
    epoch=None,
    valid_loader_name="Default",
):
    eval_metrics["mode"] = "eval"
    eval_metrics["epoch"] = epoch
    eval_metrics["head"] = valid_loader_name
    logger.log(eval_metrics)
    if epoch is None:
        inintial_phrase = "Initial"
    else:
        inintial_phrase = f"Epoch {epoch}"
    if log_errors == "PerAtomRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A"
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_stress"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_stress = eval_metrics["rmse_stress"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_stress={error_stress:8.2f} meV / A^3",
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_virials = eval_metrics["rmse_virials_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_virials_per_atom={error_virials:8.2f} meV",
        )
    elif (
        log_errors == "PerAtomMAEstressvirials"
        and eval_metrics["mae_stress_per_atom"] is not None
    ):
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        error_stress = eval_metrics["mae_stress"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A, MAE_stress={error_stress:8.2f} meV / A^3"
        )
    elif (
        log_errors == "PerAtomMAEstressvirials"
        and eval_metrics["mae_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        error_virials = eval_metrics["mae_virials"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A, MAE_virials={error_virials:8.2f} meV"
        )
    elif log_errors == "TotalRMSE":
        error_e = eval_metrics["rmse_e"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "PerAtomMAE":
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "TotalMAE":
        error_e = eval_metrics["mae_e"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "DipoleRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_MU_per_atom={error_mu:8.2f} mDebye",
        )
    elif log_errors == "DipolePolarRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        error_polarizability = eval_metrics["rmse_polarizability_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:.4f}, RMSE_MU_per_atom={error_mu:.2f} me A, RMSE_polarizability_per_atom={error_polarizability:.2f} me A^2 / V",
        )
    elif log_errors == "EnergyDipoleRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_Mu_per_atom={error_mu:8.2f} mDebye",
        )


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    valid_loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
    start_epoch: int,
    max_num_epochs: int,
    patience: int,
    checkpoint_handler: CheckpointHandler,
    logger: MetricsLogger,
    eval_interval: int,
    output_args: Dict[str, bool],
    device: torch.device,
    log_errors: str,
    swa: Optional[SWAContainer] = None,
    ema: Optional[ExponentialMovingAverage] = None,
    max_grad_norm: Optional[float] = 10.0,
    log_wandb: bool = False,
    distributed: bool = False,
    save_all_checkpoints: bool = False,
    plotter: TrainingPlotter = None,
    distributed_model: Optional[DistributedDataParallel] = None,
    train_sampler: Optional[DistributedSampler] = None,
    rank: Optional[int] = 0,
    verbose: bool = False,
):
    lowest_loss = np.inf
    valid_loss = np.inf
    patience_counter = 0
    swa_start = True
    keep_last = False
    if log_wandb:
        import wandb

    if max_grad_norm is not None:
        logging.info(f"Using gradient clipping with tolerance={max_grad_norm:.3f}")

    logging.info("")
    logging.info("===========TRAINING===========")
    logging.info("Started training, reporting errors on validation set")
    logging.info("Loss metrics on validation set")
    epoch = start_epoch

    # log validation loss before _any_ training
    for valid_loader_name, valid_loader in valid_loaders.items():
        valid_loss_head, eval_metrics = evaluate(
            model=model,
            loss_fn=loss_fn,
            data_loader=valid_loader,
            output_args=output_args,
            device=device,
        )
        valid_err_log(
            valid_loss_head, eval_metrics, logger, log_errors, None, valid_loader_name
        )
    valid_loss = valid_loss_head  # consider only the last head for the checkpoint

    # variable used for broadcast by rank == 0 if epoch loop is exited early, e.g. patience
    exit_now = torch.zeros(1, device=device) if distributed else None
    while epoch < max_num_epochs:
        # LR scheduler and SWA update
        if swa is None or epoch < swa.start:
            if epoch > start_epoch:
                lr_scheduler.step(
                    metrics=valid_loss
                )  # Can break if exponential LR, TODO fix that!
        else:
            if swa_start:
                logging.info("Changing loss based on Stage Two Weights")
                lowest_loss = np.inf
                swa_start = False
                keep_last = True
            loss_fn = swa.loss_fn
            swa.model.update_parameters(model)
            if epoch > start_epoch:
                swa.scheduler.step()

        # Train
        if distributed:
            train_sampler.set_epoch(epoch)
        if "ScheduleFree" in type(optimizer).__name__:
            optimizer.train()
        train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            ema=ema,
            logger=logger,
            device=device,
            distributed=distributed,
            distributed_model=distributed_model,
            rank=rank,
            verbose=verbose,
        )
        if distributed:
            torch.distributed.barrier()

        # Validate
        if epoch % eval_interval == 0:
            model_to_evaluate = (
                model if distributed_model is None else distributed_model
            )
            param_context = (
                ema.average_parameters() if ema is not None else nullcontext()
            )
            if "ScheduleFree" in type(optimizer).__name__:
                optimizer.eval()
            with param_context:
                wandb_log_dict = {}
                for valid_loader_name, valid_loader in valid_loaders.items():
                    valid_loss_head, eval_metrics = evaluate(
                        model=model_to_evaluate,
                        loss_fn=loss_fn,
                        data_loader=valid_loader,
                        output_args=output_args,
                        device=device,
                    )
                    if rank == 0:
                        valid_err_log(
                            valid_loss_head,
                            eval_metrics,
                            logger,
                            log_errors,
                            epoch,
                            valid_loader_name,
                        )
                        if log_wandb:
                            wandb_log_dict[valid_loader_name] = {
                                "epoch": epoch,
                                "valid_loss": valid_loss_head,
                                "valid_rmse_e_per_atom": eval_metrics[
                                    "rmse_e_per_atom"
                                ],
                                "valid_rmse_f": eval_metrics["rmse_f"],
                            }
                if plotter and epoch % plotter.plot_frequency == 0:
                    try:
                        plotter.plot(epoch, model_to_evaluate, rank)
                    except Exception as e:  # pylint: disable=broad-except
                        logging.debug(f"Plotting failed: {e}")
                valid_loss = (
                    valid_loss_head  # consider only the last head for the checkpoint
                )
            if log_wandb:
                wandb.log(wandb_log_dict)
            if rank == 0:
                if valid_loss >= lowest_loss:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if swa is not None and epoch < swa.start:
                            logging.info(
                                f"Stopping optimization after {patience_counter} epochs without improvement and starting Stage Two"
                            )
                            epoch = swa.start
                        else:
                            logging.info(
                                f"Stopping optimization after {patience_counter} epochs without improvement"
                            )
                            if exit_now is not None:
                                exit_now.fill_(1)
                    if save_all_checkpoints:
                        param_context = (
                            ema.average_parameters()
                            if ema is not None
                            else nullcontext()
                        )
                        with param_context:
                            checkpoint_handler.save(
                                state=CheckpointState(model, optimizer, lr_scheduler),
                                epochs=epoch,
                                keep_last=True,
                            )
                else:
                    lowest_loss = valid_loss
                    patience_counter = 0
                    param_context = (
                        ema.average_parameters() if ema is not None else nullcontext()
                    )
                    with param_context:
                        checkpoint_handler.save(
                            state=CheckpointState(model, optimizer, lr_scheduler),
                            epochs=epoch,
                            keep_last=keep_last,
                        )
                        keep_last = False or save_all_checkpoints
        if distributed:
            torch.distributed.barrier()
        if exit_now is not None:
            torch.distributed.broadcast(exit_now, src=0)
            if exit_now == 1:
                break

        epoch += 1

    logging.info("Training complete")


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    ema: Optional[ExponentialMovingAverage],
    logger: MetricsLogger,
    device: torch.device,
    distributed: bool,
    distributed_model: Optional[DistributedDataParallel] = None,
    rank: Optional[int] = 0,
    verbose: bool = False,
) -> None:
    model_to_train = model if distributed_model is None else distributed_model

    if isinstance(optimizer, LBFGS):
        _, opt_metrics = take_step_lbfgs(
            model=model_to_train,
            loss_fn=loss_fn,
            data_loader=data_loader,
            optimizer=optimizer,
            ema=ema,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            device=device,
            distributed=distributed,
            rank=rank,
        )
        opt_metrics["mode"] = "opt"
        opt_metrics["epoch"] = epoch
        if rank == 0:
            logger.log(opt_metrics)
    else:
        epoch_E_inter, epoch_E_qeq, epoch_natoms = 0.0, 0.0, 0.0
        for batch_idx, batch in enumerate(data_loader):
            _, opt_metrics = take_step(
                model=model_to_train,
                loss_fn=loss_fn,
                batch=batch,
                optimizer=optimizer,
                ema=ema,
                output_args=output_args,
                max_grad_norm=max_grad_norm,
                device=device,
                verbose=verbose and batch_idx == 0,
                epoch=epoch,
            )
            if "qeq_energy_sum" in opt_metrics:
                epoch_E_inter += opt_metrics["qeq_interaction_energy_sum"]
                epoch_E_qeq += opt_metrics["qeq_energy_sum"]
                epoch_natoms += opt_metrics["qeq_natoms"]
            opt_metrics["mode"] = "opt"
            opt_metrics["epoch"] = epoch
            if rank == 0:
                logger.log(opt_metrics)
        if verbose and epoch_natoms > 0:
            logging.info(
                f"  Epoch energy decomposition (mean/atom): "
                f"E_inter={epoch_E_inter / epoch_natoms:.4f} eV, "
                f"E_QEq={epoch_E_qeq / epoch_natoms:.6f} eV"
            )


def _log_qeq_diagnostics(model, output, batch, epoch: int) -> None:
    """Gradient/energy-decomposition diagnostics for MACEDefect models
    (only called when the model output contains qeq_energy)."""
    logging.info(f"--- Gradient diagnostics (epoch {epoch}, batch 0) ---")
    if output.get("forces") is not None and getattr(batch, "forces", None) is not None:
        F_pred = output["forces"].detach()
        F_ref = batch.forces.to(F_pred.dtype)
        F_err = (F_pred - F_ref).norm(dim=-1)
        logging.info(
            f"  forces: pred_rms={F_pred.norm(dim=-1).mean():.4f} eV/Å, "
            f"ref_rms={F_ref.norm(dim=-1).mean():.4f} eV/Å, "
            f"err_rms={F_err.mean():.4f} eV/Å"
        )
    if output.get("energy") is not None and getattr(batch, "energy", None) is not None:
        num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).float()
        E_pred = output["energy"].detach() / num_atoms
        E_ref = batch.energy.to(E_pred.dtype) / num_atoms
        logging.info(
            f"  energy/atom: pred_mean={E_pred.mean():.4f} eV, "
            f"ref_mean={E_ref.mean():.4f} eV, "
            f"err_mean={(E_pred - E_ref).abs().mean():.4f} eV"
        )
    E_inter_b = output.get("interaction_energy")
    if E_inter_b is not None:
        num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).float()
        na = num_atoms.mean() + 1e-30
        qeq_str = ""
        if output.get("qeq_energy") is not None:
            qeq_str = f", E_QEq={output['qeq_energy'].detach().mean() / na:.6f} eV"
        logging.info(
            f"  energy decomposition (mean/atom): "
            f"E_inter={E_inter_b.detach().mean() / na:.4f} eV{qeq_str}"
        )
    if output.get("qeq_charges") is not None:
        q = output["qeq_charges"].detach()
        elem_idx = batch.node_attrs.argmax(dim=-1)
        z_list = model.atomic_numbers
        parts = []
        for ei in elem_idx.unique().sort().values:
            qi = q[elem_idx == ei]
            z = z_list[ei.item()] if ei.item() < len(z_list) else ei.item()
            parts.append(
                f"Z={z}: mean={qi.mean():.4f}, std={qi.std():.4f}, "
                f"min={qi.min():.4f}, max={qi.max():.4f}"
            )
        logging.info("  QEq charges per element: " + " | ".join(parts))

    grad_report = sorted(
        (
            (n, p.grad.norm().item())
            for n, p in model.named_parameters()
            if p.grad is not None
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    logging.info("  Top-10 param grad norms:")
    for n, g in grad_report[:10]:
        logging.info(f"    {n}: {g:.6f}")
    zero_grads = [n for n, g in grad_report if g < 1e-12]
    if zero_grads:
        logging.warning(f"  ZERO gradients on: {zero_grads}")
    else:
        logging.info("  All parameters have nonzero gradients ✓")
    total_gnorm = sum(g**2 for _, g in grad_report) ** 0.5
    logging.info(f"  Total grad norm (before clip): {total_gnorm:.4f}")
    logging.info("--- End gradient diagnostics ---")


def take_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: torch_geometric.batch.Batch,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
    verbose: bool = False,
    epoch: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    batch = batch.to(device)
    batch_dict = batch.to_dict()
    output_holder: Dict[str, Any] = {}

    def closure():
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch_dict,
            training=True,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        loss = loss_fn(pred=output, ref=batch)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        if output.get("qeq_energy") is not None:
            output_holder["output"] = output
            if verbose:
                _log_qeq_diagnostics(model, output, batch, epoch)

        return loss

    loss = closure()
    optimizer.step()

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }
    if "output" in output_holder:
        with torch.no_grad():
            out = output_holder["output"]
            natoms = (batch.ptr[1:] - batch.ptr[:-1]).float().sum().item()
            loss_dict["qeq_interaction_energy_sum"] = out["interaction_energy"].detach().sum().item()
            loss_dict["qeq_energy_sum"] = out["qeq_energy"].detach().sum().item()
            loss_dict["qeq_natoms"] = natoms

    return loss, loss_dict


def take_step_lbfgs(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
    distributed: bool,
    rank: int,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    logging.debug(
        f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )

    total_sample_count = 0
    for batch in data_loader:
        total_sample_count += batch.num_graphs

    if distributed:
        global_sample_count = torch.tensor(total_sample_count, device=device)
        torch.distributed.all_reduce(
            global_sample_count, op=torch.distributed.ReduceOp.SUM
        )
        total_sample_count = global_sample_count.item()

    signal = torch.zeros(1, device=device) if distributed else None

    def closure():
        if distributed:
            if rank == 0:
                signal.fill_(1)
                torch.distributed.broadcast(signal, src=0)

            for param in model.parameters():
                torch.distributed.broadcast(param.data, src=0)

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)

        # Process each batch and then collect the results we pass to the optimizer
        for batch in data_loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            output = model(
                batch_dict,
                training=True,
                compute_force=output_args["forces"],
                compute_virials=output_args["virials"],
                compute_stress=output_args["stress"],
            )
            batch_loss = loss_fn(pred=output, ref=batch)
            batch_loss = batch_loss * (batch.num_graphs / total_sample_count)

            batch_loss.backward()
            total_loss += batch_loss

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        if distributed:
            torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
        return total_loss

    if distributed:
        if rank == 0:
            loss = optimizer.step(closure)
            signal.fill_(0)
            torch.distributed.broadcast(signal, src=0)
        else:
            while True:
                # Other ranks wait for signals from rank 0
                torch.distributed.broadcast(signal, src=0)
                if signal.item() == 0:
                    break
                if signal.item() == 1:
                    loss = closure()

        for param in model.parameters():
            torch.distributed.broadcast(param.data, src=0)
    else:
        loss = optimizer.step(closure)

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    return loss, loss_dict


# Keep parameters frozen/active after evaluation
@contextmanager
def preserve_grad_state(model):
    # save the original requires_grad state for all parameters
    requires_grad_backup = {param: param.requires_grad for param in model.parameters()}
    try:
        # temporarily disable gradients for all parameters
        for param in model.parameters():
            param.requires_grad = False
        yield  # perform evaluation here
    finally:
        # restore the original requires_grad states
        for param, requires_grad in requires_grad_backup.items():
            param.requires_grad = requires_grad


def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    output_args: Dict[str, bool],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:

    metrics = MACELoss(loss_fn=loss_fn).to(device)

    start_time = time.time()

    with preserve_grad_state(model):
        for batch in data_loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            output = model(
                batch_dict,
                training=False,
                compute_force=output_args["forces"],
                compute_virials=output_args["virials"],
                compute_stress=output_args["stress"],
            )
            avg_loss, aux = metrics(batch, output)
    avg_loss, aux = metrics.compute()
    aux["time"] = time.time() - start_time
    metrics.reset()

    return avg_loss, aux


class MACELoss(Metric):
    def __init__(self, loss_fn: torch.nn.Module):
        super().__init__()
        self.loss_fn = loss_fn
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_data", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("E_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta_es", default=[], dist_reduce_fx="cat")
        self.add_state("delta_es_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Fs_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fs", default=[], dist_reduce_fx="cat")
        self.add_state("delta_fs", default=[], dist_reduce_fx="cat")
        self.add_state(
            "stress_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_stress", default=[], dist_reduce_fx="cat")
        self.add_state(
            "virials_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_virials", default=[], dist_reduce_fx="cat")
        self.add_state("delta_virials_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Mus_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state(
            "polarizability_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_polarizability", default=[], dist_reduce_fx="cat")
        self.add_state(
            "delta_polarizability_per_atom", default=[], dist_reduce_fx="cat"
        )

    def update(self, batch, output):  # pylint: disable=arguments-differ
        loss = self.loss_fn(pred=output, ref=batch)
        self.total_loss += loss
        self.num_data += batch.num_graphs

        if output.get("energy") is not None and batch.energy is not None:
            self.delta_es.append(batch.energy - output["energy"])
            self.delta_es_per_atom.append(
                (batch.energy - output["energy"]) / (batch.ptr[1:] - batch.ptr[:-1])
            )
            self.E_computed += filter_nonzero_weight(
                batch, self.delta_es, batch.weight, batch.energy_weight
            )
        if output.get("forces") is not None and batch.forces is not None:
            self.fs.append(batch.forces)
            self.delta_fs.append(batch.forces - output["forces"])
            self.Fs_computed += filter_nonzero_weight(
                batch,
                self.delta_fs,
                batch.weight,
                batch.forces_weight,
                spread_atoms=True,
            )
        if output.get("stress") is not None and batch.stress is not None:
            self.delta_stress.append(batch.stress - output["stress"])
            self.stress_computed += filter_nonzero_weight(
                batch, self.delta_stress, batch.weight, batch.stress_weight
            )
        if output.get("virials") is not None and batch.virials is not None:
            self.delta_virials.append(batch.virials - output["virials"])
            self.delta_virials_per_atom.append(
                (batch.virials - output["virials"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
            self.virials_computed += filter_nonzero_weight(
                batch, self.delta_virials, batch.weight, batch.virials_weight
            )
        if output.get("dipole") is not None and batch.dipole is not None:
            self.mus.append(batch.dipole)
            self.delta_mus.append(batch.dipole - output["dipole"])
            self.delta_mus_per_atom.append(
                (batch.dipole - output["dipole"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1)
            )
            self.Mus_computed += filter_nonzero_weight(
                batch,
                self.delta_mus,
                batch.weight,
                batch.dipole_weight,
                spread_quantity_vector=False,
            )
        if (
            output.get("polarizability") is not None
            and batch.polarizability is not None
        ):
            self.delta_polarizability.append(
                batch.polarizability - output["polarizability"]
            )
            self.delta_polarizability_per_atom.append(
                (batch.polarizability - output["polarizability"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1).unsqueeze(-1)
            )
            self.polarizability_computed += filter_nonzero_weight(
                batch,
                self.delta_polarizability,
                batch.weight,
                batch.polarizability_weight,
                spread_quantity_vector=False,
            )

    def convert(self, delta: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        if isinstance(delta, list):
            delta = torch.cat(delta)
        return to_numpy(delta)

    def compute(self):

        class NoneMultiply:
            def __mul__(self, other):
                return NoneMultiply()

            def __rmul__(self, other):
                return NoneMultiply()

            def __imul__(self, other):
                return NoneMultiply()

            def __format__(self, format_spec):
                return str(None)

        aux = defaultdict(NoneMultiply)
        aux["loss"] = to_numpy(self.total_loss / self.num_data).item()
        if self.E_computed:
            delta_es = self.convert(self.delta_es)
            delta_es_per_atom = self.convert(self.delta_es_per_atom)
            aux["mae_e"] = compute_mae(delta_es)
            aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
            aux["rmse_e"] = compute_rmse(delta_es)
            aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
            aux["q95_e"] = compute_q95(delta_es)
        if self.Fs_computed:
            fs = self.convert(self.fs)
            delta_fs = self.convert(self.delta_fs)
            aux["mae_f"] = compute_mae(delta_fs)
            aux["rel_mae_f"] = compute_rel_mae(delta_fs, fs)
            aux["rmse_f"] = compute_rmse(delta_fs)
            aux["rel_rmse_f"] = compute_rel_rmse(delta_fs, fs)
            aux["q95_f"] = compute_q95(delta_fs)
        if self.stress_computed:
            delta_stress = self.convert(self.delta_stress)
            aux["mae_stress"] = compute_mae(delta_stress)
            aux["rmse_stress"] = compute_rmse(delta_stress)
            aux["q95_stress"] = compute_q95(delta_stress)
        if self.virials_computed:
            delta_virials = self.convert(self.delta_virials)
            delta_virials_per_atom = self.convert(self.delta_virials_per_atom)
            aux["mae_virials"] = compute_mae(delta_virials)
            aux["rmse_virials"] = compute_rmse(delta_virials)
            aux["rmse_virials_per_atom"] = compute_rmse(delta_virials_per_atom)
            aux["q95_virials"] = compute_q95(delta_virials)
        if self.Mus_computed:
            mus = self.convert(self.mus)
            delta_mus = self.convert(self.delta_mus)
            delta_mus_per_atom = self.convert(self.delta_mus_per_atom)
            aux["mae_mu"] = compute_mae(delta_mus)
            aux["mae_mu_per_atom"] = compute_mae(delta_mus_per_atom)
            aux["rel_mae_mu"] = compute_rel_mae(delta_mus, mus)
            aux["rmse_mu"] = compute_rmse(delta_mus)
            aux["rmse_mu_per_atom"] = compute_rmse(delta_mus_per_atom)
            aux["rel_rmse_mu"] = compute_rel_rmse(delta_mus, mus)
            aux["q95_mu"] = compute_q95(delta_mus)
        if self.polarizability_computed:
            delta_polarizability = self.convert(self.delta_polarizability)
            delta_polarizability_per_atom = self.convert(
                self.delta_polarizability_per_atom
            )
            aux["mae_polarizability"] = compute_mae(delta_polarizability)
            aux["mae_polarizability_per_atom"] = compute_mae(
                delta_polarizability_per_atom
            )
            aux["rmse_polarizability"] = compute_rmse(delta_polarizability)
            aux["rmse_polarizability_per_atom"] = compute_rmse(
                delta_polarizability_per_atom
            )
            aux["q95_polarizability"] = compute_q95(delta_polarizability)

        return aux["loss"], aux
