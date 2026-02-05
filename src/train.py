"""
Module for training Sparse Autoencoder (SAE) models.
"""

import os

import time
from collections import defaultdict

import torch
from einops import rearrange
from tqdm import tqdm

import wandb

from overcomplete.metrics import l2, r2_score, l0_eps
from overcomplete.sae.trackers import DeadCodeTracker


def extract_input(batch):
    """
    Extracts the input data from a batch. Supports different batch types:
    - If the batch is a tuple or list, returns the first element.
    - If the batch is a dict, tries to return the value with key 'data' (or 'input').
    - Otherwise, assumes the batch is the input data.
    """
    if isinstance(batch, (tuple, list)):
        return batch[0]
    elif isinstance(batch, dict):
        return batch.get('data')
    else:
        return batch

def extract_multimodal_input(batch):
    """
    Extracts the input data from a batch. Supports different batch types:
    - If the batch is a tuple or list, returns the first two element.
    - If the batch is a dict, tries to return the value with key 'data' (or 'input').
    - Otherwise, assumes the batch is the input data.
    """
    if isinstance(batch, (tuple, list)):
        return batch[0], batch[1]
    elif isinstance(batch, dict):
        return batch.get('data')
    else:
        return batch


def _compute_reconstruction_error(x, x_hat):
    """
    Try to match the shapes of x and x_hat to compute the reconstruction error.

    If the input (x) shape is 4D assume it is (n, c, w, h), if it is 3D assume
    it is (n, t, c). Else, assume it is already flattened.

    Concerning the reconstruction error, we want a measure that could be compared
    across different input shapes, so we use the R2 score: 1 means perfect
    reconstruction, 0 means no reconstruction at all.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.

    Returns
    -------
    float
        Reconstruction error.
    """
    if len(x.shape) == 4 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, 'n c w h -> (n w h) c')
    elif len(x.shape) == 3 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, 'n t c -> (n t) c')
    else:
        assert x.shape == x_hat.shape, "Input and output shapes must match."
        x_flatten = x

    r2 = r2_score(x_flatten, x_hat)

    return r2.item()


def _log_metrics(monitoring, logs, model, z, loss, optimizer):
    """
    Log training metrics for the current training step.

    Parameters
    ----------
    monitoring : int
        Monitoring level, for 1 store only basic statistics, for 2 store more detailed statistics.
    logs : defaultdict
        Logs of training statistics.
    model : nn.Module
        The SAE model.
    z : torch.Tensor
        Encoded tensor.
    loss : torch.Tensor
        Loss value.
    optimizer : optim.Optimizer
        Optimizer for updating model parameters.
    """
    if monitoring == 0:
        return

    if monitoring > 0:
        logs['lr'].append(optimizer.param_groups[0]['lr'])
        logs['step_loss'].append(loss.item())

    if monitoring > 1:
        # store directly some z values
        # and the params / gradients norms
        logs['z'].append(z.detach()[::10])
        logs['z_l2'].append(l2(z).item())

        logs['dictionary_sparsity'].append(l0_eps(model.get_dictionary()).mean().item())
        logs['dictionary_norms'].append(l2(model.get_dictionary(), -1).mean().item())

        for name, param in model.named_parameters():
            if param.grad is not None:
                logs[f'params_norm_{name}'].append(l2(param).item())
                logs[f'params_grad_norm_{name}'].append(l2(param.grad).item())


def train_sae(model, dataloader, criterion, optimizer, scheduler=None,
              nb_epochs=20, clip_grad=1.0, monitoring=1, device="cpu", verbose=True):
    """
    Train a Sparse Autoencoder (SAE) model.

    Parameters
    ----------
    model : nn.Module
        The SAE model to train.
    dataloader : DataLoader
        DataLoader providing the training data.
    criterion : callable
        Loss function.
    optimizer : optim.Optimizer
        Optimizer for updating model parameters.
    scheduler : callable, optional
        Learning rate scheduler. If None, no scheduler is used, by default None.
    nb_epochs : int, optional
        Number of training epochs, by default 20.
    clip_grad : float, optional
        Gradient clipping value, by default 1.0.
    monitoring : int, optional
        Whether to monitor and log training statistics, the options are:
         (0) silent.
         (1) monitor and log training losses.
         (2) monitor and log training losses and statistics about gradients norms and z statistics.
        By default 1.
    device : str, optional
        Device to run the training on, by default 'cpu'.

    Returns
    -------
    defaultdict
        Logs of training statistics.
    """
    logs = defaultdict(list)

    for epoch in range(nb_epochs):
        model.train()

        start_time = time.time()
        epoch_loss = 0.0
        epoch_error = 0.0
        epoch_sparsity = 0.0
        batch_count = 0

        dead_tracker = None

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{nb_epochs}", disable=not verbose):
            batch_count += 1
            x = extract_input(batch).to(device, non_blocking=True)
            optimizer.zero_grad()

            z_pre, z, x_hat = model(x)
            loss = criterion(x, x_hat, z_pre, z, model.get_dictionary())

            # init dead code tracker with first batch
            if dead_tracker is None:
                dead_tracker = DeadCodeTracker(z.shape[1], device)
            dead_tracker.update(z)

            loss.backward()

            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if monitoring:
                epoch_loss += loss.item()
                epoch_error += _compute_reconstruction_error(x, x_hat)
                epoch_sparsity += l0_eps(z, 0).sum().item()
                _log_metrics(monitoring, logs, model, z, loss, optimizer)

        if monitoring and batch_count > 0:
            avg_loss = epoch_loss / batch_count
            avg_error = epoch_error / batch_count
            avg_sparsity = epoch_sparsity / batch_count
            dead_ratio = dead_tracker.get_dead_ratio()
            epoch_duration = time.time() - start_time

            logs['avg_loss'].append(avg_loss)
            logs['r2'].append(avg_error)
            logs['time_epoch'].append(epoch_duration)
            logs['z_sparsity'].append(avg_sparsity)
            logs['dead_features'].append(dead_ratio)

            print(f"Epoch[{epoch+1}/{nb_epochs}], Loss: {avg_loss:.4f}, "
                  f"R2: {avg_error:.4f}, L0: {avg_sparsity:.4f}, "
                  f"Dead Features: {dead_ratio*100:.1f}%, "
                  f"Time: {epoch_duration:.4f} seconds")

    return logs


def train_multimodal_sae(model, dataloader, criterion, optimizer, scheduler=None,
              nb_epochs=20, clip_grad=1.0, monitoring=1, device="cpu", verbose=False,
              checkpoint_path=None, checkpoint_interval=5, checkpoint_name="checkpoint.pt",
              vision_only=False):
    """
    Train a Sparse Autoencoder (SAE) model.

    Parameters
    ----------
    model : nn.Module
        The SAE model to train.
    dataloader : DataLoader
        DataLoader providing the training data.
    criterion : callable
        Loss function.
    optimizer : optim.Optimizer
        Optimizer for updating model parameters.
    scheduler : callable, optional
        Learning rate scheduler. If None, no scheduler is used, by default None.
    nb_epochs : int, optional
        Number of training epochs, by default 20.
    clip_grad : float, optional
        Gradient clipping value, by default 1.0.
    monitoring : int, optional
        Whether to monitor and log training statistics, the options are:
         (0) silent.
         (1) monitor and log training losses.
         (2) monitor and log training losses and statistics about gradients norms and z statistics.
        By default 1.
    device : str, optional
        Device to run the training on, by default 'cpu'.

    Returns
    -------
    defaultdict
        Logs of training statistics.
    """
    logs = defaultdict(list)
    model.train()
    best_loss = None
    multimodal = not vision_only

    checkpoint = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'logs': logs
    }

    # check that checkpoint_path exists, otherwise create it.
    full_checkpoint_path = os.path.join(checkpoint_path, checkpoint_name + f"_epoch_{0}" + ".pt")
    dir_name = os.path.dirname(full_checkpoint_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    torch.save(checkpoint, full_checkpoint_path)
    
    for epoch in range(nb_epochs):
        model.train()

        start_time = time.time()
        epoch_loss = 0.0
        epoch_error = 0.0
        epoch_sparsity = 0.0
        batch_count = 0

        dead_tracker = None

        progress_bar = tqdm(dataloader, disable=not verbose)
        for batch in progress_bar:
            batch_count += 1

            if multimodal:
                image, text = extract_multimodal_input(batch) #shapes [batch_size, d_model] and [batch_size, d_model]
                image = image.to(device, non_blocking=True)
                text = text.to(device, non_blocking=True)
                x = torch.cat((image, text), dim=0) #shapes [2*batch_size, d_model]
            else:
                x = extract_input(batch)
                x = x.to(device, non_blocking=True)
                
            optimizer.zero_grad()
            z_pre, z, x_hat = model(x)
            loss = criterion(x, x_hat, z_pre, z, model.get_dictionary())

            # init dead code tracker with first batch
            if dead_tracker is None:
                dead_tracker = DeadCodeTracker(z.shape[1], device)
            dead_tracker.update(z)

            loss.backward()

            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if monitoring:
                epoch_loss += loss.item()
                epoch_error += _compute_reconstruction_error(x, x_hat)
                epoch_sparsity += l0_eps(z, 0).sum().item()
                # if verbose:
                #     print(f"Epoch[{epoch+1}/{nb_epochs}], Batch[{batch_count}], Loss: {loss.item():.2e}, "
                #             f"R2: {_compute_reconstruction_error(x, x_hat):.4f}, "
                #             f"L0: {l0_eps(z, 0).sum().item():.4f}, "
                #             f"Dead Features: {dead_tracker.get_dead_ratio()*100:.1f}%")
                _log_metrics(monitoring, logs, model, z, loss, optimizer)
            
            progress_bar.set_description(
                f"Epoch[{epoch+1}/{nb_epochs}], L: {loss.item():.2e}, "
                f"R2: {_compute_reconstruction_error(x, x_hat):.2f}, "
                f"L0: {l0_eps(z, 0).sum().item():.1f}, "
                f"Dead: {dead_tracker.get_dead_ratio()*100:.1f}%"
            )

        if monitoring and batch_count > 0:
            avg_loss = epoch_loss / batch_count
            avg_error = epoch_error / batch_count
            avg_sparsity = epoch_sparsity / batch_count
            dead_ratio = dead_tracker.get_dead_ratio()
            epoch_duration = time.time() - start_time

            logs['avg_loss'].append(avg_loss)
            logs['r2'].append(avg_error)
            logs['time_epoch'].append(epoch_duration)
            logs['z_sparsity'].append(avg_sparsity)
            logs['dead_features'].append(dead_ratio)

            print(f"Epoch[{epoch+1}/{nb_epochs}], Loss: {avg_loss:.2e}, "
                  f"R2: {avg_error:.4f}, L0: {avg_sparsity:.4f}, "
                  f"Dead Features: {dead_ratio*100:.1f}%, "
                  f"Time: {epoch_duration:.4f} seconds")
        
        if checkpoint_path is not None and ((epoch + 1) % checkpoint_interval == 0 or epoch == nb_epochs - 1):
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'logs': logs
            }
            full_checkpoint_path = os.path.join(checkpoint_path, checkpoint_name + f"_epoch_{epoch+1}" + ".pt")
            dir_name = os.path.dirname(full_checkpoint_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            torch.save(checkpoint, full_checkpoint_path)
            
            if best_loss is None or avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint, full_checkpoint_path.replace(f"_epoch_{epoch+1}", ""))

    return logs


def train_multiple_saes(
        saes, train_loaders, losses, optimizers, schedulers=None,
        nb_epochs=20, N_tokens=None, clip_grad=1.0, device="cpu",
        monitoring=1, verbose=False,
        checkpoint_path=None,
        checkpoint_interval=5, checkpoint_name="checkpoint.pt",
        run=None,
    ):
    logs = defaultdict(list)
    for model in saes: model.train()
    best_loss = None

    checkpoint = {
        'epoch': 0,
        'model_state_dict': [model.state_dict() for model in saes],
        'optimizer_state_dict': [optimizer.state_dict() for optimizer in optimizers],
        'scheduler_state_dict': [scheduler.state_dict() if scheduler else None for scheduler in schedulers] if schedulers else None,
        'logs': logs
    }

    # check that checkpoint_path exists, otherwise create it.
    full_checkpoint_path = os.path.join(checkpoint_path, checkpoint_name + f"_epoch_{0}" + ".pt")
    dir_name = os.path.dirname(full_checkpoint_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    torch.save(checkpoint, full_checkpoint_path)
    
    plot_logs = {f"sae_{i+1}/{len(saes)}": defaultdict(float) for i in range(len(saes))}
    
    global_step = 0
    global_tokens = 0
    for epoch in range(nb_epochs):

        start_time = time.time()
        epoch_loss = 0.0
        epoch_error = 0.0
        epoch_sparsity = 0.0
        batch_count = 0

        dead_trackers = None

        # train_loaders is a list of dataloaders. zip them together to iterate simultaneously.
        dataloader = zip(*train_loaders)
        progress_bar = tqdm(dataloader, disable=not verbose)
        for batch in progress_bar:
            batch_count += 1
            global_step += 1

            xs, z_pres, zs, x_hats = [], [], [], []
            for i, model in enumerate(saes):
                optimizer = optimizers[i]
                optimizer.zero_grad()
                x = extract_input(batch[i]).to(device, non_blocking=True)
                z_pre, z, x_hat = model(x)
                
                if i == 0:
                    global_tokens += x.shape[0]
                
                xs.append(x), z_pres.append(z_pre), zs.append(z), x_hats.append(x_hat)
            
            # for each model select a random other model for cyclic reconstruction
            loss = 0.0
            for i, model in enumerate(saes):
                if len(saes) > 1:
                    j = (i + torch.randint(1, len(saes), (1,)).item()) % len(saes) # ensure j != i
                else:
                    j = i
                
                batch_loss = losses[i](
                    (xs[i], xs[j]),
                    (x_hats[i], x_hats[j]),
                    (z_pres[i], z_pres[j]),
                    (zs[i], zs[j]),
                    model.get_dictionary(),
                    f_E1=model.encode, f_D1=model.decode,
                    f_E2=saes[j].encode, f_D2=saes[j].decode,
                )
                for k, v in batch_loss.items():
                    plot_logs[f"sae_{i+1}/{len(saes)}"][k] = v.item() if isinstance(v, torch.Tensor) else v
                
                loss += batch_loss["total_loss"]
            
            run.log(
                {f"{k1}/{k2}": v for k1, log_dict in plot_logs.items() for k2, v in log_dict.items()},
                step=global_step,
                # commit=True,
            )

            # init dead code tracker with first batch
            if dead_trackers is None:
                dead_trackers = [DeadCodeTracker(z.shape[1], device) for z in zs]
            for dead_tracker, z in zip(dead_trackers, zs):
                dead_tracker.update(z)

            loss.backward()

            if clip_grad:
                for model in saes:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            for optimizer in optimizers:
                optimizer.step()

            if schedulers is not None:
                for scheduler in schedulers:
                    if scheduler is not None:
                        scheduler.step()

            # TODO : monitor with wandb to track all saes
            if monitoring:
                epoch_loss += loss.item()
                epoch_error += torch.tensor([_compute_reconstruction_error(x, x_hat) for x, x_hat in zip(xs, x_hats)]).mean().item()
                epoch_sparsity += torch.tensor([l0_eps(z, 0).sum().item() for z in zs]).mean().item()
                # if verbose:
                #     print(f"Epoch[{epoch+1}/{nb_epochs}], Batch[{batch_count}], Loss: {loss.item():.2e}, "
                #             f"R2: {_compute_reconstruction_error(x, x_hat):.4f}, "
                #             f"L0: {l0_eps(z, 0).sum().item():.4f}, "
                #             f"Dead Features: {dead_tracker.get_dead_ratio()*100:.1f}%")
                _log_metrics(monitoring, logs, model, z, loss, optimizer)
            
            progress_bar.set_description(
                f"Epoch[{epoch+1}/{nb_epochs}], L: {loss.item():.2e}, "
                f"R2: {torch.tensor([_compute_reconstruction_error(x, x_hat) for x, x_hat in zip(xs, x_hats)])}, "
                f"L0: {torch.tensor([l0_eps(z, 0).sum().item() for z in zs]).mean().item():.1f}, "
                f"Dead: {torch.tensor([dead_tracker.get_dead_ratio() for dead_tracker in dead_trackers]).mean().item()*100:.1f}%"
            )
            
            if N_tokens is not None and global_tokens >= N_tokens:
                break

        if monitoring and batch_count > 0:
            avg_loss = epoch_loss / batch_count
            avg_error = epoch_error / batch_count
            avg_sparsity = epoch_sparsity / batch_count
            dead_ratio = dead_tracker.get_dead_ratio()
            epoch_duration = time.time() - start_time

            logs['avg_loss'].append(avg_loss)
            logs['r2'].append(avg_error)
            logs['time_epoch'].append(epoch_duration)
            logs['z_sparsity'].append(avg_sparsity)
            logs['dead_features'].append(dead_ratio)

            print(f"Epoch[{epoch+1}/{nb_epochs}], Loss: {avg_loss:.2e}, "
                  f"R2: {avg_error:.4f}, L0: {avg_sparsity:.4f}, "
                  f"Dead Features: {dead_ratio*100:.1f}%, "
                  f"Time: {epoch_duration:.4f} seconds")
        
        if checkpoint_path is not None and (((epoch + 1) % checkpoint_interval == 0) or (epoch == nb_epochs - 1) or (N_tokens is not None and global_tokens >= N_tokens)):
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': [model.state_dict() for model in saes],
                'optimizer_state_dict': [optimizer.state_dict() for optimizer in optimizers],
                'scheduler_state_dict': [scheduler.state_dict() if scheduler else None for scheduler in schedulers] if schedulers is not None else None,
                'logs': logs
            }
            full_checkpoint_path = os.path.join(checkpoint_path, checkpoint_name + f"_epoch_{epoch+1}" + ".pt")
            dir_name = os.path.dirname(full_checkpoint_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            torch.save(checkpoint, full_checkpoint_path)
            
            if best_loss is None or avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint, full_checkpoint_path.replace(f"_epoch_{epoch+1}", ""))
        
        if N_tokens is not None and global_tokens >= N_tokens:
            break

    return logs


def train_sae_amp(model, dataloader, criterion, optimizer, scheduler=None,
                  nb_epochs=20, clip_grad=1.0, monitoring=1, device="cuda",
                  max_nan_fallbacks=5):
    """
    Train a Sparse Autoencoder (SAE) model with AMP and NaN fallback.
    Training in fp16 with AMP and fallback to fp32 for the rest of the current
    epoch if NaNs are detected 'max_nan_fallbacks' times. Next epoch will start
    again in fp16.

    Parameters
    ----------
    model : nn.Module
        The SAE model to train.
    dataloader : DataLoader
        DataLoader providing the training data.
    criterion : callable
        Loss function.
    optimizer : optim.Optimizer
        Optimizer for updating model parameters.
    scheduler : callable, optional
        Learning rate scheduler. If None, no scheduler is used, by default None.
    nb_epochs : int, optional
        Number of training epochs, by default 20.
    clip_grad : float, optional
        Gradient clipping value, by default 1.0.
    monitoring : int, optional
        Whether to monitor and log training statistics, the options are:
         (0) silent.
         (1) monitor and log training losses.
         (2) monitor and log training losses and statistics about gradients norms and z statistics.
        By default 1.
    device : str, optional
        Device to run the training on, by default 'cuda'.
    max_nan_fallbacks : int, optional
        Maximum number of NaN fallbacks per epoch before disabling AMP, by default 5.

    Returns
    -------
    defaultdict
        Logs of training statistics.
    """
    scaler = torch.amp.GradScaler(device=device, enabled=True)
    logs = defaultdict(list)

    for epoch in range(nb_epochs):
        model.train()

        start_time = time.time()
        epoch_loss = 0.0
        epoch_error = 0.0
        epoch_sparsity = 0.0
        batch_count = 0

        dead_tracker = None

        nan_fallback_count = 0

        for batch in dataloader:
            batch_count += 1
            x = extract_input(batch).to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device, enabled=True):
                z_pre, z, x_hat = model(x)
                loss = criterion(x, x_hat, z_pre, z, model.get_dictionary())

            # init dead code tracker with first batch
            if dead_tracker is None:
                dead_tracker = DeadCodeTracker(z.shape[1], device)
            dead_tracker.update(z)

            if torch.isnan(loss) or torch.isinf(loss):
                nan_fallback_count += 1
                if monitoring:
                    print(f"[Warning] NaN detected in loss at Epoch {epoch+1}, "
                          f"Iteration {len(logs['step_loss'])+1}. Switching to full precision.")

                with torch.amp.autocast(device_type=device, enabled=False):
                    z_pre, z, x_hat = model(x)
                    loss = criterion(x, x_hat, z_pre, z, model.get_dictionary())
                    dead_tracker.update(z)

                if torch.isnan(loss) or torch.isinf(loss):
                    if monitoring:
                        print(f"[Error] Loss is NaN even in full precision at Epoch {epoch+1}, "
                              f"Iteration {len(logs['step_loss'])+1}. Skipping this batch.")
                    continue  # skip the current batch if NaN persists

                if nan_fallback_count >= max_nan_fallbacks:
                    if monitoring:
                        print(f"[Info] Maximum NaN fallbacks reached at Epoch {epoch+1}. "
                              f"Disabling AMP for the rest of this epoch.")
                    break

            scaler.scale(loss).backward()

            if clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            if monitoring:
                epoch_loss += loss.item()
                epoch_error += _compute_reconstruction_error(x, x_hat)
                epoch_sparsity += l0_eps(z, 0).sum().item()
                _log_metrics(monitoring, logs, model, z, loss, optimizer)

        if monitoring and batch_count > 0:
            avg_loss = epoch_loss / batch_count
            avg_error = epoch_error / batch_count
            avg_sparsity = epoch_sparsity / batch_count
            dead_ratio = dead_tracker.get_dead_ratio()
            epoch_duration = time.time() - start_time

            logs['avg_loss'].append(avg_loss)
            logs['r2'].append(avg_error)
            logs['z_sparsity'].append(avg_sparsity)
            logs['time_epoch'].append(epoch_duration)
            logs['epoch_nan_fallbacks'].append(nan_fallback_count)
            logs['dead_features'].append(dead_ratio)

            print(f"Epoch[{epoch+1}/{nb_epochs}], Loss: {avg_loss:.4f}, "
                  f"R2: {avg_error:.4f}, L0: {avg_sparsity:.4f}, "
                  f"Dead Features: {dead_ratio*100:.1f}%, "
                  f"Time: {epoch_duration:.4f} seconds")

    return logs
