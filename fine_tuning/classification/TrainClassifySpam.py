import torch

from fine_tuning.classification.InitModel import calc_loss_batch, calc_loss_loader, calc_accuracy_loader


def train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                            num_epochs, eval_freq, eval_iter):
    # Initializes lists to track losses and tokens seen
    train_losses, val_losses, trains_accs, val_accs = [], [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            # Reset loss gradients
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # Calculates loss gradients
            loss.backward()
            # Updates model weights using loss gradients
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )

        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )

        trains_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        print(f"Ep {epoch + 1}: Train accuracy {train_accuracy * 100:.2f}, Val accuracy {val_accuracy * 100:.2f}")

    return train_losses, val_losses, trains_accs, val_accs, tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # disable the dropout, for stable, reproducible
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


