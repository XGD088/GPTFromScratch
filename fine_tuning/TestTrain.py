import time

import torch

from fine_tuning.InitModel import model, calc_accuracy_loader
from fine_tuning.SpamDataLoader import val_loader, train_loader, test_loader
from fine_tuning.TrainClassifySpam import train_classifier_simple
from pre_training.CalculateLossWithDataSet import device
import matplotlib.pyplot as plt

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
num_epochs = 5
train_losses, val_losses, trains_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5
)

end_time = time.time()
print(f"Training took {end_time - start_time:.0f} seconds")


def plot_values(
        epochs_seen, examples_seen, train_values, val_values,
        label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(
        epochs_seen, val_values, linestyle="-.",
        label=f"Validation {label}"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()


epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)


#
epochs_tensor = torch.linspace(0, num_epochs, len(trains_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(trains_accs))
plot_values(
    epochs_tensor, examples_seen_tensor, trains_accs, val_accs,
    label="accuracy"
)

# Evaluate the model on the test set
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")


# Save the model
torch.save(model.state_dict(), "review_classifier.pth")

# Load the model
#model_state_dict = torch.load("review_classifier.pth, map_location=device")
#model.load_state_dict(model_state_dict)
