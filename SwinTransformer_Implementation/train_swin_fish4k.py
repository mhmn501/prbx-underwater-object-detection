import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import seaborn as sns
import numpy as np

# === HARD-CODED CLASS NAMES ===
class_names = [
    "Dascyllus reticulatus",
    "Plectroglyphidodon dickii",
    "Chromis chrysura",
    "Amphiprion clarkii",
    "Chaetodon lunulatus",
    "Chaetodon trifascialis",
    "Myripristis kuntee",
    "Acanthurus nigrofuscus",
    "Hemigymnus fasciatus",
    "Neoniphon sammara",
    "Abudefduf vaigiensis",
    "Canthigaster valentini",
    "Pomacentrus moluccensis",
    "Zebrasoma scopas",
    "Hemigymnus melapterus",
    "Lutjanus fulvus",
    "Scolopsis bilineata",
    "Scaridae",
    "Pempheris vanicolensis",
    "Zanclus cornutus",
    "Neoglyphidodon nigroris",
    "Balistapus undulatus",
    "Siganus fuscescens"
]

# === CONFIGURATION ===
DATASET_DIR = "C:/Users/Haiqal/Desktop/PRBX/datasets/Fish4K-Split"
BATCH_SIZE = 32
NUM_CLASSES = 23
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "C:/Users/Haiqal/Desktop/PRBX/SwinTransformer_Implementation/swin_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
RESULTS_CSV = "C:/Users/Haiqal/Desktop/PRBX/SwinTransformer_Implementation/swin_training_log.csv"
PLOTS_DIR = "C:/Users/Haiqal/Desktop/PRBX/SwinTransformer_Implementation/swin_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# === DATASET PREPARATION ===
def prepare_dataloaders():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATASET_DIR, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
        for x in ['train', 'val', 'test']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    return dataloaders, dataset_sizes, image_datasets

# === MODEL INITIALIZATION ===
def initialize_model():
    model = timm.create_model(
        'swin_tiny_patch4_window7_224', 
        pretrained=True,
        num_classes=NUM_CLASSES
        )
    model = model.to(DEVICE)
    return model

# === OPTIONAL: Resume Training From Checkpoint ===
def resume_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}.")
        return model, optimizer, start_epoch, best_acc
    else:
        print(f"\nNo checkpoint found at {checkpoint_path}. Starting fresh training.")
        return model, optimizer, 0, 0.0

# === TRAINING LOOP, EVALUATION, VISUALIZATION (same as EfficientNet version) ===
# Copy the same train_model(), evaluate(), predict_batch_images(), and plot_training_metrics() functions
# Minor edits: Save plots and checkpoints with 'swin_' prefixes to keep clean.

# === TRAINING LOOP ===
def train_model(model, optimizer, dataloaders, dataset_sizes, start_epoch=0, best_acc=0.0):
    """
    Train the model and save best and last checkpoints.
    """
    PATIENCE = 30  # Number of epochs to wait after no improvement
    counter = 0    # How many epochs since last improvement

    criterion = nn.CrossEntropyLoss()

    # If resuming, append to CSV; otherwise start fresh
    history = []
    if os.path.exists(RESULTS_CSV) and start_epoch > 0:
        history = pd.read_csv(RESULTS_CSV).to_dict('records')

    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            loop = tqdm(dataloaders[phase], desc=f"{phase.upper()} Phase")
            for inputs, labels in loop:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()

                with autocast():  # <<< Enabled mixed precision here
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Compute predictions
                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            history.append({
                'epoch': epoch,
                'phase': phase,
                'loss': epoch_loss,
                'accuracy': epoch_acc.item(),
                'learning_rate': scheduler.get_last_lr()[0]
            })

            # Early stopping logic and save best model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    counter = 0  # Reset counter because improvement
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_acc': best_acc,
                    }, os.path.join(CHECKPOINT_DIR, 'best_checkpoint.pth'))
                else:
                    counter += 1  # No improvement --> increment counter

        # Always save the latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pth'))

        pd.DataFrame(history).to_csv(RESULTS_CSV, index=False)

        # Update learning rate scheduler
        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # Early Stopping Check
        if counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {PATIENCE} epochs without improvement!")
            break


    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")

# === EVALUATE FUNCTION ===
def evaluate(model, dataloaders, class_names):
    """
    Evaluate the model on the test set.
    Computes precision, recall, F1-score, confusion matrix, and saves results.
    """

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        loop = tqdm(dataloaders['test'], desc="Testing")
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # === Compute Metrics ===
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\nTest Set Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # Save to CSV
    metrics_df = pd.DataFrame({
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1],
        'Accuracy': [accuracy]
    })
    metrics_df.to_csv(os.path.join(PLOTS_DIR, 'test_metrics_summary.csv'), index=False)
    print(f"\nTest metrics saved to {PLOTS_DIR}/test_metrics_summary.csv")

    # === Confusion Matrix ===
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    # Find non-zero rows/columns
    nonzero_rows = ~(cm.sum(axis=1) == 0)
    nonzero_cols = ~(cm.sum(axis=0) == 0)

    # Filter
    cm_clean = cm[nonzero_rows][:, nonzero_cols]
    class_names_clean = [class_names[i] for i in range(len(class_names)) if nonzero_rows[i]]
    
    # Create annotation matrix: keep only significant non-zeros
    annot_matrix = np.where(cm_clean > 0.001, cm_clean, np.nan)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_clean, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names_clean, yticklabels=class_names_clean, annot_kws={"size":8}, mask=np.isnan(annot_matrix), linewidths=0.5)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'))
    plt.show()
    print(f"Confusion matrix saved to {PLOTS_DIR}/confusion_matrix.png")

# === PREDICT FUNCTION ===
def predict_batch_images(model, dataset, class_names, num_images=16, num_grids=10):
    """
    Predict a random batch of images from the test set, visualize predictions vs true labels, and save the plot.

    """
    model.eval()

    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    for grid_num in range(1, num_grids + 1):
        fig, axes = plt.subplots(int(num_images**0.5), int(num_images**0.5), figsize=(12, 12))
        axes = axes.flatten()

        # Randomly select indices
        random_indices = random.sample(range(len(dataset)), num_images)

        for idx, img_idx in enumerate(random_indices):
            img, label = dataset[img_idx]
            img_input = img.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(img_input)
                _, pred = torch.max(output, 1)

            pred_class = class_names[pred.item()]
            true_class = class_names[label]

            # Reverse normalization for visualization
            img_show = img.cpu().permute(1, 2, 0) * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            img_show = img_show.clamp(0, 1)

            axes[idx].imshow(img_show)
            if pred_class == true_class:
                title_color = 'green'
            else:
                title_color = 'red'
            axes[idx].set_title(f"Pred: {pred_class}\nTrue: {true_class}", color=title_color, fontsize=8)
            axes[idx].axis('off')

        plt.tight_layout()
        
        save_path = os.path.join(PLOTS_DIR, f"swintransformer_test_predictions_grid{grid_num}.jpg")
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close(fig)
        print(f"Grid {grid_num} saved to {save_path}")

# === METRICS VISUALIZATION ===
def plot_training_metrics():
    """
    Plot loss and accuracy curves from training.
    """
    df = pd.read_csv(RESULTS_CSV)

    # Plot Loss
    plt.figure(figsize=(10, 6))
    for phase in ['train', 'val']:
        plt.plot(df[df['phase'] == phase]['epoch'], df[df['phase'] == phase]['loss'], label=f'{phase} loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(PLOTS_DIR, 'loss_plot.png'))
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    for phase in ['train', 'val']:
        plt.plot(df[df['phase'] == phase]['epoch'], df[df['phase'] == phase]['accuracy'], label=f'{phase} accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(PLOTS_DIR, 'accuracy_plot.png'))
    plt.show()

    # Plot Learning Rate
    plt.figure(figsize=(10, 6))
    plt.plot(df[df['phase'] == 'train']['epoch'], df[df['phase'] == 'train']['learning_rate'], label='Learning Rate', color='purple')
    plt.title('Learning Rate over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(PLOTS_DIR, 'learning_rate_plot.png'))
    plt.show()


# === MAIN ENTRY POINT ===
if __name__ == "__main__":
    model = initialize_model()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    dataloaders, dataset_sizes, image_datasets = prepare_dataloaders()

    # === Uncomment to resume training ===
    model, optimizer, start_epoch, best_acc = resume_checkpoint(model, optimizer, os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pth'))
    # train_model(model, optimizer, dataloaders, dataset_sizes, start_epoch, best_acc)

    # === Fresh Training ===
    # train_model(model, optimizer, dataloaders, dataset_sizes)

    # === Evaluate and Plot ===
    evaluate(model, dataloaders, class_names)
    # plot_training_metrics()

    # === Visualize Predictions ===
    # predict_batch_images(model, image_datasets['test'], class_names)
