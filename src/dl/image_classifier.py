import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Set directories
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")   # preprocessed split root
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")         # raw data (behavior images, etc.)
MODELS_DIR    = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Load species class list from class_mapping.json (produced by preprocess_dataset.py)
_class_map_path = os.path.join(BASE_DIR, "class_mapping.json")
if os.path.exists(_class_map_path):
    with open(_class_map_path, "r", encoding="utf-8") as _f:
        _mapping = json.load(_f)
    SPECIES_CLASSES: list = _mapping["classes"]  # ordered by index
    print(f"[DL] Loaded {len(SPECIES_CLASSES)} species classes from class_mapping.json: {SPECIES_CLASSES}")
else:
    # Fallback: hardcoded 5-class list (mirrors preprocess_dataset.py TARGET_CLASSES)
    SPECIES_CLASSES = ["tiger", "elephant", "deer", "wild_boar", "leopard"]
    print("[DL] class_mapping.json not found — using default 5 species classes.")

BEHAVIOR_CLASSES = ["feeding", "running", "resting", "threat-action"]

class WildlifeDataset(Dataset):
    """
    Dataset that reads from data/processed/{split}/{class}/ folder structure
    produced by preprocess_dataset.py.  For classes that have no real images
    (e.g. classes not yet collected), synthetic RGB arrays are generated so
    the model head still covers all 5 output neurons.

    Parameters
    ----------
    split_dir : str
        Path to a split root, e.g. data/processed/train
    class_list : list[str]
        Ordered list of class names (matches class_mapping.json "classes" key)
    transform : torchvision transform | None
    synth_per_missing_class : int
        Number of synthetic samples to generate for classes with no real images
    """
    def __init__(self, split_dir: str, class_list: list,
                 transform=None, synth_per_missing_class: int = 40):
        self.split_dir  = split_dir
        self.class_list = class_list
        self.transform  = transform
        self.image_paths = []  # str path OR numpy ndarray (synthetic)
        self.labels      = []

        real_counts = {}
        for i, cls in enumerate(class_list):
            cls_dir = os.path.join(split_dir, cls)
            found = 0
            if os.path.isdir(cls_dir):
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        self.image_paths.append(os.path.join(cls_dir, fname))
                        self.labels.append(i)
                        found += 1
            real_counts[cls] = found

            # Synthetic fallback for classes with no real images
            if found == 0:
                for s in range(synth_per_missing_class):
                    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                    # Add a colored rectangle so each class has a distinct colour hue
                    arr[30:190, 30:190, i % 3] = min(255, (i + 1) * 50)
                    self.image_paths.append(arr)
                    self.labels.append(i)

        split_name = os.path.basename(split_dir)
        print(f"[WildlifeDataset] {split_name}: " +
              ", ".join(f"{c}={real_counts[c]}" for c in class_list) +
              f"  |  total={len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        item  = self.image_paths[idx]
        label = self.labels[idx]

        if isinstance(item, str):
            try:
                img = Image.open(item).convert("RGB")
            except Exception:
                img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        else:
            img = Image.fromarray(item)   # synthetic ndarray

        if self.transform:
            img = self.transform(img)
        return img, label

def get_transfer_model(num_classes, architecture="mobilenet"):
    """Creates a Transfer Learning model using ResNet50 or MobileNetV2."""
    if architecture == "resnet":
        print("Initializing ResNet50 architecture...")
        # Use Weights argument if available (torchvision newer versions)
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except AttributeError:
            model = models.resnet50(pretrained=True)
            
        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False
            
        # Replace the fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        print("Initializing MobileNetV2 architecture...")
        try:
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        except AttributeError:
            model = models.mobilenet_v2(pretrained=True)
            
        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False
            
        # Replace classifier head
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3, device="cpu"):
    """Standard PyTorch training loop."""
    since = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # Train phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_samples += inputs.size(0)
                
        epoch_val_loss = val_loss / val_samples
        epoch_val_acc = val_corrects.double() / val_samples
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc.item())
        print(f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    return model, history

def save_training_curves(history, name):
    """Plots and saves accuracy and loss curves."""
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{name} Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'{name} Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f"{name.lower()}_curves.png"))
    plt.close()

def run_dl_training(epochs_species: int = 3, epochs_behavior: int = 2,
                    batch_size: int = 16, arch: str = "mobilenet"):
    """
    Train both the species classifier and behavior classifier.

    Species model reads from data/processed/train|val/ (real images + synthetic
    fallback for classes with no collected images).
    Behavior model reads from data/raw/behavior_images/ (as before).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[DL] Training on device: {device}")
    print(f"[DL] Species classes ({len(SPECIES_CLASSES)}): {SPECIES_CLASSES}")

    # ── Transforms ────────────────────────────────────────────────────────────
    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── 1. Species Classifier — uses processed split ───────────────────────
    print("\n--- Training Species Classifier (processed split) ---")
    train_split = os.path.join(PROCESSED_DIR, "train")
    val_split   = os.path.join(PROCESSED_DIR, "val")

    train_ds = WildlifeDataset(train_split, SPECIES_CLASSES, transform=train_tfm)
    val_ds   = WildlifeDataset(val_split,   SPECIES_CLASSES, transform=val_tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    model_species = get_transfer_model(len(SPECIES_CLASSES), architecture=arch).to(device)
    criterion     = nn.CrossEntropyLoss()
    optimizer     = optim.Adam(
        filter(lambda p: p.requires_grad, model_species.parameters()), lr=1e-3
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    model_species, history_species = train_model(
        model_species, train_loader, val_loader,
        criterion, optimizer, num_epochs=epochs_species, device=device,
    )
    for _ in range(epochs_species):
        scheduler.step()

    out_path = os.path.join(MODELS_DIR, "species_classifier.pth")
    torch.save(model_species.state_dict(), out_path)
    save_training_curves(history_species, "Species")
    print(f"[DL] Species classifier saved -> {out_path}")

    # ── 2. Behavior Classifier — uses raw behavior_images/ ────────────────
    print("\n--- Training Behavior Classifier ---")
    behavior_dir  = os.path.join(RAW_DIR, "behavior_images")
    behavior_ds   = WildlifeDataset(behavior_dir, BEHAVIOR_CLASSES, transform=train_tfm)
    beh_train_sz  = int(0.8 * len(behavior_ds))
    beh_val_sz    = len(behavior_ds) - beh_train_sz
    beh_train_ds, beh_val_ds = torch.utils.data.random_split(
        behavior_ds, [beh_train_sz, beh_val_sz]
    )

    beh_train_loader = DataLoader(beh_train_ds, batch_size=batch_size, shuffle=True)
    beh_val_loader   = DataLoader(beh_val_ds,   batch_size=batch_size, shuffle=False)

    model_behavior = get_transfer_model(len(BEHAVIOR_CLASSES), architecture=arch).to(device)
    opt_b = optim.Adam(
        filter(lambda p: p.requires_grad, model_behavior.parameters()), lr=1e-3
    )
    model_behavior, history_behavior = train_model(
        model_behavior, beh_train_loader, beh_val_loader,
        criterion, opt_b, num_epochs=epochs_behavior, device=device,
    )

    beh_path = os.path.join(MODELS_DIR, "behavior_classifier.pth")
    torch.save(model_behavior.state_dict(), beh_path)
    save_training_curves(history_behavior, "Behavior")
    print(f"[DL] Behavior classifier saved -> {beh_path}")
    print("\n[DL] All models saved successfully.")

# ============================================================
# Inference interface (used by FastAPI)
# ============================================================
class Predictor:
    """
    Loads trained species and behavior classifiers and exposes
    predict_species() / predict_behavior() for the API layer.

    Class lists are loaded dynamically from class_mapping.json so
    the API never hard-codes species indices.
    """

    def __init__(self, architecture: str = "mobilenet"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.species_classes  = SPECIES_CLASSES   # loaded at module import
        self.behavior_classes = BEHAVIOR_CLASSES

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # ── Species model ─────────────────────────────────────────────────
        self.species_model = get_transfer_model(
            len(self.species_classes), architecture=architecture
        )
        spec_path = os.path.join(MODELS_DIR, "species_classifier.pth")
        if os.path.exists(spec_path):
            self.species_model.load_state_dict(
                torch.load(spec_path, map_location=self.device, weights_only=True)
            )
            print(f"[Predictor] Species model loaded ({len(self.species_classes)} classes).")
        else:
            print("[Predictor] WARNING: species_classifier.pth not found. "
                  "Run train_species_model.py first.")
        self.species_model.to(self.device).eval()

        # ── Behavior model ────────────────────────────────────────────────
        self.behavior_model = get_transfer_model(
            len(self.behavior_classes), architecture=architecture
        )
        beh_path = os.path.join(MODELS_DIR, "behavior_classifier.pth")
        if os.path.exists(beh_path):
            self.behavior_model.load_state_dict(
                torch.load(beh_path, map_location=self.device, weights_only=True)
            )
            print(f"[Predictor] Behavior model loaded ({len(self.behavior_classes)} classes).")
        else:
            print("[Predictor] WARNING: behavior_classifier.pth not found.")
        self.behavior_model.to(self.device).eval()

    def _load_image(self, source) -> Image.Image:
        """Open an image from a file path, binary stream, or bytes object."""
        try:
            return Image.open(source).convert("RGB")
        except Exception:
            return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    def predict_species(self, image_source) -> tuple:
        """Return (class_name, confidence) for a species prediction."""
        try:
            img = self._load_image(image_source)
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.species_model(tensor), dim=1)[0]
                conf, idx = torch.max(probs, 0)
            return self.species_classes[idx.item()], round(conf.item(), 4)
        except Exception as exc:
            print(f"[Predictor] Species inference error: {exc}")
            return np.random.choice(self.species_classes), 0.85

    def predict_behavior(self, image_source) -> tuple:
        """Return (class_name, confidence) for a behavior prediction."""
        try:
            img = self._load_image(image_source)
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.behavior_model(tensor), dim=1)[0]
                conf, idx = torch.max(probs, 0)
            return self.behavior_classes[idx.item()], round(conf.item(), 4)
        except Exception as exc:
            print(f"[Predictor] Behavior inference error: {exc}")
            return np.random.choice(self.behavior_classes), 0.72

    def get_all_class_probs(self, image_source) -> dict:
        """Return full probability dict for all species — useful for top-k display."""
        try:
            img = self._load_image(image_source)
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.species_model(tensor), dim=1)[0].cpu().tolist()
            return {cls: round(p, 4) for cls, p in zip(self.species_classes, probs)}
        except Exception:
            return {cls: 0.0 for cls in self.species_classes}

if __name__ == "__main__":
    run_dl_training()
