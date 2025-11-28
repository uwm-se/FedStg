
# client.py
import argparse
import flwr as fl
import torch
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import EnhancedCNNModel, get_model_params, set_model_params
import threading
import subprocess
import sys
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

client_types = {
    0: "fast", 1: "medium", 2: "slow",
    3: "fast", 4: "medium", 5: "slow",
    6: "fast", 7: "medium", 8: "slow",
    9: "fast", 10: "medium", 11: "slow",
    12: "fast", 13: "medium", 14: "slow",
    15: "fast", 16: "medium", 17: "slow",
    18: "fast", 19: "medium"
}

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def load_datasets(cid, round_num, full_trainset, total_rounds=100):
    rotation_speed = 0.15
    phase = (round_num / total_rounds) * 2 * np.pi * rotation_speed
    client_offset = cid * 0.4
    class_weights = []
    for class_idx in range(10):
        weight = (np.cos(phase + client_offset + class_idx * 0.5) + 1.1) ** 2
        class_weights.append(weight)
    class_weights = np.array(class_weights) / sum(class_weights)
    selected_classes = np.random.choice(10, size=random.randint(4, 6), p=class_weights, replace=False).tolist()
    idxs = [i for i, (_, label) in enumerate(full_trainset) if label in selected_classes]
    trainset = torch.utils.data.Subset(full_trainset, idxs)
    class_str = ",".join(map(str, sorted(selected_classes)))
    print(f"Client {cid} (Round {round_num}) - Classes: {class_str} - Samples: {len(idxs)}")
    return DataLoader(trainset, batch_size=32, shuffle=True), class_str

class CIFARClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        self.client_type = client_types[cid]
        self.model = EnhancedCNNModel().to(device)
        self.round_num = 0
        self.performance_history = []
        self.full_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        self.testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    def _calculate_dynamic_participation(self):
        base_prob = 0.8
        raw_cap_map = {"fast": 1.00, "medium": 0.9, "slow": 0.8}
        raw_caps = [raw_cap_map.get(ct, 0.95) for ct in client_types.values()]
        mean_raw = sum(raw_caps) / len(raw_caps)
        capability_factor = raw_cap_map.get(self.client_type, 0.95) / mean_raw
        perf_factor = 1.0
        if len(self.performance_history) > 0:
            perf_factor = min(1.5, 0.7 + float(np.mean(self.performance_history[-3:])))
        decay_factor = max(0.5, 1.0 - (self.round_num * 0.01))
        dynamic_prob = base_prob * capability_factor * perf_factor * decay_factor
        return min(0.95, max(0.3, dynamic_prob))


    def get_parameters(self, config=None):
        return get_model_params(self.model)

    def fit(self, parameters, config):
        self.round_num = int(config["server_round"])
    
        # Always pick the round's class subset so we can report it even if we skip
        trainloader, class_str = load_datasets(self.cid, self.round_num, self.full_trainset)
    
        participation_prob = self._calculate_dynamic_participation()
        if random.random() > participation_prob:
            print(f"[Client {self.cid}] Skipping (Round {self.round_num}, Prob: {participation_prob:.2f})")
            return get_model_params(self.model), 0, {
                "status": "skipped",
                "cid": int(self.cid),
                "participation_prob": float(participation_prob),
                "classes": str(class_str),
                "server_round": int(self.round_num),
            }
    
        print(f"\n=== [Round {self.round_num}] Client {self.cid} Training ===")
        set_model_params(self.model, parameters)
    
        optimizer = torch.optim.SGD(self.model.parameters(), lr=float(config["lr"]), momentum=0.9, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
    
        self.model.train()
        for epoch in range(int(config["epochs"])):
            epoch_loss, correct, total = 0.0, 0, 0
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            epoch_acc = correct / max(1, total)
            epoch_loss_avg = epoch_loss / max(1, len(trainloader))
            self.performance_history.append(epoch_acc)
            print(f"[Client {self.cid}] Epoch {epoch+1} - Loss: {epoch_loss_avg:.4f}, Acc: {100*epoch_acc:.2f}%")
    
        return (
            get_model_params(self.model),
            len(trainloader.dataset),
            {
                "accuracy": float(epoch_acc),
                "loss": float(epoch_loss_avg),
                "cid": int(self.cid),
                "classes": str(class_str),
                "server_round": int(self.round_num),
            },
        )

    def evaluate(self, parameters, config=None):
        set_model_params(self.model, parameters)
        valloader = DataLoader(self.testset, batch_size=32)
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        self.model.eval()
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / max(1, total)
        return float(loss), total, {"accuracy": float(accuracy), "cid": int(self.cid)}

# ====== Multi-client launcher ======
def run_client(cid):
    print(f"\n🚀 [Main] Launching Client {cid}")
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--cid", str(cid)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True, bufsize=1)
    log_file = open(f"client_output_{cid}.log", "w")
    for line in proc.stdout:
        print(f"[Client {cid}] {line}", end="")
        log_file.write(line)
    proc.wait()
    log_file.close()
    print(f"✅ [Main] Client {cid} finished\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, help="Client ID")
    args = parser.parse_args()

    if args.cid is not None:
        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CIFARClient(args.cid))
    else:
        threads = []
        for cid in range(20):
            t = threading.Thread(target=run_client, args=(cid,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        print("\n🏁 All clients have finished training.")

