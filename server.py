# server.py
import flwr as fl
import torch
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import EnhancedCNNModel, get_model_params, set_model_params
import math

# Use these to convert between Flower Parameters and numpy arrays
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Static capability labeling
# -----------------------------
client_types = {
    0: "fast", 1: "medium", 2: "slow",
    3: "fast", 4: "medium", 5: "slow",
    6: "fast", 7: "medium", 8: "slow",
    9: "fast", 10: "medium", 11: "slow",
    12: "fast", 13: "medium", 14: "slow",
    15: "fast", 16: "medium", 17: "slow",
    18: "fast", 19: "medium"
}

# -----------------------------
# Server-side metric store
# -----------------------------
class ServerMetrics:
    def __init__(self):
        self.history = {
            "accuracy": [], "loss": [], "round_times": [],
            "participations": defaultdict(list),
            "client_metrics": [], "aggregation_freq": [],
            "config_history": [], "best_accuracy": 0.0,
            "best_model_round": 0, "best_model_params": None,
            "client_participation": defaultdict(list),
            "class_distributions": defaultdict(list),
            "aggregation_rounds": [], "round_class_distributions": defaultdict(dict),
            "client_accuracy_history": defaultdict(list),
            "global_class_count": defaultdict(int),
            "timestamps": []
        }

    def update_best_model(self, accuracy, parameters, round_num):
        if accuracy > self.history["best_accuracy"]:
            self.history["best_accuracy"] = accuracy
            self.history["best_model_round"] = round_num
            self.history["best_model_params"] = parameters
            print(f"🔥 New best model at Round {round_num} - Accuracy: {accuracy:.2%}")

    def save(self, filename="fl_results.json"):
        save_data = {
            "accuracy_history": self.history["accuracy"],
            "loss_history": self.history["loss"],
            "round_times": self.history["round_times"],
            "participations": dict(self.history["participations"]),
            "config_history": self.history["config_history"],
            "best_accuracy": self.history["best_accuracy"],
            "best_model_round": self.history["best_model_round"],
            "client_participation": dict(self.history["client_participation"]),
            "class_distributions": dict(self.history["class_distributions"]),
            "aggregation_rounds": self.history["aggregation_rounds"],
            "timestamps": self.history["timestamps"]
        }
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        if self.history["best_model_params"] is not None:
            torch.save(
                self.history["best_model_params"],
                f"best_model_r{self.history['best_model_round']}_acc{self.history['best_accuracy']:.4f}.pt"
            )
        print(f"✅ Metrics saved to {filename}")
        print(f"✅ Best model (Round {self.history['best_model_round']}) saved to PT file")

metrics = ServerMetrics()

# -----------------------------------------------------------
# LOGGING-ONLY metrics aggregation (uses same rarity logic, but
# this does NOT affect the model update — the CustomStrategy does).
# -----------------------------------------------------------
def _parse_classes_str(s: str) -> List[str]:
    if not s:
        return []
    return [c.strip() for c in s.split(",") if c.strip()]

def _build_round_class_count_from_metrics(metrics_list: List[Tuple[int, Dict[str, Any]]]) -> Counter:
    cnt = Counter()
    for _, m in metrics_list:
        if not m:
            continue
        for c in _parse_classes_str(m.get("classes", "")):
            cnt.update([c])
    return cnt

def _rarity_weight_from_counts(cls_list: List[str], round_class_freq: Counter, n_clients: int, eps: float = 1e-6) -> float:
    """Rarity weight that does NOT collapse to 0.7.
    Uses relative rarity n_clients/f and maps around 1.0 into [0.7, 1.5] with a gentle slope.
    """
    if not cls_list:
        return 1.0
    uniq = list(set(cls_list))
    rel = [n_clients / (round_class_freq.get(c, 1) + eps) for c in uniq]
    base = float(np.mean(rel)) if rel else 1.0  # ~1.0 for average frequency
    # Map base→weight: 1.0→1.0, mild slope; clip into [0.7, 1.5]
    rarity = 1.0 + 0.25 * (base - 1.0)  # base=4 → 1.75 (clipped to 1.5); base=0.5 → 0.875
    return float(min(1.5, max(0.7, rarity)))

def weighted_average(metrics_list: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, float]:
    """Aggregate metrics for logging (accuracy/loss), using capability + rarity weights."""
    if not metrics_list:
        return {}

    speed_weight_map = {"fast": 1.0, "medium": 0.9, "slow": 0.8}
    round_class_freq = _build_round_class_count_from_metrics(metrics_list)
    n_clients = sum(1 for _, m in metrics_list if m)

    sum_acc = 0.0
    sum_loss = 0.0
    total_weight = 0.0
    have_acc = False
    have_loss = False

    for num_examples, m in metrics_list:
        if not m:
            continue
        cid = m.get("cid", None)
        client_type = client_types.get(cid, "medium")
        speed_w = speed_weight_map.get(client_type, 0.9)
        classes = _parse_classes_str(m.get("classes", ""))
        rarity_w = _rarity_weight_from_counts(classes, round_class_freq, n_clients)

        adjusted_w = float(num_examples) * float(speed_w) * float(rarity_w)

        if "accuracy" in m:
            sum_acc += float(m["accuracy"]) * adjusted_w
            have_acc = True
        if "loss" in m:
            sum_loss += float(m["loss"]) * adjusted_w
            have_loss = True
        total_weight += adjusted_w

        # light logging
        if "server_round" in m and classes:
            r = m["server_round"]
            metrics.history["round_class_distributions"][r][cid] = ",".join(classes)

    out: Dict[str, float] = {}
    if total_weight > 0:
        if have_acc:
            out["accuracy"] = sum_acc / total_weight
        if have_loss:
            out["loss"] = sum_loss / total_weight
    return out

# -----------------------------------------------------------
# Custom Strategy: rarity- & capability-weighted PARAMETER aggregation
# -----------------------------------------------------------
class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self, min_clients, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_clients = min_clients
        self.ready = False
        self.current_parameters = None

        # dynamic aggregation frequency state
        self.aggregate_every_round = False
        self.last_agg_round = 0
        self.consecutive_stagnant_aggs = 0

        # capability weights
        self.speed_weight_map = {"fast": 1.0, "medium": 0.9, "slow": 0.8}

    @staticmethod
    def _parse_classes(md: Dict[str, Any]) -> List[str]:
        cs = md.get("classes", "") or ""
        return [c.strip() for c in cs.split(",") if c.strip()]

    def initialize_parameters(self, client_manager):
        while len(client_manager.all()) < self.min_clients:
            time.sleep(1)
        self.ready = True
        self.current_parameters = super().initialize_parameters(client_manager)
        return self.current_parameters

    def should_aggregate(self, server_round: int) -> bool:
        # Warm-up (rounds 0..4): aggregate every round
        if server_round <= 4:
            return True
        # After warm-up: aggregate every other round unless stagnation fallback toggled
        return True if self.aggregate_every_round else (server_round % 2 == 0)

    def _build_round_class_count(self, results) -> Counter:
        cnt = Counter()
        for _, res in results:
            md = res.metrics or {}
            for c in self._parse_classes(md):
                cnt.update([c])
        return cnt

    def _rarity_weight(self, client_classes: List[str], round_class_freq: Counter, n_clients: int) -> float:
        return _rarity_weight_from_counts(client_classes, round_class_freq, n_clients)

    def aggregate_fit(self, server_round, results, failures):
        metrics.history["timestamps"].append(datetime.now().isoformat())
    
        # ---- NEW: log participation & class distribution for every client/round ----
        participated_cids, skipped_cids = [], []
        for _, res in results:
            md = res.metrics or {}
            cid = int(md.get("cid"))
            took_part = (res.num_examples > 0) and (md.get("status") != "skipped")
    
            # per-client time series (lists indexed by round order)
            metrics.history["client_participation"][cid].append(1 if took_part else 0)
            metrics.history["class_distributions"][cid].append(md.get("classes", ""))
    
            # per-round map: round -> {cid: "cls,cls,..."}
            metrics.history["round_class_distributions"][server_round][cid] = md.get("classes", "")
    
            (participated_cids if took_part else skipped_cids).append(cid)
    
        # round summary: round -> {"participating": [...], "skipped": [...]}
        metrics.history["participations"][server_round] = {
            "participating": participated_cids,
            "skipped": skipped_cids,
        }
        # ---- END NEW ----
    
        # Respect dynamic aggregation schedule (but keep the logs above)
        if not self.should_aggregate(server_round):
            print(f"🔄 Skipping aggregation at Round {server_round}")
            return self.current_parameters, {}
    
        if len(results) == 0:
            return self.current_parameters, {}
    
        # Build per-round class counts (only from available results)
        round_class_count = self._build_round_class_count(results)
        n_clients = len(results)
    
        # Compute weights and collect all client parameter tensors
        weights = []
        ndarrays_list = []
        metrics_list = []  # for weighted metrics logging
    
        for _, res in results:
            md = res.metrics or {}
            cid = int(md.get("cid"))
            client_type = client_types.get(cid, "medium")
            speed_w = float(self.speed_weight_map.get(client_type, 0.9))
    
            classes = self._parse_classes(md)
            rarity_w = self._rarity_weight(classes, round_class_count, n_clients)
    
            w = float(res.num_examples) * speed_w * rarity_w
            weights.append(w)
            ndarrays_list.append(parameters_to_ndarrays(res.parameters))
    
            m_for_log = {}
            if "accuracy" in md:
                m_for_log["accuracy"] = float(md["accuracy"])
            if "loss" in md:
                m_for_log["loss"] = float(md["loss"])
            m_for_log["cid"] = cid
            m_for_log["classes"] = md.get("classes", "")
            m_for_log["server_round"] = server_round
            metrics_list.append((res.num_examples, m_for_log))
    
        total_w = sum(weights) if sum(weights) > 0 else 1.0
    
        weighted_layers = []
        for layers in zip(*ndarrays_list):
            acc = None
            for wi, li in zip(weights, layers):
                term = (wi / total_w) * li
                acc = term if acc is None else acc + term
            weighted_layers.append(acc)
        aggregated_params = ndarrays_to_parameters(weighted_layers)
    
        aggregated_metrics = weighted_average(metrics_list)
    
        metrics.history["aggregation_rounds"].append(server_round)
        self.current_parameters = aggregated_params
        self.last_agg_round = server_round
        return aggregated_params, aggregated_metrics


    def configure_fit(self, server_round, parameters, client_manager):
        return super().configure_fit(server_round, parameters, client_manager)

# -----------------------------------------------------------
# Evaluation hook (keeps stagnation fallback logic)
# -----------------------------------------------------------
def get_evaluate_fn():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        net = EnhancedCNNModel().to(device)
        set_model_params(net, parameters)
        criterion = torch.nn.CrossEntropyLoss()
        correct = total = 0
        total_loss = 0.0
        net.eval()
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels).item()
                total_loss += loss
                correct += (torch.argmax(outputs, 1) == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        metrics.history["accuracy"].append(accuracy)
        metrics.history["loss"].append(total_loss)

        # stagnation-aware fallback toggle
        strategy = fl.server.strategy.strategy
        prev_best = metrics.history["best_accuracy"]
        if accuracy > prev_best:
            metrics.update_best_model(accuracy, parameters, server_round)
            strategy.aggregate_every_round = False
            strategy.consecutive_stagnant_aggs = 0
        elif strategy.should_aggregate(server_round):
            strategy.consecutive_stagnant_aggs += 1
            print(f"⚠️ Stagnation at aggregation round {server_round} "
                  f"(Stagnant Count: {strategy.consecutive_stagnant_aggs})")
            if strategy.consecutive_stagnant_aggs >= 2:
                strategy.aggregate_every_round = True
                print("🔁 Switching to aggregation every round due to stagnation.")

        print(f"[{'Aggregation' if strategy.should_aggregate(server_round) else 'Non-Aggregation'} "
              f"Round {server_round}] Eval - Loss: {total_loss:.4f}, Acc: {accuracy:.2%}")
        return total_loss, {"accuracy": accuracy}

    return evaluate

# -----------------------------------------------------------
# Round config (cosine LR, as before)
# -----------------------------------------------------------
def fit_config(server_round: int):
    T = 100  # total rounds for cosine schedule
    base_lr = 0.001

    # Start with cosine schedule
    lr = 0.0005 + 0.5 * (base_lr - 0.0005) * (1 + math.cos(math.pi * server_round / T))

    # Apply late-round phase adjustments
    if server_round > 75:
        lr *= 0.75
    if server_round > 90:
        lr = max(lr * 0.5, 0.0001)

    epochs = 2

    config = {
        "server_round": server_round,
        "lr": lr,
        "epochs": epochs,
        "batch_size": 32,
    }

    metrics.history["config_history"].append(config)
    return config

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    model = EnhancedCNNModel().cpu()
    initial_params = fl.common.ndarrays_to_parameters(get_model_params(model))
    strategy = CustomStrategy(
        min_clients=20,
        initial_parameters=initial_params,
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=16,   # 80% of 20
        min_evaluate_clients=10,  # 50% of 20
        min_available_clients=20,
        evaluate_fn=get_evaluate_fn(),
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=weighted_average,       # logging only
        evaluate_metrics_aggregation_fn=weighted_average,  # logging only
    )
    # expose for evaluate() logic
    fl.server.strategy.strategy = strategy

    print("🚀 Starting Federated Learning Server...")
    start = time.time()
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=strategy
    )
    metrics.save()
    print(f"\n🌝 Training Complete!\n⏱️  Total Time: {time.time() - start:.2f}s")
    print(f"🏆 Best Accuracy: {metrics.history['best_accuracy']:.2%} "
          f"(Round {metrics.history['best_model_round']})")

if __name__ == "__main__":
    main()
