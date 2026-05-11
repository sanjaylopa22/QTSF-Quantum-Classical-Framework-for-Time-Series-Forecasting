"""
VQF-F on Electricity  —  TSLib-compatible pipeline  (GPU edition, BATCHED)
===========================================================================
Architecture identical to the original VQF-F.
KEY FIX: PennyLane native parameter broadcasting — leading batch dim is
vectorised in ONE circuit call per channel (not B sequential calls).

  Verified speedup (B=256, n=6, depth=2):
    sequential loop : 2.39 s per channel per batch
    native broadcast: 0.011 s per channel per batch     ← 200× faster
  This is what makes the script actually run instead of hanging.

  Backend choice
  ──────────────
  default.qubit (CPU, the default here) is FASTEST for 6-qubit circuits.
  Diagnostic on RTX PRO 6000 / PennyLane 0.44 / Torch 2.11:
    default.qubit (broadcast) : 0.011 s   ← USE THIS
    lightning.qubit (CPU)     : 0.101 s
    lightning.gpu             : 0.278 s   ← 25× SLOWER, do not use for n=6
  GPU statevector wins only for n_qubits ≥ ~20, well above our 6.
  Classical layers still run on your GPU via --device cuda:0.

  Dataset
  ───────
  electricity.csv  (UCI Electricity Load Diagrams 2011-2014)
  Columns : date, MT_001 … MT_320, OT   (321 client columns + date)
  Frequency: hourly  (freq='h')

  Features mode  (--features flag)
  ─────────────────────────────────
  • M  (default): multivariate → multivariate  enc_in=321 (or capped)
  • S           : univariate   → univariate    enc_in=1
                  --target column (default: last numeric column)
  • --n_channels caps channels in M mode for fast tests.

  Split borders (TSLib Dataset_Custom)
  ─────────────────────────────────────
    train : [0,        int(0.7 * T))
    val   : [0.7*T,    int(0.8 * T))   + seq_len overlap
    test  : [0.8*T,    T)              + seq_len overlap

  Architecture (UNCHANGED)
  ─────────────────────────
  RY encoding → CZ → RY/RZ × depth → CNOT → ⟨Z_i⟩ + ⟨Z_iZ_{i+1}⟩
  N_QUBITS=6, CIRCUIT_DEPTH=2, d_f=11
  RevIN per-sample per-channel + nn.Linear MIMO readout
  parameter-shift gradients, Adam, AMP, lradj='type1'

Usage
─────
  pip install pennylane torch scikit-learn matplotlib tqdm scipy

  # Univariate (FAST — recommended first run):
  python vqff_electricity_fast.py --data_path /path/to/electricity.csv \\
      --features S --device cuda:0 --batch_size 256 --epochs 10

  # Multivariate, capped at 16 channels:
  python vqff_electricity_fast.py --data_path /path/to/electricity.csv \\
      --features M --n_channels 16 --device cuda:0 \\
      --batch_size 256 --epochs 10

  # Full 321-channel multivariate (LONG — many hours):
  python vqff_electricity_fast.py --data_path /path/to/electricity.csv \\
      --features M --device cuda:0 --batch_size 256 --epochs 5
"""

import argparse, os, time, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import dct
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
torch.manual_seed(42); np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# 0.  GPU UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def setup_device(req="auto"):
    if req == "cpu": return "cpu"
    if req == "auto" or req.startswith("cuda"):
        if torch.cuda.is_available():
            d = "cuda:0" if req == "auto" else req
            p = torch.cuda.get_device_properties(d)
            print(f"  [GPU] {p.name}  |  VRAM {p.total_memory/1e9:.1f} GB"
                  f"  |  CUDA {torch.version.cuda}")
            return d
        print("  [GPU] CUDA not available — falling back to CPU.")
    return "cpu"

def gpu_mem_str(d):
    if not d.startswith("cuda"): return ""
    return (f"  GPU mem  alloc={torch.cuda.memory_allocated(d)/1e6:.0f}MB"
            f"  reserved={torch.cuda.memory_reserved(d)/1e6:.0f}MB")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    data_path     = "electricity.csv"
    features      = "M"           # 'M' or 'S'
    target        = ""            # resolved after CSV load
    freq          = "h"
    enc_in        = 321           # set dynamically
    c_out         = 321           # set dynamically
    n_channels    = -1            # M mode: -1 = use all
    train_ratio   = 0.7
    val_ratio     = 0.1
    seq_len       = 96
    label_len     = 48
    PRED_LENS     = [96, 192, 336, 720]
    batch_size    = 32
    train_epochs  = 10
    patience      = 3
    learning_rate = 1e-4
    lradj         = "type1"
    use_amp       = True
    N_QUBITS      = 6
    CIRCUIT_DEPTH = 2
    LR_QUANTUM    = 5e-3
    device        = "auto"
    num_workers   = 4
    pin_memory    = True
    checkpoints   = "./checkpoints_vqff_electricity_fast/"
    use_lightning = False         # default OFF — default.qubit is faster for n=6


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TIME FEATURES (freq='h') — 4 channels
# ─────────────────────────────────────────────────────────────────────────────
def time_features(dates):
    return np.stack([
        dates.month.values     / 12.0 - 0.5,
        dates.day.values       / 31.0 - 0.5,
        dates.dayofweek.values / 6.0  - 0.5,
        dates.hour.values      / 23.0 - 0.5,
    ], axis=-1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  COLUMN HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _find_date_col(df):
    for c in df.columns:
        if c.lower() in ("date", "datetime", "time", "timestamp"):
            return c
    return df.columns[0]


def resolve_feature_cols(df, cfg):
    """
    Sets cfg.enc_in / cfg.c_out / cfg.target and returns the list of
    feature columns based on cfg.features ('M' or 'S').
    """
    date_col = _find_date_col(df)
    all_num  = [c for c in df.columns
                if c != date_col and pd.api.types.is_numeric_dtype(df[c])]

    if cfg.features == "S":
        if cfg.target == "" or cfg.target not in df.columns:
            cfg.target = all_num[-1]
        feat_cols = [cfg.target]
    else:
        feat_cols = all_num if cfg.n_channels <= 0 else all_num[:cfg.n_channels]
        if cfg.target == "" or cfg.target not in feat_cols:
            cfg.target = feat_cols[-1]

    cfg.enc_in = len(feat_cols)
    cfg.c_out  = len(feat_cols)
    return feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DATASET
# ─────────────────────────────────────────────────────────────────────────────
class ElectricityDataset(Dataset):
    """Ratio-based 70/10/20 split, TSLib Dataset_Custom convention."""

    def __init__(self, df, split, cfg, feat_cols, scaler=None):
        self.split    = split
        self.seq_len  = cfg.seq_len
        self.pred_len = None

        date_col = _find_date_col(df)
        dates    = pd.to_datetime(df[date_col].values)
        self.time_mark_full = time_features(dates)

        data_raw = df[feat_cols].values.astype(np.float32)
        T        = len(data_raw)

        ts = int(T * cfg.train_ratio)
        vs = ts + int(T * cfg.val_ratio)
        self.SPLIT_BORDERS = {
            "train": (0,  ts),
            "val":   (ts, vs),
            "test":  (vs, T),
        }

        s, e = self.SPLIT_BORDERS[split]
        s_ov = max(0, s - cfg.seq_len) if split != "train" else s
        e    = min(e, T)

        if scaler is None:
            ts0, te0 = self.SPLIT_BORDERS["train"]
            self.scaler = StandardScaler()
            self.scaler.fit(data_raw[ts0:te0])
        else:
            self.scaler = scaler

        scaled         = self.scaler.transform(data_raw)
        self.data      = scaled[s_ov:e]
        self.time_mark = self.time_mark_full[s_ov:e]

    def set_pred_len(self, p): self.pred_len = p

    def __len__(self):
        if self.pred_len is None:
            raise RuntimeError("Call set_pred_len() before iterating.")
        n = len(self.data) - self.seq_len - self.pred_len + 1
        if n <= 0:
            raise ValueError(
                f"Split '{self.split}' has only {len(self.data)} rows — "
                f"not enough for seq_len={self.seq_len} + pred_len={self.pred_len}.")
        return n

    def __getitem__(self, i):
        s = i; e = s + self.seq_len; r = e + self.pred_len
        return (torch.from_numpy(self.data[s:e]),
                torch.from_numpy(self.time_mark[s:e]),
                torch.from_numpy(self.data[e:r]),
                torch.from_numpy(self.time_mark[e:r]))


def make_loader(ds, cfg, shuffle, device):
    uc = device.startswith("cuda")
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers if uc else 0,
                      pin_memory=cfg.pin_memory and uc, drop_last=shuffle,
                      persistent_workers=(cfg.num_workers > 0 and uc))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DCT ENCODING
# ─────────────────────────────────────────────────────────────────────────────
def dct_compress(v, n):
    c = dct(v, norm="ortho")[:n]
    lo, hi = c.min(), c.max()
    if hi - lo < 1e-10: return np.zeros(n, np.float32)
    return ((c - lo) / (hi - lo) * np.pi).astype(np.float32)

def encode_angles(x_batch, n_qubits):
    """x_batch: (B,L,M) -> (B,M,n_qubits)"""
    B, L, M = x_batch.shape
    out = np.zeros((B, M, n_qubits), np.float32)
    for b in range(B):
        for m in range(M):
            ch = x_batch[b, :, m]
            lo = ch.min(); hi = ch.max() + 1e-8
            out[b, m] = dct_compress((ch - lo) / (hi - lo) * np.pi, n_qubits)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6.  BROADCAST-CAPABLE QUANTUM CIRCUIT
#     `qml.RY(angles[..., i])` enables PennyLane native broadcasting:
#     ONE call evaluates the entire batch in parallel.
# ─────────────────────────────────────────────────────────────────────────────
class BroadcastVQC:
    def __init__(self, cfg):
        self.n     = cfg.N_QUBITS
        self.depth = cfg.CIRCUIT_DEPTH
        self.d_f   = self.n + (self.n - 1)

        if cfg.use_lightning and cfg.device.startswith("cuda"):
            try:
                self.dev = qml.device("lightning.gpu", wires=self.n)
                print(f"  [Quantum] lightning.gpu ({self.n} qubits, broadcasted)")
                print( "  [Quantum] WARNING: for n_qubits<=8 default.qubit is "
                       "usually 25× faster than lightning.gpu.")
            except Exception:
                print("  [Quantum] lightning.gpu unavailable — default.qubit")
                self.dev = qml.device("default.qubit", wires=self.n)
        else:
            self.dev = qml.device("default.qubit", wires=self.n)
            print(f"  [Quantum] default.qubit ({self.n} qubits, broadcasted)  "
                  "[FASTEST for small circuits]")

        n = self.n; depth = self.depth

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def _circuit(angles, theta_ry, phi_rz):
            # angles: (B, n) — leading batch dim broadcasts natively
            for i in range(n):
                qml.RY(angles[..., i], wires=i)
            for i in range(n - 1):
                qml.CZ(wires=[i, i + 1])
            for d in range(depth):
                for i in range(n):
                    qml.RY(theta_ry[d, i], wires=i)
                    qml.RZ(phi_rz[d, i], wires=i)
                for i in range(n - 1):
                    qml.CNOT(wires=[i, i + 1])
            obs  = [qml.expval(qml.PauliZ(i)) for i in range(n)]
            obs += [qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1))
                    for i in range(n - 1)]
            return obs

        self._circuit = _circuit

    def forward_batch(self, angles_b, theta_ry, phi_rz):
        """
        angles_b : (B, n_qubits)  CPU
        theta_ry : (depth, n)     CPU (shared across batch)
        phi_rz   : (depth, n)     CPU
        returns  : (B, d_f)       CPU
        """
        out = self._circuit(angles_b, theta_ry, phi_rz)   # list of d_f tensors, each (B,)
        return torch.stack(out, dim=-1).float()            # (B, d_f)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MODEL  (architecture UNCHANGED — works for both M and S modes)
# ─────────────────────────────────────────────────────────────────────────────
class VQFFModel(nn.Module):
    def __init__(self, cfg, pred_len):
        super().__init__()
        self.cfg = cfg
        self.n   = cfg.N_QUBITS
        self.M   = cfg.enc_in            # 1 for S mode, N for M mode
        self.pred_len = pred_len

        self.qcirc = BroadcastVQC(cfg)
        self.d_f   = self.qcirc.d_f

        # Quantum params (CPU)
        self.theta_ry = nn.Parameter(
            torch.randn(self.M, cfg.CIRCUIT_DEPTH, cfg.N_QUBITS) * 0.1)
        self.phi_rz   = nn.Parameter(
            torch.randn(self.M, cfg.CIRCUIT_DEPTH, cfg.N_QUBITS) * 0.1)

        # Classical (GPU)
        self.readout     = nn.Linear(self.M * self.d_f, self.M * pred_len)
        self.revin_gamma = nn.Parameter(torch.ones(self.M))
        self.revin_beta  = nn.Parameter(torch.zeros(self.M))

    def _revin_norm(self, x):
        mu  = x.mean(1, keepdim=True).detach()
        x   = x - mu
        std = (x.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        return x / std * self.revin_gamma + self.revin_beta, mu, std

    def _revin_denorm(self, y, mu, std):
        return (y - self.revin_beta) / (self.revin_gamma + 1e-8) * std + mu

    def forward(self, x, x_mark=None):
        device = x.device
        x_norm, mu, std = self._revin_norm(x)

        angles_np = encode_angles(
            x_norm.detach().cpu().numpy(), self.n)        # (B, M, n)
        angles_t  = torch.from_numpy(angles_np).float()   # CPU

        B, M, _ = angles_t.shape
        all_ch  = []
        for m in range(M):
            # ONE broadcast call per channel covers all B samples
            feats_m = self.qcirc.forward_batch(
                angles_t[:, m, :],     # (B, n)
                self.theta_ry[m],      # (depth, n)
                self.phi_rz[m],        # (depth, n)
            )                          # (B, d_f)
            all_ch.append(feats_m)

        feat = torch.cat(all_ch, dim=-1).float().to(device)  # (B, M*d_f)
        y    = self.readout(feat).view(B, self.pred_len, M)
        return self._revin_denorm(y, mu, std)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  LR SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
def adjust_lr(opt, epoch, cfg):
    if cfg.lradj == "type1":
        lr = cfg.learning_rate * (0.5 ** ((epoch - 1) // 1))
        for pg in opt.param_groups:
            if pg.get("is_classical"): pg["lr"] = lr
        return lr
    return cfg.learning_rate


# ─────────────────────────────────────────────────────────────────────────────
# 9.  TRAIN / EVAL  (with per-batch progress)
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, criterion, device, amp_scaler, epoch_idx):
    model.train(); total = 0.0
    use_amp = device.startswith("cuda") and model.cfg.use_amp
    n_batches = len(loader)
    print(f"    [Epoch {epoch_idx}] training {n_batches} batches...", flush=True)
    t_start = time.time()
    for bi, (x, xm, y, _) in enumerate(loader):
        x  = x.to(device,  non_blocking=True)
        xm = xm.to(device, non_blocking=True)
        y  = y.to(device,  non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            loss = criterion(model(x, xm), y)
        if use_amp:
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(opt); amp_scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += loss.item() * x.size(0)

        if bi == 0 or (bi + 1) % max(1, n_batches // 10) == 0 or bi == n_batches - 1:
            dt   = time.time() - t_start
            done = bi + 1
            eta  = dt / done * (n_batches - done) if done else 0
            print(f"      batch {done:>4}/{n_batches}  "
                  f"loss={loss.item():.5f}  "
                  f"elapsed={dt:.1f}s  ETA={eta:.1f}s", flush=True)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); total = 0.0
    for x, xm, y, _ in loader:
        x = x.to(device, non_blocking=True)
        xm = xm.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        total += criterion(model(x, xm), y).item() * x.size(0)
    return total / len(loader.dataset)


# ─────────────────────────────────────────────────────────────────────────────
# 10. METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(yt, yp):
    mse  = float(mean_squared_error(yt.ravel(), yp.ravel()))
    mae  = float(mean_absolute_error(yt.ravel(), yp.ravel()))
    rmse = float(np.sqrt(mse))
    mask = np.abs(yt.ravel()) > 1e-6
    mape = float(np.mean(np.abs(
        (yt.ravel()[mask] - yp.ravel()[mask]) / yt.ravel()[mask]
    ))) * 100 if mask.any() else float("nan")
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "MAPE%": mape}


# ─────────────────────────────────────────────────────────────────────────────
# 11. RUN HORIZON
# ─────────────────────────────────────────────────────────────────────────────
def run_horizon(df, pred_len, cfg, device, feat_cols):
    print(f"\n{'='*72}")
    print(f"  VQF-F (broadcast) | Electricity | features='{cfg.features}' | "
          f"enc_in={cfg.enc_in}")
    print(f"  pred_len={pred_len} | seq_len={cfg.seq_len} | "
          f"batch={cfg.batch_size} | device={device}")
    print(f"  n_qubits={cfg.N_QUBITS} | depth={cfg.CIRCUIT_DEPTH} | "
          f"AMP={cfg.use_amp and device.startswith('cuda')}")
    print(f"{'='*72}", flush=True)

    train_ds = ElectricityDataset(df, "train", cfg, feat_cols)
    scaler   = train_ds.scaler
    val_ds   = ElectricityDataset(df, "val",   cfg, feat_cols, scaler)
    test_ds  = ElectricityDataset(df, "test",  cfg, feat_cols, scaler)
    for ds in (train_ds, val_ds, test_ds): ds.set_pred_len(pred_len)
    print(f"  Windows → train:{len(train_ds)}  "
          f"val:{len(val_ds)}  test:{len(test_ds)}")

    tl = make_loader(train_ds, cfg, True,  device)
    vl = make_loader(val_ds,   cfg, False, device)
    el = make_loader(test_ds,  cfg, False, device)

    model = VQFFModel(cfg, pred_len)
    model.readout.to(device)
    model.revin_gamma = nn.Parameter(model.revin_gamma.to(device))
    model.revin_beta  = nn.Parameter(model.revin_beta.to(device))

    criterion  = nn.MSELoss().to(device)
    amp_scaler = GradScaler(enabled=(cfg.use_amp and device.startswith("cuda")))

    q_params = [model.theta_ry, model.phi_rz]
    c_params = list(model.readout.parameters()) + [model.revin_gamma, model.revin_beta]
    opt = optim.Adam([
        {"params": q_params, "lr": cfg.LR_QUANTUM,    "is_classical": False},
        {"params": c_params, "lr": cfg.learning_rate, "is_classical": True},
    ])

    nq = sum(p.numel() for p in q_params)
    nc = sum(p.numel() for p in c_params)
    print(f"  Params → Q(CPU):{nq}  C(GPU):{nc}  total:{nq+nc}")
    print(f"  Quantum calls per batch: {cfg.enc_in}  "
          f"(was {cfg.enc_in*cfg.batch_size} in original)")
    if device.startswith("cuda"): print(gpu_mem_str(device))
    if cfg.features == "M" and cfg.enc_in > 32:
        print(f"  ⚠  {cfg.enc_in} channels — each batch needs {cfg.enc_in} "
              f"circuit calls.\n"
              f"     Consider --n_channels 16 or --features S for faster runs.")

    os.makedirs(cfg.checkpoints, exist_ok=True)
    safe_tag = cfg.features
    if cfg.features == "S":
        safe_tag += f"_{cfg.target.replace('/', '_').replace(' ', '_')}"
    ckpt = os.path.join(cfg.checkpoints,
                        f"vqff_electricity_{safe_tag}_pl{pred_len}.pt")

    best_val, best_state, wait = float("inf"), None, 0
    train_hist, val_hist = [], []

    print(f"\n  {'Ep':>4} | {'Train MSE':>11} | {'Val MSE':>11} | "
          f"{'LR':>10} | {'Time(s)':>8}")
    print(f"  {'-'*56}", flush=True)

    t_total = time.time()
    for epoch in range(1, cfg.train_epochs + 1):
        lr  = adjust_lr(opt, epoch, cfg)
        t0  = time.time()
        tr  = train_epoch(model, tl, opt, criterion, device, amp_scaler, epoch)
        vv  = evaluate(model, vl, criterion, device)
        dt  = time.time() - t0
        train_hist.append(tr); val_hist.append(vv)
        print(f"  {epoch:>4} | {tr:>11.5f} | {vv:>11.5f} | "
              f"{lr:>10.2e} | {dt:>8.1f}{gpu_mem_str(device)}", flush=True)
        if vv < best_val - 1e-6:
            best_val   = vv
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt); wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"  Early stop epoch {epoch}  (best val={best_val:.5f})")
                break

    print(f"\n  Total: {time.time()-t_total:.1f}s  |  Best val={best_val:.5f}")

    model.load_state_dict(best_state); model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, xm, y, _ in el:
            preds.append(model(x.to(device, non_blocking=True),
                               xm.to(device, non_blocking=True)).cpu().numpy())
            trues.append(y.numpy())
    preds = np.concatenate(preds); trues = np.concatenate(trues)
    N, H, M = preds.shape
    pi = scaler.inverse_transform(preds.reshape(-1, M)).reshape(N, H, M)
    ti = scaler.inverse_transform(trues.reshape(-1, M)).reshape(N, H, M)

    m = compute_metrics(ti, pi)
    print(f"\n  [Test — original scale]")
    for k, v in m.items(): print(f"    {k:<8}: {v:.4f}")

    return {"pred_len": pred_len, "metrics": m,
            "preds": pi, "trues": ti,
            "train_hist": train_hist, "val_hist": val_hist,
            "scaler": scaler}


# ─────────────────────────────────────────────────────────────────────────────
# 12. PLOT / SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def plot_all(results, cfg):
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(18, 3.8 * n))
    if n == 1: axes = [axes]
    scatter_alpha = 0.05 if cfg.enc_in > 32 else 0.10
    scatter_label = "univariate" if cfg.features == "S" else "all channels"

    for row, res in enumerate(results):
        H = res["pred_len"]
        true = res["trues"][0, :, -1]
        pred = res["preds"][0, :, -1]

        ax = axes[row][0]
        ax.plot(true, color="#1D9E75", lw=1.6, label="Ground truth")
        ax.plot(pred, color="#D85A30", lw=1.6, ls="--", label="VQF-F")
        ax.set_title(f"H={H}  '{cfg.target}' channel, sample 0  "
                     f"(original scale)", fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=.3)
        ax.set_xlabel("Step (hourly)"); ax.set_ylabel("Load (kWh)")

        ax2 = axes[row][1]
        yf = res["trues"].ravel(); pf = res["preds"].ravel()
        ax2.scatter(yf, pf, s=1, alpha=scatter_alpha, color="#378ADD")
        lim = [min(yf.min(), pf.min()), max(yf.max(), pf.max())]
        ax2.plot(lim, lim, "r--", lw=1)
        ax2.set_title(f"H={H}  Scatter ({scatter_label})", fontsize=9)
        ax2.set_xlabel("True"); ax2.set_ylabel("Pred"); ax2.grid(alpha=.3)

        ax3 = axes[row][2]
        ax3.plot(res["train_hist"], color="#534AB7", label="Train MSE")
        ax3.plot(res["val_hist"],   color="#D85A30", ls="--", label="Val MSE")
        ax3.set_title(f"H={H}  Training curve (normalised MSE)", fontsize=9)
        ax3.set_xlabel("Epoch"); ax3.set_ylabel("MSE")
        ax3.legend(fontsize=8); ax3.grid(alpha=.3)

    mode_tag = (f"S  target='{cfg.target}'" if cfg.features == "S"
                else f"M  enc_in={cfg.enc_in}")
    plt.suptitle(f"VQF-F (broadcast) on Electricity  [{mode_tag}]",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    safe_tag = (f"S_{cfg.target.replace('/', '_').replace(' ', '_')}"
                if cfg.features == "S" else f"M_enc{cfg.enc_in}")
    fname = f"vqff_electricity_{safe_tag}_fast_results.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {fname}")
    plt.close()

def print_summary(results, cfg):
    mode_str = (f"features=S  target='{cfg.target}'" if cfg.features == "S"
                else f"features=M  enc_in={cfg.enc_in}")
    print("\n" + "=" * 74)
    print(f"{'VQF-F (broadcast) on Electricity — Summary (original scale)':^74}")
    print(f"  {mode_str}  |  n_qubits={cfg.N_QUBITS}  depth={cfg.CIRCUIT_DEPTH}")
    print("=" * 74)
    print(f"{'H':>6} | {'MSE':>10} | {'MAE':>10} | {'RMSE':>10} | {'MAPE%':>8}")
    print("-" * 74)
    for r in results:
        m = r["metrics"]
        print(f"{r['pred_len']:>6} | "
              f"{m['MSE']:>10.4f} | {m['MAE']:>10.4f} | "
              f"{m['RMSE']:>10.4f} | {m['MAPE%']:>8.2f}")
    print("=" * 74)


# ─────────────────────────────────────────────────────────────────────────────
# 13. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser(
        description="VQF-F on Electricity — broadcast batched GPU edition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("--data_path",     type=str,   required=True)
    pa.add_argument("--features",      type=str,   default="M",
                    choices=["M", "S"])
    pa.add_argument("--target",        type=str,   default="")
    pa.add_argument("--n_channels",    type=int,   default=-1,
                    help="M mode: cap number of channels (-1=all 321)")
    pa.add_argument("--pred_lens",     type=int,   nargs="+",
                    default=[96, 192, 336, 720])
    pa.add_argument("--seq_len",       type=int,   default=96)
    pa.add_argument("--batch_size",    type=int,   default=32)
    pa.add_argument("--epochs",        type=int,   default=10)
    pa.add_argument("--lr",            type=float, default=1e-4)
    pa.add_argument("--patience",      type=int,   default=3)
    pa.add_argument("--no_amp",        action="store_true")
    pa.add_argument("--n_qubits",      type=int,   default=6)
    pa.add_argument("--depth",         type=int,   default=2)
    pa.add_argument("--use_lightning", action="store_true",
                    help="Try lightning.gpu (NOT recommended for n_qubits<=8)")
    pa.add_argument("--device",        type=str,   default="auto")
    pa.add_argument("--num_workers",   type=int,   default=4)
    args = pa.parse_args()

    cfg = Config()
    cfg.data_path     = args.data_path
    cfg.features      = args.features
    cfg.target        = args.target
    cfg.n_channels    = args.n_channels
    cfg.seq_len       = args.seq_len
    cfg.PRED_LENS     = args.pred_lens
    cfg.batch_size    = args.batch_size
    cfg.train_epochs  = args.epochs
    cfg.learning_rate = args.lr
    cfg.patience      = args.patience
    cfg.use_amp       = not args.no_amp
    cfg.N_QUBITS      = args.n_qubits
    cfg.CIRCUIT_DEPTH = args.depth
    cfg.use_lightning = args.use_lightning
    cfg.device        = args.device
    cfg.num_workers   = args.num_workers
    cfg.pin_memory    = True

    device     = setup_device(cfg.device)
    cfg.device = device

    if not os.path.isfile(cfg.data_path):
        raise FileNotFoundError(
            f"electricity.csv not found at '{cfg.data_path}'.\n"
            f"Download: https://archive.ics.uci.edu/ml/datasets/"
            f"ElectricityLoadDiagrams20112014")
    print(f"\n  Loading: {os.path.abspath(cfg.data_path)}")
    df = pd.read_csv(cfg.data_path)
    print(f"  Shape: {df.shape}  columns[:5]: {list(df.columns[:5])} …")

    feat_cols = resolve_feature_cols(df, cfg)

    print("=" * 72)
    print("  VQF-F (broadcast): Variational Quantum Forecaster on Electricity")
    print("  KEY FIX: PennyLane native parameter broadcasting")
    print("           Recommended backend: default.qubit (omit --use_lightning)")
    print("=" * 72)
    print(f"  data_path  : {cfg.data_path}")
    print(f"  features   : {cfg.features}")
    if cfg.features == "S":
        print(f"  target     : '{cfg.target}'  (enc_in=1)")
    else:
        date_col = _find_date_col(df)
        total_num = sum(1 for c in df.columns
                        if c != date_col and pd.api.types.is_numeric_dtype(df[c]))
        cap = "all" if args.n_channels <= 0 else f"capped at {args.n_channels}"
        print(f"  enc_in     : {cfg.enc_in}  ({cap} of {total_num} cols)")
        print(f"  plot col   : '{cfg.target}'")
    print(f"  pred_lens  : {cfg.PRED_LENS}")
    print(f"  batch_size : {cfg.batch_size}  seq_len: {cfg.seq_len}")
    print(f"  epochs     : {cfg.train_epochs}  patience: {cfg.patience}")
    print(f"  N_QUBITS   : {cfg.N_QUBITS}  depth: {cfg.CIRCUIT_DEPTH}")
    print(f"  device     : {device}  AMP: {cfg.use_amp and device.startswith('cuda')}")
    print(f"  lightning  : {cfg.use_lightning}")

    results = []
    for H in cfg.PRED_LENS:
        results.append(run_horizon(df, H, cfg, device, feat_cols))

    print_summary(results, cfg); plot_all(results, cfg)
    if device.startswith("cuda"):
        print(f"\n  Final {gpu_mem_str(device)}")
        torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
