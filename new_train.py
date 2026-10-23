

import time, json, os, torch
from checking import symmetries
from baselineConvLSTM import ConvLSTMForecaster      

from model_d import ConvLSTMForecasterD

def build_model(cfg):
    if cfg.get("arch", "A") == "A":
        return ConvLSTMForecaster()
    return ConvLSTMForecasterD(**cfg["arch_kwargs"])

def make_windows(videos, vidx, k, t):
    """videos (N,20,1,64,64); vidx, k, t: (B,) long tensors (t = 1-based start frame).
    Returns X (B,5,1,64,64), Y (B,5,1,64,64) padded, mask (B,5): which targets exist."""
    ar = torch.arange(5, device=videos.device)
    base = (t - 1)[:, None]
    ii = base + k[:, None] * ar                    
    ti = base + k[:, None] * (5 + ar)              # 0-based target frame indices
    assert int(ii.max()) <= 19, "input window runs past frame 20"
    mask = ti <= 19
    X = videos[vidx[:, None], ii]
    Y = videos[vidx[:, None], ti.clamp(max=19)]    # padded entries are masked out
    return X, Y, mask


def masked_loss(pred, Y, mask):
    """Squared error averaged over pixels and the AVAILABLE targets of each video,
    then averaged over videos (every video counts equally)."""
    mse_h = ((pred - Y) ** 2).mean(dim=(2, 3, 4))          # (B,5)
    m = mask.float()
    return ((mse_h * m).sum(1) / m.sum(1)).mean()


def sample_batch(cfg, videos, vidx, gen, device):
    B = len(vidx)
    speeds = torch.tensor(cfg["speeds"])
    k = speeds[torch.randint(len(speeds), (B,), generator=gen)]
    last = (20 - 6 * k) if cfg["augmented"] else (20 - 9 * k)    # largest allowed start
    t = 1 + (torch.rand(B, generator=gen) * last).long()         # uniform over 1..last
    X, Y, mask = make_windows(videos, vidx.to(device), k.to(device), t.to(device))
    if cfg["sym_aug"]:
        g = torch.randint(8, (B,), generator=gen).to(device)     # same g for X and Y
        for gi in range(8):
            s = g == gi
            if s.any():
                X[s], Y[s] = symmetries(X[s], gi), symmetries(Y[s], gi)
    return X, Y, mask


@torch.no_grad()
def mse_per_horizon(model, videos, k=1, t=1, bs=32):
    """Mean over videos of the per-horizon pixel MSE (length 5; nan if never scored)."""
    model.eval()
    dev = next(model.parameters()).device
    tot, cnt = torch.zeros(5, device=dev), torch.zeros(5, device=dev)
    for s in range(0, len(videos), bs):
        vidx = torch.arange(s, min(s + bs, len(videos)), device=dev)
        kk = torch.full_like(vidx, k); tt = torch.full_like(vidx, t)
        X, Y, mask = make_windows(videos, vidx, kk, tt)
        mse_h = ((model(X) - Y) ** 2).mean(dim=(2, 3, 4))
        tot += (mse_h * mask).sum(0); cnt += mask.sum(0)
    return (tot / cnt.clamp_min(1)).where(cnt > 0, torch.full_like(tot, float("nan"))).cpu()



def train_one(cfg, seed, train, val, device, epochs=30, out="/kaggle/working/ckpt"):
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)                       # seeds initialisation (before building the model)
    model = build_model(cfg).to(device)
    gen = torch.Generator().manual_seed(seed)     # shuffling, window and augmentation sampling
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train, val = train.to(device), val.to(device)
    log = {"epoch_time": [], "train_loss": [], "val_E5": []}
    for ep in range(epochs):
        model.train()
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        perm = torch.randperm(len(train), generator=gen)
        run, nb = 0.0, 0
        for s in range(0, len(perm), 32):
            X, Y, mask = sample_batch(cfg, train, perm[s:s + 32], gen, device)
            loss = masked_loss(model(X), Y, mask)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run += loss.item(); nb += 1
        if device.type == "cuda": torch.cuda.synchronize()
        log["epoch_time"].append(time.perf_counter() - t0)
        log["train_loss"].append(run / nb)
        log["val_E5"].append(float(mse_per_horizon(model, val).mean()))
        print(f"{cfg['name']} seed{seed} ep{ep+1:2d}  train {log['train_loss'][-1]:.5f}  "
              f"val E5 {log['val_E5'][-1]:.5f}  {log['epoch_time'][-1]:.1f}s")
    log["mean_epoch_time"] = sum(log["epoch_time"]) / epochs
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "seed": seed, "log": log},
               f"{out}/{cfg['name']}_s{seed}.pt")     # final-epoch weights only
    return model, log
