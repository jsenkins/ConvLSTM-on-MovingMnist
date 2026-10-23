import numpy as np, cv2, matplotlib.pyplot as plt

vids = (train[:50, :, 0].numpy() * 255).round().astype(np.uint8)   

dx_all, dy_all = [], []
for v in vids:
    for s in range(19):                       
        prev, nxt = v[s], v[s + 1]
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx, dy = flow[..., 0], flow[..., 1]
        keep = (prev > 51) & (nxt > 51) & (np.hypot(dx, dy) > 0.5)
        dx_all.append(dx[keep]); dy_all.append(dy[keep])

dx = np.concatenate(dx_all); dy = np.concatenate(dy_all)
n_pairs = 50 * 19
speed = np.hypot(dx, dy)
theta = np.degrees(np.arctan2(-dy, dx)) % 360
median = np.median(speed)
print(f"pairs={n_pairs}  kept pixels={len(speed)}  median speed={median:.3f} px/frame")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
edges = np.linspace(0, 360, 19)
ax[0].hist(theta, bins=edges, weights=np.full(len(theta), 1 / len(theta)), edgecolor="k")
ax[0].axhline(1 / 18, color="r", ls="--", label="uniform (1/18)")
ax[0].set_xticks(range(0, 361, 60)); ax[0].set_xlabel("flow direction θ (deg; 0 = right, 90 = up)")
ax[0].set_ylabel("fraction of kept pixels"); ax[0].set_title("(a) direction"); ax[0].legend()

ax[1].hist(speed, bins=20, weights=np.full(len(speed), 1 / len(speed)), edgecolor="k")
ax[1].axvline(median, color="r", ls="--", label=f"median = {median:.3g}")
ax[1].set_xlabel("flow speed (pixels / frame)"); ax[1].set_ylabel("fraction of kept pixels")
ax[1].set_title("(b) speed"); ax[1].legend()
plt.tight_layout(); plt.savefig("figure1.png", dpi=200); plt.show()

h, _ = np.histogram(theta, bins=edges); h = h / h.sum()
print("bin fractions:", np.round(h, 3))
print("max |bin - 1/18| =", np.abs(h - 1/18).max().round(4))
print("share with speed > 3 px/frame:", (speed > 3).mean().round(4))
