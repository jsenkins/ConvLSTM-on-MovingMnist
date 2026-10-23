import torch

T_FRAMES = 20 

def window_frames(t, k=1):

    inp=[t+k*i for i in range(5)]
    assert inp[-1] <= T_FRAMES, f"input frames {inp} exceed {T_FRAMES}"
    tgt = [t + k * (5 + i) for i in range(5)]
    return inp, [f for f in tgt if f <= T_FRAMES]
def allowed_starts(k, augmented):
    last = T_FRAMES - 6 * k if augmented else T_FRAMES - 9 * k
    return list(range(1, last + 1))

def window(video, t, k=1, dim=0):
    inp, tgt = window_frames(t, k)
    take = lambda fs: video.index_select(dim, torch.tensor([f - 1 for f in fs], device=video.device))
    return take(inp), take(tgt)


def M(x):
    mirror=torch.flip(x, dims=(-1,))
    return mirror

def R(x, a=1):
    rotate=torch.rot90(x, k=a, dims=(-2, -1))
    return rotate

def symmetries(x,g):
    y=R(x,g%4)
    if g>=4:
        return M(y)
    return y

def inverse(y,g):
    if g>=4:
        y=M(y)
    return R(y, -(g%4))


