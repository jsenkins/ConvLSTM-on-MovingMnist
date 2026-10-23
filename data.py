import torch

TRAIN = slice(0, 8000)
VAL = slice(8000, 9000)
TEST = slice(9000, 10000)

def _to_video_tensor(raw):
    raw = torch.as_tensor(raw)
    if raw.ndim == 4:                       
        raw = raw.unsqueeze(2)
    if raw.ndim == 5 and raw.shape[1] == 1 and raw.shape[2] == 20:
        raw = raw.transpose(1, 2)           
    assert raw.ndim == 5 and tuple(raw.shape[1:]) == (20, 1, 64, 64), tuple(raw.shape)
    return raw


def load_moving_mnist(root="./data", download=True):
    
    from torchvision.datasets import MovingMNIST
    ds = MovingMNIST(root=root, split=None, download=download)
    raw = getattr(ds, "data", None)
    if raw is None:                         
        raw = torch.stack([torch.as_tensor(ds[i]) for i in range(len(ds))])
    videos = _to_video_tensor(raw)
    assert videos.shape[0] == 10_000, videos.shape
    return videos.to(torch.float32) / 255.0

def split(videos):
    
    
    assert videos.shape[0] == 10_000
    return videos[TRAIN], videos[VAL], videos[TEST]

