import torch
import socket
import numpy as np
from model import _Encoder
from torchvision import transforms
from PIL import Image
import os
import time

# ================= device =================
device = torch.device("cuda:0")

# ================= Encoder =================
encoder = _Encoder(c=4).to(device)
encoder.load_state_dict(
    torch.load("./out_split_model/split_deepjscc.pth", map_location=device),
    strict=False
)
encoder.eval()

# ================= Dataset =================
data_root = "./kodak_test"
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

images = [
    transform(Image.open(os.path.join(data_root, f)).convert("RGB")).unsqueeze(0)
    for f in sorted(os.listdir(data_root)) if f.endswith(".png")
]

# ================= Socket =================
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', 12345))
print("[Client] Connected")

# ================= Timing =================
encode_times = []
total_times = []

for idx, img in enumerate(images):
    img = img.to(device)

    t0 = time.perf_counter()

    # ---------- encode ----------
    with torch.no_grad():
        t_enc_start = time.perf_counter()
        feature = encoder(img)
        torch.cuda.synchronize()          # CUDA 同步，保证时间准确
        t_enc_end = time.perf_counter()

    # ---------- send ----------
    feature_bytes = feature.cpu().numpy().astype(np.float32).tobytes()
    client_socket.sendall(feature_bytes)

    t1 = time.perf_counter()

    enc_t = t_enc_end - t_enc_start
    tot_t = t1 - t0

    # ⚠️ 跳过第 0 帧的统计
    if idx > 0:
        encode_times.append(enc_t)
        total_times.append(tot_t)

    print(
        f"[Client] Frame {idx:03d} | "
        f"Encode: {enc_t*1000:.2f} ms | "
        f"Total: {tot_t*1000:.2f} ms"
    )

client_socket.close()
print("[Client] Socket closed")

# ================= Summary =================
print("\n===== Timing Summary (excluding first frame) =====")
print(f"Avg Encode Time : {np.mean(encode_times)*1000:.2f} ms")
print(f"Avg Total Time  : {np.mean(total_times)*1000:.2f} ms")
