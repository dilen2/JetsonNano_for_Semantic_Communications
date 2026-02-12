import torch
import socket
import numpy as np
from model import _Decoder
from torchvision.utils import save_image
import os
import time

# ================= device =================
device = torch.device("cuda:0")

# ================= Decoder =================
decoder = _Decoder(c=4).to(device)
decoder.load_state_dict(
    torch.load("./out_split_model/split_deepjscc.pth", map_location=device),
    strict=False
)
decoder.eval()
print("[Server] Decoder loaded")

# ================= socket =================
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 12345))
server_socket.listen(1)

print("[Server] Waiting for connection...")
conn, addr = server_socket.accept()
print("[Server] Connected from:", addr)

# ================= fixed feature shape =================
B, C, H, W = 1, 8, 32, 32
feature_size = B * C * H * W * 4  # float32 = 4 bytes

# ================= timing =================
decode_times = []

frame_id = 0

try:
    while True:
        # ---------- recv ----------
        raw = b''
        while len(raw) < feature_size:
            packet = conn.recv(feature_size - len(raw))
            if not packet:
                raise ConnectionError("Socket closed")
            raw += packet

        feature = torch.from_numpy(
            np.frombuffer(raw, dtype=np.float32).copy()
        ).to(device)
        feature = feature.view(B, C, H, W)

        # ---------- decode ----------
        with torch.no_grad():
            t0 = time.perf_counter()
            recon = decoder(feature)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

        dec_t = t1 - t0

        # ⚠️ 跳过第 0 帧统计
        if frame_id > 0:
            decode_times.append(dec_t)

        print(
            f"[Server] Frame {frame_id:03d} | "
            f"Decode: {dec_t*1000:.2f} ms"
        )

        # ---------- optional save ----------
        # os.makedirs("./decoded_images", exist_ok=True)
        # save_image(recon, f"./decoded_images/decoded_{frame_id}.png")

        frame_id += 1

except Exception as e:
    print("[Server] Error:", e)

finally:
    conn.close()
    server_socket.close()

    if len(decode_times) > 0:
        print("\n===== Decode Timing Summary (excluding first frame) =====")
        print(f"Avg Decode Time: {np.mean(decode_times)*1000:.2f} ms")
