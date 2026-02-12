import torch
import socket
import numpy as np
import os
import time
from torchvision.utils import save_image
from model import _Decoder
from CPAM import CPAM

# ================= device =================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ================= model params =================
B, C, H, W = 1, 8, 32, 32
c = 4

# ================= models =================
decoder = _Decoder(c=c).to(device)
cpam = CPAM(
    num_in_ch=3,
    num_out_ch=3,
    feature_channels=48,
    upscale=1
).to(device)

decoder.load_state_dict(
    torch.load("./deepjscc.pth", map_location=device),
    strict=False
)
cpam.load_state_dict(
    torch.load("./CPAM.pth", map_location=device),
    strict=False
)

decoder.eval()
cpam.eval()

print("[Server] Models loaded")

# ================= socket utils =================
def recv_all(conn, length):
    data = b''
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet:
            raise ConnectionError("Socket closed")
        data += packet
    return data

# ================= socket =================
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 12345))
server_socket.listen(1)

print("[Server] Waiting for connection...")
conn, addr = server_socket.accept()
print("[Server] Connected from", addr)

# ================= timing =================
decode_times = []
frame_id = 0
feature_bytes = C * H * W * 4  # float32

try:
    while True:
        #recv
        raw = recv_all(conn, feature_bytes)

        feature = torch.from_numpy(
            np.frombuffer(raw, dtype=np.float32).copy()
        ).to(device)
        feature = feature.view(B, C, H, W)

        #decode
        with torch.no_grad():
            t0 = time.perf_counter()
            recon = decoder(feature)
            snr = torch.tensor([13.0], device=device)
            recon = cpam(recon, snr)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

        dec_time = t1 - t0

        if frame_id > 0:
            decode_times.append(dec_time)

        print(
            f"[Server] Frame {frame_id:03d} | "
            f"Decode+CPAM: {dec_time*1000:.2f} ms"
        )

        frame_id += 1

except Exception as e:
    print("[Server] Error:", e)

finally:
    conn.close()
    server_socket.close()
    print("[Server] Socket closed")

    if len(decode_times) > 0:
        print("\n===== Decode Timing Summary (excluding first frame) =====")
        print(f"Average Decode+CPAM Time: {np.mean(decode_times)*1000:.2f} ms")
        print(f"FPS (server side): {1.0 / np.mean(decode_times):.2f}")
