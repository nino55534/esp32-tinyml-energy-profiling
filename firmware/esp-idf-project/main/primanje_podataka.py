import socket, json, datetime

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5005))
print("Slušam na portu 5005...")

while True:
    data, addr = sock.recvfrom(512)
    d = json.loads(data.decode())
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] temp={d.get('temp')}°C  hum={d.get('hum')}%  "
      f"CNN={d.get('cnn', 0):.4f} ({d.get('cnn_us', 0)}µs)")