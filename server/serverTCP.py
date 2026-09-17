import socket

TCP_IP = "0.0.0.0"
TCP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   
sock.bind((TCP_IP, TCP_PORT))
sock.listen(1)  

print(f"TCP server pokrenut, slusa na portu {TCP_PORT}...")

while True:
    print("cekam novu vezu od ESP32...")
    conn, addr = sock.accept()
    print(f"Veza uspostavljena sa: {addr}")

    try:
        data = conn.recv(1024)
        if data:
            print(f"Primljeno: {data.decode()}")
    except Exception as e:
        print(f"Greška: {e}")
    finally:
        conn.close()
        print("Veza zatvorena.")