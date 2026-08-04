import socket

ip = "10.108.31.151"
ports = [80, 443, 3000, 5000, 8000, 8080, 8765, 8888]

print(f"Scanning {ip} for open ports...")
for port in ports:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    result = s.connect_ex((ip, port))
    if result == 0:
        print(f"Port {port}: OPEN")
    else:
        print(f"Port {port}: closed")
    s.close()
