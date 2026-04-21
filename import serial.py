import serial
import time

port = "COM3"
print(f"Otváram {port}...")

with serial.Serial(port, 460800, timeout=2) as ser:
    ser.dtr = True    # ← toto ESP32-C3 potrebuje!
    time.sleep(7)
    ser.flushInput()
    print("Čítam dáta (Ctrl+C pre stop):")
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print(f">>> {line}")
        else:
            print("(timeout - žiadne dáta)")