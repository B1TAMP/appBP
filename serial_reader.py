import time
import serial
import serial.tools.list_ports
from PyQt6.QtCore import QThread, pyqtSignal


class SerialReader(QThread):
    """Čítanie sériových dát z ESP32 v samostatnom vlákne."""

    raw_data_received = pyqtSignal(float, float)  # theta1, theta2 (stupne)

    def __init__(self, port, baudrate=921600):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self.ser = None
        self.calibrate_requested = False

    def run(self):
        try:
            with serial.Serial(self.port, self.baudrate, timeout=0.01) as ser:
                self.ser = ser
                time.sleep(2)
                ser.flushInput()
                self.running = True
                buffer = b""
                while self.running:
                    if self.calibrate_requested:
                        try:
                            ser.write(b"CAL\n")
                            self.calibrate_requested = False
                        except Exception as e:
                            print(f"Send CAL failed: {e}")

                    data = ser.read(ser.in_waiting or 1)
                    if data:
                        buffer += data
                        while b'\n' in buffer:
                            line, buffer = buffer.split(b'\n', 1)
                            line = line.decode('utf-8', errors='ignore').strip()
                            if line:
                                try:
                                    # CSV formát: "theta1_remote,theta2_local"
                                    t2, t1 = map(float, line.split(','))
                                    self.raw_data_received.emit(t1, t2)
                                except ValueError:
                                    continue
        except Exception as e:
            print(f"Serial Error: {e}")
        finally:
            self.ser = None

    def send_calibration(self):
        """Nastaví flag — CAL príkaz pošle run() loop v ďalšej iterácii."""
        self.calibrate_requested = True

    def stop(self):
        self.running = False
        self.wait()
