"""
Double Pendulum Simulator - PyQt6 Version
Advanced GUI with real-time physics simulation
Multi-threaded architecture for smooth performance
"""

import sys
from anyio import current_time
import numpy as np
from collections import deque
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QLabel, QSlider, QPushButton,
                             QLineEdit, QCheckBox, QGridLayout, QGroupBox, QComboBox,
                             QSpinBox, QDoubleSpinBox, QFileDialog)
import csv
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush
import pyqtgraph as pg
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QLocale
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient
import serial
import serial.tools.list_ports
from PyQt6.QtCore import QThread
import time
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QLocale
from pyqtgraph.exporters import ImageExporter


class SerialReader(QThread):
    raw_data_received = pyqtSignal(float, float)

    def __init__(self, port, baudrate=921600):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self.ser = None  # NOVÉ - referencia na otvorený port
        self.calibrate_requested = False  # NOVÉ - flag

    def run(self):
        try:
            with serial.Serial(self.port, self.baudrate, timeout=0.01) as ser:
                self.ser = ser  # NOVÉ - ulož referenciu
                time.sleep(2)
                ser.flushInput()
                self.running = True
                buffer = b""
                while self.running:
                    # NOVÉ - ak bola požiadavka na kalibráciu, pošli
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
                                    t2,t1 = map(float, line.split(','))                        
                                    self.raw_data_received.emit(t1, t2)
                                except ValueError:
                                    continue
        except Exception as e:
            print(f"Serial Error: {e}")
        finally:
            self.ser = None

    def send_calibration(self):
        """Nastaví flag - signál CAL pošle run() loop."""
        self.calibrate_requested = True

   

    def stop(self):
        self.running = False
        self.wait()
      

class PendulumSimulator(QObject):
    """Physics simulation running in separate thread"""
    
    data_ready = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        
        # Simulation parameters - kalibrované pre fyzické kyvadlo
        self.L1 = 0.135
        self.L2 = 0.087
        self.m1 = 0.50
        self.m2 = 0.34
        self.g = 9.81
        self.dt = 0.001  # 1ms timestep
        self.damping = 0.70
        # State vector [theta1, theta2, omega1, omega2]
        self.state = np.array([120 * np.pi / 180, -100 * np.pi / 180, 0.0, 0.0])
        self.time = 0.0
        
        # Data buffers (circular buffers)
        self.buffer_size = 100000
        self.buffer_index = 0
        self.time_buffer = np.zeros(self.buffer_size)
        self.theta1_buffer = np.zeros(self.buffer_size)
        self.theta2_buffer = np.zeros(self.buffer_size)
        self.omega1_buffer = np.zeros(self.buffer_size)
        self.omega2_buffer = np.zeros(self.buffer_size)
        
        # Control flags
        self.is_running = False
        
        # Timer for simulation
        self.timer = QTimer()
        self.timer.timeout.connect(self.step)

        # comunication with serial thread
        self.serial_thread = None
        
    def start(self):
        """Start simulation"""
        self.is_running = True
        self.timer.start(1)  # 1ms timer for 1000 Hz
        
    def stop(self):
        """Stop simulation"""
        self.is_running = False
        self.timer.stop()
        
    def reset(self, theta1, theta2):
        """Reset simulation with new initial conditions"""
        self.state = np.array([theta1, theta2, 0.0, 0.0])
        self.time = 0.0
        self.buffer_index = 0
        
    def set_parameters(self, L1, L2, m1, m2, damping):
        """Update pendulum parameters"""
        self.L1 = L1
        self.L2 = L2
        self.m1 = m1
        self.m2 = m2
        self.damping = damping
        
    def pendulum_derivatives(self, state):
        """Calculate derivatives for double pendulum equations"""
        th1, th2, w1, w2 = state
        delta = th2 - th1
        
        den1 = (self.m1 + self.m2) * self.L1 - self.m2 * self.L1 * np.cos(delta)**2
        den2 = (self.L2 / self.L1) * den1
        
        dth1 = w1
        dth2 = w2
        
        dw1 = (self.m2 * self.L1 * w1**2 * np.sin(delta) * np.cos(delta) +
               self.m2 * self.g * np.sin(th2) * np.cos(delta) +
               self.m2 * self.L2 * w2**2 * np.sin(delta) -
               (self.m1 + self.m2) * self.g * np.sin(th1)) / den1
        
        dw2 = (-self.m2 * self.L2 * w2**2 * np.sin(delta) * np.cos(delta) +
               (self.m1 + self.m2) * (self.g * np.sin(th1) * np.cos(delta) -
               self.L1 * w1**2 * np.sin(delta) - self.g * np.sin(th2))) / den2
        
        dw1 -= self.damping * w1
        dw2 -= self.damping * w2
        
        return np.array([dth1, dth2, dw1, dw2])
    
    def rk4_step(self):
        """Runge-Kutta 4th order integration"""
        k1 = self.pendulum_derivatives(self.state)
        k2 = self.pendulum_derivatives(self.state + 0.5 * self.dt * k1)
        k3 = self.pendulum_derivatives(self.state + 0.5 * self.dt * k2)
        k4 = self.pendulum_derivatives(self.state + self.dt * k3)
        
        self.state += self.dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        self.time += self.dt
        
    def step(self):
        """Single simulation step"""
        if not self.is_running:
            return
            
        # Perform RK4 integration
        self.rk4_step()
        
        # Store in circular buffer
        idx = self.buffer_index % self.buffer_size
        self.time_buffer[idx] = self.time
        self.theta1_buffer[idx] = self.state[0]
        self.theta2_buffer[idx] = self.state[1]
        self.omega1_buffer[idx] = self.state[2]
        self.omega2_buffer[idx] = self.state[3]
        self.buffer_index += 1
        
        # Emit data every 30 steps (30 Hz update rate)
        if self.buffer_index % 30 == 0:
            self.emit_data()
    
    def emit_data(self):
        """Send current state to GUI"""
        # Calculate positions
        x1 = self.L1 * np.sin(self.state[0])
        y1 = self.L1 * np.cos(self.state[0])
        x2 = x1 + self.L2 * np.sin(self.state[1])
        y2 = y1 + self.L2 * np.cos(self.state[1])
        
        # Calculate energy
        E_kin = self.calculate_kinetic_energy()
        E_pot = self.calculate_potential_energy()
        
        data = {
            'time': self.time,
            'theta1': self.state[0],
            'theta2': self.state[1],
            'omega1': self.state[2],
            'omega2': self.state[3],
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
            'E_kinetic': E_kin,
            'E_potential': E_pot,
            'E_total': E_kin + E_pot
        }
        
        self.data_ready.emit(data)
    
    def calculate_kinetic_energy(self):
        """Calculate total kinetic energy"""
        th1, th2, w1, w2 = self.state
        v1_sq = (self.L1 * w1)**2
        v2_sq = (self.L1 * w1)**2 + (self.L2 * w2)**2 + \
                2 * self.L1 * self.L2 * w1 * w2 * np.cos(th1 - th2)
        return 0.5 * self.m1 * v1_sq + 0.5 * self.m2 * v2_sq
    
    def calculate_potential_energy(self):
        """Calculate total potential energy"""
        th1, th2 = self.state[0], self.state[1]
        y1 = -self.L1 * np.cos(th1)
        y2 = y1 - self.L2 * np.cos(th2)
        return self.m1 * self.g * y1 + self.m2 * self.g * y2
    
    def get_history(self, max_points=10000):
        """Get simulation history for plotting"""
        n = min(self.buffer_index, self.buffer_size)
        if n > max_points:
            indices = np.linspace(0, n-1, max_points, dtype=int)
        else:
            indices = np.arange(n)
        
        return {
            'time': self.time_buffer[indices],
            'theta1': self.theta1_buffer[indices],
            'theta2': self.theta2_buffer[indices],
            'omega1': self.omega1_buffer[indices],
            'omega2': self.omega2_buffer[indices]
        }


class PendulumCanvas(QWidget):
    """Custom widget for drawing pendulum animation"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 600)
        self.show_rods = True
        # Pendulum state
        self.x1, self.y1 = 0, 0
        self.x2, self.y2 = 0, 0
        self.L1, self.L2 = 2.0, 2.0
        
        # Trail
        self.trail = deque(maxlen=100) #100 length of a trail
        self.show_pendulum = True
        self.show_trail = True
        
        # Inicializuj počiatočnú pozíciu (120°, -10°)
        self.set_initial_position(120, -100)

        self.esp_coords = None
        
    def set_initial_position(self, theta1_deg, theta2_deg):
        """Set initial pendulum position (in degrees)"""
        theta1 = theta1_deg * np.pi / 180
        theta2 = theta2_deg * np.pi / 180
        
        # Vypočítaj pozície
        self.x1 = self.L1 * np.sin(theta1)
        self.y1 = self.L1 * np.cos(theta1)
        self.x2 = self.x1 + self.L2 * np.sin(theta2)
        self.y2 = self.y1 + self.L2 * np.cos(theta2)
        
        self.update()  # Prekreslí canvas
    
    def update_state(self, x1, y1, x2, y2):
        """Update pendulum positions"""
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        if self.show_trail:
            self.trail.append((x2, y2))
        self.update()
    def update_esp_state(self, x1, y1, x2, y2):
        self.esp_coords = (x1, y1, x2, y2)
        self.update()

    def paintEvent(self, event):
        """Vykreslenie kyvadla s 3D efektom (tiene a gradienty)"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # settup 
        w, h = self.width(), self.height()
        center_x, center_y = w / 2, h / 2
        scale = min(w, h) / (2.6 * (self.L1 + self.L2))
        
        
        pivot_x, pivot_y = center_x, center_y
        bob1_x = center_x + self.x1 * scale
        bob1_y = center_y + self.y1 * scale
        bob2_x = center_x + self.x2 * scale
        bob2_y = center_y + self.y2 * scale
        if self.show_pendulum:
            # shadow effect renderring
            shadow_offset = 5
            painter.setOpacity(0.2) # Priehľadnosť pre tieň
            painter.setPen(QPen(Qt.GlobalColor.black, 5))

            if self.show_rods:
                painter.drawLine(int(pivot_x + shadow_offset), int(pivot_y + shadow_offset), 
                                int(bob1_x + shadow_offset), int(bob1_y + shadow_offset))
                painter.drawLine(int(bob1_x + shadow_offset), int(bob1_y + shadow_offset), 
                                int(bob2_x + shadow_offset), int(bob2_y + shadow_offset))
            painter.setOpacity(1.0) 

                # 3d render for rods
            if self.show_rods:
                painter.setPen(QPen(QColor(51, 0, 51), 5))
                painter.drawLine(int(pivot_x), int(pivot_y), int(bob1_x), int(bob1_y))
                painter.drawLine(int(bob1_x), int(bob1_y), int(bob2_x), int(bob2_y))

             # 3d render for bobs
            for bx, by in [(bob1_x, bob1_y), (bob2_x, bob2_y)]:
            # Vytvorenie odlesku: stred je mierne vľavo hore (-5, -5)
                grad = QRadialGradient(bx - 4, by - 4, 12)
                grad.setColorAt(0, QColor(100, 40, 100)) # Svetlý bod (odlesk)
                grad.setColorAt(0.8, QColor(51, 0, 51))  # Základná farba
                grad.setColorAt(1, QColor(20, 0, 20))    # Tmavý okraj (tieň na guli)
                
                painter.setBrush(QBrush(grad))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(int(bx - 12), int(by - 12), 24, 24)
            
                # sted
            painter.setBrush(QBrush(QColor(150, 150, 150)))
            painter.drawEllipse(int(pivot_x - 6), int(pivot_y - 6), 12, 12)
        if self.esp_coords:
            ex1, ey1, ex2, ey2 = self.esp_coords

            # Modrá prerušovaná čiara pre ESP
            esp_pen = QPen(QColor(0, 150, 255, 200), 3, Qt.PenStyle.DashLine)
            painter.setPen(esp_pen)

            eb1x, eb1y = center_x + ex1 * scale, center_y + ey1 * scale
            eb2x, eb2y = center_x + ex2 * scale, center_y + ey2 * scale

            painter.drawLine(int(pivot_x), int(pivot_y), int(eb1x), int(eb1y))
            painter.drawLine(int(eb1x), int(eb1y), int(eb2x), int(eb2y))

            # Modré kĺby ESP kyvadla
            painter.setBrush(QBrush(QColor(0, 150, 255, 180)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(int(eb1x - 6), int(eb1y - 6), 12, 12)
            painter.drawEllipse(int(eb2x - 6), int(eb2y - 6), 12, 12)

        if self.show_trail and len(self.trail) > 1:
            painter.setPen(QPen(QColor(255, 255, 255, 150), 1))
            for i in range(1, len(self.trail)):
                x1_px = center_x + self.trail[i-1][0] * scale
                y1_px = center_y + self.trail[i-1][1] * scale
                x2_px = center_x + self.trail[i][0] * scale
                y2_px = center_y + self.trail[i][1] * scale
                painter.drawLine(int(x1_px), int(y1_px), int(x2_px), int(y2_px))

class DoublePendulumApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Double Pendulum Simulator - PyQt6")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create simulator
        self.simulator = PendulumSimulator()
        self.simulator.data_ready.connect(self.update_visualization)
        
        self.serial_thread = None
        self.last_esp_t1 = None          # ← PRIDAJ sem
        self.last_esp_t2 = None          # ← PRIDAJ sem
        self.last_canvas_update = 0  
        # Setup UI

        self.esp_omega1_data = []
        self.esp_omega2_data = []
        self.show_esp_on_graphs = False  # flag pre vykreslovanie
        self.esp_needs_graph_update = False
        self.graph_update_timer = QTimer()
        self.graph_update_timer.timeout.connect(self.update_esp_graphs)
        self.graph_update_timer.start(50)  # 20 FPS pre grafy
        self.setup_ui()
        
        # Initialize plots
        self.init_plots()
        
        
    def setup_ui(self):
        """Create user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_simulation_tab()
        self.create_graphs_tab()
        self.create_analysis_tab()
        self.create_esp32_tab()
        self.create_multi_measurement_tab()
        
    def create_simulation_tab(self):
        """Tab 1: Simulation and Control"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Left panel - controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Parameters group
        params_group = QGroupBox("Parameters")
        params_layout = QGridLayout()
        # Time slider
        params_layout.addWidget(QLabel("Simulation Time (s):"), 6, 0)
        self.time_limit_slider = QLineEdit("0.0")
        self.time_limit_slider.setFixedWidth(50)

        validator_time = QDoubleValidator(0.0, 3600.0, 1)
        validator_time.setLocale(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
        self.time_limit_slider.setValidator(validator_time)
        params_layout.addWidget(self.time_limit_slider, 6, 2)

        # Length sliders
        params_layout.addWidget(QLabel("L1 (m):"), 0, 0)
        self.L1_slider = QSlider(Qt.Orientation.Horizontal)
        self.L1_slider.setRange(10, 500)
        self.L1_slider.setValue(14)  # default 0.14 m, presný 0.135 cez edit field
        self.L1_slider.valueChanged.connect(self.update_parameters)
        params_layout.addWidget(self.L1_slider, 0, 1)
        self.L1_label = QLabel("0.140")
        params_layout.addWidget(self.L1_label, 0, 2)
        
        params_layout.addWidget(QLabel("L2 (m):"), 1, 0)
        self.L2_slider = QSlider(Qt.Orientation.Horizontal)
        self.L2_slider.setRange(10, 500)
        self.L2_slider.setValue(9)   # default 0.09 m, presný 0.087 cez edit field
        self.L2_slider.valueChanged.connect(self.update_parameters)
        params_layout.addWidget(self.L2_slider, 1, 1)
        self.L2_label = QLabel("0.090")
        params_layout.addWidget(self.L2_label, 1, 2)
        
        # Mass sliders
        params_layout.addWidget(QLabel("m1 (kg):"), 2, 0)
        self.m1_slider = QSlider(Qt.Orientation.Horizontal)
        self.m1_slider.setRange(10, 2000)
        self.m1_slider.setValue(50)  # 0.50 kg
        self.m1_slider.valueChanged.connect(self.update_parameters)
        params_layout.addWidget(self.m1_slider, 2, 1)
        self.m1_label = QLabel("0.50")
        params_layout.addWidget(self.m1_label, 2, 2)
        
        params_layout.addWidget(QLabel("m2 (kg):"), 3, 0)
        self.m2_slider = QSlider(Qt.Orientation.Horizontal)
        self.m2_slider.setRange(10, 2000)
        self.m2_slider.setValue(34)  # 0.34 kg
        self.m2_slider.valueChanged.connect(self.update_parameters)
        params_layout.addWidget(self.m2_slider, 3, 1)
        self.m2_label = QLabel("0.34")
        params_layout.addWidget(self.m2_label, 3, 2)
    
        # Angle sliders
        params_layout.addWidget(QLabel("θ1 (°):"), 4, 0)
        self.theta1_slider = QSlider(Qt.Orientation.Horizontal)
        self.theta1_slider.setRange(-180, 180)
        self.theta1_slider.setValue(120)
        self.theta1_slider.valueChanged.connect(self.update_initial_angles)
        params_layout.addWidget(self.theta1_slider, 4, 1)
        self.theta1_label = QLabel("120")
        params_layout.addWidget(self.theta1_label, 4, 2)
        
        params_layout.addWidget(QLabel("θ2 (°):"), 5, 0)
        self.theta2_slider = QSlider(Qt.Orientation.Horizontal)
        self.theta2_slider.setRange(-180, 180)
        self.theta2_slider.setValue(-100)
        self.theta2_slider.valueChanged.connect(self.update_initial_angles)
        params_layout.addWidget(self.theta2_slider, 5, 1)
        self.theta2_label = QLabel("-100")
        params_layout.addWidget(self.theta2_label, 5, 2)

        # Damping
        params_layout.addWidget(QLabel("Damping (c):"), 7, 0)
        self.damping_slider = QSlider(Qt.Orientation.Horizontal)
        self.damping_slider.setRange(0, 200)
        self.damping_slider.setValue(70)  # damping 0.70
        self.damping_slider.valueChanged.connect(self.update_parameters)
        params_layout.addWidget(self.damping_slider, 7, 1)
        self.damping_label = QLabel("0.70")
        params_layout.addWidget(self.damping_label, 7, 2)

        params_group.setLayout(params_layout)
        control_layout.addWidget(params_group)

        # Control buttons
        btn_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("Start Simulation")
        self.start_btn.clicked.connect(self.start_simulation)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Simulation")
        self.stop_btn.clicked.connect(self.stop_simulation)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_simulation)
        btn_layout.addWidget(self.reset_btn)
        
        self.default_btn = QPushButton("Default Settings")
        self.default_btn.clicked.connect(self.set_default_parameters)
        btn_layout.addWidget(self.default_btn)

        self.align_esp_btn = QPushButton("Align to ESP kyvadlo")
        self.align_esp_btn.clicked.connect(self.align_to_esp)
        self.align_esp_btn.setEnabled(False)
        btn_layout.addWidget(self.align_esp_btn)

        self.export_btn = QPushButton("Export data (CSV)")
        self.export_btn.clicked.connect(self.export_data)
        btn_layout.addWidget(self.export_btn)

        self.calibrate_btn = QPushButton("Rekalibrovať senzory (nula)")
        self.calibrate_btn.clicked.connect(self.calibrate_sensors)
        self.calibrate_btn.setEnabled(False)  # Vypnuté kým nie je ESP pripojené
        btn_layout.addWidget(self.calibrate_btn)
        control_layout.addLayout(btn_layout)

        # Edit fields validators
        validator_L = QDoubleValidator(0.1, 5.0, 2)
        validator_L.setLocale(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
        validator_L.setNotation(QDoubleValidator.Notation.StandardNotation)

        validator_m = QDoubleValidator(0.1, 20, 2)
        validator_m.setLocale(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
        validator_m.setNotation(QDoubleValidator.Notation.StandardNotation)

        # Edit fields
        self.L1_edit_field = QLineEdit("0.135")
        self.L1_edit_field.setFixedWidth(50)
        self.L1_edit_field.setValidator(validator_L)
        params_layout.addWidget(self.L1_edit_field, 0, 2)
        
        self.L2_edit_field = QLineEdit("0.087")
        self.L2_edit_field.setFixedWidth(50)
        self.L2_edit_field.setValidator(validator_L)
        params_layout.addWidget(self.L2_edit_field, 1, 2)
    
        self.m1_edit_field = QLineEdit("0.50")
        self.m1_edit_field.setFixedWidth(50)
        self.m1_edit_field.setValidator(validator_m)
        params_layout.addWidget(self.m1_edit_field, 2, 2)       
        
        self.m2_edit_field = QLineEdit("0.34")
        self.m2_edit_field.setFixedWidth(50)
        self.m2_edit_field.setValidator(validator_m)
        params_layout.addWidget(self.m2_edit_field, 3, 2)   
    
        self.L1_edit_field.returnPressed.connect(self.update_params_from_edit)
        self.L2_edit_field.returnPressed.connect(self.update_params_from_edit)
        self.m1_edit_field.returnPressed.connect(self.update_params_from_edit)
        self.m2_edit_field.returnPressed.connect(self.update_params_from_edit) 

        # Display options - CHECKBOXES
        display_group = QGroupBox("Display")
        display_main_layout = QVBoxLayout()
        checkbox_row_layout = QHBoxLayout()

        self.show_pendulum_cb = QCheckBox("Show Pendulum")
        self.show_pendulum_cb.setChecked(True)
        self.show_pendulum_cb.stateChanged.connect(self.toggle_pendulum)
        checkbox_row_layout.addWidget(self.show_pendulum_cb)
        
        self.show_trail_cb = QCheckBox("Show Trail")
        self.show_trail_cb.setChecked(True)
        self.show_trail_cb.stateChanged.connect(self.toggle_trail)
        checkbox_row_layout.addWidget(self.show_trail_cb)

        self.update_graphs_cb = QCheckBox("Allow Graphs")
        self.update_graphs_cb.setChecked(True)
        self.update_graphs_cb.stateChanged.connect(self.clear_all_graph_data)
        checkbox_row_layout.addWidget(self.update_graphs_cb)
        
        self.show_rods_cb = QCheckBox("Show Rods")
        self.show_rods_cb.setChecked(True)
        self.show_rods_cb.stateChanged.connect(self.toggle_rods)
        checkbox_row_layout.addWidget(self.show_rods_cb)

        # NOVÝ checkbox - Compare with ESP
        self.show_esp_graphs_cb = QCheckBox("Compare with ESP")
        self.show_esp_graphs_cb.setChecked(False)
        self.show_esp_graphs_cb.stateChanged.connect(self.toggle_esp_graphs)
        checkbox_row_layout.addWidget(self.show_esp_graphs_cb)
        
        display_main_layout.addLayout(checkbox_row_layout)
        display_group.setLayout(display_main_layout)
        control_layout.addWidget(display_group)
        
        # Status display
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        self.time_label = QLabel("Time: 0.00 s")
        self.energy_label = QLabel("Energy: 0.00 J")
        status_layout.addWidget(self.time_label)
        status_layout.addWidget(self.energy_label)
        status_group.setLayout(status_layout)
        control_layout.addWidget(status_group)
        
        control_layout.addStretch()
        
        # Right panel - visualization
        self.canvas = PendulumCanvas()
        
        # Add to tab layout
        layout.addWidget(control_panel, 1)
        layout.addWidget(self.canvas, 2)
        
        self.tabs.addTab(tab, "Simulation")
    def create_graphs_tab(self):
        tab = QWidget()
        layout = QGridLayout(tab)
        
        # θ1 Plot
        self.plot_theta1 = pg.PlotWidget(title="θ1 vs Čas")
        self.plot_theta1.setBackground('w')
        self.plot_theta1.setLabel('left', 'Uhol θ1', units='rad')
        self.plot_theta1.setLabel('bottom', 'Čas', units='s')
        self.plot_theta1.addLegend()
        self.curve_theta1 = self.plot_theta1.plot(pen=pg.mkPen('b', width=3), name='Simulácia')
        self.curve_esp_theta1 = self.plot_theta1.plot(pen=pg.mkPen('orange', width=2, style=Qt.PenStyle.DashLine), name='ESP')

        # θ2 Plot
        self.plot_theta2 = pg.PlotWidget(title="θ2 vs Čas")
        self.plot_theta2.setBackground('w')
        self.plot_theta2.setLabel('left', 'Uhol θ2', units='rad')
        self.plot_theta2.setLabel('bottom', 'Čas', units='s')
        self.plot_theta2.addLegend()
        self.curve_theta2 = self.plot_theta2.plot(pen=pg.mkPen('r', width=3), name='Simulácia')
        self.curve_esp_theta2 = self.plot_theta2.plot(pen=pg.mkPen('orange', width=2, style=Qt.PenStyle.DashLine), name='ESP')

        # ω1 Plot
        self.plot_omega1 = pg.PlotWidget(title="ω1 vs Čas")
        self.plot_omega1.setBackground('w')
        self.plot_omega1.setLabel('left', 'Uhlová rýchlosť ω1', units='rad/s')
        self.plot_omega1.setLabel('bottom', 'Čas', units='s')
        self.plot_omega1.addLegend()
        self.curve_omega1 = self.plot_omega1.plot(pen=pg.mkPen('g', width=3), name='Simulácia')
        self.curve_esp_omega1 = self.plot_omega1.plot(pen=pg.mkPen('orange', width=2, style=Qt.PenStyle.DashLine), name='ESP')

        # ω2 Plot
        self.plot_omega2 = pg.PlotWidget(title="ω2 vs Čas")
        self.plot_omega2.setBackground('w')
        self.plot_omega2.setLabel('left', 'Uhlová rýchlosť ω2', units='rad/s')
        self.plot_omega2.setLabel('bottom', 'Čas', units='s')
        self.plot_omega2.addLegend()
        self.curve_omega2 = self.plot_omega2.plot(pen=pg.mkPen('m', width=3), name='Simulácia')
        self.curve_esp_omega2 = self.plot_omega2.plot(pen=pg.mkPen('orange', width=2, style=Qt.PenStyle.DashLine), name='ESP')
        
        layout.addWidget(self.plot_theta1, 0, 0)
        layout.addWidget(self.plot_theta2, 0, 1)
        layout.addWidget(self.plot_omega1, 1, 0)
        layout.addWidget(self.plot_omega2, 1, 1)
        
        self.tabs.addTab(tab, "Time Graphs")
            
    def create_analysis_tab(self):
        """Tab 3: Phase Diagrams & Energy"""
        tab = QWidget()
        layout = QGridLayout(tab)
        
        # Phase space 1
        self.plot_phase1 = pg.PlotWidget(title="Fázový priestor: θ1 vs ω1")
        self.plot_phase1.setBackground('w')
        self.plot_phase1.setLabel('left', 'Uhlová rýchlosť ω1', units='rad/s')
        self.plot_phase1.setLabel('bottom', 'Uhol θ1', units='rad')
        self.plot_phase1.addLegend()
        self.curve_phase1 = self.plot_phase1.plot(pen=pg.mkPen('b', width=3), name='Simulácia')
        self.curve_esp_phase1 = self.plot_phase1.plot(pen=pg.mkPen('orange', width=2, style=Qt.PenStyle.DashLine), name='ESP')

        # Phase space 2
        self.plot_phase2 = pg.PlotWidget(title="Fázový priestor: θ2 vs ω2")
        self.plot_phase2.setBackground('w')
        self.plot_phase2.setLabel('left', 'Uhlová rýchlosť ω2', units='rad/s')
        self.plot_phase2.setLabel('bottom', 'Uhol θ2', units='rad')
        self.plot_phase2.addLegend()
        self.curve_phase2 = self.plot_phase2.plot(pen=pg.mkPen('r', width=3), name='Simulácia')
        self.curve_esp_phase2 = self.plot_phase2.plot(pen=pg.mkPen('orange', width=2, style=Qt.PenStyle.DashLine), name='ESP')

        # Configuration space
        self.plot_config = pg.PlotWidget(title="Konfiguračný priestor: θ1 vs θ2")
        self.plot_config.setBackground('w')
        self.plot_config.setLabel('left', 'Uhol θ2', units='rad')
        self.plot_config.setLabel('bottom', 'Uhol θ1', units='rad')
        self.plot_config.addLegend()
        self.curve_config = self.plot_config.plot(pen=pg.mkPen('m', width=3), name='Simulácia')
        self.curve_esp_config = self.plot_config.plot(pen=pg.mkPen('orange', width=2, style=Qt.PenStyle.DashLine), name='ESP')

        # Energy plot
        self.plot_energy = pg.PlotWidget(title="Energia vs Čas")
        self.plot_energy.setBackground('w')
        self.plot_energy.setLabel('left', 'Energia', units='J')
        self.plot_energy.setLabel('bottom', 'Čas', units='s')
        self.plot_energy.addLegend()
        
        layout.addWidget(self.plot_phase1, 0, 0)
        layout.addWidget(self.plot_phase2, 0, 1)
        layout.addWidget(self.plot_config, 1, 0)
        layout.addWidget(self.plot_energy, 1, 1)
        
        self.tabs.addTab(tab, "Analysis")

    def init_plots(self):
        """Initialize plot data"""
        self.time_data = []
        self.E_kin_data = []
        self.E_pot_data = []
        self.E_tot_data = []

        self.esp_time_data = []
        self.esp_theta1_data = []
        self.esp_theta2_data = []
        self.esp_start_time = None
        
    def update_parameters(self):
            
        L1 = self.L1_slider.value() / 100.0
        L2 = self.L2_slider.value() / 100.0
        m1 = self.m1_slider.value() / 100.0
        m2 = self.m2_slider.value() / 100.0
        # NOVÉ: Čítanie damping slidera
        damping = self.damping_slider.value() / 100.0 

        # Aktualizácia textových polí (ak ich používaš)
        self.L1_edit_field.setText(f"{L1:.2f}")
        self.L2_edit_field.setText(f"{L2:.2f}")
        self.m1_edit_field.setText(f"{m1:.2f}")
        self.m2_edit_field.setText(f"{m2:.2f}")
        # Aktualizácia labelu pre damping
        self.damping_label.setText(f"{damping:.2f}") 

        # Poslanie všetkých parametrov do simulátora (vrátane damping)
        self.simulator.set_parameters(L1, L2, m1, m2, damping)
        self.canvas.L1, self.canvas.L2 = L1, L2

        if not self.simulator.is_running:
            self.update_initial_angles()
    
    def update_initial_angles(self):
        """Update initial angles from sliders (len keď simulácia nebeží)"""
        if not self.simulator.is_running:
            theta1 = self.theta1_slider.value()
            theta2 = self.theta2_slider.value()
            
            self.theta1_label.setText(str(theta1))
            self.theta2_label.setText(str(theta2))
            
            # Aktualizuj canvas
            self.canvas.set_initial_position(theta1, theta2)
        
    def start_simulation(self):
        # Reset simulátora na aktuálnu pozíciu zo sliderov
        theta1 = self.theta1_slider.value() * np.pi / 180
        theta2 = self.theta2_slider.value() * np.pi / 180
        self.simulator.reset(theta1, theta2)

        # Vyčisti všetky bufferé pre čistý graf od t=0
        self.time_data = []
        self.E_kin_data = []
        self.E_pot_data = []
        self.E_tot_data = []
        self.canvas.trail.clear()

        # ESP bufferé - aby sim aj meranie začali od rovnakého t=0
        self.esp_time_data = []
        self.esp_theta1_data = []
        self.esp_theta2_data = []
        self.esp_start_time = time.time()

        self.simulator.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.theta1_slider.setEnabled(False)
        self.theta2_slider.setEnabled(False)
        self.default_btn.setEnabled(False)
        self.time_limit_slider.setEnabled(False)
            
    def stop_simulation(self):
        self.simulator.stop()
    
    # Grafy ZOSTÁVAJÚ pre porovnanie a export
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.theta1_slider.setEnabled(True)
        self.theta2_slider.setEnabled(True)
        self.default_btn.setEnabled(True)
        self.time_limit_slider.setEnabled(True)
        
    def reset_simulation(self):
        self.simulator.stop()
        self.canvas.trail.clear()
        self.default_btn.setEnabled(True)
        self.init_plots()
        
        # Simulačné buffery
        self.time_data = []
        self.E_kin_data = []
        self.E_pot_data = []
        self.E_tot_data = []        

        # ESP buffery
        self.esp_time_data = []
        self.esp_theta1_data = []
        self.esp_theta2_data = []
        self.esp_omega1_data = []
        self.esp_omega2_data = []

        theta1 = self.theta1_slider.value() * np.pi / 180
        theta2 = self.theta2_slider.value() * np.pi / 180
        self.simulator.reset(theta1, theta2)
        self.canvas.set_initial_position(self.theta1_slider.value(), self.theta2_slider.value())
        
        # Vymaž všetky simulačné krivky
        self.curve_theta1.setData([], [])
        self.curve_theta2.setData([], [])
        self.curve_omega1.setData([], [])
        self.curve_omega2.setData([], [])
        self.curve_phase1.setData([], [])
        self.curve_phase2.setData([], [])
        self.curve_config.setData([], [])
        self.plot_energy.clear()
        
        # Vymaž všetky ESP krivky
        self.esp_curve1.setData([], [])
        self.esp_curve2.setData([], [])
        self.curve_esp_theta1.setData([], [])
        self.curve_esp_theta2.setData([], [])
        self.curve_esp_omega1.setData([], [])
        self.curve_esp_omega2.setData([], [])
        self.curve_esp_phase1.setData([], [])
        self.curve_esp_phase2.setData([], [])
        self.curve_esp_config.setData([], [])
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.theta1_slider.setEnabled(True)
        self.theta2_slider.setEnabled(True)
        
    def toggle_pendulum(self, state):
        """Toggle pendulum visibility"""
        self.canvas.show_pendulum = state == Qt.CheckState.Checked.value
        self.canvas.update()
        
    def toggle_trail(self, state):
        """Toggle trail visibility"""
        self.canvas.show_trail = state == Qt.CheckState.Checked.value
        if not self.canvas.show_trail:
            self.canvas.trail.clear()
        self.canvas.update()

    def toggle_rods(self, state):
        # rods visibility
        self.canvas.show_rods = (state == 2 or state == Qt.CheckState.Checked.value)
        self.canvas.update()
    def update_visualization(self, data):
        """Update all visualizations with new data"""
        current_time = data['time']
        
       # Try block for checking time limit
        try:
            limit_text = self.time_limit_slider.text().replace(',', '.')
            limit = float(limit_text)
 
            if limit > 0 and current_time >= limit:
                self.stop_simulation() 
                
                self.time_label.setText(f"Time: {limit:.2f} s (Limit reached)")
                return 
        except ValueError:
            pass 

        # Update canvas
        self.canvas.update_state(data['x1'], data['y1'], data['x2'], data['y2'])
        # Update status
        self.time_label.setText(f"Time: {data['time']:.2f} s")
        self.energy_label.setText(f"Energy: {data['E_total']:.2f} J")
        
        allow_plotting = self.update_graphs_cb.isChecked()
        if not allow_plotting:
            self.clear_all_graph_data()
            return

        # Time graphs (vždy keď beží, nie len keď je tab otvorený)
        history = self.simulator.get_history()
        self.curve_theta1.setData(history['time'], history['theta1'])
        self.curve_theta2.setData(history['time'], history['theta2'])
        self.curve_omega1.setData(history['time'], history['omega1'])
        self.curve_omega2.setData(history['time'], history['omega2'])

        # Analysis graphs
        history_short = self.simulator.get_history(max_points=5000)
        self.curve_phase1.setData(history_short['theta1'], history_short['omega1'])
        self.curve_phase2.setData(history_short['theta2'], history_short['omega2'])
        self.curve_config.setData(history_short['theta1'], history_short['theta2'])

        # Energy plot
        self.time_data.append(data['time'])
        self.E_kin_data.append(data['E_kinetic'])
        self.E_pot_data.append(data['E_potential'])
        self.E_tot_data.append(data['E_total'])

        if len(self.time_data) > 10000:
            self.time_data = self.time_data[-10000:]
            self.E_kin_data = self.E_kin_data[-10000:]
            self.E_pot_data = self.E_pot_data[-10000:]
            self.E_tot_data = self.E_tot_data[-10000:]

        self.plot_energy.clear()
        self.plot_energy.plot(self.time_data, self.E_kin_data, pen='g', name='Kinetic')
        self.plot_energy.plot(self.time_data, self.E_pot_data, pen='b', name='Potential')
        self.plot_energy.plot(self.time_data, self.E_tot_data, pen='r', name='Total')

    def set_default_parameters(self):
        """Nastaví kalibrované parametre pre fyzické kyvadlo."""
        self.L1_slider.setValue(14)   # ~0.14 m
        self.L2_slider.setValue(9)    # ~0.09 m
        self.m1_slider.setValue(50)   # 0.50 kg
        self.m2_slider.setValue(34)   # 0.34 kg
        self.damping_slider.setValue(70)  # 0.70

        self.L1_edit_field.setText("0.135")
        self.L2_edit_field.setText("0.087")
        self.m1_edit_field.setText("0.50")
        self.m2_edit_field.setText("0.34")

        self.update_initial_angles()
        self.update_params_from_edit_for_L1()
        self.update_params_from_edit_for_L2()
        self.update_parameters()

    def align_to_esp(self):
        """Zarovná simulované kyvadlo na aktuálnu pozíciu fyzického kyvadla."""
        if self.last_esp_t1 is None or self.last_esp_t2 is None:
            print("Žiadne ESP dáta nedostupné.")
            return
        
        # Zastav simuláciu ak beží
        if self.simulator.is_running:
            self.stop_simulation()
        
        # Zaokrúhli na celé stupne (slidery sú integer)
        t1 = int(round(self.last_esp_t1))
        t2 = int(round(self.last_esp_t2))
        
        # Obmedz na rozsah sliderov (-180 až 180)
        t1 = max(-180, min(180, t1))
        t2 = max(-180, min(180, t2))
        
        # Nastav slidery
        self.theta1_slider.setValue(t1)
        self.theta2_slider.setValue(t2)
        
        # Aktualizuj labely a canvas
        self.update_initial_angles()
        
        print(f"Zarovnané na ESP: θ1={t1}°, θ2={t2}°")

    def clear_all_graph_data(self):
       
        self.curve_theta1.setData([], [])
        self.curve_theta2.setData([], [])
        self.curve_omega1.setData([], [])
        self.curve_omega2.setData([], [])
        self.curve_phase1.setData([], [])
        self.curve_phase2.setData([], [])
        self.curve_config.setData([], [])
        self.plot_energy.clear()

        self.curve_esp_theta1.setData([], [])
        self.curve_esp_theta2.setData([], [])
        self.curve_esp_omega1.setData([], [])
        self.curve_esp_omega2.setData([], [])
        self.curve_esp_phase1.setData([], [])
        self.curve_esp_phase2.setData([], [])
        self.curve_esp_config.setData([], [])

    def update_params_from_edit_for_L1(self):
        """Pomocná metóda - vyvolá update parametrov z L1 edit fieldu."""
        try:
            val = float(self.L1_edit_field.text().replace(',', '.'))
            self.L1_slider.setValue(int(val * 100))
        except ValueError:
            pass

    def update_params_from_edit_for_L2(self):
        """Pomocná metóda - vyvolá update parametrov z L2 edit fieldu."""
        try:
            val = float(self.L2_edit_field.text().replace(',', '.'))
            self.L2_slider.setValue(int(val * 100))
        except ValueError:
            pass

    def update_params_from_edit(self):

        sender = self.sender()  #
        try:
            # text to float
            val = float(sender.text().replace(',', '.'))
            
            # edit field slct
            if sender == self.L1_edit_field:
                self.L1_slider.setValue(int(val * 100))
            elif sender == self.L2_edit_field:
                self.L2_slider.setValue(int(val * 100))
            elif sender == self.m1_edit_field:
                self.m1_slider.setValue(int(val * 100))
            elif sender == self.m2_edit_field:
                self.m2_slider.setValue(int(val * 100))
                
           
            self.update_parameters()
            
        except ValueError:
           
            self.update_parameters()

    def create_esp32_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        
        # --- Horný panel (Zmenšený) ---
        connection_group = QGroupBox("Pripojenie ESP32-C3 Mini")
        connection_group.setMaximumHeight(150) # Obmedzíme výšku panelu
        g_layout = QGridLayout()
        
        self.port_selector = QComboBox()
        self.refresh_ports()
        
        self.connect_btn = QPushButton("Pripojiť ESP32")
        self.connect_btn.clicked.connect(self.toggle_esp_connection)
        
        g_layout.addWidget(QLabel("Port:"), 0, 0)
        g_layout.addWidget(self.port_selector, 0, 1)
        g_layout.addWidget(self.connect_btn, 1, 0, 1, 2)
        
        self.live_data_label = QLabel("Uhly z ESP32: θ1: 0°, θ2: 0°")
        g_layout.addWidget(self.live_data_label, 2, 0, 1, 2)
        
        connection_group.setLayout(g_layout)
        main_layout.addWidget(connection_group)
        
        # --- Grafy pre ESP32 Dáta ---
        # Graf pre Theta 1
        self.esp_plot1 = pg.PlotWidget(title="Live θ1 [rad]")
        self.esp_plot1.setBackground('k')
        self.esp_curve1 = self.esp_plot1.plot(pen='b') # Modrá čiara
        main_layout.addWidget(self.esp_plot1)
        
        # Graf pre Theta 2
        self.esp_plot2 = pg.PlotWidget(title="Live θ2 [rad]")
        self.esp_plot2.setBackground('k')
        self.esp_curve2 = self.esp_plot2.plot(pen='r') # Červená čiara
        main_layout.addWidget(self.esp_plot2)
        # checkbox for second pendulum fromesp32
        self.show_esp_pendulum_cb = QCheckBox("Zobraziť ESP kyvadlo v simulácii")
        self.show_esp_pendulum_cb.setChecked(False)
        g_layout.addWidget(self.show_esp_pendulum_cb, 3, 0, 1, 2)

        self.tabs.addTab(tab, "ESP32 Dáta")

    def create_multi_measurement_tab(self):
        """Tab pre nahrávanie a porovnávanie viacerých ESP meraní."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Inicializuj buffer pre merania
        self.multi_measurements = []
        self.current_measurement_index = 0
        self.is_recording_multi = False
        self.multi_record_start_time = None
        self.multi_target_count = 10
        self.multi_target_duration = 3.0
        self.multi_target_theta1 = 60.0
        self.multi_target_theta2 = 60.0
        self.multi_target_tolerance = 3.0
        self.multi_series_active = False

        # ===== Ovládanie =====
        control_group = QGroupBox("Nastavenie série meraní")
        control_layout = QGridLayout()

        control_layout.addWidget(QLabel("Počet meraní:"), 0, 0)
        self.multi_count_spin = QSpinBox()
        self.multi_count_spin.setRange(2, 50)
        self.multi_count_spin.setValue(10)
        control_layout.addWidget(self.multi_count_spin, 0, 1)

        control_layout.addWidget(QLabel("Trvanie merania (s):"), 0, 2)
        self.multi_duration_spin = QSpinBox()
        self.multi_duration_spin.setRange(1, 30)
        self.multi_duration_spin.setValue(3)
        control_layout.addWidget(self.multi_duration_spin, 0, 3)

        control_layout.addWidget(QLabel("Cieľový θ1 [°]:"), 1, 0)
        self.multi_target_theta1_spin = QDoubleSpinBox()
        self.multi_target_theta1_spin.setRange(-180.0, 180.0)
        self.multi_target_theta1_spin.setSingleStep(1.0)
        self.multi_target_theta1_spin.setDecimals(1)
        self.multi_target_theta1_spin.setValue(60.0)
        control_layout.addWidget(self.multi_target_theta1_spin, 1, 1)

        control_layout.addWidget(QLabel("Cieľový θ2 [°]:"), 1, 2)
        self.multi_target_theta2_spin = QDoubleSpinBox()
        self.multi_target_theta2_spin.setRange(-180.0, 180.0)
        self.multi_target_theta2_spin.setSingleStep(1.0)
        self.multi_target_theta2_spin.setDecimals(1)
        self.multi_target_theta2_spin.setValue(60.0)
        control_layout.addWidget(self.multi_target_theta2_spin, 1, 3)

        control_layout.addWidget(QLabel("Tolerancia [±°]:"), 2, 0)
        self.multi_tolerance_spin = QDoubleSpinBox()
        self.multi_tolerance_spin.setRange(0.5, 30.0)
        self.multi_tolerance_spin.setSingleStep(0.5)
        self.multi_tolerance_spin.setDecimals(1)
        self.multi_tolerance_spin.setValue(3.0)
        control_layout.addWidget(self.multi_tolerance_spin, 2, 1)

        self.multi_position_label = QLabel(
            "ESP θ1: ---°  θ2: ---°  |  Status: čakám na ESP dáta"
        )
        self.multi_position_label.setStyleSheet(
            "font-weight: bold; padding: 5px; background-color: #444; color: white;"
        )
        control_layout.addWidget(self.multi_position_label, 3, 0, 1, 4)

        self.multi_start_btn = QPushButton("Štart série meraní")
        self.multi_start_btn.clicked.connect(self.start_multi_series)
        control_layout.addWidget(self.multi_start_btn, 4, 0, 1, 2)

        self.multi_record_btn = QPushButton("Spustiť meranie 1/10")
        self.multi_record_btn.clicked.connect(self.record_single_measurement)
        self.multi_record_btn.setEnabled(False)
        control_layout.addWidget(self.multi_record_btn, 4, 2, 1, 2)

        self.multi_reset_btn = QPushButton("Reset série")
        self.multi_reset_btn.clicked.connect(self.reset_multi_series)
        control_layout.addWidget(self.multi_reset_btn, 5, 0, 1, 2)

        self.multi_export_btn = QPushButton("Exportovať CSV")
        self.multi_export_btn.clicked.connect(self.export_multi_csv)
        self.multi_export_btn.setEnabled(False)
        control_layout.addWidget(self.multi_export_btn, 5, 2, 1, 2)

        self.multi_status_label = QLabel("Pripravený - klikni 'Štart série meraní'")
        control_layout.addWidget(self.multi_status_label, 6, 0, 1, 4)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # ===== Grafy θ1 a θ2 =====
        graph_layout = QHBoxLayout()

        self.multi_plot_theta1 = pg.PlotWidget(title="θ1 - všetky merania")
        self.multi_plot_theta1.setBackground('w')
        self.multi_plot_theta1.getAxis('left').enableAutoSIPrefix(False)
        self.multi_plot_theta1.getAxis('bottom').enableAutoSIPrefix(False)
        self.multi_plot_theta1.setLabel('left', 'θ1 [rad]')
        self.multi_plot_theta1.setLabel('bottom', 'čas [s]')
        self.multi_plot_theta1.addLegend()
        graph_layout.addWidget(self.multi_plot_theta1)

        self.multi_plot_theta2 = pg.PlotWidget(title="θ2 - všetky merania")
        self.multi_plot_theta2.setBackground('w')
        self.multi_plot_theta2.getAxis('left').enableAutoSIPrefix(False)
        self.multi_plot_theta2.getAxis('bottom').enableAutoSIPrefix(False)
        self.multi_plot_theta2.setLabel('left', 'θ2 [rad]')
        self.multi_plot_theta2.setLabel('bottom', 'čas [s]')
        self.multi_plot_theta2.addLegend()
        graph_layout.addWidget(self.multi_plot_theta2)

        layout.addLayout(graph_layout)

        self.tabs.addTab(tab, "Multi-meranie")

    def start_multi_series(self):
        """Inicializuje sériu meraní."""
        if self.serial_thread is None or not self.serial_thread.isRunning():
            self.multi_status_label.setText("CHYBA: Najprv pripoj ESP na tabe 'ESP32 Dáta'!")
            return

        self.multi_target_count = self.multi_count_spin.value()
        self.multi_target_duration = float(self.multi_duration_spin.value())
        self.multi_target_theta1 = self.multi_target_theta1_spin.value()
        self.multi_target_theta2 = self.multi_target_theta2_spin.value()
        self.multi_target_tolerance = self.multi_tolerance_spin.value()
        self.multi_measurements = []
        self.current_measurement_index = 0
        self.multi_series_active = True

        self.multi_plot_theta1.clear()
        self.multi_plot_theta2.clear()

        self.multi_count_spin.setEnabled(False)
        self.multi_duration_spin.setEnabled(False)
        self.multi_target_theta1_spin.setEnabled(False)
        self.multi_target_theta2_spin.setEnabled(False)
        self.multi_tolerance_spin.setEnabled(False)
        self.multi_start_btn.setEnabled(False)
        self.multi_record_btn.setEnabled(True)
        self.multi_record_btn.setText(f"Spustiť meranie 1/{self.multi_target_count}")
        self.multi_export_btn.setEnabled(False)

        self.multi_status_label.setText(
            f"Séria pripravená. Zdvihni kyvadlo na θ1≈{self.multi_target_theta1:.0f}°, "
            f"θ2≈{self.multi_target_theta2:.0f}° (±{self.multi_target_tolerance:.0f}°)"
        )

    def check_target_position(self):
        """Skontroluje či sú aktuálne ESP uhly v tolerancii voči cieľu."""
        if not hasattr(self, 'esp_theta1_data') or len(self.esp_theta1_data) == 0:
            return (False, None, None, None, None)

        N_avg = min(5, len(self.esp_theta1_data))
        current_t1_rad = np.mean(self.esp_theta1_data[-N_avg:])
        current_t2_rad = np.mean(self.esp_theta2_data[-N_avg:])
        current_t1_deg = np.degrees(current_t1_rad)
        current_t2_deg = np.degrees(current_t2_rad)

        diff1 = abs(current_t1_deg - self.multi_target_theta1)
        diff2 = abs(current_t2_deg - self.multi_target_theta2)

        in_tolerance = (diff1 <= self.multi_target_tolerance and
                        diff2 <= self.multi_target_tolerance)

        return (in_tolerance, current_t1_deg, current_t2_deg, diff1, diff2)

    def update_multi_position_indicator(self):
        """Aktualizuje live indikátor pozície ESP kyvadla."""
        if not hasattr(self, 'multi_position_label'):
            return

        result = self.check_target_position()
        in_tol, t1, t2, d1, d2 = result

        if t1 is None:
            self.multi_position_label.setText(
                "ESP θ1: ---°  θ2: ---°  |  Status: čakám na ESP dáta"
            )
            self.multi_position_label.setStyleSheet(
                "font-weight: bold; padding: 5px; background-color: #444; color: white;"
            )
            return

        if not self.multi_series_active:
            self.multi_position_label.setText(
                f"ESP θ1: {t1:+.1f}°  θ2: {t2:+.1f}°  |  Séria neaktívna"
            )
            self.multi_position_label.setStyleSheet(
                "font-weight: bold; padding: 5px; background-color: #444; color: white;"
            )
            return

        if in_tol:
            self.multi_position_label.setText(
                f"ESP θ1: {t1:+.1f}° (Δ{d1:.1f}°)  "
                f"θ2: {t2:+.1f}° (Δ{d2:.1f}°)  |  ✓ V TOLERANCII"
            )
            self.multi_position_label.setStyleSheet(
                "font-weight: bold; padding: 5px; "
                "background-color: #2d8a2d; color: white;"
            )
        else:
            self.multi_position_label.setText(
                f"ESP θ1: {t1:+.1f}° (Δ{d1:.1f}°)  "
                f"θ2: {t2:+.1f}° (Δ{d2:.1f}°)  |  ✗ MIMO TOLERANCIE"
            )
            self.multi_position_label.setStyleSheet(
                "font-weight: bold; padding: 5px; "
                "background-color: #aa2222; color: white;"
            )

    def record_single_measurement(self):
        """Začne nahrávanie jedného merania - len ak je v tolerancii."""
        if self.is_recording_multi:
            return

        in_tol, t1, t2, d1, d2 = self.check_target_position()

        if not in_tol:
            if t1 is None:
                self.multi_status_label.setText(
                    "CHYBA: Žiadne ESP dáta. Skontroluj pripojenie."
                )
            else:
                self.multi_status_label.setText(
                    f"✗ ODMIETNUTÉ: Aktuálne θ1={t1:+.1f}° (Δ{d1:.1f}°), "
                    f"θ2={t2:+.1f}° (Δ{d2:.1f}°). "
                    f"Cieľ θ1={self.multi_target_theta1:+.0f}°, "
                    f"θ2={self.multi_target_theta2:+.0f}° "
                    f"±{self.multi_target_tolerance:.0f}°. Uprav polohu kyvadla."
                )
            return

        self.current_measurement_data = {'time': [], 'theta1': [], 'theta2': []}
        self.is_recording_multi = True
        self.multi_record_start_time = time.time()

        self.multi_record_btn.setEnabled(False)
        self.multi_status_label.setText(
            f"NAHRÁVAM meranie {self.current_measurement_index + 1}/{self.multi_target_count} "
            f"(štart θ1={t1:+.1f}°, θ2={t2:+.1f}°) - "
            f"PUSTI KYVADLO! ({self.multi_target_duration:.0f} s)"
        )

        QTimer.singleShot(int(self.multi_target_duration * 1000), self.finish_single_measurement)

    def finish_single_measurement(self):
        """Ukončí aktuálne meranie a pripraví ďalšie alebo dokončí sériu."""
        if not self.is_recording_multi:
            return

        self.is_recording_multi = False

        if len(self.current_measurement_data['time']) > 5:
            self.multi_measurements.append(self.current_measurement_data.copy())

            color = self.get_measurement_color(self.current_measurement_index)
            t_arr = np.array(self.current_measurement_data['time'])
            th1_arr = np.array(self.current_measurement_data['theta1'])
            th2_arr = np.array(self.current_measurement_data['theta2'])

            self.multi_plot_theta1.plot(
                t_arr, th1_arr,
                pen=pg.mkPen(color=color, width=2),
                name=f'M{self.current_measurement_index + 1}'
            )
            self.multi_plot_theta2.plot(
                t_arr, th2_arr,
                pen=pg.mkPen(color=color, width=2),
                name=f'M{self.current_measurement_index + 1}'
            )

        self.current_measurement_index += 1

        if self.current_measurement_index >= self.multi_target_count:
            self.multi_series_active = False
            self.multi_record_btn.setEnabled(False)
            self.multi_record_btn.setText("Séria dokončená")
            self.multi_export_btn.setEnabled(True)
            self.multi_status_label.setText(
                f"✓ Séria dokončená! {self.current_measurement_index} meraní zaznamenaných. "
                f"Môžeš exportovať CSV."
            )
        else:
            self.multi_record_btn.setEnabled(True)
            self.multi_record_btn.setText(
                f"Spustiť meranie {self.current_measurement_index + 1}/{self.multi_target_count}"
            )
            self.multi_status_label.setText(
                f"Meranie {self.current_measurement_index} ukončené. "
                f"Vráť kyvadlo na cieľovú pozíciu pre ďalšie meranie."
            )

    def get_measurement_color(self, index):
        """Vráti farbu pre dané meranie (cyklicky)."""
        colors = [
            (228, 26, 28),
            (55, 126, 184),
            (77, 175, 74),
            (152, 78, 163),
            (255, 127, 0),
            (255, 255, 51),
            (166, 86, 40),
            (247, 129, 191),
            (153, 153, 153),
            (0, 0, 0),
        ]
        return colors[index % len(colors)]

    def reset_multi_series(self):
        """Reset série meraní."""
        self.multi_measurements = []
        self.current_measurement_index = 0
        self.is_recording_multi = False
        self.multi_series_active = False

        self.multi_plot_theta1.clear()
        self.multi_plot_theta2.clear()

        self.multi_count_spin.setEnabled(True)
        self.multi_duration_spin.setEnabled(True)
        self.multi_target_theta1_spin.setEnabled(True)
        self.multi_target_theta2_spin.setEnabled(True)
        self.multi_tolerance_spin.setEnabled(True)
        self.multi_start_btn.setEnabled(True)
        self.multi_record_btn.setEnabled(False)
        self.multi_record_btn.setText("Spustiť meranie 1/10")
        self.multi_export_btn.setEnabled(False)

        self.multi_status_label.setText("Pripravený - klikni 'Štart série meraní'")

    def export_multi_csv(self):
        """Export všetkých meraní do jedného CSV súboru."""
        if not self.multi_measurements:
            self.multi_status_label.setText("Žiadne dáta na export.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Uložiť multi-meranie ako CSV",
            f"multi_meranie_{len(self.multi_measurements)}.csv",
            "CSV Files (*.csv)"
        )

        if not filename:
            return

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['measurement_id', 'time_s', 'theta1_rad', 'theta2_rad'])

                for idx, m in enumerate(self.multi_measurements):
                    for i in range(len(m['time'])):
                        writer.writerow([idx + 1, m['time'][i], m['theta1'][i], m['theta2'][i]])

            self.multi_status_label.setText(f"✓ Exportované do: {filename}")
        except Exception as e:
            self.multi_status_label.setText(f"CHYBA pri exporte: {e}")

    def refresh_ports(self):

        self.port_selector.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_selector.addItems(ports)

    def toggle_esp_connection(self):
        # ─── ODPOJENIE ───
        if self.serial_thread is not None and self.serial_thread.isRunning():
            self.serial_thread.stop()
            self.serial_thread = None

            self.connect_btn.setText("Pripojiť ESP32")
            self.connect_btn.setStyleSheet("")

            self.align_esp_btn.setEnabled(False)
            self.calibrate_btn.setEnabled(False)

            self.canvas.esp_coords = None
            self.canvas.update()
            print("ESP32 odpojené.")
            return

        # ─── PRIPOJENIE ───
        port = self.port_selector.currentText()
        if not port:
            return

        self.esp_time_data = []
        self.esp_theta1_data = []
        self.esp_theta2_data = []
        self.esp_omega1_data = []
        self.esp_omega2_data = []
        self.esp_start_time = time.time()

        self.serial_thread = SerialReader(port)
        self.serial_thread.raw_data_received.connect(self.process_esp_data)
        self.serial_thread.start()

        self.connect_btn.setText("Odpojiť ESP32")
        self.connect_btn.setStyleSheet("background-color: #ff4c4c; color: white;")
        self.align_esp_btn.setEnabled(True)
        self.calibrate_btn.setEnabled(True)
        print(f"Pripájanie k {port}...")

    def toggle_esp_graphs(self, state):
        self.show_esp_on_graphs = (state == 2 or state == Qt.CheckState.Checked.value)
        
        if not self.show_esp_on_graphs:
            # Vymaž ESP krivky
            self.curve_esp_theta1.setData([], [])
            self.curve_esp_theta2.setData([], [])
            self.curve_esp_omega1.setData([], [])
            self.curve_esp_omega2.setData([], [])
            self.curve_esp_phase1.setData([], [])
            self.curve_esp_phase2.setData([], [])
            self.curve_esp_config.setData([], [])

    def process_esp_data(self, t1, t2):

        if self.esp_start_time is None:
            self.esp_start_time = time.time()
        current_time = time.time() - self.esp_start_time

        # Dead zone - aktívna IBA keď simulácia nebeží
        if not self.simulator.is_running:
            THRESHOLD = 1.0
            if abs(t1) < THRESHOLD:
                t1 = 0.0
            if abs(t2) < THRESHOLD:
                t2 = 0.0

        self.last_esp_t1 = t1
        self.last_esp_t2 = t2
        r1, r2 = np.radians(t1), np.radians(t2)

        # Kreslenie ESP kyvadla (vždy, aj keď sim nebeží)
        if self.show_esp_pendulum_cb.isChecked():
            x1 = self.simulator.L1 * np.sin(r1)
            y1 = self.simulator.L1 * np.cos(r1)
            x2 = x1 + self.simulator.L2 * np.sin(r2)
            y2 = y1 + self.simulator.L2 * np.cos(r2)
            self.canvas.update_esp_state(x1, y1, x2, y2)
        else:
            self.canvas.esp_coords = None
            self.canvas.update()

        # Multi-meranie - aktualizuj live indikátor pozície (vždy, aj bez simulácie)
        if hasattr(self, 'multi_position_label'):
            self.update_multi_position_indicator()

        # Multi-meranie - ak práve nahráme, ulož aj sem (nezávisle od simulácie)
        if hasattr(self, 'is_recording_multi') and self.is_recording_multi:
            elapsed = time.time() - self.multi_record_start_time
            if elapsed <= self.multi_target_duration:
                self.current_measurement_data['time'].append(elapsed)
                self.current_measurement_data['theta1'].append(r1)
                self.current_measurement_data['theta2'].append(r2)

        # Buffer plníme iba keď sim beží
        if not self.simulator.is_running:
            return

        self.esp_time_data.append(current_time)
        self.esp_theta1_data.append(r1)
        self.esp_theta2_data.append(r2)

        # Omega — centrálna diferencia cez WINDOW bodov
        WINDOW = 5
        if len(self.esp_theta1_data) >= WINDOW:
            dt_total = self.esp_time_data[-1] - self.esp_time_data[-WINDOW]
            if dt_total > 0:
                w1 = (self.esp_theta1_data[-1] - self.esp_theta1_data[-WINDOW]) / dt_total
                w2 = (self.esp_theta2_data[-1] - self.esp_theta2_data[-WINDOW]) / dt_total
            else:
                w1 = w2 = 0.0
        else:
            w1 = w2 = 0.0
        self.esp_omega1_data.append(w1)
        self.esp_omega2_data.append(w2)

        # Limit bufferu
        MAX_POINTS = 2000
        if len(self.esp_time_data) > MAX_POINTS:
            self.esp_time_data = self.esp_time_data[-MAX_POINTS:]
            self.esp_theta1_data = self.esp_theta1_data[-MAX_POINTS:]
            self.esp_theta2_data = self.esp_theta2_data[-MAX_POINTS:]
            self.esp_omega1_data = self.esp_omega1_data[-MAX_POINTS:]
            self.esp_omega2_data = self.esp_omega2_data[-MAX_POINTS:]

        self.esp_needs_graph_update = True

        self.esp_needs_graph_update = True

    def update_esp_graphs(self):
        """Timer callback - prekresľuje ESP grafy len počas simulácie."""
        if not self.esp_needs_graph_update:
            return
        self.esp_needs_graph_update = False
        
        if not self.simulator.is_running:
            return
        
        if not self.esp_time_data:
            return
        
        # Downsampling
        n = len(self.esp_time_data)
        if n > 500:
            step = n // 500
            t_slice = self.esp_time_data[::step]
            th1_slice = self.esp_theta1_data[::step]
            th2_slice = self.esp_theta2_data[::step]
            w1_slice = self.esp_omega1_data[::step]
            w2_slice = self.esp_omega2_data[::step]
        else:
            t_slice = self.esp_time_data
            th1_slice = self.esp_theta1_data
            th2_slice = self.esp_theta2_data
            w1_slice = self.esp_omega1_data
            w2_slice = self.esp_omega2_data
        
        # ESP Dáta tab
        self.esp_curve1.setData(t_slice, th1_slice)
        self.esp_curve2.setData(t_slice, th2_slice)
        
        # Time Graphs + Analysis - len ak zaškrtnutý Compare with ESP
        if self.show_esp_on_graphs:
            self.curve_esp_theta1.setData(t_slice, th1_slice)
            self.curve_esp_theta2.setData(t_slice, th2_slice)
            self.curve_esp_omega1.setData(t_slice, w1_slice)
            self.curve_esp_omega2.setData(t_slice, w2_slice)
            self.curve_esp_phase1.setData(th1_slice, w1_slice)
            self.curve_esp_phase2.setData(th2_slice, w2_slice)
            self.curve_esp_config.setData(th1_slice, th2_slice)

    def export_data(self):
        """Exportuje simulačné aj ESP dáta do CSV súborov."""
        from PyQt6.QtWidgets import QFileDialog
        import csv
        
        folder = QFileDialog.getExistingDirectory(self, "Vyber priečinok pre export")
        if not folder:
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Simulačné dáta
        sim_history = self.simulator.get_history()
        sim_file = f"{folder}/simulation_{timestamp}.csv"
        
        try:
            with open(sim_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['time_s', 'theta1_rad', 'theta2_rad', 'omega1_rad_s', 'omega2_rad_s'])
                for i in range(len(sim_history['time'])):
                    writer.writerow([
                        sim_history['time'][i],
                        sim_history['theta1'][i],
                        sim_history['theta2'][i],
                        sim_history['omega1'][i],
                        sim_history['omega2'][i]
                    ])
            print(f"✓ Simulation exported: {sim_file}")
        except Exception as e:
            print(f"Export error (sim): {e}")
        
        # ESP dáta
        if self.esp_time_data:
            esp_file = f"{folder}/esp_{timestamp}.csv"
            try:
                with open(esp_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['time_s', 'theta1_rad', 'theta2_rad', 'omega1_rad_s', 'omega2_rad_s'])
                    for i in range(len(self.esp_time_data)):
                        writer.writerow([
                            self.esp_time_data[i],
                            self.esp_theta1_data[i],
                            self.esp_theta2_data[i],
                            self.esp_omega1_data[i] if i < len(self.esp_omega1_data) else 0,
                            self.esp_omega2_data[i] if i < len(self.esp_omega2_data) else 0
                        ])
                print(f"✓ ESP exported: {esp_file}")
            except Exception as e:
                print(f"Export error (ESP): {e}")
        
        # PNG export grafov
        plots_to_export = [
            (self.plot_theta1,  f"{folder}/graf_theta1_{timestamp}.png"),
            (self.plot_theta2,  f"{folder}/graf_theta2_{timestamp}.png"),
            (self.plot_omega1,  f"{folder}/graf_omega1_{timestamp}.png"),
            (self.plot_omega2,  f"{folder}/graf_omega2_{timestamp}.png"),
            (self.plot_phase1,  f"{folder}/fazovy_theta1_{timestamp}.png"),
            (self.plot_phase2,  f"{folder}/fazovy_theta2_{timestamp}.png"),
            (self.plot_config,  f"{folder}/konfiguracny_{timestamp}.png"),
            (self.plot_energy,  f"{folder}/energia_{timestamp}.png"),
        ]
        for plot_widget, png_path in plots_to_export:
            try:
                self.export_plot_hires(plot_widget, png_path)
                print(f"✓ PNG exported: {png_path}")
            except Exception as e:
                print(f"Export error (PNG {png_path}): {e}")

        self.time_label.setText(f"Exportované do: {folder}")

    def calibrate_sensors(self):
        """Pošle CAL signál do ESP32 aby sa rekalibroval offset."""
        if self.serial_thread is None or not self.serial_thread.isRunning():
            print("ESP32 nie je pripojené.")
            return
        
        try:
            # Získaj prístup k serial portu cez thread
            self.serial_thread.send_calibration()
            self.time_label.setText("Rekalibrácia poslaná — podrž kyvadlo dole 1s")
            print("CAL signál poslaný do ESP32")
        except Exception as e:
            print(f"Chyba kalibrácie: {e}")

    def export_plot_hires(self, plot_widget, filename, width=3200):
        exporter = ImageExporter(plot_widget.plotItem)
        exporter.parameters()['width'] = width
        exporter.export(filename)
        
def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = DoublePendulumApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()