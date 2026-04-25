import sys
import csv
import time

import numpy as np
import pyqtgraph as pg
import serial.tools.list_ports
from pyqtgraph.exporters import ImageExporter

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QSlider, QPushButton, QLineEdit, QCheckBox,
    QGridLayout, QGroupBox, QComboBox, QFileDialog,
)
from PyQt6.QtCore import Qt, QTimer, QLocale
from PyQt6.QtGui import QDoubleValidator

from serial_reader import SerialReader
from simulator import PendulumSimulator
from canvas import PendulumCanvas


class DoublePendulumApp(QMainWindow):
    """Hlavné okno aplikácie — obsahuje GUI, ESP komunikáciu a export."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Double Pendulum Simulator")
        self.setGeometry(100, 100, 1200, 800)

        # --- Jadro simulácie ---
        self.simulator = PendulumSimulator()
        self.simulator.data_ready.connect(self.update_visualization)

        # --- ESP stav ---
        self.serial_thread = None
        self.last_esp_t1 = None
        self.last_esp_t2 = None
        self.esp_start_time = None
        self.esp_time_data   = []
        self.esp_theta1_data = []
        self.esp_theta2_data = []
        self.esp_omega1_data = []
        self.esp_omega2_data = []
        self.esp_needs_graph_update = False
        self.show_esp_on_graphs = False

        # --- Energetické buffery (pre graf energie) ---
        self.time_data  = []
        self.E_kin_data = []
        self.E_pot_data = []
        self.E_tot_data = []

        # --- Timer pre ESP grafy (20 FPS) ---
        self.graph_update_timer = QTimer()
        self.graph_update_timer.timeout.connect(self.update_esp_graphs)
        self.graph_update_timer.start(50)

        self.setup_ui()
        self.init_plots()

    # ================================================================== #
    #  Budovanie UI                                                        #
    # ================================================================== #

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.create_simulation_tab()
        self.create_graphs_tab()
        self.create_analysis_tab()
        self.create_esp32_tab()

    # ------------------------------------------------------------------ #
    #  Tab 1: Simulácia                                                    #
    # ------------------------------------------------------------------ #

    def create_simulation_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        control_panel = QWidget()
        ctrl = QVBoxLayout(control_panel)
        ctrl.setAlignment(Qt.AlignmentFlag.AlignTop)

        ctrl.addWidget(self._build_params_group())
        ctrl.addLayout(self._build_control_buttons())
        ctrl.addWidget(self._build_display_group())
        ctrl.addWidget(self._build_status_group())
        ctrl.addStretch()

        self.canvas = PendulumCanvas()
        layout.addWidget(control_panel, 1)
        layout.addWidget(self.canvas, 2)
        self.tabs.addTab(tab, "Simulácia")

    def _build_params_group(self):
        group = QGroupBox("Parametre")
        grid = QGridLayout()

        # Pomocná funkcia: slider + edit pole v col 2 (bez duplikatného labelu)
        def add_param_row(row, label, attr, lo, hi, val, edit_field_attr, edit_text, validator):
            grid.addWidget(QLabel(label), row, 0)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(lo, hi)
            slider.setValue(val)
            slider.valueChanged.connect(self.update_parameters)
            setattr(self, attr + '_slider', slider)
            grid.addWidget(slider, row, 1)
            edit = QLineEdit(edit_text)
            edit.setFixedWidth(50)
            edit.setValidator(validator)
            edit.returnPressed.connect(self.update_params_from_edit)
            setattr(self, edit_field_attr, edit)
            grid.addWidget(edit, row, 2)

        # Validators
        val_L = QDoubleValidator(0.01, 5.0, 3)
        val_L.setLocale(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
        val_m = QDoubleValidator(0.01, 5.0, 3)
        val_m.setLocale(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))

        add_param_row(0, "L1 (m):",  "L1", 10,  5000,   56, "L1_edit_field", "0.056", val_L)
        add_param_row(1, "L2 (m):",  "L2", 10,  5000,   49, "L2_edit_field", "0.049", val_L)
        add_param_row(2, "m1 (kg):", "m1", 10,  5000,  500, "m1_edit_field", "0.500", val_m)
        add_param_row(3, "m2 (kg):", "m2", 10,  5000,  342, "m2_edit_field", "0.342", val_m)

        # Uhly (integer stupne)
        grid.addWidget(QLabel("θ1 (°):"), 4, 0)
        self.theta1_slider = QSlider(Qt.Orientation.Horizontal)
        self.theta1_slider.setRange(-180, 180)
        self.theta1_slider.setValue(120)
        self.theta1_slider.valueChanged.connect(self.update_initial_angles)
        grid.addWidget(self.theta1_slider, 4, 1)
        self.theta1_label = QLabel("120")
        grid.addWidget(self.theta1_label, 4, 2)

        grid.addWidget(QLabel("θ2 (°):"), 5, 0)
        self.theta2_slider = QSlider(Qt.Orientation.Horizontal)
        self.theta2_slider.setRange(-180, 180)
        self.theta2_slider.setValue(-100)
        self.theta2_slider.valueChanged.connect(self.update_initial_angles)
        grid.addWidget(self.theta2_slider, 5, 1)
        self.theta2_label = QLabel("-100")
        grid.addWidget(self.theta2_label, 5, 2)

        # Čas simulácie
        grid.addWidget(QLabel("Čas simulácie (s):"), 6, 0)
        self.time_limit_slider = QLineEdit("0.0")
        self.time_limit_slider.setFixedWidth(50)
        val_t = QDoubleValidator(0.0, 3600.0, 1)
        val_t.setLocale(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
        self.time_limit_slider.setValidator(val_t)
        grid.addWidget(self.time_limit_slider, 6, 2)

        # Tlmenie
        grid.addWidget(QLabel("Tlmenie (c):"), 7, 0)
        self.damping_slider = QSlider(Qt.Orientation.Horizontal)
        self.damping_slider.setRange(0, 200)
        self.damping_slider.setValue(2)
        self.damping_slider.valueChanged.connect(self.update_parameters)
        grid.addWidget(self.damping_slider, 7, 1)
        self.damping_label = QLabel("0.02")
        grid.addWidget(self.damping_label, 7, 2)

        group.setLayout(grid)
        return group

    def _build_control_buttons(self):
        layout = QVBoxLayout()

        self.start_btn = QPushButton("Spustiť simuláciu")
        self.start_btn.clicked.connect(self.start_simulation)
        layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Zastaviť simuláciu")
        self.stop_btn.clicked.connect(self.stop_simulation)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_simulation)
        layout.addWidget(self.reset_btn)

        self.default_btn = QPushButton("Predvolené nastavenia")
        self.default_btn.clicked.connect(self.set_default_parameters)
        layout.addWidget(self.default_btn)

        self.align_esp_btn = QPushButton("Zarovnať na ESP kyvadlo")
        self.align_esp_btn.clicked.connect(self.align_to_esp)
        self.align_esp_btn.setEnabled(False)
        layout.addWidget(self.align_esp_btn)

        self.export_btn = QPushButton("Exportovať dáta (CSV + PNG)")
        self.export_btn.clicked.connect(self.export_data)
        layout.addWidget(self.export_btn)

        self.calibrate_btn = QPushButton("Rekalibrovať senzory (nula)")
        self.calibrate_btn.clicked.connect(self.calibrate_sensors)
        self.calibrate_btn.setEnabled(False)
        layout.addWidget(self.calibrate_btn)

        return layout

    def _build_display_group(self):
        group = QGroupBox("Zobrazenie")
        row = QHBoxLayout()

        self.show_pendulum_cb = QCheckBox("Kyvadlo")
        self.show_pendulum_cb.setChecked(True)
        self.show_pendulum_cb.stateChanged.connect(self.toggle_pendulum)
        row.addWidget(self.show_pendulum_cb)

        self.show_trail_cb = QCheckBox("Stopa")
        self.show_trail_cb.setChecked(True)
        self.show_trail_cb.stateChanged.connect(self.toggle_trail)
        row.addWidget(self.show_trail_cb)

        self.update_graphs_cb = QCheckBox("Grafy")
        self.update_graphs_cb.setChecked(True)
        self.update_graphs_cb.stateChanged.connect(self.clear_all_graph_data)
        row.addWidget(self.update_graphs_cb)

        self.show_rods_cb = QCheckBox("Tyče")
        self.show_rods_cb.setChecked(True)
        self.show_rods_cb.stateChanged.connect(self.toggle_rods)
        row.addWidget(self.show_rods_cb)

        self.show_esp_graphs_cb = QCheckBox("Porovnať s ESP")
        self.show_esp_graphs_cb.setChecked(False)
        self.show_esp_graphs_cb.stateChanged.connect(self.toggle_esp_graphs)
        row.addWidget(self.show_esp_graphs_cb)

        group.setLayout(row)
        return group

    def _build_status_group(self):
        group = QGroupBox("Stav")
        layout = QVBoxLayout()
        self.time_label   = QLabel("Čas: 0.00 s")
        self.energy_label = QLabel("Energia: 0.00 J")
        layout.addWidget(self.time_label)
        layout.addWidget(self.energy_label)
        group.setLayout(layout)
        return group

    # ------------------------------------------------------------------ #
    #  Tab 2: Časové grafy                                                 #
    # ------------------------------------------------------------------ #

    def create_graphs_tab(self):
        tab = QWidget()
        grid = QGridLayout(tab)

        self.plot_theta1, self.curve_theta1, self.curve_esp_theta1 = self._make_plot(
            "θ1 vs Čas", "Uhol θ1 (rad)", "Čas (s)", 'b')
        self.plot_theta2, self.curve_theta2, self.curve_esp_theta2 = self._make_plot(
            "θ2 vs Čas", "Uhol θ2 (rad)", "Čas (s)", 'r')
        self.plot_omega1, self.curve_omega1, self.curve_esp_omega1 = self._make_plot(
            "ω1 vs Čas", "Uhlová rýchlosť ω1 (rad/s)", "Čas (s)", 'g')
        self.plot_omega2, self.curve_omega2, self.curve_esp_omega2 = self._make_plot(
            "ω2 vs Čas", "Uhlová rýchlosť ω2 (rad/s)", "Čas (s)", 'm')

        grid.addWidget(self.plot_theta1, 0, 0)
        grid.addWidget(self.plot_theta2, 0, 1)
        grid.addWidget(self.plot_omega1, 1, 0)
        grid.addWidget(self.plot_omega2, 1, 1)
        self.tabs.addTab(tab, "Časové grafy")

    def _make_plot(self, title, ylabel, xlabel, sim_color):
        """Vytvorí PlotWidget so simulačnou a ESP krivkou."""
        plot = pg.PlotWidget(title=title)
        plot.setBackground('w')
        plot.setLabel('left', ylabel)
        plot.setLabel('bottom', xlabel)
        plot.addLegend()
        sim_curve = plot.plot(pen=pg.mkPen(sim_color, width=3), name='Simulácia')
        esp_curve = plot.plot(
            pen=pg.mkPen('orange', width=2, style=Qt.PenStyle.DashLine), name='ESP')
        return plot, sim_curve, esp_curve

    # ------------------------------------------------------------------ #
    #  Tab 3: Analýza                                                      #
    # ------------------------------------------------------------------ #

    def create_analysis_tab(self):
        tab = QWidget()
        grid = QGridLayout(tab)

        self.plot_phase1, self.curve_phase1, self.curve_esp_phase1 = self._make_plot(
            "Fázový priestor: θ1 vs ω1", "Uhlová rýchlosť ω1 (rad/s)", "Uhol θ1 (rad)", 'b')
        self.plot_phase2, self.curve_phase2, self.curve_esp_phase2 = self._make_plot(
            "Fázový priestor: θ2 vs ω2", "Uhlová rýchlosť ω2 (rad/s)", "Uhol θ2 (rad)", 'r')
        self.plot_config, self.curve_config, self.curve_esp_config = self._make_plot(
            "Konfiguračný priestor: θ1 vs θ2", "Uhol θ2 (rad)", "Uhol θ1 (rad)", 'm')

        self.plot_energy = pg.PlotWidget(title="Energia vs Čas")
        self.plot_energy.setBackground('w')
        self.plot_energy.setLabel('left', "Energia (J)")
        self.plot_energy.setLabel('bottom', "Čas (s)")
        self.plot_energy.addLegend()

        grid.addWidget(self.plot_phase1, 0, 0)
        grid.addWidget(self.plot_phase2, 0, 1)
        grid.addWidget(self.plot_config, 1, 0)
        grid.addWidget(self.plot_energy, 1, 1)
        self.tabs.addTab(tab, "Analýza")

    # ------------------------------------------------------------------ #
    #  Tab 4: ESP32 Dáta                                                   #
    # ------------------------------------------------------------------ #

    def create_esp32_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)

        conn_group = QGroupBox("Pripojenie ESP32-C3")
        conn_group.setMaximumHeight(150)
        g = QGridLayout()

        self.port_selector = QComboBox()
        self.refresh_ports()

        self.connect_btn = QPushButton("Pripojiť ESP32")
        self.connect_btn.clicked.connect(self.toggle_esp_connection)

        g.addWidget(QLabel("Port:"), 0, 0)
        g.addWidget(self.port_selector, 0, 1)
        g.addWidget(self.connect_btn, 1, 0, 1, 2)

        self.live_data_label = QLabel("Uhly z ESP32: θ1: 0°, θ2: 0°")
        g.addWidget(self.live_data_label, 2, 0, 1, 2)

        self.show_esp_pendulum_cb = QCheckBox("Zobraziť ESP kyvadlo v simulácii")
        self.show_esp_pendulum_cb.setChecked(False)
        g.addWidget(self.show_esp_pendulum_cb, 3, 0, 1, 2)

        conn_group.setLayout(g)
        main_layout.addWidget(conn_group)

        self.esp_plot1 = pg.PlotWidget(title="Live θ1 (rad)")
        self.esp_plot1.setBackground('k')
        self.esp_curve1 = self.esp_plot1.plot(pen='b')
        main_layout.addWidget(self.esp_plot1)

        self.esp_plot2 = pg.PlotWidget(title="Live θ2 (rad)")
        self.esp_plot2.setBackground('k')
        self.esp_curve2 = self.esp_plot2.plot(pen='r')
        main_layout.addWidget(self.esp_plot2)

        self.tabs.addTab(tab, "ESP32 Dáta")

    def init_plots(self):
        self.time_data  = []
        self.E_kin_data = []
        self.E_pot_data = []
        self.E_tot_data = []
        self.esp_time_data   = []
        self.esp_theta1_data = []
        self.esp_theta2_data = []
        self.esp_start_time  = None

    # ================================================================== #
    #  Parametre a počiatočné podmienky                                    #
    # ================================================================== #

    def update_parameters(self):
        L1 = self.L1_slider.value() / 1000.0
        L2 = self.L2_slider.value() / 1000.0
        m1 = self.m1_slider.value() / 1000.0
        m2 = self.m2_slider.value() / 1000.0
        damping = self.damping_slider.value() / 100.0

        self.L1_edit_field.setText(f"{L1:.3f}")
        self.L2_edit_field.setText(f"{L2:.3f}")
        self.m1_edit_field.setText(f"{m1:.3f}")
        self.m2_edit_field.setText(f"{m2:.3f}")
        self.damping_label.setText(f"{damping:.2f}")

        self.simulator.set_parameters(L1, L2, m1, m2, damping)
        self.canvas.L1, self.canvas.L2 = L1, L2

        if not self.simulator.is_running:
            self.update_initial_angles()

    def update_params_from_edit(self):
        sender = self.sender()
        try:
            val = float(sender.text().replace(',', '.'))
            mapping = {
                self.L1_edit_field: self.L1_slider,
                self.L2_edit_field: self.L2_slider,
                self.m1_edit_field: self.m1_slider,
                self.m2_edit_field: self.m2_slider,
            }
            if sender in mapping:
                mapping[sender].setValue(int(val * 1000))
        except ValueError:
            pass
        self.update_parameters()

    def update_initial_angles(self):
        if self.simulator.is_running:
            return
        t1 = self.theta1_slider.value()
        t2 = self.theta2_slider.value()
        self.theta1_label.setText(str(t1))
        self.theta2_label.setText(str(t2))
        self.canvas.set_initial_position(t1, t2)

    def set_default_parameters(self):
        self.L1_slider.setValue(56)
        self.L2_slider.setValue(49)
        self.m1_slider.setValue(500)
        self.m2_slider.setValue(342)
        self.damping_slider.setValue(2)
        self.update_parameters()
        self.update_initial_angles()

    # ================================================================== #
    #  Riadenie simulácie                                                  #
    # ================================================================== #

    def start_simulation(self):
        theta1 = self.theta1_slider.value() * np.pi / 180
        theta2 = self.theta2_slider.value() * np.pi / 180
        self.simulator.reset(theta1, theta2)

        self.time_data  = []
        self.E_kin_data = []
        self.E_pot_data = []
        self.E_tot_data = []
        self.canvas.trail.clear()

        self.esp_time_data   = []
        self.esp_theta1_data = []
        self.esp_theta2_data = []
        self.esp_start_time  = time.time()

        self.simulator.start()
        self._set_sim_controls(running=True)

    def stop_simulation(self):
        self.simulator.stop()
        self._set_sim_controls(running=False)

    def reset_simulation(self):
        self.simulator.stop()
        self.canvas.trail.clear()
        self.init_plots()

        self.esp_omega1_data = []
        self.esp_omega2_data = []

        theta1 = self.theta1_slider.value() * np.pi / 180
        theta2 = self.theta2_slider.value() * np.pi / 180
        self.simulator.reset(theta1, theta2)
        self.canvas.set_initial_position(
            self.theta1_slider.value(), self.theta2_slider.value())

        for curve in [self.curve_theta1, self.curve_theta2,
                      self.curve_omega1, self.curve_omega2,
                      self.curve_phase1, self.curve_phase2,
                      self.curve_config,
                      self.esp_curve1, self.esp_curve2,
                      self.curve_esp_theta1, self.curve_esp_theta2,
                      self.curve_esp_omega1, self.curve_esp_omega2,
                      self.curve_esp_phase1, self.curve_esp_phase2,
                      self.curve_esp_config]:
            curve.setData([], [])
        self.plot_energy.clear()

        self._set_sim_controls(running=False)

    def _set_sim_controls(self, running: bool):
        """Zapne/vypne tlačidlá podľa toho, či simulácia beží."""
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.theta1_slider.setEnabled(not running)
        self.theta2_slider.setEnabled(not running)
        self.default_btn.setEnabled(not running)
        self.time_limit_slider.setEnabled(not running)

    # ================================================================== #
    #  Aktualizácia vizualizácie (slot signálu simulátora)                 #
    # ================================================================== #

    def update_visualization(self, data):
        current_time = data['time']

        # Kontrola časového limitu
        try:
            limit = float(self.time_limit_slider.text().replace(',', '.'))
            if limit > 0 and current_time >= limit:
                self.stop_simulation()
                self.time_label.setText(f"Čas: {limit:.2f} s (limit dosiahnutý)")
                return
        except ValueError:
            pass

        self.canvas.update_state(data['x1'], data['y1'], data['x2'], data['y2'])
        self.time_label.setText(f"Čas: {current_time:.2f} s")
        self.energy_label.setText(f"Energia: {data['E_total']:.2f} J")

        if not self.update_graphs_cb.isChecked():
            self.clear_all_graph_data()
            return

        # Časové grafy
        history = self.simulator.get_history()
        self.curve_theta1.setData(history['time'], history['theta1'])
        self.curve_theta2.setData(history['time'], history['theta2'])
        self.curve_omega1.setData(history['time'], history['omega1'])
        self.curve_omega2.setData(history['time'], history['omega2'])

        # Analýza
        hs = self.simulator.get_history(max_points=5000)
        self.curve_phase1.setData(hs['theta1'], hs['omega1'])
        self.curve_phase2.setData(hs['theta2'], hs['omega2'])
        self.curve_config.setData(hs['theta1'], hs['theta2'])

        # Energia
        self.time_data.append(data['time'])
        self.E_kin_data.append(data['E_kinetic'])
        self.E_pot_data.append(data['E_potential'])
        self.E_tot_data.append(data['E_total'])
        if len(self.time_data) > 10000:
            self.time_data  = self.time_data[-10000:]
            self.E_kin_data = self.E_kin_data[-10000:]
            self.E_pot_data = self.E_pot_data[-10000:]
            self.E_tot_data = self.E_tot_data[-10000:]

        self.plot_energy.clear()
        self.plot_energy.plot(self.time_data, self.E_kin_data, pen='g', name='Kinetická')
        self.plot_energy.plot(self.time_data, self.E_pot_data, pen='b', name='Potenciálna')
        self.plot_energy.plot(self.time_data, self.E_tot_data, pen='r', name='Celková')

    def clear_all_graph_data(self):
        for curve in [self.curve_theta1, self.curve_theta2,
                      self.curve_omega1, self.curve_omega2,
                      self.curve_phase1, self.curve_phase2,
                      self.curve_config,
                      self.curve_esp_theta1, self.curve_esp_theta2,
                      self.curve_esp_omega1, self.curve_esp_omega2,
                      self.curve_esp_phase1, self.curve_esp_phase2,
                      self.curve_esp_config]:
            curve.setData([], [])
        self.plot_energy.clear()

    # ================================================================== #
    #  Toggle / checkbox handlery                                          #
    # ================================================================== #

    def toggle_pendulum(self, state):
        self.canvas.show_pendulum = (state == Qt.CheckState.Checked.value)
        self.canvas.update()

    def toggle_trail(self, state):
        self.canvas.show_trail = (state == Qt.CheckState.Checked.value)
        if not self.canvas.show_trail:
            self.canvas.trail.clear()
        self.canvas.update()

    def toggle_rods(self, state):
        self.canvas.show_rods = (state == Qt.CheckState.Checked.value)
        self.canvas.update()

    def toggle_esp_graphs(self, state):
        self.show_esp_on_graphs = (state == Qt.CheckState.Checked.value)
        if not self.show_esp_on_graphs:
            for curve in [self.curve_esp_theta1, self.curve_esp_theta2,
                          self.curve_esp_omega1, self.curve_esp_omega2,
                          self.curve_esp_phase1, self.curve_esp_phase2,
                          self.curve_esp_config]:
                curve.setData([], [])

    # ================================================================== #
    #  ESP32 komunikácia                                                   #
    # ================================================================== #

    def refresh_ports(self):
        self.port_selector.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_selector.addItems(ports)

    def toggle_esp_connection(self):
        if self.serial_thread is not None and self.serial_thread.isRunning():
            self._disconnect_esp()
        else:
            self._connect_esp()

    def _disconnect_esp(self):
        self.serial_thread.stop()
        self.serial_thread = None
        self.connect_btn.setText("Pripojiť ESP32")
        self.connect_btn.setStyleSheet("")
        self.align_esp_btn.setEnabled(False)
        self.calibrate_btn.setEnabled(False)
        self.canvas.esp_coords = None
        self.canvas.update()
        print("ESP32 odpojené.")

    def _connect_esp(self):
        port = self.port_selector.currentText()
        if not port:
            return
        self.esp_time_data   = []
        self.esp_theta1_data = []
        self.esp_theta2_data = []
        self.esp_omega1_data = []
        self.esp_omega2_data = []
        self.esp_start_time  = time.time()

        self.serial_thread = SerialReader(port)
        self.serial_thread.raw_data_received.connect(self.process_esp_data)
        self.serial_thread.start()

        self.connect_btn.setText("Odpojiť ESP32")
        self.connect_btn.setStyleSheet("background-color: #ff4c4c; color: white;")
        self.align_esp_btn.setEnabled(True)
        self.calibrate_btn.setEnabled(True)
        print(f"Pripájanie k {port}...")

    def process_esp_data(self, t1, t2):
        if self.esp_start_time is None:
            self.esp_start_time = time.time()
        current_time = time.time() - self.esp_start_time

        # Konverzia relatívneho θ2 na absolútny uhol od zvislice.
        # Senzor na dolnom ramene meria uhol voči hornému ramenu,
        # nie voči zvislici. Prevod: θ2_abs = θ1 + θ2_rel
        # Ak dolné rameno ide zrkadlovo, zmeň + na -  (viď instrukcie).
        t2 = t1 + t2
        if t2 > 180:
            t2 -= 360
        elif t2 < -180:
            t2 += 360

        # Dead zone — len keď simulácia nebeží (potlačí šum v pokoji)
        if not self.simulator.is_running:
            THRESHOLD = 1.0
            t1 = 0.0 if abs(t1) < THRESHOLD else t1
            t2 = 0.0 if abs(t2) < THRESHOLD else t2

        self.last_esp_t1 = t1
        self.last_esp_t2 = t2
        self.live_data_label.setText(f"Uhly z ESP32: θ1: {t1:.1f}°, θ2: {t2:.1f}°")

        r1, r2 = np.radians(t1), np.radians(t2)

        # Kreslenie ESP kyvadla na canvas
        if self.show_esp_pendulum_cb.isChecked():
            x1 = self.simulator.L1 * np.sin(r1)
            y1 = self.simulator.L1 * np.cos(r1)
            x2 = x1 + self.simulator.L2 * np.sin(r2)
            y2 = y1 + self.simulator.L2 * np.cos(r2)
            self.canvas.update_esp_state(x1, y1, x2, y2)
        else:
            self.canvas.esp_coords = None
            self.canvas.update()

        # Buffer plníme len keď simulácia beží
        if not self.simulator.is_running:
            return

        self.esp_time_data.append(current_time)
        self.esp_theta1_data.append(r1)
        self.esp_theta2_data.append(r2)

        # Omega — centrálna diferencia cez 5 bodov
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

        MAX_POINTS = 2000
        if len(self.esp_time_data) > MAX_POINTS:
            self.esp_time_data   = self.esp_time_data[-MAX_POINTS:]
            self.esp_theta1_data = self.esp_theta1_data[-MAX_POINTS:]
            self.esp_theta2_data = self.esp_theta2_data[-MAX_POINTS:]
            self.esp_omega1_data = self.esp_omega1_data[-MAX_POINTS:]
            self.esp_omega2_data = self.esp_omega2_data[-MAX_POINTS:]

        self.esp_needs_graph_update = True

    def update_esp_graphs(self):
        """Timer callback (20 FPS) — prekreslí ESP grafy počas simulácie."""
        if not self.esp_needs_graph_update or not self.simulator.is_running:
            return
        if not self.esp_time_data:
            return
        self.esp_needs_graph_update = False

        # Downsampling na max 500 bodov
        n = len(self.esp_time_data)
        if n > 500:
            step = n // 500
            t  = self.esp_time_data[::step]
            t1 = self.esp_theta1_data[::step]
            t2 = self.esp_theta2_data[::step]
            w1 = self.esp_omega1_data[::step]
            w2 = self.esp_omega2_data[::step]
        else:
            t, t1, t2 = self.esp_time_data, self.esp_theta1_data, self.esp_theta2_data
            w1, w2    = self.esp_omega1_data, self.esp_omega2_data

        self.esp_curve1.setData(t, t1)
        self.esp_curve2.setData(t, t2)

        if self.show_esp_on_graphs:
            self.curve_esp_theta1.setData(t, t1)
            self.curve_esp_theta2.setData(t, t2)
            self.curve_esp_omega1.setData(t, w1)
            self.curve_esp_omega2.setData(t, w2)
            self.curve_esp_phase1.setData(t1, w1)
            self.curve_esp_phase2.setData(t2, w2)
            self.curve_esp_config.setData(t1, t2)

    def align_to_esp(self):
        """Nastaví slidery simulácie na aktuálnu pozíciu fyzického kyvadla."""
        if self.last_esp_t1 is None or self.last_esp_t2 is None:
            print("Žiadne ESP dáta nedostupné.")
            return
        if self.simulator.is_running:
            self.stop_simulation()
        t1 = max(-180, min(180, int(round(self.last_esp_t1))))
        t2 = max(-180, min(180, int(round(self.last_esp_t2))))
        self.theta1_slider.setValue(t1)
        self.theta2_slider.setValue(t2)
        self.update_initial_angles()
        print(f"Zarovnané na ESP: θ1={t1}°, θ2={t2}°")

    def calibrate_sensors(self):
        """Pošle CAL príkaz do ESP32 na rekalibrciu offsetu."""
        if self.serial_thread is None or not self.serial_thread.isRunning():
            print("ESP32 nie je pripojené.")
            return
        self.serial_thread.send_calibration()
        self.time_label.setText("Rekalibrácia poslaná — podrž kyvadlo dole 1 s")
        print("CAL signál poslaný do ESP32")

    # ================================================================== #
    #  Export                                                              #
    # ================================================================== #

    def export_data(self):
        """Exportuje CSV dáta + PNG grafy do vybraného priečinka."""
        folder = QFileDialog.getExistingDirectory(self, "Vyber priečinok pre export")
        if not folder:
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        self._export_csv_sim(folder, ts)
        self._export_csv_esp(folder, ts)
        self._export_png_graphs(folder, ts)
        self.time_label.setText(f"Exportované do: {folder}")

    def _export_csv_sim(self, folder, ts):
        history = self.simulator.get_history()
        path = f"{folder}/simulation_{ts}.csv"
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['time_s', 'theta1_rad', 'theta2_rad',
                            'omega1_rad_s', 'omega2_rad_s'])
                for i in range(len(history['time'])):
                    w.writerow([history['time'][i],   history['theta1'][i],
                                history['theta2'][i], history['omega1'][i],
                                history['omega2'][i]])
            print(f"✓ Simulácia CSV: {path}")
        except Exception as e:
            print(f"Export error (sim CSV): {e}")

    def _export_csv_esp(self, folder, ts):
        if not self.esp_time_data:
            return
        path = f"{folder}/esp_{ts}.csv"
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['time_s', 'theta1_rad', 'theta2_rad',
                            'omega1_rad_s', 'omega2_rad_s'])
                for i in range(len(self.esp_time_data)):
                    w.writerow([
                        self.esp_time_data[i],
                        self.esp_theta1_data[i],
                        self.esp_theta2_data[i],
                        self.esp_omega1_data[i] if i < len(self.esp_omega1_data) else 0,
                        self.esp_omega2_data[i] if i < len(self.esp_omega2_data) else 0,
                    ])
            print(f"✓ ESP CSV: {path}")
        except Exception as e:
            print(f"Export error (ESP CSV): {e}")

    def _export_png_graphs(self, folder, ts):
        plots = [
            (self.plot_theta1, f"{folder}/graf_theta1_{ts}.png"),
            (self.plot_theta2, f"{folder}/graf_theta2_{ts}.png"),
            (self.plot_omega1, f"{folder}/graf_omega1_{ts}.png"),
            (self.plot_omega2, f"{folder}/graf_omega2_{ts}.png"),
            (self.plot_phase1, f"{folder}/fazovy_theta1_{ts}.png"),
            (self.plot_phase2, f"{folder}/fazovy_theta2_{ts}.png"),
            (self.plot_config, f"{folder}/konfiguracny_{ts}.png"),
            (self.plot_energy, f"{folder}/energia_{ts}.png"),
        ]
        for plot_widget, path in plots:
            try:
                self.export_plot_hires(plot_widget, path)
                print(f"✓ PNG: {path}")
            except Exception as e:
                print(f"Export error (PNG): {e}")

    def export_plot_hires(self, plot_widget, filename, width=3200):
        """Exportuje pyqtgraph PlotWidget ako PNG s vysokým rozlíšením."""
        exporter = ImageExporter(plot_widget.plotItem)
        exporter.parameters()['width'] = width
        exporter.export(filename)
