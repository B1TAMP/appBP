import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal


class PendulumSimulator(QObject):
    """
    Fyzikálna simulácia dvojkyvadla.
    Integrácia Runge-Kutta 4. rádu, časový krok 1 ms.
    Výsledky sa emitujú signálom data_ready 30x za sekundu.
    """

    data_ready = pyqtSignal(dict)

    def __init__(self):
        super().__init__()

        # Fyzikálne parametre (zodpovedajú fyzickému modelu z BP)
        self.L1 = 0.056     # dĺžka prvého ramena [m]
        self.L2 = 0.049     # dĺžka druhého ramena [m]
        self.m1 = 0.500     # hmotnosť prvého kývača [kg]
        self.m2 = 0.342     # hmotnosť druhého kývača [kg]
        self.g = 9.81       # gravitačné zrýchlenie [m/s²]
        self.dt = 0.001     # časový krok [s]
        self.damping = 0.01 # koeficient tlmenia

        # Stavový vektor: [theta1, theta2, omega1, omega2]
        self.state = np.array([120 * np.pi / 180, -100 * np.pi / 180, 0.0, 0.0])
        self.time = 0.0

        # Kruhový buffer pre históriu (100 000 krokov ≈ 100 s pri dt=1ms)
        self.buffer_size = 100000
        self.buffer_index = 0
        self.time_buffer   = np.zeros(self.buffer_size)
        self.theta1_buffer = np.zeros(self.buffer_size)
        self.theta2_buffer = np.zeros(self.buffer_size)
        self.omega1_buffer = np.zeros(self.buffer_size)
        self.omega2_buffer = np.zeros(self.buffer_size)

        self.is_running = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.step)

    # ------------------------------------------------------------------ #
    #  Riadenie simulácie                                                  #
    # ------------------------------------------------------------------ #

    def start(self):
        self.is_running = True
        self.timer.start(1)  # 1 ms timer → ~1000 krokov/s

    def stop(self):
        self.is_running = False
        self.timer.stop()

    def reset(self, theta1, theta2):
        """Resetuje simuláciu na zadané počiatočné podmienky."""
        self.state = np.array([theta1, theta2, 0.0, 0.0])
        self.time = 0.0
        self.buffer_index = 0

    def set_parameters(self, L1, L2, m1, m2, damping):
        self.L1 = L1
        self.L2 = L2
        self.m1 = m1
        self.m2 = m2
        self.damping = damping

    # ------------------------------------------------------------------ #
    #  Fyzika                                                              #
    # ------------------------------------------------------------------ #

    def pendulum_derivatives(self, state):
        """Rovnice pohybu dvojkyvadla — derivácie stavového vektora."""
        th1, th2, w1, w2 = state
        delta = th2 - th1

        den1 = (self.m1 + self.m2) * self.L1 - self.m2 * self.L1 * np.cos(delta) ** 2
        den2 = (self.L2 / self.L1) * den1

        dw1 = (self.m2 * self.L1 * w1 ** 2 * np.sin(delta) * np.cos(delta)
               + self.m2 * self.g * np.sin(th2) * np.cos(delta)
               + self.m2 * self.L2 * w2 ** 2 * np.sin(delta)
               - (self.m1 + self.m2) * self.g * np.sin(th1)) / den1

        dw2 = (-self.m2 * self.L2 * w2 ** 2 * np.sin(delta) * np.cos(delta)
               + (self.m1 + self.m2) * (self.g * np.sin(th1) * np.cos(delta)
               - self.L1 * w1 ** 2 * np.sin(delta)
               - self.g * np.sin(th2))) / den2

        dw1 -= self.damping * w1
        dw2 -= self.damping * w2

        return np.array([w1, w2, dw1, dw2])

    def rk4_step(self):
        """Jeden krok Runge-Kutta 4. rádu."""
        k1 = self.pendulum_derivatives(self.state)
        k2 = self.pendulum_derivatives(self.state + 0.5 * self.dt * k1)
        k3 = self.pendulum_derivatives(self.state + 0.5 * self.dt * k2)
        k4 = self.pendulum_derivatives(self.state + self.dt * k3)
        self.state += self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.time += self.dt

    def step(self):
        """Jeden simulačný krok — volá timer každú 1 ms."""
        if not self.is_running:
            return

        self.rk4_step()

        idx = self.buffer_index % self.buffer_size
        self.time_buffer[idx]   = self.time
        self.theta1_buffer[idx] = self.state[0]
        self.theta2_buffer[idx] = self.state[1]
        self.omega1_buffer[idx] = self.state[2]
        self.omega2_buffer[idx] = self.state[3]
        self.buffer_index += 1

        if self.buffer_index % 30 == 0:
            self.emit_data()

    def emit_data(self):
        """Emituje aktuálny stav do GUI (30 Hz)."""
        x1 = self.L1 * np.sin(self.state[0])
        y1 = self.L1 * np.cos(self.state[0])
        x2 = x1 + self.L2 * np.sin(self.state[1])
        y2 = y1 + self.L2 * np.cos(self.state[1])

        E_kin = self.calculate_kinetic_energy()
        E_pot = self.calculate_potential_energy()

        self.data_ready.emit({
            'time': self.time,
            'theta1': self.state[0],
            'theta2': self.state[1],
            'omega1': self.state[2],
            'omega2': self.state[3],
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
            'E_kinetic': E_kin,
            'E_potential': E_pot,
            'E_total': E_kin + E_pot,
        })

    # ------------------------------------------------------------------ #
    #  Energia a história                                                  #
    # ------------------------------------------------------------------ #

    def calculate_kinetic_energy(self):
        th1, th2, w1, w2 = self.state
        v1_sq = (self.L1 * w1) ** 2
        v2_sq = ((self.L1 * w1) ** 2 + (self.L2 * w2) ** 2
                 + 2 * self.L1 * self.L2 * w1 * w2 * np.cos(th1 - th2))
        return 0.5 * self.m1 * v1_sq + 0.5 * self.m2 * v2_sq

    def calculate_potential_energy(self):
        th1, th2 = self.state[0], self.state[1]
        y1 = -self.L1 * np.cos(th1)
        y2 = y1 - self.L2 * np.cos(th2)
        return self.m1 * self.g * y1 + self.m2 * self.g * y2

    def get_history(self, max_points=10000):
        """Vráti históriu simulácie pre vykreslenie grafov."""
        n = min(self.buffer_index, self.buffer_size)
        indices = (np.linspace(0, n - 1, max_points, dtype=int)
                   if n > max_points else np.arange(n))
        return {
            'time':   self.time_buffer[indices],
            'theta1': self.theta1_buffer[indices],
            'theta2': self.theta2_buffer[indices],
            'omega1': self.omega1_buffer[indices],
            'omega2': self.omega2_buffer[indices],
        }
