import numpy as np
from collections import deque
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient


class PendulumCanvas(QWidget):
    """
    Widget pre animáciu kyvadla.
    Kreslí simulované kyvadlo (fialové) aj ESP kyvadlo (modré prerušované).
    Súradnicová konvencia: x = L*sin(θ), y = L*cos(θ)  →  θ=0 je dole.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 600)

        self.L1, self.L2 = 2.0, 2.0
        self.x1, self.y1 = 0.0, 0.0
        self.x2, self.y2 = 0.0, 0.0

        self.trail = deque(maxlen=100)
        self.show_pendulum = True
        self.show_trail = True
        self.show_rods = True

        self.esp_coords = None  # (ex1, ey1, ex2, ey2) alebo None

        self.set_initial_position(120, -100)

    # ------------------------------------------------------------------ #
    #  Aktualizácia stavu                                                  #
    # ------------------------------------------------------------------ #

    def set_initial_position(self, theta1_deg, theta2_deg):
        t1 = theta1_deg * np.pi / 180
        t2 = theta2_deg * np.pi / 180
        self.x1 = self.L1 * np.sin(t1)
        self.y1 = self.L1 * np.cos(t1)
        self.x2 = self.x1 + self.L2 * np.sin(t2)
        self.y2 = self.y1 + self.L2 * np.cos(t2)
        self.update()

    def update_state(self, x1, y1, x2, y2):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        if self.show_trail:
            self.trail.append((x2, y2))
        self.update()

    def update_esp_state(self, x1, y1, x2, y2):
        self.esp_coords = (x1, y1, x2, y2)
        self.update()

    # ------------------------------------------------------------------ #
    #  Kreslenie                                                           #
    # ------------------------------------------------------------------ #

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        scale = min(w, h) / (2.6 * (self.L1 + self.L2))

        # Pixelové súradnice
        px, py   = cx, cy
        b1x, b1y = cx + self.x1 * scale, cy + self.y1 * scale
        b2x, b2y = cx + self.x2 * scale, cy + self.y2 * scale

        if self.show_pendulum:
            self._draw_shadow(painter, px, py, b1x, b1y, b2x, b2y)
            self._draw_rods(painter, px, py, b1x, b1y, b2x, b2y)
            self._draw_bobs(painter, b1x, b1y, b2x, b2y)
            self._draw_pivot(painter, px, py)

        if self.esp_coords:
            self._draw_esp_pendulum(painter, cx, cy, scale)

        if self.show_trail and len(self.trail) > 1:
            self._draw_trail(painter, cx, cy, scale)

    def _draw_shadow(self, painter, px, py, b1x, b1y, b2x, b2y):
        off = 5
        painter.setOpacity(0.2)
        painter.setPen(QPen(Qt.GlobalColor.black, 5))
        if self.show_rods:
            painter.drawLine(int(px + off), int(py + off), int(b1x + off), int(b1y + off))
            painter.drawLine(int(b1x + off), int(b1y + off), int(b2x + off), int(b2y + off))
        painter.setOpacity(1.0)

    def _draw_rods(self, painter, px, py, b1x, b1y, b2x, b2y):
        if not self.show_rods:
            return
        painter.setPen(QPen(QColor(51, 0, 51), 5))
        painter.drawLine(int(px), int(py), int(b1x), int(b1y))
        painter.drawLine(int(b1x), int(b1y), int(b2x), int(b2y))

    def _draw_bobs(self, painter, b1x, b1y, b2x, b2y):
        for bx, by in [(b1x, b1y), (b2x, b2y)]:
            grad = QRadialGradient(bx - 4, by - 4, 12)
            grad.setColorAt(0,   QColor(100, 40, 100))
            grad.setColorAt(0.8, QColor(51, 0, 51))
            grad.setColorAt(1,   QColor(20, 0, 20))
            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(int(bx - 12), int(by - 12), 24, 24)

    def _draw_pivot(self, painter, px, py):
        painter.setBrush(QBrush(QColor(150, 150, 150)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(int(px - 6), int(py - 6), 12, 12)

    def _draw_esp_pendulum(self, painter, cx, cy, scale):
        ex1, ey1, ex2, ey2 = self.esp_coords
        eb1x, eb1y = cx + ex1 * scale, cy + ey1 * scale
        eb2x, eb2y = cx + ex2 * scale, cy + ey2 * scale
        pivot_x, pivot_y = cx, cy

        painter.setPen(QPen(QColor(0, 150, 255, 200), 3, Qt.PenStyle.DashLine))
        painter.drawLine(int(pivot_x), int(pivot_y), int(eb1x), int(eb1y))
        painter.drawLine(int(eb1x), int(eb1y), int(eb2x), int(eb2y))

        painter.setBrush(QBrush(QColor(0, 150, 255, 180)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(int(eb1x - 6), int(eb1y - 6), 12, 12)
        painter.drawEllipse(int(eb2x - 6), int(eb2y - 6), 12, 12)

    def _draw_trail(self, painter, cx, cy, scale):
        painter.setPen(QPen(QColor(255, 255, 255, 150), 1))
        for i in range(1, len(self.trail)):
            x1 = cx + self.trail[i - 1][0] * scale
            y1 = cy + self.trail[i - 1][1] * scale
            x2 = cx + self.trail[i][0] * scale
            y2 = cy + self.trail[i][1] * scale
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))
