import sys
from PyQt6.QtWidgets import QApplication
from app import DoublePendulumApp


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = DoublePendulumApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
