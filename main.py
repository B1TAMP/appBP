import sys
from PyQt6.QtWidgets import QApplication
from app import DoublePendulumApp

 # Parametre kyvadla a simulácie 
  # l1  = 125
   # l2 = 85
    # m1 = 
     # m2 = 
      # potrebujem upravit toolbary l1 a l2  momentalne v metroch idealne ale od 3cm do 1m .


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = DoublePendulumApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
