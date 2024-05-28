import numpy as np
import matplotlib.pyplot as plt

# Erzeuge einen Bereich von x-Werten von 0 bis 2π
x = np.linspace(0, 2 * np.pi, 1000)

# Berechne die y-Werte der Sinusfunktion
y = np.sin(x)

# Erstelle das Diagramm
plt.plot(x, y)

# Beschrifte die Achsen
plt.xlabel('x')
plt.ylabel('sin(x)')

# Titel des Diagramms
plt.title('Darstellung der Sinusfunktion')

# Setze die x-Achse auf den Bereich 0 bis 2π
plt.xlim(0, 2 * np.pi)

# Setze die y-Achse auf den Bereich -1 bis 1
plt.ylim(-3, 3)

# Zeige ein Gitter an
plt.grid(True)

# Zeige das Diagramm
plt.show()
