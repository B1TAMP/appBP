# Copilot Instructions for Double Pendulum Simulator

## Project Overview

**Double Pendulum Simulator** is a PyQt6-based real-time physics simulation with GUI visualization. The application simulates the chaotic motion of a double pendulum system with configurable parameters (lengths, masses, initial angles) and provides interactive analysis tools.

## Architecture

### Core Components

1. **PendulumSimulator** - Physics engine (separate conceptual thread)
   - Implements Runge-Kutta 4th order (RK4) numerical integration
   - Manages simulation state: `[theta1, theta2, omega1, omega2]`
   - Uses circular buffers (100,000 elements) to store history efficiently
   - Emits data via Qt signal (`data_ready`) at 30 Hz update rate
   - Key file: `Pendulum_app.py` lines 20-158

2. **PendulumCanvas** - Custom Qt widget for real-time visualization
   - Renders pendulum rods, bobs (masses), and pivot point with different colors
   - Maintains motion trail (deque, max 100 points) for path visualization
   - Uses pixel scaling to fit any window size
   - Key file: `Pendulum_app.py` lines 161-244

3. **DoublePendulumApp** - Main QMainWindow with tabbed interface
   - Tab 1: Simulation control (sliders for L1, L2, m1, m2, θ1, θ2) + canvas
   - Tab 2: Time-series graphs (θ1, θ2, ω1, ω2 vs time)
   - Tab 3: Phase space diagrams and energy plots
   - Key file: `Pendulum_app.py` lines 247-661

### Data Flow

```
[Parameters] → PendulumSimulator.set_parameters()
              ↓
          RK4 Integration (1ms timestep) → pendulum_derivatives()
              ↓
          Circular Buffer (100K points) → emit_data() signal (30 Hz)
              ↓
          DoublePendulumApp.update_visualization()
              ↓
          [Canvas + Graphs + Energy display]
```

## Physics Implementation

- **Equations**: Classical double pendulum Lagrangian equations with gravitational potential
- **Integration method**: RK4 with fixed timestep (dt = 0.001s)
- **Energy conservation**: Kinetic + Potential energy calculated and displayed
- **Coordinate system**: θ measured from vertical; x,y computed relative to pivot
- **Key formulas**: See `pendulum_derivatives()` (lines 75-95) for state derivatives

## Development Patterns

### Slider-to-Parameter Mapping
Sliders use integer ranges (e.g., 10-500) that are divided by 100 when applied:
```python
L1 = self.L1_slider.value() / 100.0  # slider(200) → 2.0 m
```

### Simulation State Management
- Only allow angle slider changes when simulation is stopped
- Reset clears buffer and trail; start fresh from selected initial conditions
- Stop preserves all plot data for analysis

### Performance Optimization
- Update graphs only when tab is active (`if self.tabs.currentIndex() == X`)
- Downsample history to 10,000 points max for phase diagrams
- Emit UI updates at 30 Hz (every 30 simulation steps) to balance responsiveness

### Visualization Scaling
Canvas automatically scales based on window size and pendulum lengths:
```python
scale = min(w, h) / (2.6 * (self.L1 + self.L2))
```

## Running the Application

```bash
python Pendulum_app.py
```

**Dependencies**: PyQt6, PyQtGraph, NumPy

**Notes**:
- Application starts with paused state (press "Start Simulation")
- Trail visualization uses white lines; disable for cleaner view
- All sliders update in real-time when simulation is stopped

## Common Modifications

- **Trail color**: Line 205 (`painter.setPen(QPen(QColor(255, 255, 255), 1))`)
- **Trail length**: Line 196 (`deque(maxlen=100)`)
- **Simulation timestep**: Line 34 (`self.dt = 0.001`)
- **Update rate**: Line 120 (`if self.buffer_index % 30 == 0`) - change 30 for different Hz
- **Initial position**: Lines 422-424 (modify theta1_slider/theta2_slider default values)
