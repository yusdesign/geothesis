import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

st.set_page_config(layout="wide")
st.title("🌀 Geothesis — T₃ Waveform Surface")

# Sidebar controls
st.sidebar.header("Controls")
omega1 = st.sidebar.slider("ω₁ (T₁ vertices)", 0.1, 3.0, 1.0, 0.1)
omega2 = st.sidebar.slider("ω₂ (T₂ tangents)", 0.1, 3.0, 1.5, 0.1)
elevation = st.sidebar.slider("View elevation", 0, 90, 30, 5)
azimuth_speed = st.sidebar.slider("Auto-rotate speed", 0.0, 2.0, 0.3, 0.1)
surface_opacity = st.sidebar.slider("Surface opacity", 0.0, 1.0, 0.6, 0.05)
show_wireframe = st.sidebar.checkbox("Show wireframe", True)
show_circle = st.sidebar.checkbox("Show circle", True)
time_window = st.sidebar.slider("Time window", 5, 50, 20, 1)

# Constants
R = 1.0

# Initialize session state
if 'azimuth' not in st.session_state:
    st.session_state.azimuth = 45
if 'playing' not in st.session_state:
    st.session_state.playing = True

# Play/Pause
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("▶️ Play" if not st.session_state.playing else "⏸️ Pause"):
        st.session_state.playing = not st.session_state.playing
with col2:
    if st.button("🔄 Reset"):
        st.session_state.azimuth = 45

def get_t1_vertices(t, omega1, R):
    angles = np.array([0, 2*np.pi/3, 4*np.pi/3]) + omega1 * t
    return np.array([[R*np.cos(a), R*np.sin(a), 0] for a in angles])

def get_t2_vertices(t, omega2, R):
    angles = np.array([0, 2*np.pi/3, 4*np.pi/3]) + omega2 * t
    tangency_points = np.array([[R*np.cos(a), R*np.sin(a), 0] for a in angles])
    
    vertices = []
    for i in range(3):
        a1, a2 = angles[i], angles[(i+1) % 3]
        A = np.array([[np.cos(a1), np.sin(a1)],
                      [np.cos(a2), np.sin(a2)]])
        b = np.array([R, R])
        try:
            intersection = np.linalg.solve(A, b)
            vertices.append([intersection[0], intersection[1], 0])
        except:
            vertices.append([R*3*np.cos(a1), R*3*np.sin(a1), 0])
    
    return np.array(vertices), tangency_points

def get_t3(t1_vertices, t2_vertices, R):
    connecting_lines = []
    for i in range(3):
        start = t1_vertices[i]
        end = t2_vertices[i]
        direction = end - start
        extended_start = start - direction * 50
        extended_end = end + direction * 50
        connecting_lines.append([extended_start, extended_end])
    
    t3_vertices = []
    for i in range(3):
        line1 = connecting_lines[i]
        line2 = connecting_lines[(i+1) % 3]
        
        p1, p2 = line1
        p3, p4 = line2
        
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        x3, y3 = p3[0], p3[1]
        x4, y4 = p4[0], p4[1]
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 0.001:
            return None, None
        
        t_val = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        ix = x1 + t_val * (x2 - x1)
        iy = y1 + t_val * (y2 - y1)
        t3_vertices.append([ix, iy, 0])
    
    t3_vertices = np.array(t3_vertices)
    
    # Area (2D cross product)
    v1 = t3_vertices[1][:2] - t3_vertices[0][:2]
    v2 = t3_vertices[2][:2] - t3_vertices[0][:2]
    area = abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2
    
    if area < 0.01:
        return None, None
    
    # Check crossing
    crosses = False
    for i in range(3):
        p1 = t3_vertices[i][:2]
        p2 = t3_vertices[(i+1) % 3][:2]
        d = p2 - p1
        f = p1 - np.array([0.0, 0.0])
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - R**2
        disc = b**2 - 4*a*c
        
        if disc >= 0:
            disc_sqrt = np.sqrt(disc)
            t1_int = (-b - disc_sqrt) / (2*a)
            t2_int = (-b + disc_sqrt) / (2*a)
            if (0 <= t1_int <= 1) or (0 <= t2_int <= 1):
                crosses = True
                break
    
    return t3_vertices, crosses

# Auto-rotate
if st.session_state.playing:
    st.session_state.azimuth += azimuth_speed

# Create figure
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Time range for surface (past → future, centered at t=0)
t_start = -time_window / 2
t_end = time_window / 2
t_steps = 200
t_range = np.linspace(t_start, t_end, t_steps)

# Collect T₃ data over time
t3_data = []
t3_times = []
t3_areas = []
t3_crosses_list = []

for t in t_range:
    t1_v = get_t1_vertices(t, omega1, R)
    t2_v, _ = get_t2_vertices(t, omega2, R)
    t3_v, crosses = get_t3(t1_v, t2_v, R)
    
    if t3_v is not None:
        t3_data.append(t3_v)
        t3_times.append(t)
        v1 = t3_v[1][:2] - t3_v[0][:2]
        v2 = t3_v[2][:2] - t3_v[0][:2]
        area = abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2
        t3_areas.append(area)
        t3_crosses_list.append(crosses)

# Build surface from T₃ triangles
if len(t3_data) > 2:
    # Create surface vertices
    surface_vertices = []
    surface_faces = []
    
    for idx, (t3_v, t_val) in enumerate(zip(t3_data, t3_times)):
        for vertex in t3_v:
            surface_vertices.append([vertex[0], vertex[1], t_val])
    
    # Create faces (triangles connecting consecutive T₃ triangles)
    n_triangles = len(t3_data)
    for idx in range(n_triangles - 1):
        base = idx * 3
        for i in range(3):
            next_i = (i + 1) % 3
            face = [base + i, base + next_i, base + 3 + next_i]
            surface_faces.append(face)
            face = [base + i, base + 3 + next_i, base + 3 + i]
            surface_faces.append(face)
    
    # Convert to numpy
    surface_vertices = np.array(surface_vertices)
    
    # Create surface mesh
    if show_wireframe:
        mesh = Poly3DCollection([surface_vertices[face] for face in surface_faces], 
                                alpha=surface_opacity, 
                                facecolor='#00ccff', 
                                edgecolor='#006688',
                                linewidth=0.3)
        ax.add_collection3d(mesh)
    
    # Color by crossing status
    for idx, (t3_v, t_val, crosses) in enumerate(zip(t3_data, t3_times, t3_crosses_list)):
        color = '#00ccff' if crosses else '#ff00ff'
        alpha = 0.5 if crosses else 0.3
        for i in range(3):
            ax.plot([t3_v[i, 0], t3_v[(i+1)%3, 0]], 
                    [t3_v[i, 1], t3_v[(i+1)%3, 1]], 
                    [t_val, t_val], 
                    color, alpha=alpha, linewidth=1.5)

# Draw circle at center (z=0)
if show_circle:
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(R*np.cos(theta), R*np.sin(theta), np.zeros_like(theta), 
            'blue', linewidth=2, alpha=0.8)

# Draw time axis
ax.plot([0, 0], [0, 0], [t_start, t_end], 
        'white', alpha=0.5, linewidth=1, linestyle='--')

# Set limits
limit = 3
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([t_start, t_end])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Time')

# Title
n_alive = len(t3_data)
n_cross = sum(t3_crosses_list)
ax.set_title(f'T₃ Waveform Surface | ω₁={omega1:.1f}, ω₂={omega2:.1f} | {n_alive}/{len(t_range)} alive, {n_cross} crossing', 
             fontsize=14)

# Set view
ax.view_init(elev=elevation, azim=st.session_state.azimuth)

st.pyplot(fig)

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Speed ratio", f"{omega2/omega1:.2f}")
col2.metric("T₃ alive", f"{n_alive}/{len(t_range)}")
col3.metric("Crossing", f"{n_cross}")
col4.metric("Surface vertices", f"{len(surface_vertices)}")

# Auto-refresh
if st.session_state.playing:
    import time as time_module
    time_module.sleep(0.05)
    st.rerun()
