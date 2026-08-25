import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

st.set_page_config(layout="wide")
st.title("🌀 Geothesis — 3D Wave Function")

# Sidebar controls
st.sidebar.header("Controls")
omega1 = st.sidebar.slider("ω₁ (T₁ vertices)", 0.1, 3.0, 1.0, 0.1)
omega2 = st.sidebar.slider("ω₂ (T₂ tangents)", 0.1, 3.0, 1.5, 0.1)
elevation = st.sidebar.slider("View elevation", 0, 90, 20, 5)
azimuth_speed = st.sidebar.slider("Rotation speed", 0.0, 2.0, 0.5, 0.1)
time_speed = st.sidebar.slider("Time speed", 0.01, 0.2, 0.05, 0.01)
show_helices = st.sidebar.checkbox("Show helical paths", True)
show_waveform = st.sidebar.checkbox("Show T₃ area waveform", True)

# Constants
R = 1.0
TIME_LIMIT = 20

# Initialize session state
if 'time' not in st.session_state:
    st.session_state.time = 0.0
if 'playing' not in st.session_state:
    st.session_state.playing = True
if 'azimuth' not in st.session_state:
    st.session_state.azimuth = 45

# Play/Pause buttons
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("▶️ Play" if not st.session_state.playing else "⏸️ Pause"):
        st.session_state.playing = not st.session_state.playing
with col2:
    if st.button("🔄 Reset"):
        st.session_state.time = 0.0
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
        extended_start = start - direction * 20
        extended_end = end + direction * 20
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
    
    # Check area using 2D cross product (scalar)
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

# Update time
if st.session_state.playing:
    st.session_state.time += time_speed
    st.session_state.azimuth += azimuth_speed

t = st.session_state.time

# Create 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

limit = 3.5
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (time)')

# Circle at z=0
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(R*np.cos(theta), R*np.sin(theta), np.zeros_like(theta), 
        'b-', alpha=0.3, linewidth=1)

# Circle at current time
ax.plot(R*np.cos(theta), R*np.sin(theta), np.ones_like(theta)*t, 
        'b-', alpha=0.5, linewidth=1.5)

# Helical paths
if show_helices:
    t_range = np.linspace(max(0, t-5), t, 50)
    for i in range(3):
        path = []
        for t_path in t_range:
            v = get_t1_vertices(t_path, omega1, R)
            path.append(v[i])
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], t_range, 
                '#00ff88', alpha=0.3, linewidth=1)
        
        path = []
        for t_path in t_range:
            v, _ = get_t2_vertices(t_path, omega2, R)
            path.append(v[i])
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], t_range, 
                '#ff4444', alpha=0.3, linewidth=1, linestyle='--')

# Get current vertices
t1_vertices = get_t1_vertices(t, omega1, R)
t2_vertices, tangency_points = get_t2_vertices(t, omega2, R)

# Draw T1
t1_current = t1_vertices.copy()
t1_current[:, 2] = t
for i in range(3):
    ax.plot([t1_current[i, 0], t1_current[(i+1)%3, 0]], 
            [t1_current[i, 1], t1_current[(i+1)%3, 1]], 
            [t1_current[i, 2], t1_current[(i+1)%3, 2]], 
            '#00ff88', linewidth=2)
ax.scatter(t1_current[:, 0], t1_current[:, 1], t1_current[:, 2], 
           c='#00ff88', s=50)

# Draw T2
t2_current = t2_vertices.copy()
t2_current[:, 2] = t
for i in range(3):
    ax.plot([t2_current[i, 0], t2_current[(i+1)%3, 0]], 
            [t2_current[i, 1], t2_current[(i+1)%3, 1]], 
            [t2_current[i, 2], t2_current[(i+1)%3, 2]], 
            '#ff4444', linewidth=2, linestyle='--')
ax.scatter(t2_current[:, 0], t2_current[:, 1], t2_current[:, 2], 
           c='#ff4444', s=50)

# Draw tangency points
tangency_current = tangency_points.copy()
tangency_current[:, 2] = t
ax.scatter(tangency_current[:, 0], tangency_current[:, 1], tangency_current[:, 2], 
           c='#ffaa00', s=30)

# Draw T3
t3_vertices, t3_crosses = get_t3(t1_vertices, t2_vertices, R)

if t3_vertices is not None:
    t3_current = t3_vertices.copy()
    t3_current[:, 2] = t
    
    # Connecting lines
    for i in range(3):
        start = t1_vertices[i]
        end = t2_vertices[i]
        direction = end - start
        ext_start = start - direction * 20
        ext_end = end + direction * 20
        ax.plot([ext_start[0], ext_end[0]], 
                [ext_start[1], ext_end[1]], 
                [t, t], 
                '#aa44ff', alpha=0.5, linewidth=0.8, linestyle=':')
    
    # T3 triangle
    t3_color = '#00ccff' if t3_crosses else '#ff00ff'
    for i in range(3):
        ax.plot([t3_current[i, 0], t3_current[(i+1)%3, 0]], 
                [t3_current[i, 1], t3_current[(i+1)%3, 1]], 
                [t3_current[i, 2], t3_current[(i+1)%3, 2]], 
                t3_color, linewidth=3)
    ax.scatter(t3_current[:, 0], t3_current[:, 1], t3_current[:, 2], 
               c=t3_color, s=80)
    
    status = "CROSSES circle" if t3_crosses else "NO cross"
    st.sidebar.success(f"T₃ ALIVE — {status}")
else:
    st.sidebar.error("T₃ not alive")

# Waveform
if show_waveform:
    t_range = np.linspace(max(0, t-5), t, 30)
    areas = []
    times = []
    for t_path in t_range:
        t1_v = get_t1_vertices(t_path, omega1, R)
        t2_v, _ = get_t2_vertices(t_path, omega2, R)
        t3_v, _ = get_t3(t1_v, t2_v, R)
        if t3_v is not None:
            v1 = t3_v[1][:2] - t3_v[0][:2]
            v2 = t3_v[2][:2] - t3_v[0][:2]
            area = abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2
            areas.append(area)
            times.append(t_path)
    
    if len(areas) > 1:
        ax.plot(areas, np.zeros_like(areas), times, 
                '#00ffaa', alpha=0.8, linewidth=2)

ax.set_title(f'Geothesis 3D | t={t:.2f} | ω₁={omega1:.1f}, ω₂={omega2:.1f}', 
             fontsize=14)

ax.view_init(elev=elevation, azim=st.session_state.azimuth)

st.pyplot(fig)

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Time", f"{t:.2f}")
col2.metric("Speed ratio", f"{omega2/omega1:.2f}")
col3.metric("T₃ status", "Alive" if t3_vertices is not None else "Dead")

# Auto-refresh
if st.session_state.playing:
    time.sleep(0.05)
    st.rerun()
