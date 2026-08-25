import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Parameters
R = 1.0  # Circle radius
omega1 = 1.0  # Angular speed of T1 vertices
omega2 = 1.5  # Angular speed of T2 tangency points
time_steps = 200
time = np.linspace(0, 4*np.pi, time_steps)

# Set up figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1.5])

# Colors
T1_COLOR = '#00ff88'
T2_COLOR = '#ff4444'
TANGENT_COLOR = '#ffaa00'
LINE_COLOR = '#aa44ff'
T3_CROSS_COLOR = '#00ccff'
T3_NO_CROSS_COLOR = '#ff00ff'

def get_t1_vertices(t, omega1, R):
    """T1 inscribed triangle vertices (on circle)"""
    angles = np.array([0, 2*np.pi/3, 4*np.pi/3]) + omega1 * t
    vertices = np.array([[R*np.cos(a), R*np.sin(a), 0] for a in angles])
    return vertices

def get_t2_vertices(t, omega2, R):
    """T2 circumscribed triangle vertices (tangent lines intersections)"""
    angles = np.array([0, 2*np.pi/3, 4*np.pi/3]) + omega2 * t
    
    # Tangency points on circle
    tangency_points = np.array([[R*np.cos(a), R*np.sin(a), 0] for a in angles])
    
    # Calculate T2 vertices (intersection of tangent lines)
    vertices = []
    for i in range(3):
        a1, a2 = angles[i], angles[(i+1) % 3]
        # Tangent line at angle a: cos(a)*x + sin(a)*y = R
        A = np.array([[np.cos(a1), np.sin(a1)],
                      [np.cos(a2), np.sin(a2)]])
        b = np.array([R, R])
        try:
            intersection = np.linalg.solve(A, b)
            vertices.append([intersection[0], intersection[1], 0])
        except:
            # Parallel tangents (shouldn't happen)
            vertices.append([R*3*np.cos(a1), R*3*np.sin(a1), 0])
    
    return np.array(vertices), tangency_points

def get_t3_from_connecting_lines(t1_vertices, t2_vertices, R):
    """Calculate T3 from intersections of connecting lines"""
    connecting_lines = []
    
    for i in range(3):
        start = t1_vertices[i]
        end = t2_vertices[i]
        direction = end - start
        # Extend line
        extended_start = start - direction * 10
        extended_end = end + direction * 10
        connecting_lines.append([extended_start, extended_end])
    
    # Find intersections
    t3_vertices = []
    for i in range(3):
        line1 = connecting_lines[i]
        line2 = connecting_lines[(i+1) % 3]
        
        p1, p2 = line1
        p3, p4 = line2
        
        # 2D line intersection
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        x3, y3 = p3[0], p3[1]
        x4, y4 = p4[0], p4[1]
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 0.001:
            return None, None  # Parallel lines, no T3
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        
        t3_vertices.append([ix, iy, 0])
    
    t3_vertices = np.array(t3_vertices)
    
    # Check if T3 is valid (not degenerate)
    area = np.abs(np.cross(t3_vertices[1] - t3_vertices[0], 
                           t3_vertices[2] - t3_vertices[0])) / 2
    
    if area < 0.01:
        return None, None
    
    # Check if T3 crosses circle
    crosses_circle = False
    for i in range(3):
        p1 = t3_vertices[i][:2]
        p2 = t3_vertices[(i+1) % 3][:2]
        
        # Check if segment intersects circle
        d = p2 - p1
        f = p1 - np.array([0, 0])
        
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - R**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant >= 0:
            disc_sqrt = np.sqrt(discriminant)
            t1_intersect = (-b - disc_sqrt) / (2*a)
            t2_intersect = (-b + disc_sqrt) / (2*a)
            
            if (0 <= t1_intersect <= 1) or (0 <= t2_intersect <= 1):
                crosses_circle = True
                break
    
    return t3_vertices, crosses_circle

def update(frame):
    ax.clear()
    t = time[frame]
    
    # Set limits and labels
    limit = 3.5
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (time)')
    ax.set_title(f'Wave Function T₃ | t = {t:.2f} | ω₁={omega1:.1f}, ω₂={omega2:.1f}', 
                 fontsize=12, pad=20)
    
    # Draw circle at z=0 (wireframe cylinder to show time evolution)
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = R * np.cos(theta)
    circle_y = R * np.sin(theta)
    circle_z = np.zeros_like(theta)
    ax.plot(circle_x, circle_y, circle_z, 'b-', alpha=0.3, linewidth=1)
    
    # Draw time axis (Z-axis as time)
    z_time = np.linspace(-limit, limit, 100)
    ax.plot(np.zeros_like(z_time), np.zeros_like(z_time), z_time, 
            'gray', alpha=0.3, linewidth=1, linestyle='--')
    
    # Draw circle at current time (horizontal cross-section)
    ax.plot(circle_x, circle_y, np.ones_like(theta) * t, 
            'b-', alpha=0.5, linewidth=1.5)
    
    # Get T1 and T2 vertices at current time
    t1_vertices = get_t1_vertices(t, omega1, R)
    t2_vertices, tangency_points = get_t2_vertices(t, omega2, R)
    
    # Extend vertices to show helical paths (past and future)
    past_time = time[max(0, frame-30):frame]
    future_time = time[frame:min(time_steps, frame+30)]
    
    # Draw helical paths for T1 vertices
    for i in range(3):
        path_points = []
        for t_path in time:
            v = get_t1_vertices(t_path, omega1, R)
            path_points.append(v[i])
        path_points = np.array(path_points)
        ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2] + t_path, 
                T1_COLOR, alpha=0.2, linewidth=0.8)
    
    # Draw helical paths for T2 vertices
    for i in range(3):
        path_points = []
        for t_path in time:
            v, _ = get_t2_vertices(t_path, omega2, R)
            path_points.append(v[i])
        path_points = np.array(path_points)
        ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2] + t_path, 
                T2_COLOR, alpha=0.2, linewidth=0.8, linestyle='--')
    
    # Draw T1 triangle at current time (at z=t)
    t1_current = t1_vertices.copy()
    t1_current[:, 2] = t
    for i in range(3):
        ax.plot([t1_current[i, 0], t1_current[(i+1)%3, 0]], 
                [t1_current[i, 1], t1_current[(i+1)%3, 1]], 
                [t1_current[i, 2], t1_current[(i+1)%3, 2]], 
                T1_COLOR, linewidth=2)
    ax.scatter(t1_current[:, 0], t1_current[:, 1], t1_current[:, 2], 
               c=T1_COLOR, s=50, label='T₁ vertices')
    
    # Draw T2 triangle at current time (at z=t)
    t2_current = t2_vertices.copy()
    t2_current[:, 2] = t
    for i in range(3):
        ax.plot([t2_current[i, 0], t2_current[(i+1)%3, 0]], 
                [t2_current[i, 1], t2_current[(i+1)%3, 1]], 
                [t2_current[i, 2], t2_current[(i+1)%3, 2]], 
                T2_COLOR, linewidth=2, linestyle='--')
    ax.scatter(t2_current[:, 0], t2_current[:, 1], t2_current[:, 2], 
               c=T2_COLOR, s=50, label='T₂ vertices')
    
    # Draw tangency points
    tangency_current = tangency_points.copy()
    tangency_current[:, 2] = t
    ax.scatter(tangency_current[:, 0], tangency_current[:, 1], tangency_current[:, 2], 
               c=TANGENT_COLOR, s=30, label='Tangency')
    
    # Draw connecting lines and T3
    t3_vertices, t3_crosses = get_t3_from_connecting_lines(t1_vertices, t2_vertices, R)
    
    if t3_vertices is not None:
        t3_current = t3_vertices.copy()
        t3_current[:, 2] = t
        
        # Draw connecting lines (infinite)
        for i in range(3):
            start = t1_vertices[i]
            end = t2_vertices[i]
            direction = end - start
            extended_start = start - direction * 20
            extended_end = end + direction * 20
            extended_start[2] = t
            extended_end[2] = t
            
            ax.plot([extended_start[0], extended_end[0]], 
                    [extended_start[1], extended_end[1]], 
                    [extended_start[2], extended_end[2]], 
                    LINE_COLOR, alpha=0.5, linewidth=0.8, linestyle=':')
        
        # Draw T3
        t3_color = T3_CROSS_COLOR if t3_crosses else T3_NO_CROSS_COLOR
        for i in range(3):
            ax.plot([t3_current[i, 0], t3_current[(i+1)%3, 0]], 
                    [t3_current[i, 1], t3_current[(i+1)%3, 1]], 
                    [t3_current[i, 2], t3_current[(i+1)%3, 2]], 
                    t3_color, linewidth=3)
        ax.scatter(t3_current[:, 0], t3_current[:, 1], t3_current[:, 2], 
                   c=t3_color, s=80, label='T₃')
        
        # Update title with T3 status
        status = "CROSSES" if t3_crosses else "NO CROSS"
        ax.set_title(f'T₃ ALIVE ({status}) | t = {t:.2f} | ω₁={omega1:.1f}, ω₂={omega2:.1f}', 
                     fontsize=12, pad=20)
    else:
        ax.set_title(f'T₃ NOT ALIVE | t = {t:.2f} | ω₁={omega1:.1f}, ω₂={omega2:.1f}', 
                     fontsize=12, pad=20)
    
    # Draw waveform surface (T3 area over time as translucent surface)
    # Collect T3 existence over time window
    window = 40
    start_idx = max(0, frame - window)
    end_idx = min(time_steps, frame + 1)
    
    t3_areas = []
    t3_times = []
    
    for idx in range(start_idx, end_idx):
        t_window = time[idx]
        t1_v = get_t1_vertices(t_window, omega1, R)
        t2_v, _ = get_t2_vertices(t_window, omega2, R)
        t3_v, crosses = get_t3_from_connecting_lines(t1_v, t2_v, R)
        
        if t3_v is not None:
            area = np.abs(np.cross(t3_v[1] - t3_v[0], t3_v[2] - t3_v[0])) / 2
            t3_areas.append(area)
            t3_times.append(t_window)
    
    # Draw area waveform
    if len(t3_areas) > 1:
        ax.plot(np.zeros_like(t3_areas), np.zeros_like(t3_areas), t3_times, 
                'w-', alpha=0.3, linewidth=0.5)
        ax.plot(t3_areas, np.zeros_like(t3_areas), t3_times, 
                '#00ffaa', alpha=0.7, linewidth=2, label='T₃ area')
    
    # Legend
    ax.legend(loc='upper left', fontsize=8)
    
    # Set view angle (rotate around Y axis)
    ax.view_init(elev=20, azim=45 + frame * 0.5)

# Create animation
anim = FuncAnimation(fig, update, frames=time_steps, interval=50, blit=False)

# Save animation
anim.save('geothesis_3d_wave.gif', writer='pillow', fps=20)

plt.show()

print("Animation saved as 'geothesis_3d_wave.gif'")
