import numpy as np
import pyvista as pv

def vis_3d_scalar(density, grad):  
    S = density.shape[-1]

    ones = np.ones(S)
    slices = np.linspace(0, 1, S)
    px = np.kron(ones, np.kron(ones, slices))
    py = np.kron(ones, np.kron(slices, ones))
    pz = np.kron(slices, np.kron(ones, ones))
    pos = np.vstack([px.ravel(), py.ravel(), pz.ravel()]).T   
    
    density = density.ravel()
    grad = grad.ravel()
    
    mask_d = density > 0.5
    pos_d = pos[mask_d, :]
    density = density[mask_d]
    
    mask_g = grad < np.percentile(grad, 0.1)
    pos_g = pos[mask_g, :]
    grad = grad[mask_g]
    
    grid_d = pv.PolyData(pos_d)
    grid_d.point_data['scalar_values'] = density
    grid_g = pv.PolyData(pos_g)
    grid_g.point_data['scalar_values'] = grad
    
    plotter = pv.Plotter()
    plotter.add_points(grid_d, color='darkgray', point_size=2)
    plotter.add_mesh(grid_g, 
                     scalars='scalar_values',
                     cmap='viridis',
                     point_size=10,
                     render_points_as_spheres=True,
                     show_scalar_bar=True)
    
    plotter.add_axes()
    plotter.show()
    
def vis_3d_orientation(density, vec):
    S = density.shape[-1]

    ones = np.ones(S)
    slices = np.linspace(0, 1, S)
    px = np.kron(ones, np.kron(ones, slices))
    py = np.kron(ones, np.kron(slices, ones))
    pz = np.kron(slices, np.kron(ones, ones))
    pos = np.vstack([px.ravel(), py.ravel(), pz.ravel()]).T   
    
    density = density.ravel()
    vec = vec.reshape(-1, 3)
    mask = density > 0.5
    pos = pos[mask, :]
    vec = vec[mask, :]
    density = density[mask]
    
    n_points = len(pos)
    sample_size = n_points // 10
    indices = np.random.choice(n_points, sample_size, replace=False)
    pos = pos[indices, :]
    vec = vec[indices, :]
    density = density[indices]
    
    vec_normalized = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    x = vec_normalized[:, 0]
    y = vec_normalized[:, 1]
    z = vec_normalized[:, 2]
    
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    orientation = theta + phi 
    
    grid = pv.PolyData(pos)

    grid['vectors'] = vec_normalized
    grid['orientation'] = orientation

    plotter = pv.Plotter()

    glyphs = grid.glyph(
        orient='vectors',
        scale=False,
        factor=2e-2, 
        geom=pv.Arrow(shaft_radius=0.03, tip_radius=0.05, tip_length=0.15) 
    )

    plotter.add_mesh(glyphs, 
                    scalars='orientation',
                    cmap='hsv',
                    show_scalar_bar=True,
                    lighting=True)

    plotter.add_mesh(grid, 
                    color='darkgray',
                    point_size=2,
                    render_points_as_spheres=True)

    plotter.add_axes()
    plotter.add_title("Vector Field Orientation")

    plotter.show()
    
def vis_3d_magnitude(density, vec):
    S = density.shape[-1]

    ones = np.ones(S)
    slices = np.linspace(0, 1, S)
    px = np.kron(ones, np.kron(ones, slices))
    py = np.kron(ones, np.kron(slices, ones))
    pz = np.kron(slices, np.kron(ones, ones))
    pos = np.vstack([px.ravel(), py.ravel(), pz.ravel()]).T   
    
    density = density.ravel()
    vec = vec.reshape(-1, 3)
    mask = density > 0.5
    pos = pos[mask, :]
    vec = vec[mask, :]
    density = density[mask]
    
    n_points = len(pos)
    sample_size = n_points // 10
    indices = np.random.choice(n_points, sample_size, replace=False)
    pos = pos[indices, :]
    vec = vec[indices, :]
    density = density[indices]

    grid = pv.PolyData(pos)
    grid['disp_x'] = vec[:, 0]
    grid['disp_y'] = vec[:, 1]
    grid['disp_z'] = vec[:, 2]

    plotter = pv.Plotter()
    
    plotter.add_mesh(grid, 
                            scalars="disp_x", 
                            cmap="viridis", 
                            point_size=5, 
                            render_points_as_spheres=True,
                            show_scalar_bar=False)

    plotter.add_scalar_bar(
        title="Displacement X",
        n_labels=5,
        vertical=True,
        position_x=0.85,
        position_y=0.1,
        width=0.02,
        height=0.8
    )
    
    directions = ['disp_x', 'disp_y', 'disp_z']
    direction_titles = {
        'disp_x': 'Displacement X',
        'disp_y': 'Displacement Y',
        'disp_z': 'Displacement Z'
    }

    current_direction = {'index': 0}
    
    direction_label = plotter.add_text(
        f"Current: {direction_titles[directions[current_direction['index']]]}",
        position=(600, 50), 
        font_size=12,
        color='black'
    )

    def toggle_direction(value):
        idx = int(round(value))
        current_direction['index'] = idx % len(directions)
        current_dir = directions[current_direction['index']]
        current_title = direction_titles[current_dir]

        grid.set_active_scalars(current_dir)

        plotter.scalar_bar.SetTitle(current_title)
        plotter.update_scalar_bar_range([grid[current_dir].min(), grid[current_dir].max()])

        direction_label.SetInput(f"Current: {current_title}")

        plotter.render()

    plotter.add_slider_widget(
        callback=toggle_direction,
        rng=[0, 2],
        value=0,
        title="Select Direction",
        style='modern',
        pointa=(0.05, 0.9),
        pointb=(0.25, 0.9),
        color='lightblue'
    )

    plotter.add_axes()
    plotter.show()


def vis_3d_pc(vertices, force, BC, title):
    mask1 = BC == 1
    mask2 = BC == 0
    vertices1 = vertices[mask1, :]
    vertices2 = vertices[mask2, :]
    force1 = force[mask1, :]
    force2 = force[mask2, :]
    grid1 = pv.PolyData(vertices1)
    grid1['force'] = force1
    grid2 = pv.PolyData(vertices2)
    grid2['force'] = force2
    
    grid1['magnitude'] = np.linalg.norm(force1, axis=1)
    grid2['magnitude'] = np.linalg.norm(force2, axis=1)
    
    plotter = pv.Plotter()
    
    plotter.add_points(grid1, color='red', point_size=10, render_points_as_spheres=True)
    plotter.add_points(grid2, color='darkgray', point_size=5, render_points_as_spheres=True)
    
    glyphs1 = grid1.glyph(
        orient='force',
        scale=False,
        factor=2e-2,  
        geom=pv.Arrow(shaft_radius=0.03, tip_radius=0.05, tip_length=0.15)
    )
    glyphs2 = grid2.glyph(
        orient='force',
        scale=False,
        factor=2e-2,
        geom=pv.Arrow(shaft_radius=0.03, tip_radius=0.05, tip_length=0.15)
    )
    
    plotter.add_mesh(glyphs1, scalars='magnitude', cmap='viridis', lighting=True)
    plotter.add_mesh(glyphs2, scalars='magnitude', cmap='viridis', lighting=True)
    
    plotter.add_axes()
    plotter.add_title(title)
    plotter.show()
    
        
if __name__ == '__main__':
    
    # nodes-wise info
    density_path = 'computer_chair_density.npy'
    grad_path = 'computer_chair_grad.npy'
    ur_path = 'computer_chair_ur.npy'
    u_path = 'computer_chair_u.npy'
    
    density = np.load(density_path)
    grad = np.load(grad_path)
    ur = np.load(ur_path)
    u = np.load(u_path)
    
    vis_3d_scalar(density, grad)
    # vis_3d_orientation(density, u)
    # vis_3d_orientation(density, ur)
    # vis_3d_magnitude(density, u)
    # vis_3d_magnitude(density, ur)
    
    # vertices-wise info
    vu_path = 'computer_chair_vu.npy'
    vertices_path = 'computer_chair_vertices.npy'
    force_path = 'computer_chair_force.npy'
    BC_path = 'computer_chair_BC.npy'
    
    vu = np.load(vu_path)
    vertices = np.load(vertices_path)
    force = np.load(force_path)
    BC = np.load(BC_path)
    
    # vis_3d_pc(vertices, force, BC, title='vertices forces)
    # vis_3d_pc(vertices, vu, BC, title='vertices displacement')