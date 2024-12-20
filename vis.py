import numpy as np
import pyvista as pv

def vis_3d_scalar(density, grad):  
    S = density.shape[-1]
    line = np.linspace(0, 1, S)
    x, y, z = np.meshgrid(line, line, line)
    pos = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    
    density = density.ravel()
    grad = grad.ravel()
    
    mask_d = density > 0.5
    pos_d = pos[mask_d, :]
    density = density[mask_d]
    
    mask_g = grad < np.percentile(grad, 1)
    pos_g = pos[mask_g, :]
    grad = grad[mask_g]
    
    grid_d = pv.PolyData(pos_d)
    grid_d.point_data['scalar_values'] = density
    grid_g = pv.PolyData(pos_g)
    grid_g.point_data['scalar_values'] = grad
    
    plotter = pv.Plotter()
    plotter.add_points(grid_d, color='darkgray', point_size=4)
    plotter.add_points(grid_g, color='cornflowerblue', point_size=4)
     
    Inter = 1 / (S-1)
    test_pos = np.array([85, 110, 60])
    test_grid = pv.PolyData(test_pos * Inter)
    plotter.add_points(test_grid, color='red', point_size=10)
    
    plotter.add_axes()
    plotter.show()
    
def vis_3d_orientation(density, vec):
    S = density.shape[-1]
    line = np.linspace(0, 1, S)
    x, y, z = np.meshgrid(line, line, line)
    pos = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T   
    
    density = density.ravel()
    vec = vec.reshape(-1, 3)
    mask = density > 0.5
    pos = pos[mask, :]
    vec = vec[mask, :]
    density = density[mask]
    
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
        factor=5e-3, 
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
    line = np.linspace(0, 1, S)
    x, y, z = np.meshgrid(line, line, line)
    pos = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    
    density = density.ravel()
    vec = vec.reshape(-1, 3)
    mask = density > 0.5
    pos = pos[mask, :]
    vec = vec[mask, :]
    density = density[mask]
    
    magnitude = np.linalg.norm(vec, axis=1)
    low_5 = np.percentile(magnitude, 5)
    high_5 = np.percentile(magnitude, 95)
    magnitude = np.clip(magnitude, low_5, high_5)
    
    grid = pv.PolyData(pos)
    grid['vectors'] = vec
    grid['magnitude'] = magnitude

    plotter = pv.Plotter()
    
    glyphs = grid.glyph(
        orient='vectors',
        scale=False,
        factor=2e-3,  
        geom=pv.Arrow(shaft_radius=0.03, tip_radius=0.05, tip_length=0.15)
    )
    
    plotter.add_mesh(glyphs,
                     scalars='magnitude',
                     cmap='viridis',
                     show_scalar_bar=True,
                     lighting=True)
    
    plotter.add_mesh(grid, 
                    color='darkgray',
                    point_size=3,
                    render_points_as_spheres=True)

    plotter.add_axes()
    plotter.add_title("Vector Field Magnitude")

    plotter.show()

def vis_3d_pc(vertices, force, BC):
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
    plotter.add_title("Point Cloud Force")
    plotter.show()
    
        
if __name__ == '__main__':
    
    density_path = 'computer_chair_density.npy'
    grad_path = 'computer_chair_grad.npy'
    ur_path = 'computer_chair_ur.npy'
    
    u_path = 'computer_chair_u.npy'
    vertices_path = 'computer_chair_vertices.npy'
    force_path = 'computer_chair_force.npy'
    BC_path = 'computer_chair_BC.npy'
    
    density = np.load(density_path)
    grad = np.load(grad_path)
    ur = np.load(ur_path)
    
    u = np.load(u_path)
    vertices = np.load(vertices_path)
    force = np.load(force_path)
    BC = np.load(BC_path)

    # vis_3d_scalar(density, grad)
    # vis_3d_orientation(density, u)
    # vis_3d_orientation(density, ur)
    # vis_3d_magnitude(density, u)
    # vis_3d_magnitude(density, ur)
    
    # vis_3d_pc(vertices, force, BC)

    vis_3d_pc(vertices, u, BC)