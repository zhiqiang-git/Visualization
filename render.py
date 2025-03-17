import os
from datetime import datetime
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import argparse
import subprocess

ps.init()

# scp_command1 = "scp stu4@501:~/PS/log/render_data.npz data/render_data.npz"
# scp_command2 = "scp stu4@501:~/PS/log/grad_data.npz data/grad_data.npz"
scp_command1 = "scp zhiqiang@qixing-ut:~/PS/log/render_data.npz data/render_data.npz"
scp_command2 = "scp zhiqiang@qixing-ut:~/PS/log/grad_data.npz data/grad_data.npz"
subprocess.run(scp_command1, shell=True, check=True)
subprocess.run(scp_command2, shell=True, check=True)

def callback():
    global percentage, cg, grad, grad_pc
    global origin_pc, u_pc, e_pc, density_thres
    global c, cu, ce, l, d, f

    psim.PushItemWidth(150)
    changed, percentage = psim.SliderFloat("Gradient percentage Slide", percentage, v_min=0, v_max=1)
    if changed:
        origin_pc.set_enabled(True)
        u_pc.set_enabled(False)
        e_pc.set_enabled(False)

        threshold = np.percentile(grad, 100*percentage)
        mask = grad < threshold
        cg_masked = cg[mask]
        grad_masked = grad[mask]
        grad_pc = ps.register_point_cloud("Gradient", cg_masked, radius=0.02, color=(0.8, 0.8, 0.8), transparency=0.8)
        grad_pc.add_scalar_quantity("Gradient", grad_masked, cmap='viridis', enabled=True)
        grad_pc.set_point_render_mode("quad")
        grad_pc.set_enabled()

    changed, percentage = psim.InputFloat("Gradient percentage Input", percentage)
    if changed:
        origin_pc.set_enabled(True)
        u_pc.set_enabled(False)
        e_pc.set_enabled(False)

        threshold = np.percentile(grad, 100*percentage)
        mask = grad < threshold
        cg_masked = cg[mask]
        grad_masked = grad[mask]
        grad_pc = ps.register_point_cloud("Gradient", cg_masked, radius=0.02, color=(0.8, 0.8, 0.8), transparency=0.8)
        grad_pc.add_scalar_quantity("Gradient", grad_masked, cmap='viridis', enabled=True)
        grad_pc.set_point_render_mode("quad")
        grad_pc.set_enabled()
    
    changed, density_thres = psim.SliderFloat("Density threshold Slide", density_thres, v_min=0, v_max=1)
    if changed:
        grad_pc.set_enabled(False)

        mask_d = d > density_thres
        cm = c[mask_d]
        cum = cu[mask_d]
        cem = ce[mask_d]
        lm = l[mask_d]
        dm = d[mask_d]
        fm = f[mask_d]

        origin_pc = ps.register_point_cloud("Origin", cm, radius=0.02, color=(0.8, 0.8, 0.8), transparency=0.8)
        u_pc = ps.register_point_cloud("Displacement", cum, radius=0.02, color=(0.8, 0.8, 0.8), transparency=0.8)
        e_pc = ps.register_point_cloud("Estimation", cem, radius=0.02, color=(0.8, 0.8, 0.8), transparency=0.8)
        origin_pc.set_point_render_mode("quad")
        u_pc.set_point_render_mode("quad")
        e_pc.set_point_render_mode("quad")
        origin_pc.set_enabled()
        u_pc.set_enabled(False)
        e_pc.set_enabled(False)

        origin_pc.add_scalar_quantity("Loss term", lm, cmap='viridis', enabled=True)
        origin_pc.add_scalar_quantity("Density", dm, cmap='viridis', enabled=False)
        origin_pc.add_vector_quantity("Force", fm, enabled=False)

    if(psim.Button("Optimization results")):
        grad_pc.set_enabled(False)
        origin_pc.set_enabled(False)
        u_pc.set_enabled(False)
        e_pc.set_enabled(False)

    psim.PopItemWidth()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_type", type=str, required=True, 
                        choices=("relative", "rigid", "direct"),
                        help="The type of loss used in loss computation")
    args = parser.parse_args()

    loss_type = args.loss_type

    percentage = 0.02
    density_thres = 0.5
    
    if loss_type == "relative" or loss_type == "rigid":
        # shape and displaced shape
        render_data = np.load("data/render_data.npz")
        c = render_data["c"]
        cu = render_data["cu"]
        ce = render_data["ce"]
        l = render_data["l"]
        d = render_data["d"]
        f = render_data["f"]

        mask_d = d > density_thres
        cm = c[mask_d]
        cum = cu[mask_d]
        cem = ce[mask_d]
        lm = l[mask_d]
        dm = d[mask_d]
        fm = f[mask_d]

        origin_pc = ps.register_point_cloud("Origin", cm, radius=0.02, color=(0.8, 0.8, 0.8), transparency=0.8)
        u_pc = ps.register_point_cloud("Displacement", cum, radius=0.02, color=(0.8, 0.8, 0.8), transparency=0.8)
        e_pc = ps.register_point_cloud("Estimation", cem, radius=0.02, color=(0.8, 0.8, 0.8), transparency=0.8)
        origin_pc.set_point_render_mode("quad")
        u_pc.set_point_render_mode("quad")
        e_pc.set_point_render_mode("quad")
        origin_pc.set_enabled()
        u_pc.set_enabled(False)
        e_pc.set_enabled(False)

        origin_pc.add_scalar_quantity("Loss term", lm, cmap='viridis', enabled=True)
        origin_pc.add_scalar_quantity("Density", dm, cmap='viridis', enabled=False)
        origin_pc.add_vector_quantity("Force", fm, enabled=False)

    elif loss_type == "direct":
        # shape and displaced shape
        render_data = np.load("data/render_data.npz")
        c = render_data["c"]
        cu = render_data["cu"]
        l = render_data["l"]
        d = render_data["d"]
        f = render_data["f"]

        mask_d = d > density_thres
        cm = c[mask_d]
        cum = cu[mask_d]
        lm = l[mask_d]
        dm = d[mask_d]
        fm = f[mask_d]

        origin_pc = ps.register_point_cloud("Origin", cm, radius=0.02, color=(0.8, 0.8, 0.8), transparency=0.8)
        u_pc = ps.register_point_cloud("Displacement", cum, radius=0.02, color=(0.8, 0.8, 0.8), transparency=0.8)
        origin_pc.set_point_render_mode("quad")
        u_pc.set_point_render_mode("quad")
        origin_pc.set_enabled()
        u_pc.set_enabled(False)

        origin_pc.add_scalar_quantity("Loss term", lm, cmap='viridis', enabled=True)
        origin_pc.add_scalar_quantity("Density", dm, cmap='viridis', enabled=False)

    else:
        pass

    # grad data
    grad_data = np.load("data/grad_data.npz")
    cg = grad_data["cg"]
    grad = grad_data["grad"]
    init_threshold = np.percentile(grad, 100*percentage)
    init_mask = grad < init_threshold
    init_cg_masked = cg[init_mask]
    init_grad_masked = grad[init_mask]

    grad_pc = ps.register_point_cloud("Gradient", init_cg_masked, radius=0.02, color=(0.8, 0.8, 0.8), transparency=0.8)
    grad_pc.add_scalar_quantity("Gradient", init_grad_masked, cmap='viridis', enabled=True)
    grad_pc.set_point_render_mode("quad")

    ps.set_user_callback(callback)

    # camera params
    ps.set_view_projection_mode("orthographic")
    ps.look_at((0.05, 0.75, 1.0), (0.5, 0.5, 0.5))
    ps.show()

    