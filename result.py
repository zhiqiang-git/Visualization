import polyscope as ps
import numpy as np
import subprocess

ps.init()

# scp_command = "scp stu4@501:~/PS/log/result_data.npz data/result_data.npz"
scp_command = "scp zhiqiang@qixing-ut:~/PS/log/result_data.npz data/result_data.npz"
subprocess.run(scp_command, shell=True, check=True)

result_data = np.load("data/result_data.npz")
result = result_data["result"]
res = result.shape[0] + 1
dim = (res, res, res)
bounds_low = (0., 0., 0.)
bounds_high = (1., 1., 1.)
result_grid = ps.register_volume_grid("Result", dim, bounds_low, bounds_high)
result_grid.add_scalar_quantity("density", result, defined_on='cells', cmap='viridis', enabled=True)

ps.show()