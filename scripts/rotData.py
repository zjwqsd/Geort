import numpy as np

data = np.load("data/human_alex.npy")  # shape: (N, 21, 3)

converted_data = data[:, :, [1, 0, 2]]
converted_data[:, :, 2] *= -1

np.save("data/rot_alex.npy", converted_data)
print("转换完成，已保存为 data/rot_alex.npy")
