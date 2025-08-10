import time
import argparse
from geort.env import hand
from geort.mocap.replay_mocap import ReplayMocap
from geort.env.hand import HandKinematicModel
from geort import load_model, get_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-hand', type=str, default='allegro')
    parser.add_argument('-ckpt_tag', type=str, default='alex')
    parser.add_argument('-data', type=str, default='human')

    args = parser.parse_args()

    # GeoRT Model.
    model = load_model(args.ckpt_tag)
    
    # Motion Capture.
    mocap = ReplayMocap(args.data)
    
    # Robot Simulation.
    config = get_config(args.hand)
    hand = HandKinematicModel.build_from_config(config, render=True)
    viewer_env = hand.get_viewer_env()
    
    # Run!
    frame_time = 0.01  # 每帧的时间间隔（秒），固定为 1/100 秒
    while True:
        first_checked = False
        lower, upper = hand.get_joint_limit()
        start_time = time.time()

        for i in range(10):
            viewer_env.update()

        result = mocap.get()
        if result['status'] == 'recording' and result["result"] is not None:
            raw = result["result"]

            if not first_checked:
                first_checked = True
                import numpy as np
                arr = np.asarray(raw)
                if arr.ndim == 2 and arr.shape[-1] == 3:
                    mins, maxs = arr.min(0), arr.max(0)
                    span = float((maxs - mins).max())
                    if span > 3.0:
                        print("[WARN] 输入点跨度>3m，疑似毫米(mm)，请确认是否需要/1000。")

            qpos = model.forward(raw)             # 仍然直接喂入原始 [N,3]
            import numpy as np
            qpos = np.clip(qpos, lower, upper)    # 限幅
            hand.set_qpos_target(qpos)

        if result['status'] == 'quit':
            break

        # 计算当前帧的时间
        elapsed_time = time.time() - start_time

        # 保证每秒 100 帧
        sleep_time = frame_time - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == '__main__':
    main()
