import time
import argparse
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
        start_time = time.time()

        for i in range(10):
            viewer_env.update()

        result = mocap.get()
        # print("[Debug] Mocap Result:", result)  # 打印结果，确保数据是有效的

        if result['status'] == 'recording' and result["result"] is not None:
            qpos = model.forward(result["result"])
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
