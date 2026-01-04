import h5py
import numpy as np

INPUT_H5 = "/home/jiji/workspace/isaac/project/PushT_PD/diffusion_policy-main/data/dataset.hdf5"
OUTPUT_H5 = "/home/jiji/workspace/isaac/project/PushT_PD/data/robomimic_style_dataset.hdf5"


def main():
    with h5py.File(INPUT_H5, "r") as fin, \
         h5py.File(OUTPUT_H5, "w") as fout:

        data_grp = fout.create_group("data")

        obs_state = []
        obs_img_global = []
        obs_img_wrist = []
        actions = []
        rewards = []
        dones = []
        episode_ends = []

        step_count = 0

        for ep_key in sorted(fin.keys()):
            ep = fin[ep_key]
            T = ep["action"].shape[0]

            obs_state.append(ep["obs/state"][:])
            obs_img_global.append(ep["obs/img_global"][:])
            obs_img_wrist.append(ep["obs/img_wrist"][:])
            actions.append(ep["action"][:])

            rewards.append(np.zeros(T, dtype=np.float32))
            dones.append(np.zeros(T, dtype=bool))

            step_count += T
            episode_ends.append(step_count)

        # concat
        data_grp.create_dataset("obs/state", data=np.concatenate(obs_state))
        data_grp.create_dataset("obs/img_global", data=np.concatenate(obs_img_global))
        data_grp.create_dataset("obs/img_wrist", data=np.concatenate(obs_img_wrist))
        data_grp.create_dataset("actions", data=np.concatenate(actions))
        data_grp.create_dataset("rewards", data=np.concatenate(rewards))
        data_grp.create_dataset("dones", data=np.concatenate(dones))
        data_grp.create_dataset("episode_ends", data=np.array(episode_ends))

    print(f"✅ Saved robomimic-style HDF5 to {OUTPUT_H5}")


if __name__ == "__main__":
    main()
