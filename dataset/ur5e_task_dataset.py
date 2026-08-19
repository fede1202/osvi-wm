import glob
import os
import pickle
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, Sampler

from dataset.agent_dataset import adjust_augmentations
from utils.projection_utils import compute_crop_adjustment
from utils.utils import crop, randomize_video, resize

'''
train_tasks/test_tasks -> task_XX
teacher pkl -> camera_front_image -> decode JPEG -> BGR to RGB -> video context
agent pkl -> front_camera_image/camera_front_image -> BGR to RGB -> images
obs["eef_pos"] -> traj_points[:, :3]
action[-1] / gripper_qpos -> traj_points[:, 3]
projection YAML -> agent_traj["projection_matrix"]

'''


def _as_task_name(task):
    if isinstance(task, int):
        return f"task_{task:02d}"
    return str(task)


def _add_ur5_savers_path(agent_dir=None, ur5_repo=None):
    candidates = []
    if ur5_repo:
        candidates.append(Path(os.path.expanduser(ur5_repo)))
    if agent_dir:
        p = Path(os.path.expanduser(agent_dir)).resolve()
        # agent_dir is UR5e-2f-85/ai_controller/saved_rollouts/script_controller.
        if len(p.parents) >= 3:
            candidates.append(p.parents[2])

    for root in candidates:
        savers_dir = root / "dataset_collector" / "dataset_collector_pkg" / "scripts"
        if savers_dir.exists() and str(savers_dir) not in sys.path:
            sys.path.insert(0, str(savers_dir))


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Trajectory":
            try:
                import savers

                return savers.Trajectory
            except Exception:
                pass
        return super().find_class(module, name)


def _load_pickle(path):
    with open(path, "rb") as f:
        return _CompatUnpickler(f).load()


def _unwrap_trajectory(payload):
    if isinstance(payload, dict) and "traj" in payload:
        return payload["traj"]
    return payload


def _traj_len(traj):
    return len(traj)


def _traj_step(traj, index):
    if hasattr(traj, "get"):
        return traj.get(index)
    return traj[index]


def _step_obs(step):
    if isinstance(step, dict) and "obs" in step:
        return step["obs"]
    return step


def _resolve_repo_relative(path):
    path = os.path.expanduser(path)
    if os.path.isabs(path) or os.path.exists(path):
        return path
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / path)


def _load_projection_matrix(path):
    resolved = _resolve_repo_relative(path)
    with open(resolved, "r") as f:
        data = yaml.safe_load(f)
    matrix = data["projection_matrix"] if isinstance(data, dict) else data
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"projection_matrix must have shape (4, 4), got {matrix.shape}")
    return matrix


class UR5eTrajectoryDataset(Dataset):
    def __init__(
        self,
        files,
        *,
        image_keys,
        projection_matrix,
        height=240,
        width=320,
        crop=(0, 0, 0, 0),
        T_context=10,
        T_pair=1,
        color_jitter=None,
        rand_crop=None,
        rand_translate=None,
        rand_flip=False,
        randomize_vid_frames=False,
        sample_sides=True,
        no_context_jitter=True,
        normalize=True,
        reduce_bits=False,
        decode_compressed_images=True,
        convert_bgr_to_rgb=True,
        eef_pos_key="eef_pos",
        action_gripper_index=-1,
        gripper_qpos_key="gripper_qpos",
        grasp_threshold=0.5,
        grasp_scale=0.2,
        high_ent=False,
        head_label=0,
        **unused_params,
    ):
        self._files = list(files)
        self._image_keys = tuple(image_keys)
        self._projection_matrix = np.asarray(projection_matrix, dtype=np.float32)
        self._im_dims = (width, height)
        self._crop = tuple(crop) if crop is not None else (0, 0, 0, 0)
        self._T_context = T_context
        self._T_pair = T_pair
        self._color_jitter = color_jitter
        self._rand_crop = rand_crop
        self._rand_trans = np.array(rand_translate if rand_translate is not None else [0, 0])
        self._rand_flip = rand_flip
        self._randomize_vid_frames = randomize_vid_frames
        self._sample_sides = sample_sides
        self.no_context_jitter = no_context_jitter
        self._normalize = normalize
        self._reduce_bits = reduce_bits
        self._decode_compressed_images = decode_compressed_images
        self._convert_bgr_to_rgb = convert_bgr_to_rgb
        self._eef_pos_key = eef_pos_key
        self._action_gripper_index = action_gripper_index
        self._gripper_qpos_key = gripper_qpos_key
        self._grasp_threshold = grasp_threshold
        self._grasp_scale = grasp_scale
        self.high_ent = high_ent
        self.head_label = head_label

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index):
        return self.load_traj(index)

    def load_traj(self, index):
        path = self._files[index]
        payload = _load_pickle(path)
        traj = _unwrap_trajectory(payload)
        return traj, path

    def make_context(self, index, force_flip=None):
        traj, fname = self.load_traj(index)
        frames = self._sample_context_frames(traj)
        frames, stats = randomize_video(
            frames,
            self._color_jitter,
            None,
            self._rand_crop,
            0,
            self._rand_trans,
            self._normalize,
            rand_flip=self._rand_flip,
            force_flip=force_flip,
        )
        projection = self._adjust_projection(stats, frames.shape[-3:-1])
        return {
            "video": np.transpose(frames, (0, 3, 1, 2)),
            "projection_matrix": projection,
            "fname": fname,
        }

    def make_pairs(self, index, force_flip=None):
        traj, fname = self.load_traj(index)
        len_traj = _traj_len(traj)
        chosen_t = np.linspace(0, len_traj - 1, num=self._T_pair + 1, endpoint=True, dtype=int)

        images = []
        for t in chosen_t:
            obs = _step_obs(_traj_step(traj, int(t)))
            images.append(self._crop_and_resize(self._get_image(obs))[None])
        images = np.concatenate(images, axis=0).astype(np.float32)
        images, stats = self._randomize_frames(images, force_flip=force_flip)

        out_inds = np.linspace(0, len_traj - 1, num=50, endpoint=True, dtype=int)
        poses = []
        grasps = []
        for i in out_inds:
            step = _traj_step(traj, int(i))
            obs = _step_obs(step)
            poses.append(np.asarray(obs[self._eef_pos_key], dtype=np.float32)[:3])
            grasps.append(self._get_grasp(step, obs))

        grasps = (np.asarray(grasps, dtype=np.float32) > self._grasp_threshold).astype(np.float32)
        traj_points = np.concatenate(
            [np.stack(poses, axis=0), (grasps[:, None] * self._grasp_scale)],
            axis=-1,
        )

        return {
            "images": np.transpose(images, (0, 3, 1, 2)),
            "traj_points": traj_points,
            "projection_matrix": self._adjust_projection(stats[0], images.shape[-2:]),
            "setting_name": "ur5e_pick_place",
            "start0": True,
            "high_ent": int(self.high_ent),
            "head_label": int(self.head_label),
            "fname": fname,
        }

    def _sample_context_frames(self, traj):
        len_traj = _traj_len(traj)
        if self.no_context_jitter:
            inds = np.linspace(0, len_traj - 1, num=self._T_context, endpoint=True, dtype=int)
        else:
            per_bracket = max(len_traj / self._T_context, 1)
            inds = []
            for i in range(self._T_context):
                lo = int(i * per_bracket)
                hi = int((i + 1) * per_bracket)
                n = int(max(0, min(np.random.randint(lo, max(lo + 1, hi)), len_traj - 1)))
                if self._sample_sides and i == 0:
                    n = 0
                elif self._sample_sides and i == self._T_context - 1:
                    n = len_traj - 1
                inds.append(n)

        frames = []
        for i in inds:
            obs = _step_obs(_traj_step(traj, int(i)))
            frames.append(self._crop_and_resize(self._get_image(obs))[None])
        return np.concatenate(frames, axis=0)

    def _get_image(self, obs):
        for key in self._image_keys:
            if key in obs:
                return self._decode_image(obs[key])
        raise KeyError(f"None of image keys {self._image_keys} found in obs keys {list(obs.keys())}")

    def _decode_image(self, value):
        img = value
        if isinstance(img, (bytes, bytearray)):
            img = np.frombuffer(img, dtype=np.uint8)
        if isinstance(img, np.ndarray) and img.ndim == 1 and self._decode_compressed_images:
            decoded = cv2.imdecode(img.astype(np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError("cv2.imdecode failed for compressed image")
            img = decoded
        img = np.asarray(img)
        if img.ndim == 2:
            img = img[:, :, None]
        if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))
        if self._convert_bgr_to_rgb and img.ndim == 3 and img.shape[-1] >= 3:
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.uint8)

    def _get_grasp(self, step, obs):
        if isinstance(step, dict) and "action" in step and step["action"] is not None:
            action = np.asarray(step["action"]).reshape(-1)
            return float(action[self._action_gripper_index])
        if self._gripper_qpos_key in obs:
            return float(np.asarray(obs[self._gripper_qpos_key]).reshape(-1)[0])
        return 0.0

    def _crop_and_resize(self, img):
        return resize(crop(img, self._crop), self._im_dims, False, self._reduce_bits)

    def _randomize_frames(self, frames, force_flip=None):
        if self._randomize_vid_frames:
            result = [
                randomize_video(
                    f[None],
                    self._color_jitter,
                    None,
                    self._rand_crop,
                    0,
                    self._rand_trans,
                    self._normalize,
                    rand_flip=self._rand_flip,
                    force_flip=force_flip,
                )
                for f in frames
            ]
            frames, stats = zip(*result)
            return np.concatenate(frames, axis=0), list(stats)

        frames, stats = randomize_video(
            frames,
            self._color_jitter,
            None,
            self._rand_crop,
            0,
            self._rand_trans,
            self._normalize,
            rand_flip=self._rand_flip,
            force_flip=force_flip,
        )
        return frames, [stats]

    def _adjust_projection(self, stats, size):
        projection = self._projection_matrix.copy()
        crop_adjust = compute_crop_adjustment(self._crop, size)
        projection = projection @ np.linalg.inv(crop_adjust)
        projection = projection @ adjust_augmentations(stats, size)
        return projection.astype(np.float32)


class UR5eAgentTeacherDataset(Dataset):
    def __init__(
        self,
        agent_dir,
        teacher_dir,
        agent_context=0,
        epoch_repeat=1,
        mode="train",
        train_tasks=None,
        test_tasks=None,
        novar=False,
        flip_sync=False,
        epoch_repeat_train=None,
        projection_matrix_path="configs/ur5e_zed_front_projection.yaml",
        teacher_image_keys=("camera_front_image",),
        agent_image_keys=("camera_front_image", "front_camera_image"),
        ur5_repo=None,
        **params,
    ):
        _add_ur5_savers_path(agent_dir=agent_dir, ur5_repo=ur5_repo)

        teacher_context = params.pop("T_context", 15)
        convert_bgr_to_rgb = params.pop("convert_bgr_to_rgb", True)
        agent_convert_bgr_to_rgb = params.pop("agent_convert_bgr_to_rgb", convert_bgr_to_rgb)
        teacher_convert_bgr_to_rgb = params.pop("teacher_convert_bgr_to_rgb", convert_bgr_to_rgb)
        self._agent_context = agent_context if agent_context is not None else teacher_context
        self.flip_sync = flip_sync
        self.novar = novar
        self._epoch_repeat = epoch_repeat_train if (epoch_repeat_train is not None and mode == "train") else epoch_repeat

        if mode != "train":
            params = params.copy()
            params["rand_translate"] = None
            params["color_jitter"] = None
            params["rand_crop"] = None
            params["rand_flip"] = False

        projection_matrix = _load_projection_matrix(projection_matrix_path)
        tasks = self._select_tasks(agent_dir, train_tasks or [], test_tasks or [], mode)

        (
            agent_files,
            teacher_files,
            self._pairs,
            self._pair_tasks,
            self._task_to_indices,
        ) = self._build_task_pairs(agent_dir, teacher_dir, tasks)
        self._task_names = sorted(self._task_to_indices.keys())
        self._agent_dataset = UR5eTrajectoryDataset(
            agent_files,
            image_keys=agent_image_keys,
            projection_matrix=projection_matrix,
            T_context=self._agent_context,
            convert_bgr_to_rgb=agent_convert_bgr_to_rgb,
            **params,
        )
        self._teacher_dataset = UR5eTrajectoryDataset(
            teacher_files,
            image_keys=teacher_image_keys,
            projection_matrix=projection_matrix,
            T_context=teacher_context,
            convert_bgr_to_rgb=teacher_convert_bgr_to_rgb,
            **params,
        )

        print(
            f"Loaded UR5e task dataset ({mode}): "
            f"{len(tasks)} tasks, {len(agent_files)} agent trajs, "
            f"{len(teacher_files)} teacher trajs, {len(self._pairs)} pairs"
        )

    def __len__(self):
        return len(self._pairs) * self._epoch_repeat

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self), "invalid index!"

        a_i, t_i = self._pairs[index % len(self._pairs)]
        label = 0
        force_flip = None
        if self.flip_sync:
            force_flip = [
                -1 if random.random() > 0.5 else 1,
                -1 if random.random() > 0.5 else 1,
            ]

        teacher_context = self._teacher_dataset.make_context(t_i, force_flip=force_flip)
        agent_pairs = self._agent_dataset.make_pairs(a_i, force_flip=force_flip)
        if self._agent_context:
            agent_context = self._agent_dataset.make_context(a_i, force_flip=force_flip)
            return teacher_context, agent_context, agent_pairs
        return teacher_context, agent_pairs, label

    @property
    def task_names(self):
        return list(self._task_names)

    @property
    def task_to_indices(self):
        return {task: list(indices) for task, indices in self._task_to_indices.items()}

    def task_for_index(self, index):
        return self._pair_tasks[index % len(self._pairs)]

    def _select_tasks(self, agent_dir, train_tasks, test_tasks, mode):
        all_tasks = sorted(
            d for d in os.listdir(os.path.expanduser(agent_dir))
            if os.path.isdir(os.path.join(os.path.expanduser(agent_dir), d)) and d.startswith("task_")
        )
        train_tasks = [_as_task_name(t) for t in train_tasks]
        test_tasks = [_as_task_name(t) for t in test_tasks]
        if mode == "train":
            tasks = train_tasks or sorted(set(all_tasks) - set(test_tasks))
        else:
            tasks = test_tasks
        if not tasks:
            raise ValueError(f"No tasks selected for mode={mode}. Available tasks: {all_tasks}")
        return sorted(tasks)

    def _build_task_pairs(self, agent_dir, teacher_dir, tasks):
        agent_files = []
        teacher_files = []
        pairs = []
        pair_tasks = []
        task_to_indices = {}
        agent_root = os.path.expanduser(agent_dir)
        teacher_root = os.path.expanduser(teacher_dir)

        for task in tasks:
            task_agent_files = sorted(glob.glob(os.path.join(agent_root, task, "*.pkl")))
            task_teacher_files = sorted(glob.glob(os.path.join(teacher_root, task, "*.pkl")))
            if not task_agent_files:
                raise FileNotFoundError(f"No agent .pkl files found for {task} in {agent_root}")
            if not task_teacher_files:
                raise FileNotFoundError(f"No teacher .pkl files found for {task} in {teacher_root}")

            agent_inds = np.arange(len(task_agent_files)) + len(agent_files)
            teacher_inds = np.arange(len(task_teacher_files)) + len(teacher_files)
            agent_files.extend(task_agent_files)
            teacher_files.extend(task_teacher_files)
            if self.novar:
                task_pairs = list(zip(agent_inds, teacher_inds))
            else:
                task_pairs = [(a, t) for a in agent_inds for t in teacher_inds]

            start = len(pairs)
            pairs.extend(task_pairs)
            end = len(pairs)
            task_to_indices[task] = list(range(start, end))
            pair_tasks.extend([task] * len(task_pairs))

        return agent_files, teacher_files, pairs, pair_tasks, task_to_indices
def _unwrap_ur5e_dataset(dataset):
    delegate = getattr(dataset, "_delegate", None)
    return delegate if delegate is not None else dataset


class BalancedTaskBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        *,
        batch_size=None,
        samples_per_task_per_batch=None,
        drop_last=True,
        shuffle=True,
        seed=0,
        epoch_strategy="max",
    ):
        self.dataset = _unwrap_ur5e_dataset(dataset)
        if not hasattr(self.dataset, "task_to_indices"):
            raise TypeError("BalancedTaskBatchSampler requires a UR5eAgentTeacherDataset")

        self.task_to_indices = self.dataset.task_to_indices
        self.tasks = sorted(self.task_to_indices.keys())
        if not self.tasks:
            raise ValueError("Cannot build balanced batches without tasks")

        if samples_per_task_per_batch is None:
            if batch_size is None:
                samples_per_task_per_batch = 1
            else:
                if batch_size % len(self.tasks) != 0:
                    raise ValueError(
                        f"batch_size={batch_size} is not divisible by num_tasks={len(self.tasks)}. "
                        "For balanced batches use batch_size = num_tasks * samples_per_task_per_batch."
                    )
                samples_per_task_per_batch = batch_size // len(self.tasks)
        elif batch_size is not None and batch_size != len(self.tasks) * samples_per_task_per_batch:
            raise ValueError(
                f"batch_size={batch_size} does not match "
                f"num_tasks({len(self.tasks)}) * samples_per_task_per_batch({samples_per_task_per_batch})."
            )

        self.samples_per_task_per_batch = int(samples_per_task_per_batch)
        if self.samples_per_task_per_batch <= 0:
            raise ValueError("samples_per_task_per_batch must be positive")

        self.batch_size = len(self.tasks) * self.samples_per_task_per_batch
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        if epoch_strategy not in {"max", "min"}:
            raise ValueError("epoch_strategy must be 'max' or 'min'")
        self.epoch_strategy = epoch_strategy

        counts = [len(self.task_to_indices[task]) for task in self.tasks]
        reference = max(counts) if epoch_strategy == "max" else min(counts)
        if drop_last:
            self._num_batches = reference // self.samples_per_task_per_batch
        else:
            self._num_batches = int(np.ceil(reference / self.samples_per_task_per_batch))
        if self._num_batches <= 0:
            raise ValueError(
                "Not enough samples to build one balanced batch. "
                f"Task counts: {dict(zip(self.tasks, counts))}"
            )

    def __iter__(self):
        rng = random.Random(self.seed)
        task_indices = {}
        task_positions = {}

        for task in self.tasks:
            indices = list(self.task_to_indices[task])
            if self.shuffle:
                rng.shuffle(indices)
            task_indices[task] = indices
            task_positions[task] = 0

        for _ in range(self._num_batches):
            batch = []
            for task in self.tasks:
                indices = task_indices[task]
                for _ in range(self.samples_per_task_per_batch):
                    pos = task_positions[task]
                    if pos >= len(indices):
                        pos = 0
                        if self.shuffle:
                            rng.shuffle(indices)
                    batch.append(int(indices[pos]))
                    task_positions[task] = pos + 1
            if self.shuffle:
                rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self._num_batches


def make_balanced_task_batch_sampler(
    dataset,
    *,
    batch_size=None,
    samples_per_task_per_batch=None,
    drop_last=True,
    shuffle=True,
    seed=0,
    epoch_strategy="max",
):
    return BalancedTaskBatchSampler(
        dataset,
        batch_size=batch_size,
        samples_per_task_per_batch=samples_per_task_per_batch,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
        epoch_strategy=epoch_strategy,
    )

