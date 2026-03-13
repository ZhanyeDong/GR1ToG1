import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

# Workaround for duplicate OpenMP runtime on Windows. Must be set before importing torch/numpy.
if os.environ.get("KMP_DUPLICATE_LIB_OK") is None:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pinocchio as pin
import torch


@dataclass
class IKResult:
    success: bool
    q: np.ndarray
    err_norm: float
    iters: int


def _strip_collisions(urdf_text: str) -> str:
    # Remove all collision blocks to avoid unsupported geometries (e.g., capsule).
    out = []
    i = 0
    while i < len(urdf_text):
        start = urdf_text.find("<collision", i)
        if start == -1:
            out.append(urdf_text[i:])
            break
        out.append(urdf_text[i:start])
        end = urdf_text.find("</collision>", start)
        if end == -1:
            # Malformed URDF; keep remaining text.
            break
        i = end + len("</collision>")
    return "".join(out)


def load_model(urdf_path: str, strip_collision: bool = False) -> Tuple[pin.Model, pin.Data]:
    if strip_collision:
        urdf_path = str(urdf_path)
        src = Path(urdf_path)
        text = src.read_text(encoding="utf-8")
        stripped = _strip_collisions(text)
        tmp = src.with_suffix(".nocol.urdf")
        tmp.write_text(stripped, encoding="utf-8")
        model = pin.buildModelFromUrdf(str(tmp))
    else:
        model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return model, data


def _require_joint(model: pin.Model, joint_name: str) -> int:
    joint_id = model.getJointId(joint_name)
    if joint_id == 0 and model.joints[0].name != joint_name:
        raise ValueError(f"joint '{joint_name}' not found in model")
    return joint_id


def _require_frame(model: pin.Model, frame_name: str) -> int:
    frame_id = model.getFrameId(frame_name)
    if frame_id == len(model.frames):
        candidates = [f.name for f in model.frames if frame_name in f.name or f.name in frame_name]
        hint = f" close: {candidates[:10]}" if candidates else ""
        raise ValueError(f"frame '{frame_name}' not found in model.{hint}")
    return frame_id


def _clip_to_limits(model: pin.Model, q: np.ndarray) -> np.ndarray:
    q_clipped = q.copy()
    for i in range(model.nq):
        low = model.lowerPositionLimit[i]
        high = model.upperPositionLimit[i]
        if math.isfinite(low):
            q_clipped[i] = max(q_clipped[i], low)
        if math.isfinite(high):
            q_clipped[i] = min(q_clipped[i], high)
    return q_clipped


def set_right_arm_q(
    model: pin.Model,
    right_arm_joint_names: Iterable[str],
    q_right: np.ndarray,
    q_full: Optional[np.ndarray] = None,
) -> np.ndarray:
    right_arm_joint_names = list(right_arm_joint_names)
    if q_right.shape != (len(right_arm_joint_names),):
        raise ValueError("q_right size does not match right_arm_joint_names")
    if q_full is None:
        q_full = pin.neutral(model)
    else:
        if q_full.shape[0] != model.nq:
            raise ValueError(f"q_full size {q_full.shape[0]} != model.nq {model.nq}")
        q_full = q_full.copy()

    for name, value in zip(right_arm_joint_names, q_right):
        jid = _require_joint(model, name)
        idx_q = model.joints[jid].idx_q
        nq = model.joints[jid].nq
        if nq != 1:
            raise ValueError(f"joint '{name}' has nq={nq}, expected 1")
        q_full[idx_q] = value

    return _clip_to_limits(model, q_full)


def compute_ee_pose(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    ee_frame: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if q.shape[0] != model.nq:
        raise ValueError(f"q size {q.shape[0]} != model.nq {model.nq}")
    frame_id = _require_frame(model, ee_frame)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pose = data.oMf[frame_id]
    return pose.translation.copy(), pose.rotation.copy()


def transform_gr1_pos_to_g1(pos_gr1: np.ndarray, R_g1_to_gr1: np.ndarray, t_g1_to_gr1: np.ndarray) -> np.ndarray:
    # p_gr1 = R * p_g1 + t  ->  p_g1 = R.T * (p_gr1 - t)
    return R_g1_to_gr1.T @ (pos_gr1 - t_g1_to_gr1)


def filter_dataset_gr1_to_g1_ik(
    dataset,
    g1_urdf: str,
    gr1_urdf: str,
    g1_right_ee: str,
    gr1_right_ee: str,
    g1_right_arm: Sequence[str],
    gr1_right_arm: Sequence[str],
    g1_base: str = "pelvis",
    gr1_base: str = "base",
    g1_left_shoulder: str = "left_shoulder_pitch_link",
    g1_right_shoulder: str = "right_shoulder_pitch_link",
    gr1_left_shoulder: str = "l_upper_arm_pitch",
    gr1_right_shoulder: str = "r_upper_arm_pitch",
    tol: float = 1e-4,
    max_iters: int = 200,
    damping: float = 1e-3,
):
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(dataset, torch.Tensor):
        data_np = dataset.detach().cpu().numpy()
        is_torch = True
    else:
        data_np = np.asarray(dataset)
        is_torch = False

    if data_np.ndim != 2 or data_np.shape[1] != 7:
        raise ValueError(f"dataset must have shape (N,7), got {data_np.shape}")

    model_g1, data_g1 = load_model(g1_urdf)
    model_gr1, data_gr1 = load_model(gr1_urdf, strip_collision=True)
    q_full_g1 = pin.neutral(model_g1)
    R_align, t_align = estimate_base_alignment(
        model_g1,
        data_g1,
        g1_base,
        g1_left_shoulder,
        g1_right_shoulder,
        model_gr1,
        data_gr1,
        gr1_base,
        gr1_left_shoulder,
        gr1_right_shoulder,
    )

    keep = np.zeros((data_np.shape[0],), dtype=bool)
    for i in range(data_np.shape[0]):
        q_right_gr1 = data_np[i]
        q_full_gr1 = set_right_arm_q(model_gr1, gr1_right_arm, q_right_gr1)
        gr1_pos, _ = compute_ee_pose(model_gr1, data_gr1, q_full_gr1, gr1_right_ee)
        g1_target = transform_gr1_pos_to_g1(gr1_pos, R_align, t_align)
        ik_res = ik_right_arm_position(
            model_g1,
            data_g1,
            g1_right_arm,
            g1_right_ee,
            g1_target,
            q0_full=q_full_g1,
            tol=tol,
            max_iters=max_iters,
            damping=damping,
        )
        keep[i] = ik_res.success

    if is_torch:
        return dataset[torch.as_tensor(keep)]
    return data_np[keep]


def ik_right_arm_position(
    model: pin.Model,
    data: pin.Data,
    right_arm_joint_names: Iterable[str],
    ee_frame: str,
    target_pos: np.ndarray,
    q0_full: Optional[np.ndarray] = None,
    tol: float = 1e-4,
    max_iters: int = 200,
    damping: float = 1e-3,
) -> IKResult:
    if target_pos.shape != (3,):
        raise ValueError("target_pos must be shape (3,)")
    frame_id = _require_frame(model, ee_frame)
    if q0_full is None:
        q = pin.neutral(model)
    else:
        if q0_full.shape[0] != model.nq:
            raise ValueError(f"q0_full size {q0_full.shape[0]} != model.nq {model.nq}")
        q = q0_full.copy()

    arm_joint_ids = [_require_joint(model, n) for n in right_arm_joint_names]
    arm_idx_v = [model.joints[j].idx_v for j in arm_joint_ids]

    for it in range(max_iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        cur = data.oMf[frame_id].translation
        err = cur - target_pos
        err_norm = float(np.linalg.norm(err))
        if err_norm <= tol:
            return IKResult(True, q, err_norm, it)

        J = pin.computeFrameJacobian(
            model, data, q, frame_id, pin.ReferenceFrame.WORLD
        )
        Jpos = J[:3, :]
        Jarm = Jpos[:, arm_idx_v]
        JJt = Jarm @ Jarm.T
        damp = (damping ** 2) * np.eye(3)
        dq_arm = -Jarm.T @ np.linalg.solve(JJt + damp, err)
        dq = np.zeros(model.nv)
        for col, idx in enumerate(arm_idx_v):
            dq[idx] = dq_arm[col]
        q = pin.integrate(model, q, dq)
        q = _clip_to_limits(model, q)

    return IKResult(False, q, err_norm, max_iters)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("zero-length vector for normalization")
    return v / n


def estimate_base_alignment(
    model_a: pin.Model,
    data_a: pin.Data,
    base_a: str,
    left_shoulder_a: str,
    right_shoulder_a: str,
    model_b: pin.Model,
    data_b: pin.Data,
    base_b: str,
    left_shoulder_b: str,
    right_shoulder_b: str,
) -> Tuple[np.ndarray, np.ndarray]:
    q_a = pin.neutral(model_a)
    q_b = pin.neutral(model_b)

    def link_pos_in_base(model, data, q, base_name, link_name):
        base_id = _require_frame(model, base_name)
        link_id = _require_frame(model, link_name)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        oMb = data.oMf[base_id]
        oMl = data.oMf[link_id]
        bMl = oMb.inverse() * oMl
        return bMl.translation.copy()

    la = link_pos_in_base(model_a, data_a, q_a, base_a, left_shoulder_a)
    ra = link_pos_in_base(model_a, data_a, q_a, base_a, right_shoulder_a)
    lb = link_pos_in_base(model_b, data_b, q_b, base_b, left_shoulder_b)
    rb = link_pos_in_base(model_b, data_b, q_b, base_b, right_shoulder_b)

    xa = _normalize(ra - la)
    xb = _normalize(rb - lb)
    ca = _normalize((ra + la) * 0.5)
    cb = _normalize((rb + lb) * 0.5)
    za = _normalize(np.cross(xa, ca))
    zb = _normalize(np.cross(xb, cb))
    ya = _normalize(np.cross(za, xa))
    yb = _normalize(np.cross(zb, xb))
    Ra = np.column_stack([xa, ya, za])
    Rb = np.column_stack([xb, yb, zb])
    R = Rb @ Ra.T
    ta = (ra + la) * 0.5
    tb = (rb + lb) * 0.5
    t = tb - R @ ta
    # Adjust: make G1 shoulders lower than GR1 by 0.07 in GR1 base Z.
    t = t + np.array([0.10, 0.0, 0.])
    return R, t


def example(q_right_gr1):
    urdf_g1 = "g1_29dof.urdf"
    urdf_gr1 = "GR1T2.urdf"

    g1_right_arm = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    gr1_right_arm = [
        "r_shoulder_pitch",
        "r_shoulder_roll",
        "r_shoulder_yaw",
        "r_elbow_pitch",
        "r_wrist_yaw",
        "r_wrist_roll",
        "r_wrist_pitch",
    ]

    g1_base = "pelvis"
    gr1_base = "base"
    g1_right_ee = "right_rubber_hand"
    gr1_right_ee = "r_hand_pitch"

    model_g1, data_g1 = load_model(urdf_g1)
    model_gr1, data_gr1 = load_model(urdf_gr1, strip_collision=True)

    q_full_gr1 = set_right_arm_q(model_gr1, gr1_right_arm, q_right_gr1)
    gr1_pos, gr1_rot = compute_ee_pose(model_gr1, data_gr1, q_full_gr1, gr1_right_ee)

    R_align, t_align = estimate_base_alignment(
        model_g1,
        data_g1,
        g1_base,
        "left_shoulder_pitch_link",
        "right_shoulder_pitch_link",
        model_gr1,
        data_gr1,
        gr1_base,
        "l_upper_arm_pitch",
        "r_upper_arm_pitch",
    )
    g1_target = transform_gr1_pos_to_g1(gr1_pos, R_align, t_align)

    ik_result = ik_right_arm_position(
        model_g1,
        data_g1,
        g1_right_arm,
        g1_right_ee,
        g1_target,
        q0_full=pin.neutral(model_g1),
    )

    return ik_result.success
    # print("gr1 right ee pos:", gr1_pos)
    # print("gr1 right ee rot:\n", gr1_rot)
    # print("g1 reachable:", ik_result.success, "iters:", ik_result.iters, "err:", ik_result.err_norm)
    # print("estimated R (g1->gr1):\n", R_align)
    # print("estimated t (g1->gr1):", t_align)


if __name__ == "__main__":
    # q_right_gr1 = np.array([-3.2054e-01, -1.9381e-01, 1.4541e-02, -1.5300e+00, -2.7155e-01,
    # -1.4067e-01, 1.1202e-01])
    # print(example(q_right_gr1))
    tensor = torch.load("train_dataset_seq_bias.pt")
    t=0
    for i in range(len(tensor)):
        q_right_gr1 = tensor[i]
        if example(q_right_gr1)==True:
            t+=1
    print(t)
