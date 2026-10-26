"""多视角融合：同一个姿势从不同角度拍（或 AI 生成）的几张图，各自识别后合成一份更可信的姿势。

依据：关节的局部旋转（相对父关节）与相机无关，不需要知道辅助图是从哪个角度拍的。
每个关节按"这根骨段在该视角里有多正对镜头"加权平均：正对镜头时深度最难判断（透视缩短），
同一根骨段在另一张图里与画面平行就更可信。主图决定整体朝向、位置与构图，体型取主图。
辅助图常是 AI 生成的，可能自作主张改动作，权重再乘 aux_weight。

输入输出都是会话里的 human 字典（键同 image_to_pose 的 npz：joint_global_rots、joint_coords、rest_joint_coords、
rest_joint_rots、focal_length、cam_t 等）。
"""

from __future__ import annotations

import json
from functools import cache

import numpy as np
from scipy.spatial.transform import Rotation as R

from .retarget.ffxiv import DATA_DIR

FLIP = np.array([1.0, -1.0, -1.0])   # SAM 3D Body 的关节坐标在相机系（y、z 取反），关节旋转没有翻转
MIN_WEIGHT = 0.05


@cache
def _skeleton() -> tuple[list[str], np.ndarray, np.ndarray]:
    """MHR 关节名、父关节、每个关节用来量"朝向镜头程度"的子关节（第一个子关节；末端用父关节那段）。"""
    rest = json.loads((DATA_DIR / "mhr_rest.json").read_text(encoding="utf-8"))
    names, parents = rest["names"], np.array(rest["parents"])
    child = np.full(len(names), -1)
    for j in range(len(names) - 1, -1, -1):   # 倒序遍历，留下的是编号最小的子关节
        if parents[j] >= 0:
            child[parents[j]] = j
    return names, parents, child


def _local_rotations(human: dict) -> R:
    G = R.from_matrix(human["joint_global_rots"])
    _, parents, _ = _skeleton()
    loc = [G[j] if p < 0 else G[p].inv() * G[j] for j, p in enumerate(parents)]
    return R.concatenate(loc)


def foreshortening(human: dict) -> np.ndarray:
    """每个关节所驱动骨段与相机视线夹角的余弦绝对值（1 = 正对镜头，0 = 与画面平行）。末端关节沿用父关节的值。"""
    _, parents, child = _skeleton()
    P = np.asarray(human["joint_coords"])
    out = np.zeros(len(parents))
    for j in range(len(parents)):
        c = child[j]
        if c < 0:
            continue
        d = P[c] - P[j]
        n = np.linalg.norm(d)
        out[j] = abs(d[2]) / n if n > 1e-6 else 0.0
    for j in range(len(parents)):   # 父在前，末端取父关节的值
        if child[j] < 0 and parents[j] >= 0:
            out[j] = out[parents[j]]
    return out


def weights(human: dict, bias: float = 1.0) -> np.ndarray:
    return bias * np.maximum(1.0 - foreshortening(human), MIN_WEIGHT) ** 2


def _bone_dirs(human: dict, to_main: R | None = None) -> np.ndarray:
    """每个关节指向第一个子关节的单位方向（主图相机系，已翻转 y、z）；末端为 0。"""
    _, _, child = _skeleton()
    P = np.asarray(human["joint_coords"]) * FLIP
    d = np.zeros_like(P)
    has = child >= 0
    d[has] = P[child[has]] - P[has]
    n = np.linalg.norm(d, axis=1, keepdims=True)
    d = np.divide(d, n, out=np.zeros_like(d), where=n > 1e-6)
    return to_main.apply(d) if to_main is not None else d


MAX_TILT = np.radians(85)
# 统计"采纳了多少"和"动作差异"用的主要骨段
LIMBS = ("c_spine1", "l_upleg", "l_lowleg", "r_upleg", "r_lowleg", "l_uparm", "l_lowarm", "r_uparm", "r_lowarm")
SAME_VIEW_DEG = 30     # 与主图相差小于这个角度：几乎同一角度，补不了深度
BACK_VIEW_DEG = 150
CONFLICT_DEG = 25      # 平均画面方向差异超过这个：辅助图的动作和主图对不上


def facing_yaw(human: dict) -> float:
    """人物朝向相对相机的水平角（度）：0 = 正对镜头，+90 = 面朝画面右方（镜头看到的是人物右侧）。"""
    names, _, _ = _skeleton()
    P = np.asarray(human["joint_coords"]) * FLIP   # y 朝上，镜头看向 -z
    left = P[names.index("l_upleg")] - P[names.index("r_upleg")]
    up = P[names.index("c_neck")] - P[names.index("root")]
    f = np.cross(left, up)
    return float(np.degrees(np.arctan2(f[0], f[2])))


def describe(main: dict, aux: list[dict], aux_weight: float = 0.8) -> list[dict]:
    """每张辅助图相对主图的情况，给界面显示和提示用：
    yaw：辅助图机位绕人物转了多少度（正 = 转到人物右侧）；view：same / left / right / back；
    share：主要骨段平均有多少权重来自这张图；conflict：主要骨段在主图画面里的方向与辅助图平均差几度（AI 改了动作时会很大），
    worst：差得最多的骨段。"""
    names, _, _ = _skeleton()
    main_rot = R.from_matrix(main["joint_global_rots"])
    dm = _bone_dirs(main)
    w_main = weights(main)
    yaw0 = facing_yaw(main)
    limbs = [names.index(n) for n in LIMBS]
    out = []
    for a in aux:
        yaw = (facing_yaw(a) - yaw0 + 180) % 360 - 180
        view = ("same" if abs(yaw) < SAME_VIEW_DEG else "back" if abs(yaw) > BACK_VIEW_DEG
                else "right" if yaw > 0 else "left")
        w_aux = weights(a, aux_weight)
        share = float(np.mean(w_aux[limbs] / (w_aux[limbs] + w_main[limbs])))
        da = _bone_dirs(a, main_rot[1] * R.from_matrix(a["joint_global_rots"])[1].inv())
        conflicts = []
        for j in limbs:
            u = np.array([dm[j, 0], dm[j, 1], 0.0])
            n = np.linalg.norm(u)
            if n < 0.3:   # 主图里几乎正对镜头，画面方向本来就不可靠
                continue
            normal = np.cross(u / n, [0.0, 0.0, 1.0])
            conflicts.append((float(np.degrees(np.arcsin(min(1.0, abs(da[j] @ normal))))), names[j]))
        mean = float(np.mean([c for c, _ in conflicts])) if conflicts else 0.0
        worst = max(conflicts) if conflicts else (0.0, "")
        out.append({"yaw": round(yaw, 1), "view": view, "share": round(share, 3), "conflict": round(mean, 1),
                    "worst": worst[1], "worstDeg": round(worst[0], 1)})
    return out


def _depth_only(d: np.ndarray, w: np.ndarray) -> np.ndarray | None:
    """d：各视角的骨段方向（主图相机系，主图在第 0 行），w：权重。
    保留主图的画面方向 u，在 u 与视线轴构成的平面里按权重平均倾角。骨段几乎正对镜头（画面方向不可靠）时返回 None。"""
    u = np.array([d[0, 0], d[0, 1], 0.0])
    n = np.linalg.norm(u)
    if n < 0.05:
        return None
    u /= n
    tilt = np.clip(np.arctan2(d[:, 2], d @ u), -MAX_TILT, MAX_TILT)   # 辅助图画面方向反了（与主图矛盾）时夹到边界
    t = float((w * tilt).sum() / w.sum())
    return np.cos(t) * u + np.array([0.0, 0.0, np.sin(t)])


def fuse(main: dict, aux: list[dict], aux_weight: float = 0.8, report: dict | None = None,
         keep_2d: bool = True) -> dict:
    """返回融合后的 human 字典（复制自主图，替换 joint_global_rots、joint_coords 与网格顶点）。
    keep_2d：每根骨段在主图里的画面方向保持不变，辅助图只决定它往镜头里 / 外倾斜多少。
    单张图唯一猜不准的就是这个深度，辅助图（尤其 AI 生成、可能改了动作的）也就不会把主图里看得见的姿势改掉。
    report 给出时写入每个关节各视角的权重占比（主图在第 0 列）。"""
    names, parents, child = _skeleton()
    locs = [_local_rotations(main)] + [_local_rotations(a) for a in aux]
    W = np.stack([weights(main)] + [weights(a, aux_weight) for a in aux])   # (视角, 关节)
    # 整体朝向只用主图：MHR 的 body_world 恒为单位旋转，相机系下的朝向在它的子关节 root（骨盆）上，
    # 所以 root 的"局部旋转"也随相机变，不能跨视角平均
    W[:, :2] = 0
    W[0, :2] = 1

    fused = []
    for j in range(len(parents)):
        quats = np.stack([loc[j].as_quat() for loc in locs])
        quats *= np.sign(quats @ quats[0])[:, None]   # 同半球，避免 q 与 -q 互相抵消
        q = (W[:, j:j + 1] * quats).sum(0)
        fused.append(q / np.linalg.norm(q))
    L = R.from_quat(np.array(fused))

    # 前向运动学：旋转逐级相乘；骨段偏移取主图实际姿势里的（父关节坐标系下），
    # MHR 摆姿势时骨段会伸缩，用零姿态的骨长会差几厘米
    main_pos = np.asarray(main["joint_coords"]) * FLIP
    main_rot = R.from_matrix(main["joint_global_rots"])
    if keep_2d:
        # 辅助图的相机系转到主图：按骨盆朝向对齐
        dirs = [_bone_dirs(main)] + [_bone_dirs(a, main_rot[1] * R.from_matrix(a["joint_global_rots"])[1].inv())
                                     for a in aux]
    G: list[R] = [None] * len(parents)  # type: ignore[list-item]
    P = np.zeros((len(parents), 3))
    for j, p in enumerate(parents):
        if p < 0:
            G[j], P[j] = L[j], main_pos[j]
            continue
        G[j] = G[p] * L[j]
        P[j] = P[p] + G[p].apply(main_rot[p].inv().apply(main_pos[j] - main_pos[p]))
        if keep_2d and child[j] >= 0 and p > 0:
            target = _depth_only(np.stack([d[j] for d in dirs]), W[:, j])
            if target is not None:
                c = child[j]
                cur = G[j].apply(main_rot[j].inv().apply(main_pos[c] - main_pos[j]))
                if np.linalg.norm(cur) > 1e-6:
                    G[j] = R.align_vectors([target], [cur])[0] * G[j]

    out = dict(main)
    G = R.concatenate(G)
    out["joint_global_rots"] = G.as_matrix()
    out["joint_coords"] = P * FLIP
    if main.get("vertices") is not None:
        # 网格顶点跟随最近的关节刚性移动（重定向的站姿重力校正要用网格估算重心）
        from scipy.spatial import cKDTree
        V = np.asarray(main["vertices"]) * FLIP
        _, idx = cKDTree(main_pos).query(V)
        delta = G[idx] * main_rot[idx].inv()
        out["vertices"] = (P[idx] + delta.apply(V - main_pos[idx])) * FLIP
    if report is not None:
        share = W / W.sum(0, keepdims=True)
        report.update({n: share[:, j].round(3).tolist() for j, n in enumerate(names)})
    return out
