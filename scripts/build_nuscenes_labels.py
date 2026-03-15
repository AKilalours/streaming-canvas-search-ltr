from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import Polygon, box as sbox
from nuscenes.nuscenes import NuScenes

# BEV config (toy but consistent)
BEV = 64
X_MIN, X_MAX = -32.0, 32.0   # meters (forward)
Y_MIN, Y_MAX = -32.0, 32.0   # meters (left)
DX = (X_MAX - X_MIN) / BEV
DY = (Y_MAX - Y_MIN) / BEV

KEEP_PREFIX = (
    "vehicle.", "human.", "animal.", "movable_object.", "static_object."
)

def ego_T_inv(pose):
    """Transform global -> ego using ego_pose."""
    t = np.array(pose["translation"], dtype=np.float32)
    R = Quaternion(pose["rotation"]).rotation_matrix.astype(np.float32)
    # global point p -> ego: R^T (p - t)
    return t, R

def poly_to_grid(poly: Polygon) -> np.ndarray:
    occ = np.zeros((BEV, BEV), dtype=np.float32)
    if poly.is_empty:
        return occ

    # compute candidate cell range
    minx, miny, maxx, maxy = poly.bounds
    ix0 = int(np.floor((minx - X_MIN) / DX))
    ix1 = int(np.ceil((maxx - X_MIN) / DX))
    iy0 = int(np.floor((miny - Y_MIN) / DY))
    iy1 = int(np.ceil((maxy - Y_MIN) / DY))
    ix0 = max(0, min(BEV - 1, ix0)); ix1 = max(0, min(BEV, ix1))
    iy0 = max(0, min(BEV - 1, iy0)); iy1 = max(0, min(BEV, iy1))

    for ix in range(ix0, ix1):
        x0 = X_MIN + ix * DX
        x1 = x0 + DX
        for iy in range(iy0, iy1):
            y0 = Y_MIN + iy * DY
            y1 = y0 + DY
            cell = sbox(x0, y0, x1, y1)
            if poly.intersects(cell):
                occ[iy, ix] = 1.0  # note: row=y, col=x
    return occ

def build_occ(nusc: NuScenes, sample) -> np.ndarray:
    # use ego pose from LIDAR_TOP as reference frame
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    t0, R0 = ego_T_inv(pose)

    union_poly = None

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        if not ann["category_name"].startswith(KEEP_PREFIX):
            continue

        box = nusc.get_box(ann_token)  # global
        # global -> ego
        corners = box.corners().T.astype(np.float32)  # (8,3)
        corners_ego = (corners - t0[None, :]) @ R0  # (8,3) using R^T implicit via @R0? (R0 is rotation_matrix; we need R^T)
        # correction: ego = (p - t) @ R^T
        corners_ego = (corners - t0[None, :]) @ (R0.T)

        # take 4 lowest-z corners for ground footprint
        z = corners_ego[:, 2]
        idx = np.argsort(z)[:4]
        pts = corners_ego[idx][:, :2]
        poly = Polygon(pts).convex_hull
        if union_poly is None:
            union_poly = poly
        else:
            union_poly = union_poly.union(poly)

    if union_poly is None:
        occ = np.zeros((BEV, BEV), dtype=np.float32)
    else:
        occ = poly_to_grid(union_poly)

    return occ[None, :, :]  # (1,64,64)

def build_traj(nusc: NuScenes, sample, horizon: int = 12) -> np.ndarray:
    # current ego pose (LIDAR_TOP)
    sd0 = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pose0 = nusc.get("ego_pose", sd0["ego_pose_token"])
    t0 = np.array(pose0["translation"], dtype=np.float32)
    R0 = Quaternion(pose0["rotation"]).rotation_matrix.astype(np.float32)

    traj = np.zeros((horizon, 2), dtype=np.float32)

    tok = sample["token"]
    cur = sample
    for k in range(horizon):
        nxt = cur["next"]
        if nxt == "":
            # pad with last available
            traj[k:] = traj[k-1] if k > 0 else 0.0
            break
        cur = nusc.get("sample", nxt)
        sd = nusc.get("sample_data", cur["data"]["LIDAR_TOP"])
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        t = np.array(pose["translation"], dtype=np.float32)

        # global delta to ego0 frame: (t - t0) in ego0 coords
        d = t - t0
        ego_xy = (R0.T @ d)[:2]  # x forward, y left
        traj[k] = ego_xy

    return traj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--version", type=str, default="v1.0-mini")
    ap.add_argument("--manifest", type=str, default="artifacts/nuscenes_mini_manifest.jsonl")
    ap.add_argument("--out_dir", type=str, default="artifacts/nuscenes_labels")
    ap.add_argument("--horizon", type=int, default=12)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    for i, r in enumerate(rows):
        sample = nusc.get("sample", r["sample_token"])
        occ = build_occ(nusc, sample)
        traj = build_traj(nusc, sample, horizon=args.horizon)
        np.savez_compressed(out_dir / f'{r["sample_token"]}.npz', occ=occ, traj=traj)
        if (i + 1) % 25 == 0:
            print(f"labels: {i+1}/{len(rows)}")

    print("WROTE:", out_dir, "files=", len(rows))

if __name__ == "__main__":
    main()
