"""
SMPL-X mesh visualization helpers for RAG-Gesture.

Standalone module — only imports torch/numpy at module load. Renderer deps
(pyrender, trimesh, cv2) are imported lazily inside the render functions so
that other tools in this repo don't pull them in.

Adapted from miburi/.../dataloaders/utils/visualize.py — see that file for the
original (and a wider set of helpers including joint-debug and body-part
filtering).
"""
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np
import torch


LOG = logging.getLogger(__name__)


def create_checkerboard_floor(
    y: float = 0.0,
    length: float = 10.0,
    tile_size: float = 1.0,
    color_a: tuple = (170, 170, 170, 255),
    color_b: tuple = (120, 120, 120, 255),
):
    import trimesh

    half = length * 0.5
    nx = max(1, int(length / tile_size))
    nz = max(1, int(length / tile_size))

    vertices = []
    faces = []
    face_colors = []
    idx = 0
    for ix in range(nx):
        for iz in range(nz):
            x0 = -half + ix * tile_size
            x1 = x0 + tile_size
            z0 = -half + iz * tile_size
            z1 = z0 + tile_size
            vertices.extend([
                [x0, y, z0],
                [x1, y, z0],
                [x1, y, z1],
                [x0, y, z1],
            ])
            faces.extend([
                [idx + 0, idx + 2, idx + 1],
                [idx + 0, idx + 3, idx + 2],
            ])
            c = color_a if ((ix + iz) % 2 == 0) else color_b
            face_colors.extend([c, c])
            idx += 4

    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        face_colors=np.asarray(face_colors, dtype=np.uint8),
        process=False,
    )


def mux_audio_into_video(
    video_path: str,
    audio_path: str,
    output_path: Optional[str] = None,
) -> str:
    """ffmpeg-based audio mux. Overwrites video_path in place on success."""
    if not audio_path or not os.path.exists(audio_path):
        LOG.warning("Audio file missing for muxing: %s", audio_path)
        return video_path

    if output_path is None:
        output_path = f"{os.path.splitext(video_path)[0]}_audio_tmp.mp4"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True)
        os.replace(output_path, video_path)
    except FileNotFoundError:
        LOG.warning("ffmpeg not found, keeping silent video: %s", video_path)
    except subprocess.CalledProcessError as exc:
        LOG.warning("ffmpeg mux failed for %s: %s", video_path, exc)
        if os.path.exists(output_path):
            os.remove(output_path)
    return video_path


def ensure_vscode_compatible_video(
    video_path: str,
    output_path: Optional[str] = None,
) -> str:
    """Re-encode to H.264/yuv420p for broad editor playback support."""
    if output_path is None:
        output_path = f"{os.path.splitext(video_path)[0]}_vscode_tmp.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True)
        os.replace(output_path, video_path)
    except FileNotFoundError:
        LOG.warning("ffmpeg not found, keeping original video codec: %s", video_path)
    except subprocess.CalledProcessError as exc:
        LOG.warning("ffmpeg re-encode failed for %s: %s", video_path, exc)
        if os.path.exists(output_path):
            os.remove(output_path)
    return video_path


def stitch_videos_hstack(video_paths, output_path: str) -> str:
    """Horizontally stack videos using ffmpeg."""
    if len(video_paths) < 2:
        raise ValueError("Need at least 2 videos to hstack.")
    for p in video_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input video for hstack: {p}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for p in video_paths:
        cmd.extend(["-i", p])
    inputs = "".join(f"[{i}:v]" for i in range(len(video_paths)))
    filter_complex = f"{inputs}hstack=inputs={len(video_paths)}[v]"
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        output_path,
    ])
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        LOG.warning("ffmpeg not found; cannot stitch videos: %s", output_path)
    except subprocess.CalledProcessError as exc:
        LOG.warning("ffmpeg hstack failed for %s: %s", output_path, exc)
    return output_path


def _prepare_betas(betas: Optional[np.ndarray], nframes: int, device: torch.device):
    if betas is None:
        return torch.zeros((nframes, 300), device=device, dtype=torch.float32)
    betas_np = np.asarray(betas)
    if betas_np.ndim == 1:
        betas_t = torch.tensor(betas_np, device=device, dtype=torch.float32).unsqueeze(0)
        return betas_t.repeat(nframes, 1)
    betas_t = torch.tensor(betas_np, device=device, dtype=torch.float32)
    if betas_t.shape[0] == 1:
        return betas_t.repeat(nframes, 1)
    if betas_t.shape[0] != nframes:
        return betas_t[:1].repeat(nframes, 1)
    return betas_t


def _smplx_vertices_from_params(
    smplx_model,
    poses: np.ndarray,
    transl: np.ndarray,
    expressions: Optional[np.ndarray],
    betas: Optional[np.ndarray],
    batch_size: int = 256,
) -> np.ndarray:
    device = next(smplx_model.parameters()).device
    nframes = poses.shape[0]
    expr_dim = 100 if expressions is None else int(expressions.shape[1])
    verts_all = []

    for start in range(0, nframes, batch_size):
        end = min(start + batch_size, nframes)
        p = torch.tensor(poses[start:end], device=device, dtype=torch.float32)
        t = torch.tensor(transl[start:end], device=device, dtype=torch.float32)
        if expressions is None:
            e = torch.zeros((end - start, expr_dim), device=device, dtype=torch.float32)
        else:
            e = torch.tensor(expressions[start:end], device=device, dtype=torch.float32)
        b = _prepare_betas(betas, nframes, device)[start:end]

        with torch.no_grad():
            out = smplx_model(
                betas=b,
                transl=t,
                expression=e,
                jaw_pose=p[:, 66:69],
                global_orient=p[:, :3],
                body_pose=p[:, 3:21 * 3 + 3],
                left_hand_pose=p[:, 25 * 3:40 * 3],
                right_hand_pose=p[:, 40 * 3:55 * 3],
                leye_pose=p[:, 69:72],
                reye_pose=p[:, 72:75],
                return_verts=True,
            )
        verts_all.append(out.vertices.detach().cpu().numpy())

    return np.concatenate(verts_all, axis=0)


def _active_frame_mask(poses: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Frames where the pose vector has any non-zero entry.

    Retrieved clips are zero-padded outside the inserted snippet; those frames
    decode to the SMPL-X v_template at world origin and would skew floor/
    camera framing if mixed with the real motion.
    """
    poses = np.asarray(poses)
    flat = poses.reshape(poses.shape[0], -1)
    return np.any(np.abs(flat) > tol, axis=1)


def smplx_min_vertex_y(
    smplx_model,
    poses: np.ndarray,
    transl: np.ndarray,
    expressions: Optional[np.ndarray],
    betas: Optional[np.ndarray],
    active_only: bool = True,
) -> float:
    """Return the lowest vertex Y after decoding the given SMPL-X params.

    With ``active_only=True``, frames where the pose is all-zero are
    excluded — useful for retrieval clips whose padding would otherwise
    return the v_template floor instead of the inserted snippet's floor.
    """
    verts = _smplx_vertices_from_params(
        smplx_model=smplx_model,
        poses=np.asarray(poses),
        transl=np.asarray(transl),
        expressions=expressions,
        betas=betas,
    )
    if active_only:
        mask = _active_frame_mask(poses)
        if mask.any():
            verts = verts[mask]
    return float(verts[..., 1].min())


def smplx_active_anchor(
    smplx_model,
    poses: np.ndarray,
    transl: np.ndarray,
    expressions: Optional[np.ndarray],
    betas: Optional[np.ndarray],
    active_only: bool = True,
) -> np.ndarray:
    """Return ``[mean_x, min_y, mean_z]`` of (active) decoded SMPL-X vertices.

    Use this as a single 3D "anchor" for the character: subtract a retrieval
    clip's anchor from a pred clip's anchor and add the delta to the
    retrieval's translation — that puts the retrieval character at the same
    floor level and horizontal position as pred, so both panels auto-frame
    to the same screen region.
    """
    verts = _smplx_vertices_from_params(
        smplx_model=smplx_model,
        poses=np.asarray(poses),
        transl=np.asarray(transl),
        expressions=expressions,
        betas=betas,
    )
    if active_only:
        mask = _active_frame_mask(poses)
        if mask.any():
            verts = verts[mask]
    return np.array([
        float(verts[..., 0].mean()),
        float(verts[..., 1].min()),
        float(verts[..., 2].mean()),
    ], dtype=np.float32)


def compute_auto_framing(
    vertices: np.ndarray,
    active_mask: Optional[np.ndarray] = None,
    width: int = 640,
    height: int = 960,
    cam_y_offset: float = 0.4,
) -> tuple:
    """Compute (camera_pose, floor_y) from a vertex sequence.

    Used as the per-panel auto-framing inside ``render_smplx_debug_video``
    and exposed so callers (e.g. side-by-side renders) can compute framing
    from a *reference* panel once and pass it to multiple panels so all
    characters land in the same screen region.
    """
    if active_mask is not None and active_mask.any():
        framing_verts = vertices[active_mask]
    else:
        framing_verts = vertices

    floor_y = float(framing_verts[..., 1].min()) - 0.02
    char_top_y = float(framing_verts[..., 1].max())
    char_x = float(framing_verts[..., 0].mean())
    char_z = float(framing_verts[..., 2].mean())
    char_mid_y = 0.5 * (floor_y + char_top_y)

    cam_pitch = np.deg2rad(-8.0)
    c = float(np.cos(cam_pitch))
    s = float(np.sin(cam_pitch))
    camera_pose = np.array([
        [1.0, 0.0, 0.0, char_x],
        [0.0, c, -s, char_mid_y + cam_y_offset],
        [0.0, s, c, char_z + 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return camera_pose, floor_y


def render_smplx_debug_video(
    smplx_model,
    poses: np.ndarray,
    transl: np.ndarray,
    expressions: Optional[np.ndarray],
    betas: Optional[np.ndarray],
    output_path: str,
    fps: int,
    width: int = 640,
    height: int = 960,
    audio_path: Optional[str] = None,
    mesh_color: tuple = (36, 73, 156, 255),
    camera_pose: Optional[np.ndarray] = None,
    floor_y: Optional[float] = None,
) -> str:
    """Render an SMPL-X mesh sequence to an mp4 with a checkerboard floor.

    By default, floor and camera framing are derived from *active* (non-zero
    pose) frames only, so zero-padded retrieval clips and large absolute-Y
    translations (as in BEAT2) don't push the floor off-frustum. Pass
    ``camera_pose`` (4x4 ndarray) and/or ``floor_y`` (float) to override the
    auto-framing -- useful when stitching panels where you want every panel
    to share the same camera/floor (so characters land in the same screen
    region across panels).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import cv2
    import pyrender
    import trimesh

    vertices = _smplx_vertices_from_params(
        smplx_model=smplx_model,
        poses=poses,
        transl=transl,
        expressions=expressions,
        betas=betas,
    )
    faces = smplx_model.faces

    active_mask = _active_frame_mask(poses)

    scene = pyrender.Scene(
        bg_color=np.array([0.75, 0.75, 0.75, 1.0]),
        ambient_light=np.array([0.35, 0.35, 0.35]),
    )

    auto_camera_pose, auto_floor_y = compute_auto_framing(
        vertices=vertices,
        active_mask=active_mask,
        width=width,
        height=height,
    )
    if camera_pose is None:
        camera_pose = auto_camera_pose
    else:
        camera_pose = np.asarray(camera_pose, dtype=np.float32)
    if floor_y is None:
        floor_y = auto_floor_y

    floor_mesh = create_checkerboard_floor(y=floor_y, length=12.0, tile_size=1.0)
    scene.add(pyrender.Mesh.from_trimesh(floor_mesh, smooth=False))

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=float(width) / float(height))
    scene.add(camera, pose=camera_pose)

    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
    scene.add(key_light, pose=camera_pose)
    fill_pose = camera_pose.copy()
    fill_pose[0, 3] = 1.5
    fill_pose[1, 3] = 2.0
    scene.add(fill_light, pose=fill_pose)

    renderer = pyrender.OffscreenRenderer(width, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        for fidx in range(vertices.shape[0]):
            # Keep frame count intact but skip drawing the mesh on inactive
            # (zero-pose) frames — empty floor/background is rendered instead.
            mesh_node = None
            if bool(active_mask[fidx]):
                mesh = trimesh.Trimesh(vertices=vertices[fidx], faces=faces, process=False)
                mesh.visual.vertex_colors = np.tile(
                    np.asarray(mesh_color, dtype=np.uint8),
                    (mesh.vertices.shape[0], 1),
                )
                mesh_node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
            color, _ = renderer.render(scene)
            writer.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            if mesh_node is not None:
                scene.remove_node(mesh_node)
    finally:
        writer.release()
        renderer.delete()

    if audio_path:
        return mux_audio_into_video(output_path, audio_path)
    return ensure_vscode_compatible_video(output_path)


def render_gt_pred_side_by_side(
    smplx_model,
    gt_poses: np.ndarray,
    gt_transl: np.ndarray,
    gt_expressions: Optional[np.ndarray],
    pred_poses: np.ndarray,
    pred_transl: np.ndarray,
    pred_expressions: Optional[np.ndarray],
    betas: Optional[np.ndarray],
    output_path: str,
    fps: int,
    panel_width: int = 640,
    panel_height: int = 960,
    audio_path: Optional[str] = None,
    gt_color: tuple = (180, 54, 54, 255),
    pred_color: tuple = (36, 73, 156, 255),
) -> str:
    """Render GT (red, left) and Pred (blue, right) into one stitched mp4.

    Both panels share the *same* camera + floor (computed from GT) so they
    appear at the same screen location.
    """
    gt_verts = _smplx_vertices_from_params(
        smplx_model=smplx_model,
        poses=gt_poses, transl=gt_transl,
        expressions=gt_expressions, betas=betas,
    )
    shared_camera, shared_floor = compute_auto_framing(
        vertices=gt_verts,
        active_mask=_active_frame_mask(gt_poses),
        width=panel_width,
        height=panel_height,
    )
    with tempfile.TemporaryDirectory(prefix="ragvis_sbs_") as tmpdir:
        gt_path = os.path.join(tmpdir, "gt.mp4")
        pred_path = os.path.join(tmpdir, "pred.mp4")
        stitched_path = os.path.join(tmpdir, "stitched.mp4")

        render_smplx_debug_video(
            smplx_model=smplx_model,
            poses=gt_poses, transl=gt_transl, expressions=gt_expressions,
            betas=betas, output_path=gt_path, fps=fps,
            width=panel_width, height=panel_height, mesh_color=gt_color,
            camera_pose=shared_camera, floor_y=shared_floor,
        )
        render_smplx_debug_video(
            smplx_model=smplx_model,
            poses=pred_poses, transl=pred_transl, expressions=pred_expressions,
            betas=betas, output_path=pred_path, fps=fps,
            width=panel_width, height=panel_height, mesh_color=pred_color,
            camera_pose=shared_camera, floor_y=shared_floor,
        )
        stitch_videos_hstack([gt_path, pred_path], stitched_path)
        if not os.path.exists(stitched_path):
            raise RuntimeError(f"hstack failed; no output at {stitched_path}")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        shutil.move(stitched_path, output_path)

        if audio_path and os.path.exists(audio_path):
            mux_audio_into_video(output_path, audio_path)
    return output_path


def render_pred_vs_retrieval_side_by_side(
    smplx_model,
    pred_poses: np.ndarray,
    pred_transl: np.ndarray,
    pred_expressions: Optional[np.ndarray],
    retr_poses: np.ndarray,
    retr_transl: np.ndarray,
    retr_expressions: Optional[np.ndarray],
    betas: Optional[np.ndarray],
    output_path: str,
    fps: int,
    panel_width: int = 640,
    panel_height: int = 960,
    audio_path: Optional[str] = None,
    pred_color: tuple = (36, 73, 156, 255),
    retr_color: tuple = (54, 156, 73, 255),
) -> str:
    """Render Pred (blue, left) and Retrieval (green, right) into one stitched mp4.

    Both panels share the *same* camera + floor (computed from pred) so the
    retrieval character appears at the same screen location as pred -- assumes
    the caller has already aligned ``retr_transl`` to pred's anchor (via
    ``smplx_active_anchor``). The retrieval panel preserves the full temporal
    length; ``render_smplx_debug_video`` automatically hides the mesh on
    zero-pose frames so the retrieved snippet appears only where it was
    actually inserted.
    """
    pred_verts = _smplx_vertices_from_params(
        smplx_model=smplx_model,
        poses=pred_poses, transl=pred_transl,
        expressions=pred_expressions, betas=betas,
    )
    shared_camera, shared_floor = compute_auto_framing(
        vertices=pred_verts,
        active_mask=_active_frame_mask(pred_poses),
        width=panel_width,
        height=panel_height,
    )
    with tempfile.TemporaryDirectory(prefix="ragvis_pvr_") as tmpdir:
        pred_path = os.path.join(tmpdir, "pred.mp4")
        retr_path = os.path.join(tmpdir, "retr.mp4")
        stitched_path = os.path.join(tmpdir, "stitched.mp4")

        render_smplx_debug_video(
            smplx_model=smplx_model,
            poses=pred_poses, transl=pred_transl, expressions=pred_expressions,
            betas=betas, output_path=pred_path, fps=fps,
            width=panel_width, height=panel_height, mesh_color=pred_color,
            camera_pose=shared_camera, floor_y=shared_floor,
        )
        render_smplx_debug_video(
            smplx_model=smplx_model,
            poses=retr_poses, transl=retr_transl, expressions=retr_expressions,
            betas=betas, output_path=retr_path, fps=fps,
            width=panel_width, height=panel_height, mesh_color=retr_color,
            camera_pose=shared_camera, floor_y=shared_floor,
        )
        stitch_videos_hstack([pred_path, retr_path], stitched_path)
        if not os.path.exists(stitched_path):
            raise RuntimeError(f"hstack failed; no output at {stitched_path}")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        shutil.move(stitched_path, output_path)

        if audio_path and os.path.exists(audio_path):
            mux_audio_into_video(output_path, audio_path)
    return output_path
