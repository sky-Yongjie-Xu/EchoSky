import argparse
from typing import Union
from argparse import ArgumentParser
import cv2
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from matplotlib import animation
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.signal import find_peaks
import sys
from threading import Thread, Lock
import click
from typing import Iterable, Tuple
import torchvision


plt_thread_lock = Lock()

# relative paths to weights for various models
weights_path = Path(__file__).parent / 'weights'
model_paths = {
    'plax': weights_path / 'hypertrophy_model.pt',
    'amyloid': weights_path / 'amyloid.pt',
    'as': weights_path / 'as_model.pt'
}


class Model(torch.nn.Module):

    """Model used for prediction of PLAX measurement points.
    Output channels correspond to heatmaps for the endpoints of
    measurements of interest.
    """

    def __init__(self, 
            measurements=['LVPW', 'LVID', 'IVS'], 
        ) -> None:
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=len(measurements) + 1)

    def forward(self, x):
        return torch.sigmoid(self.model(x)['out'])
    

class BoolAction(argparse.Action):

    """Class used by argparse to parse binary arguements.
    Yes, Y, y, True, T, t are all accepted as True. Any other
    arguement is evaluated as False.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        b = values.lower()[0] in ['t', 'y', '1']
        setattr(namespace, self.dest, b)
        print(parser)


def get_clip_dims(paths: Iterable[Union[Path, str]]) -> Tuple[np.ndarray, list]:
    """Gets the dimentions of all the videos in a list of paths.

    Args:
        paths (Iterable[Union[Path, str]]): List of paths to iterrate through

    Returns:
        dims (np.ndarray): array of clip dims (frames, width, height). shape=(n, 3)
        filenames (list): list of filenames. len=n
    """
    
    dims = []
    fnames = []
    for p in paths:
        if isinstance(p, str):
            p = Path(p)
        if '.avi' not in p.name:
            continue
        cap = cv2.VideoCapture(str(p))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dims.append((frame_count, w, h))
        fnames.append(p.name)
    return np.array(dims).T, fnames

def read_clip(path, res=None, max_len=None) -> np.ndarray:
    """Reads a clip and returns it as a numpy array

    Args:
        path ([Path, str]): Path to video to read
        res (Tuple[int], optional): Resolution of video to return. If None, 
            original resolution will be returned otherwise the video will be 
            cropped and downsampled. Defaults to None.
        max_len (int, optional): Max length of video to read. Only the first n 
            frames of longer videos will be returned. Defaults to None.

    Returns:
        np.ndarray: Numpy array of video. shape=(n, h, w, 3)
    """

    cap = cv2.VideoCapture(str(path))
    frames = []
    i = 0
    while True:
        if max_len is not None and i >= max_len:
            break
        i += 1
        ret, frame = cap.read()
        if not ret:
            break
        if res is not None:
            frame = crop_and_scale(frame, res)
        frames.append(frame)
    cap.release()
    return np.array(frames)

def get_systole_diastole(lvid: np.ndarray, kernel=[1, 2, 3, 2, 1], distance: int=25) -> Tuple[np.ndarray]:
    """Finds heart phase from a representative signal. Signal must be maximum at end diastole and
    minimum at end systole.

    Args:
        lvid (np.ndarray): Signal representing heart phase. shape=(n,)
        kernel (list, optional): Smoothing kernel used before finding peaks. Defaults to [1, 2, 3, 2, 1].
        distance (int, optional): Minimum distance between peaks in find_peaks(). Defaults to 25.

    Returns:
        systole_i (np.ndarray): Indices of end systole. shape=(n_sys,)
        diastole_i (np.ndarray): Indices of end diastole. shape=(n_dia,)
    """

    # Smooth input
    kernel = np.array(kernel)
    kernel = kernel / kernel.sum()
    lvid_filt = np.convolve(lvid, kernel, mode='same')

    # Find peaks
    diastole_i, _ = find_peaks(lvid_filt, distance=distance)
    systole_i, _ = find_peaks(-lvid_filt, distance=distance)

    # Ignore first/last index if possible
    if len(systole_i) != 0 and len(diastole_i) != 0:
        start_minmax = np.concatenate([diastole_i, systole_i]).min()
        end_minmax = np.concatenate([diastole_i, systole_i]).max()
        diastole_i = np.delete(diastole_i, np.where((diastole_i == start_minmax) | (diastole_i == end_minmax)))
        systole_i = np.delete(systole_i, np.where((systole_i == start_minmax) | (systole_i == end_minmax)))
    
    return systole_i, diastole_i

def get_lens_np(pts: np.ndarray) -> np.ndarray:
    """Used to get the euclidean distance between consecutive points.

    Args:
        pts (np.ndarray): Input points. shape=(..., n, 2)

    Returns:
        np.ndarray: Distances. shape=(..., n-1)
    """
    return np.sum((pts[..., 1:, :] - pts[..., :-1, :]) ** 2, axis=-1) ** 0.5

def get_points_np(preds: np.ndarray, threshold: float=0.3) -> np.ndarray:
    """Gets the centroid of heatmaps.

    Args:
        preds (np.ndarray): Input heatmaps. shape=(n, h, w, c)
        threshold (float, optional): Value below which input pixels are ignored. Defaults to 0.3.

    Returns:
        np.ndarray: Centroid locations. shape=(n, c, 2)
    """

    preds = np.copy(preds)
    preds[preds < threshold] = 0
    Y, X = np.mgrid[:preds.shape[-3], :preds.shape[-2]]
    np.seterr(divide='ignore', invalid='ignore')
    x_pts = np.sum(X[None, ..., None] * preds, axis=(-3, -2)) / np.sum(preds, axis=(-3, -2))
    y_pts = np.sum(Y[None, ..., None] * preds, axis=(-3, -2)) / np.sum(preds, axis=(-3, -2))
    return np.moveaxis(np.array([x_pts, y_pts]), 0, -1)

def get_angles_np(pts: np.ndarray) -> np.ndarray:
    """Returns the angles between corresponding segments of a polyline.

    Args:
        pts (np.ndarray): Input polyline. shape=(..., n, 2)

    Returns:
        np.ndarray: Angles in degrees. Constrained to [-180, 180]. shape=(..., n-1)
    """

    a_m = np.arctan2(*np.moveaxis(pts[..., 1:, :] - pts[..., :-1, :], -1, 0))
    a = (a_m[..., 1:] - a_m[..., :-1]) * 180 / np.pi
    a[a > 180] -= 360
    a[a < -180] += 360
    return a

def get_pred_measurements(preds: np.ndarray, scale: float=1) -> Tuple[np.ndarray]:
    """Given PLAX heatmap predictions, generate values of interest.

    Args:
        preds (np.ndarray): PLAX model heatmap predictions. shape=(n, h, w, 4)
        scale (int, optional): Image scale [cm/px]. Defaults to 1.

    Returns:
        pred_pts (np.ndarray): Centroids of heatmaps. shape=(n, 4, 2)
        pred_lens (np.ndarray): Measurement lengths. shape=(n, 3)
        sys_i (np.ndarray): Indices of end systole. shape=(n_sys,)
        dia_i (np.ndarray): Indices of end diastole. shape=(n_dia,)
        angles (np.ndarray): Angles between measurements in degrees. shape=(n, 2)
    """

    pred_pts = get_points_np(preds)
    pred_lens = get_lens_np(pred_pts) * scale
    sys_i, dia_i = get_systole_diastole(pred_lens[:, 1])
    angles = get_angles_np(pred_pts)
    return pred_pts, pred_lens, sys_i, dia_i, angles

def overlay_preds(
            a: np.ndarray, 
            background=None, 
            c=np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0]])
        ) -> np.ndarray:
    """Used to visualize PLAX model predictions over echo frames

    Args:
        a (np.ndarray): Predicted heatmaps. shape=(h, w, 4)
        background (np.ndarray, optional): Echo frame to overlay on top of. shape=(h, w, 3) Defaults to None.
        c (np.ndarray, optional): RGB colors corresponding to each channel of the predictions. shape=(4, 3)
            Defaults to np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0]]).

    Returns:
        np.ndarray: RGB image visualization of heatmaps. shape=(h, w, 3)
    """

    if background is None:
        background = np.zeros((a.shape[0], a.shape[1], 3))
    np.seterr(divide='ignore', invalid='ignore')
    color = (a ** 2).dot(c) / np.sum(a, axis=-1)[..., None]
    alpha = (1 - np.prod(1 - a, axis=-1))[..., None]
    alpha = np.nan_to_num(alpha)
    color = np.nan_to_num(color)
    return alpha * color + (1 - alpha) * background

def crop_and_scale(img: np.ndarray, res=(640, 480)) -> np.ndarray:
    """Scales and cropts an numpy array image to specified resolution.
    Image is first cropped to correct aspect ratio and then scaled using
    bicubic interpolation.

    Args:
        img (np.ndarray): Image to be resized. shape=(h, w, 3)
        res (tuple, optional): Resolution to be scaled to. Defaults to (640, 480).

    Returns:
        np.ndarray: Scaled image. shape=(res[1], res[0], 3)
    """

    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]
    
    img = cv2.resize(img, res)

    return img


def save_preds(
            p: Union[str, Path], fn: str, clip: np.ndarray, preds: np.ndarray,
            csv=True, avi=True, plot=True, npy=False, angle_threshold=30
        ) -> None:
    folder_name = fn.replace('.avi', '').replace('.', '_')
    inf_path = p / folder_name
    if not inf_path.exists():
        inf_path.mkdir()

    if npy:
        np.save(inf_path / (folder_name + '.npy'), preds)
    pred_pts, pred_lens, sys_i, dia_i, angles = get_pred_measurements(preds)

    if csv:
        phase = np.array([''] * len(pred_pts), dtype=object)
        phase[sys_i] = 'ES'
        phase[dia_i] = 'ED'
        df = pd.DataFrame({
            'frame': np.arange(len(pred_pts)),
            'X1': pred_pts[:, 0, 0],
            'Y1': pred_pts[:, 0, 1],
            'X2': pred_pts[:, 1, 0],
            'Y2': pred_pts[:, 1, 1],
            'X3': pred_pts[:, 2, 0],
            'Y3': pred_pts[:, 2, 1],
            'X4': pred_pts[:, 3, 0],
            'Y4': pred_pts[:, 3, 1],
            'LVPW': pred_lens[:, 0],
            'LVID': pred_lens[:, 1],
            'IVS': pred_lens[:, 2],
            'predicted_phase': phase,
            'LVPW_LVID_angle': angles[:, 0],
            'LVID_IVS_angle': angles[:, 1],
            'bad_angle': (abs(angles[:, 0]) > angle_threshold) | (abs(angles[:, 1]) > angle_threshold)
        })
        df.to_csv(inf_path / (folder_name + '.csv'), index=False)

    if avi:
        with plt_thread_lock:
            make_animation_cv2(inf_path / (folder_name + '.avi'), clip, preds, pred_pts)

    if plot:
        make_plot(inf_path / (folder_name + '.png'), folder_name, pred_lens, sys_i, dia_i)


def make_animation(
            save_path: Union[Path, str], clip: np.ndarray, preds: np.ndarray,
            pred_pts: np.ndarray, pred_lens: np.ndarray, sys_i, dia_i,
            figsize=(12, 12), units='PX', fps=50
        ) -> None:
    if isinstance(save_path, str):
        save_path = Path(save_path)

    grid = plt.GridSpec(4, 1)
    fig = plt.figure(0, figsize=figsize)
    ax1 = fig.add_subplot(grid[3:, 0])
    ax2 = fig.add_subplot(grid[:3, 0])
    for l, n in zip(pred_lens.T, ['LVPW', 'LVID', 'IVS']):
        ax1.plot(l, label=n)
    l1, = ax1.plot([0, 0, 0], pred_lens[0], 'ro')
    ax1.legend()
    ax1.set_xlabel('Frame')
    ax1.set_ylabel(f'Measurement [{units}]')
    ax1.vlines(sys_i, pred_lens.min(), pred_lens.max(), linestyles='dashed', colors='b', label='Systole')
    ax1.vlines(dia_i, pred_lens.min(), pred_lens.max(), linestyles='dashed', colors='g', label='Diastole')
    im = ax2.imshow(overlay_preds(preds[0], clip[0] / 255))
    l2, = ax2.plot(*pred_pts[0].T, 'C1o-')
    ax2.set_title(save_path.name)

    def animate(i):
        im.set_data(overlay_preds(preds[i], clip[i] / 255))
        l1.set_data([i, i, i], pred_lens[i])
        l2.set_data(*pred_pts[i].T)

    ani = animation.FuncAnimation(fig, animate, frames=len(clip), interval=1000 / fps)
    writer = animation.FFMpegWriter(fps)
    ani.save(save_path, writer)
    plt.close(fig)


def make_plot(
            save_path: Union[Path, str], title: str, pred_lens: np.ndarray,
            sys_i, dia_i, figsize=(8, 6)
        ) -> None:
    plt.figure(1, figsize=figsize)
    plt.clf()
    for l, n in zip(pred_lens.T, ['LVPW', 'LVID', 'IVS']):
        plt.plot(l, label=n)
    plt.plot(sys_i, pred_lens[sys_i], 'r+', label='Systole')
    plt.plot(dia_i, pred_lens[dia_i], 'rx', label='Diastole')
    plt.title(title)
    plt.xlabel('Frame')
    plt.ylabel('Measurement [px]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def make_animation_cv2(
            save_path: Union[Path, str], clip: np.ndarray, preds: np.ndarray, pred_pts: np.ndarray,
            fps=30.0, line_color=(1, 1, 0), point_color=(1, 0.5, 0), linewidth=2, markersize=4
        ) -> None:
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (clip.shape[2], clip.shape[1]))
    for frame, pred, line in zip(clip, preds, pred_pts):
        img = overlay_preds(pred, frame / 255)
        if not np.isnan(line).any():
            line = line.round().astype(int)
            for pt0, pt1 in zip(line[:-1], line[1:]):
                img = cv2.line(img, tuple(pt0), tuple(pt1), line_color, linewidth)
            for pt in line:
                img = cv2.circle(img, tuple(pt), radius=markersize, color=point_color, thickness=-1)
        img = (img * 255).astype(np.uint8)
        out.write(img[:, :, ::-1])
    out.release()


class PlaxHypertrophyInferenceEngine:
    def __init__(self, model_path: Union[Path, str] = model_paths['plax'], device='cuda:0'):
        self.device = device
        self.model = None
        self.model_path = Path(model_path)

    def load_model(self):
        self.model = Model().eval().to(self.device)
        return self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

    def run_on_dir(
                self, in_dir: Union[Path, str], out_dir: Union[Path, str], batch_size=100,
                h=480, w=640, channels_in=3, channels_out=4,
                n_threads=16, verbose=True, save_csv=True, save_avi=True, save_npy=False, save_plot=True
            ):
        in_dir = Path(in_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)

        threads = []
        for fn, clip, preds in self._run_on_clips(list(in_dir.iterdir()), verbose=verbose,
                                                  h=h, w=w, channels_in=channels_in, channels_out=channels_out, batch_size=batch_size):
            if len(threads) >= n_threads:
                threads.pop(0).join()
            t = Thread(target=save_preds, args=(out_dir, fn.name, clip, preds, save_csv, save_avi, save_plot, save_npy))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    def _run_on_clips(self, paths, batch_size=100, h=480, w=640, channels_in=3, channels_out=4, verbose=True):
        (n, w_all, h_all), fnames = get_clip_dims(paths)
        frame_map = pd.DataFrame({
            'frame': np.concatenate([np.arange(ni) for ni in n]),
            'path': np.concatenate([np.full(ni, str(p)) for ni, p in zip(n, paths)]),
        })

        clips = {}
        total = len(frame_map)
        for si in tqdm(range(0, total, batch_size), disable=not verbose):
            batch_map = frame_map.iloc[si:si + batch_size]
            batch_paths = batch_map['path'].unique()

            for k in list(clips.keys()):
                if k not in batch_paths:
                    yield Path(k), *clips.pop(k)

            for p in batch_paths:
                if p not in clips:
                    c = read_clip(p, res=(w, h))
                    clips[p] = (c, np.zeros((len(c), h, w, channels_out), dtype=np.float32))
                mask = batch_map['path'] == p
                batch = clips[p][0][batch_map.loc[mask, 'frame']]
                preds = self.run_model_np(batch)
                clips[p][1][batch_map.loc[mask, 'frame']] = preds

        for k, v in clips.items():
            yield Path(k), v[0], v[1]

    def run_model_np(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load_model()
        x = x.astype(np.float32) / 255.0
        x = x.transpose(0, 3, 1, 2)
        tensor = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        out = out.cpu().numpy().transpose(0, 2, 3, 1)
        return out


# ======================== Click 命令行封装 ========================
@click.command("plax_inference")
@click.option("--in_dir", type=click.Path(exists=True), required=True)
@click.option("--out_dir", type=click.Path(), required=True)
@click.option("--device", default="cuda:0", help="cuda:0 or cpu")
@click.option("--batch-size", default=100)
@click.option("--n-threads", default=16)
@click.option("--verbose/--quiet", default=True)
@click.option("--save-csv/--skip-csv", default=True)
@click.option("--save-avi/--skip-avi", default=True)
@click.option("--save-plot/--skip-plot", default=True)
@click.option("--save-npy/--skip-npy", default=False)
def run(
        in_dir, out_dir,
        device="cuda:0",
        batch_size=100,
        n_threads=16,
        verbose=True,
        save_csv=True,
        save_avi=True,
        save_plot=True,
        save_npy=False
):
    engine = PlaxHypertrophyInferenceEngine(device=device)
    engine.run_on_dir(
        in_dir=in_dir,
        out_dir=out_dir,
        batch_size=batch_size,
        n_threads=n_threads,
        verbose=verbose,
        save_csv=save_csv,
        save_avi=save_avi,
        save_plot=save_plot,
        save_npy=save_npy
    )


# ======================== 注册到你的引擎 ========================
def register():
    return {
        "name": "plax_inference",
        "entry": run,
        "description": "PLAX 心脏超声自动测量（LVPW、LVID、IVS）"
    }


if __name__ == "__main__":
    run()