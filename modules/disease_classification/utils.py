############################################################################ liver disease prediction ############################################################################
import pandas as pd
import numpy as np
import os
from pathlib import Path

from typing import Tuple, Union, List
from numpy.typing import ArrayLike

from tqdm import tqdm
import math

from sklearn import metrics
from sklearn.metrics import roc_curve

import cv2
import torch
# import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from scipy.spatial.transform import Rotation as R


def sensivity_specifity_cutoff(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]

def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

def crop_and_scale(
    img: ArrayLike, res: Tuple[int], interpolation=cv2.INTER_CUBIC, zoom: float = 0.0
) -> ArrayLike:
    """Takes an image, a resolution, and a zoom factor as input, returns the
    zoomed/cropped image."""
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    # Crop to correct aspect ratio
    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]

    # Apply zoom
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]

    # Resize image
    img = cv2.resize(img, res, interpolation=interpolation)

    return img

def read_video(
    path: Union[str, Path],
    n_frames: int = None,
    sample_period: int = 1,
    out_fps: float = None,  # Output fps
    fps: float = None,  # input fps of video (default to avi metadata)
    frame_interpolation: bool = True,
    random_start: bool = False,
    res: Tuple[int] = None,  # (width, height)
    interpolation=cv2.INTER_CUBIC,
    zoom: float = 0):

    # Check path
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Get video properties
    cap = cv2.VideoCapture(str(path))
    vid_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    if out_fps is not None:
        sample_period = 1
        # Figuring out how many frames to read, and at what stride, to achieve the target
        # output FPS if one is given.
        if n_frames is not None:
            out_n_frames = n_frames
            n_frames = int(np.ceil((n_frames - 1) * fps / out_fps + 1))
        else:
            out_n_frames = int(np.floor((vid_size[0] - 1) * out_fps / fps + 1))

    # Setup output array
    if n_frames is None:
        n_frames = vid_size[0] // sample_period
    if n_frames * sample_period > vid_size[0]:
        raise Exception(
            f"{n_frames} frames requested (with sample period {sample_period}) but video length is only {vid_size[0]} frames"
        )
    if res is None:
        out = np.zeros((n_frames, *vid_size[1:], 3), dtype=np.uint8)
    else:
        out = np.zeros((n_frames, res[1], res[0], 3), dtype=np.uint8)

    # Read video, skipping sample_period frames each time
    if random_start:
        si = np.random.randint(vid_size[0] - n_frames * sample_period + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, si)
    for frame_i in range(n_frames):
        _, frame = cap.read()
        if res is not None:
            frame = crop_and_scale(frame, res, interpolation, zoom)
        out[frame_i] = frame
        for _ in range(sample_period - 1):
            cap.read()
    cap.release()

    # if a particular output fps is desired, either get the closest frames from the input video
    # or interpolate neighboring frames to achieve the fps without frame stutters.
    if out_fps is not None:
        i = np.arange(out_n_frames) * fps / out_fps
        if frame_interpolation:
            out_0 = out[np.floor(i).astype(int)]
            out_1 = out[np.ceil(i).astype(int)]
            t = (i % 1)[:, None, None, None]
            out = (1 - t) * out_0 + t * out_1
        else:
            out = out[np.round(i).astype(int)]

    if n_frames == 1:
        out = np.squeeze(out)
    return out, vid_size, fps



class EchoDataset(Dataset):
    def __init__(
        self,
        data_path: Union[Path, str],
        manifest_path: Union[Path, str] = None,
        split: str = None,
        labels: List[str] = None,
        verify_existing: bool = True,
        drop_na_labels: bool = True,
        n_frames: int = 16,
        random_start: bool = False,
        sample_rate: Union[int, Tuple[int], float] = 2,
        verbose: bool = True,
        resize_res: Tuple[int] = None,
        zoom: float = 0
    ):
        self.verbose = verbose
        self.data_path = Path(data_path)
        self.split = split
        self.verify_existing = verify_existing
        self.n_frames = n_frames
        self.random_start = random_start
        self.sample_rate = sample_rate
        self.resize_res = resize_res
        self.zoom = zoom
       
        # Read manifest file
        if manifest_path is not None:
            self.manifest_path = Path(manifest_path)
        else:
            self.manifest_path = self.data_path / "manifest.csv"

        if self.manifest_path.exists():
            self.manifest = pd.read_csv(self.manifest_path, low_memory=False)
        else:
            self.manifest = pd.DataFrame(
                {
                    "filename": os.listdir(self.data_path),
                }
            )
        if self.split is not None:
            self.manifest = self.manifest[self.manifest["split"] == self.split]
        if self.verbose:
            print(
                f"Manifest loaded. \nSplit: {self.split}\nLength: {len(self.manifest):,}"
            )

        # Make sure all files actually exist. This can be disabled for efficiency if
        # you have an especially large dataset
        if self.verify_existing:
            old_len = len(self.manifest)
            existing_files = os.listdir(self.data_path)
            self.manifest = self.manifest[
                self.manifest["filename"].isin(existing_files)
            ]
            new_len = len(self.manifest)
            if self.verbose:
                print(
                    f"{old_len - new_len} files in the manifest are missing from {self.data_path}."
                )
        elif (not self.verify_existing) and self.verbose:
            print(
                f"self.verify_existing is set to False, so it's possible for the manifest to contain filenames which are not present in {data_path}"
            )

        
    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        output = {}
        row = self.manifest.iloc[index]
        filename = row["filename"]
        output["filename"] = filename

        # self.read_file expected in child classes
        primary_input = self.read_file(self.data_path / filename, row)
        output["primary_input"] = primary_input
        return output
   
    def read_file(self, filepath, row=None):

        if isinstance(self.sample_rate, int):  # Simple single int sample period
            vid, vid_shape, fps = read_video(
                filepath,
                self.n_frames,
                res=self.resize_res,
                zoom=self.zoom,
                sample_period=self.sample_rate,
                random_start=self.random_start,
            )
        elif isinstance(self.sample_rate, float):  # Fixed fps
            target_fps = self.sample_rate
            fps = row["fps"]

            vid, vid_shape, fps = read_video(
                filepath,
                self.n_frames,
                1,
                fps=row["fps"],
                out_fps=target_fps,
                frame_interpolation=self.interpolate_frames,
                random_start=self.random_start,
                res=self.resize_res,
                zoom=self.zoom,
            )
        else:  # Tuple sample period ints to be randomly sampled from (1, 2, 3)
            sample_period = np.random.choice(
                [x for x in self.sample_rate if row["frames"] > x * self.n_frames]
            )
            vid, vid_shape, fps = read_video(
                filepath,
                self.n_frames,
                res=self.resize_res,
                zoom=self.zoom,
                sample_period=sample_period,
                random_start=self.random_start,
            )
        vid = torch.from_numpy(vid)
        vid = torch.movedim(vid / 255, -1, 0).to(torch.float32)
        return vid

def get_frame_count(filename):
    cap = cv2.VideoCapture(str(filename))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return frame_count


############################################################################  MS Disease Prediction ############################################################################
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision
import pydicom
import cv2
from typing import Tuple, Union, List
from numpy.typing import ArrayLike
from PIL import Image
import pandas as pd
import os


def load_model(device, weights_path, num_classes):
    weights = torch.load(weights_path, map_location=device)
    weights = {key[2:]: val for key, val in weights.items()}
    model = torchvision.models.video.r2plus1d_18(num_classes=num_classes)
    model.load_state_dict(weights)
    model.to(device)
    return model.eval()

def load_image_model(device, weights_path, num_classes):
    weights = torch.load(weights_path, map_location=device)
    weights = {key[2:]: val for key, val in weights.items()}
    model = torchvision.models.resnet50(num_classes=num_classes)
    model.load_state_dict(weights)
    model.to(device)
    return model.eval()

def mask_outside_ultrasound(original_pixels: np.array) -> np.array:
    """
    Masks all pixels outside the ultrasound region in a video.

    Args:
    vid (np.ndarray): A numpy array representing the video frames. FxHxWxC

    Returns:
    np.ndarray: A numpy array with pixels outside the ultrasound region masked.
    """
    try:
        testarray=np.copy(original_pixels)
        vid=np.copy(original_pixels)
        ##################### CREATE MASK #####################
        # Sum all the frames
        frame_sum = testarray[0].astype(np.float32)  # Start off the frameSum with the first frame
        frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_YUV2RGB)
        frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_RGB2GRAY)
        frame_sum = np.where(frame_sum > 0, 1, 0) # make all non-zero values 1
        frames = testarray.shape[0]
        for i in range(frames): # Go through every frame
            frame = testarray[i, :, :, :].astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = np.where(frame>0,1,0) # make all non-zero values 1
            frame_sum = np.add(frame_sum,frame)

        # Erode to get rid of the EKG tracing
        kernel = np.ones((3,3), np.uint8)
        frame_sum = cv2.erode(np.uint8(frame_sum), kernel, iterations=10)

        # Make binary
        frame_sum = np.where(frame_sum > 0, 1, 0)

        # Make the difference frame fr difference between 1st and last frame
        # This gets rid of static elements
        frame0 = testarray[0].astype(np.uint8)
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_YUV2RGB)
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
        frame_last = testarray[testarray.shape[0] - 1].astype(np.uint8)
        frame_last = cv2.cvtColor(frame_last, cv2.COLOR_YUV2RGB)
        frame_last = cv2.cvtColor(frame_last, cv2.COLOR_RGB2GRAY)
        frame_diff = abs(np.subtract(frame0, frame_last))
        frame_diff = np.where(frame_diff > 0, 1, 0)

        # Ensure the upper left hand corner 20x20 box all 0s.
        # There is a weird dot that appears here some frames on Stanford echoes
        frame_diff[0:20, 0:20] = np.zeros([20, 20])

        # Take the overlap of the sum frame and the difference frame
        frame_overlap = np.add(frame_sum,frame_diff)
        frame_overlap = np.where(frame_overlap > 1, 1, 0)

        # Dilate
        kernel = np.ones((3,3), np.uint8)
        frame_overlap = cv2.dilate(np.uint8(frame_overlap), kernel, iterations=10).astype(np.uint8)

        # Fill everything that's outside the mask sector with some other number like 100
        cv2.floodFill(frame_overlap, None, (0,0), 100)
        # make all non-100 values 255. The rest are 0
        frame_overlap = np.where(frame_overlap!=100,255,0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(frame_overlap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours[0] has shape (445, 1, 2). 445 coordinates. each coord is 1 row, 2 numbers
        # Find the convex hull
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            cv2.drawContours(frame_overlap, [hull], -1, (255, 0, 0), 3)
        frame_overlap = np.where(frame_overlap > 0, 1, 0).astype(np.uint8) #make all non-0 values 1
        # Fill everything that's outside hull with some other number like 100
        cv2.floodFill(frame_overlap, None, (0,0), 100)
        # make all non-100 values 255. The rest are 0
        frame_overlap = np.array(np.where(frame_overlap != 100, 255, 0),dtype=bool)
        ################## Create your .avi file and apply mask ##################
        # Store the dimension values

        # Apply the mask to every frame and channel (changing in place)
        for i in range(len(vid)):
            frame = vid[i, :, :, :].astype('uint8')
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
            frame = cv2.bitwise_and(frame, frame, mask = frame_overlap.astype(np.uint8))
            vid[i,:,:,:]=frame
        return vid
    except Exception as e:
        print("Error masking returned as is.")
        return vid

def read_video(
    path: Union[str, Path],
    n_frames: int = None,
    sample_period: int = 1,
    out_fps: float = None,  # Output fps
    fps: float = None,  # input fps of video (default to avi metadata)
    frame_interpolation: bool = True,
    random_start: bool = False,
    res: Tuple[int] = None,  # (width, height)
    interpolation=cv2.INTER_CUBIC,
    zoom: float = 0):

    # Check path
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Get video properties
    cap = cv2.VideoCapture(str(path))
    vid_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    if out_fps is not None:
        sample_period = 1
        # Figuring out how many frames to read, and at what stride, to achieve the target
        # output FPS if one is given.
        if n_frames is not None:
            out_n_frames = n_frames
            n_frames = int(np.ceil((n_frames - 1) * fps / out_fps + 1))
        else:
            out_n_frames = int(np.floor((vid_size[0] - 1) * out_fps / fps + 1))

    # Setup output array
    if n_frames is None:
        n_frames = vid_size[0] // sample_period
    if n_frames * sample_period > vid_size[0]:
        raise Exception(
            f"{n_frames} frames requested (with sample period {sample_period}) but video length is only {vid_size[0]} frames"
        )
    if res is None:
        out = np.zeros((n_frames, *vid_size[1:], 3), dtype=np.uint8)
    else:
        out = np.zeros((n_frames, res[1], res[0], 3), dtype=np.uint8)

    # Read video, skipping sample_period frames each time
    if random_start:
        si = np.random.randint(vid_size[0] - n_frames * sample_period + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, si)
    for frame_i in range(n_frames):
        _, frame = cap.read()
        if res is not None:
            frame = crop_and_scale(frame, res, interpolation, zoom)
        out[frame_i] = frame
        for _ in range(sample_period - 1):
            cap.read()
    cap.release()

    # if a particular output fps is desired, either get the closest frames from the input video
    # or interpolate neighboring frames to achieve the fps without frame stutters.
    if out_fps is not None:
        i = np.arange(out_n_frames) * fps / out_fps
        if frame_interpolation:
            out_0 = out[np.floor(i).astype(int)]
            out_1 = out[np.ceil(i).astype(int)]
            t = (i % 1)[:, None, None, None]
            out = (1 - t) * out_0 + t * out_1
        else:
            out = out[np.round(i).astype(int)]

    if n_frames == 1:
        out = np.squeeze(out)
    return out, vid_size, fps

def crop_and_scale(
    img: ArrayLike, res: Tuple[int], interpolation=cv2.INTER_CUBIC, zoom: float = 0.0
) -> ArrayLike:
    """Takes an image, a resolution, and a zoom factor as input, returns the
    zoomed/cropped image."""
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    # Crop to correct aspect ratio
    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]

    # Apply zoom
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]

    # Resize image
    img = cv2.resize(img, res, interpolation=interpolation)

    return img

def write_to_avi(frames: np.ndarray, out_file, fps=30):
    out = cv2.VideoWriter(str(out_file), cv2.VideoWriter_fourcc(*'MJPG'), fps, (frames.shape[2], frames.shape[1]))
    for frame in frames:
        out.write(frame.astype(np.uint8))
    out.release()

def write_to_jpg(frames: np.ndarray, out_file: Path):
    for i, frame in enumerate(frames):
        cv2.imwrite(str(out_file.parent / f"{out_file.stem}_{i:04d}.jpg"), frame.astype(np.uint8))
    


class EchoDataset(Dataset):
    def __init__(
        self,
        data_path: Union[Path, str],
        manifest_path: Union[Path, str] = None,
        split: str = None,
        labels: List[str] = None,
        verify_existing: bool = True,
        drop_na_labels: bool = True,
        n_frames: int = 16,
        random_start: bool = False,
        sample_rate: Union[int, Tuple[int], float] = 2,
        verbose: bool = True,
        resize_res: Tuple[int] = (112, 112),
        zoom: float = 0
    ):
        self.verbose = verbose
        self.data_path = Path(data_path)
        self.split = split
        self.verify_existing = verify_existing
        self.n_frames = n_frames
        self.random_start = random_start
        self.sample_rate = sample_rate
        self.resize_res = resize_res
        self.zoom = zoom
       
        # Read manifest file
        if manifest_path is not None:
            self.manifest_path = Path(manifest_path)
        else:
            self.manifest_path = self.data_path / "manifest.csv"

        if self.manifest_path.exists():
            self.manifest = pd.read_csv(self.manifest_path, low_memory=False)
        else:
            self.manifest = pd.DataFrame(
                {
                    "filename": os.listdir(self.data_path),
                }
            )
        if self.split is not None:
            self.manifest = self.manifest[self.manifest["split"] == self.split]
        if self.verbose:
            print(
                f"Manifest loaded. \nSplit: {self.split}\nLength: {len(self.manifest):,}"
            )

        # Make sure all files actually exist. This can be disabled for efficiency if
        # you have an especially large dataset
        if self.verify_existing:
            old_len = len(self.manifest)
            existing_files = [f for f in self.manifest['filename'] if os.path.exists(f)]
            self.manifest = self.manifest[
                self.manifest["filename"].isin(existing_files)
            ]
            new_len = len(self.manifest)
            if self.verbose:
                print(
                    f"{old_len - new_len} files in the manifest are missing."
                )
        elif (not self.verify_existing) and self.verbose:
            print(
                f"self.verify_existing is set to False, so it's possible for the manifest to contain filenames which are not present"
            )

        
    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        output = {}
        row = self.manifest.iloc[index]
        filename = row["filename"]
        output["filename"] = filename

        # self.read_file expected in child classes
        primary_input = self.read_file(self.data_path / filename, row)
        output["primary_input"] = primary_input
        return output
   
    def read_file(self, filepath, row=None):

        if isinstance(self.sample_rate, int):  # Simple single int sample period
            vid, vid_shape, fps = read_video(
                filepath,
                self.n_frames,
                res=self.resize_res,
                zoom=self.zoom,
                sample_period=self.sample_rate,
                random_start=self.random_start,
            )
        elif isinstance(self.sample_rate, float):  # Fixed fps
            target_fps = self.sample_rate
            fps = row["fps"]

            vid, vid_shape, fps = read_video(
                filepath,
                self.n_frames,
                1,
                fps=row["fps"],
                out_fps=target_fps,
                frame_interpolation=self.interpolate_frames,
                random_start=self.random_start,
                res=self.resize_res,
                zoom=self.zoom,
            )
        else:  # Tuple sample period ints to be randomly sampled from (1, 2, 3)
            sample_period = np.random.choice(
                [x for x in self.sample_rate if row["frames"] > x * self.n_frames]
            )
            vid, vid_shape, fps = read_video(
                filepath,
                self.n_frames,
                res=self.resize_res,
                zoom=self.zoom,
                sample_period=sample_period,
                random_start=self.random_start,
            )
        vid = torch.from_numpy(vid)
        vid = torch.movedim(vid / 255, -1, 0).to(torch.float32)
        return vid


class ImageDataset(Dataset):
    """
    Dataset class for image-based echo views (e.g., MV_CW Doppler images)
    """
    def __init__(
        self,
        data_path: Union[Path, str],
        manifest_path: Union[Path, str] = None,
        split: str = None,
        verify_existing: bool = True,
        verbose: bool = True,
        resize_res: Tuple[int] = (224, 224),
    ):
        self.verbose = verbose
        self.data_path = Path(data_path)
        self.split = split
        self.verify_existing = verify_existing
        self.resize_res = resize_res
       
        # Read manifest file
        if manifest_path is not None:
            self.manifest_path = Path(manifest_path)
        else:
            self.manifest_path = self.data_path / "manifest.csv"

        if self.manifest_path.exists():
            self.manifest = pd.read_csv(self.manifest_path, low_memory=False)
        else:
            self.manifest = pd.DataFrame(
                {
                    "filename": os.listdir(self.data_path),
                }
            )
        
        if self.split is not None:
            self.manifest = self.manifest[self.manifest["split"] == self.split]
        
        if self.verbose:
            print(
                f"Manifest loaded. \nSplit: {self.split}\nLength: {len(self.manifest):,}"
            )

        # Make sure all files actually exist
        if self.verify_existing:
            old_len = len(self.manifest)
            existing_files = [f for f in self.manifest['filename'] if os.path.exists(f)]
            self.manifest = self.manifest[
                self.manifest["filename"].isin(existing_files)
            ]
            new_len = len(self.manifest)
            if self.verbose:
                print(
                    f"{old_len - new_len} files in the manifest are missing."
                )
        elif (not self.verify_existing) and self.verbose:
            print(
                f"self.verify_existing is set to False, so it's possible for the manifest to contain filenames which are not present"
            )

        
    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        output = {}
        row = self.manifest.iloc[index]
        filename = row["filename"]
        output["filename"] = filename

        # Read image file
        primary_input = self.read_file(self.data_path / filename)
        output["primary_input"] = primary_input
        return output
   
    def read_file(self, filepath):
        """
        Read image file and convert to tensor.
        
        Args:
            filepath: Path to image file (.jpg)
            
        Returns:
            Tensor of shape (C, H, W) with values in [0, 1]
        """
        # Load image
        img = Image.open(filepath).convert('RGB')
        
        # Resize
        if self.resize_res is not None:
            img = img.resize(self.resize_res, Image.BILINEAR)
        
        # Convert to numpy array
        img_array = np.array(img)  # Shape: (H, W, C)
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img_array)
        img_tensor = torch.movedim(img_tensor / 255.0, -1, 0).to(torch.float32)  # (C, H, W)
        
        return img_tensor




import cv2
import numpy as np
import pydicom as dicom
import PIL.Image
from pathlib import Path
from typing import Tuple, List

# ========================
# 1. Pixel extraction and color conversion
# ========================

def ybr_to_rgb(pixels: np.ndarray) -> np.ndarray:
    """Convert YBR color space to RGB"""
    if len(pixels.shape) == 3:  # Single frame HxWxC
        return cv2.cvtColor(pixels.astype(np.uint8), cv2.COLOR_YUV2RGB)
    elif len(pixels.shape) == 4:  # Multiple frames FxHxWxC
        rgb_frames = []
        for frame in pixels:
            rgb_frames.append(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_YUV2RGB))
        return np.array(rgb_frames)
    return pixels


def get_pixels(dcm: dicom.Dataset) -> Tuple[np.ndarray, bool]:
    """
    Extract pixels from DICOM and convert to RGB format.
    
    Args:
        dcm: DICOM dataset
        
    Returns:
        pixels: Pixel array in RGB format
        is_grayscale: Whether the original was grayscale
    """
    pixels = dcm.pixel_array
    is_grayscale = (len(pixels.shape) < 3 or pixels.shape[-1] != 3)
    current = dcm['PhotometricInterpretation'].value
    
    if current in ["YBR_FULL", "YBR_FULL_422"]:
        pixels = ybr_to_rgb(pixels)
    else:
        if current in ['MONOCHROME2', 'PALETTE COLOR']:
            # Convert grayscale to RGB
            if len(pixels.shape) == 2:  # HxW
                pixels = np.stack([pixels, pixels, pixels], axis=-1)
            elif len(pixels.shape) == 3:  # FxHxW
                pixels = np.stack([pixels, pixels, pixels], axis=-1)
    
    return pixels, is_grayscale


# ========================
# 2. DICOM Tag De-identification
# ========================

SIMPLIFIED_TAG_ACTIONS = {
    "(0008,0020)": "Z",  # Study Date
    "(0008,0030)": "Z",  # Study Time
    "(0010,0010)": "Z",  # Patient's Name
    "(0010,0020)": "Z",  # Patient ID
    "(0010,0030)": "Z",  # Patient's Birth Date
}

removed_tags = None


def clean_tag(dcm: dicom.Dataset, tag: tuple, action: str):
    """Clean a single DICOM tag"""
    global removed_tags
    
    if tag not in dcm:
        return
    
    try:
        if action == "Z":
            dcm[tag].value = ""
        elif action == "X":
            del dcm[tag]
    except Exception:
        pass


def clean_tags(dcm: dicom.Dataset) -> List[str]:
    """De-identify DICOM tags"""
    global removed_tags
    removed_tags = []
    
    for tag_str, action in SIMPLIFIED_TAG_ACTIONS.items():
        tag_tuple = tuple(int("0x" + x, 0) for x in tag_str[1:-1].split(","))
        clean_tag(dcm, tag_tuple, action)
    
    out = removed_tags
    removed_tags = None
    return out


# ========================
# 3. Simple Text Masking (No OCR)
# ========================

def simple_mask_regions(pixels: np.ndarray, mask_color=(0, 0, 0)) -> np.ndarray:
    """
    Mask common regions where PHI appears (top banner and bottom strip).
    
    Args:
        pixels: Pixel array (HxWxC)
        mask_color: Color to use for masking (default: black)
        
    Returns:
        Masked pixel array
    """
    H, W = pixels.shape[:2]
    pixels_masked = pixels.copy()
    
    # Mask top banner (first 10% of image)
    top_height = int(0.1 * H)
    pixels_masked[0:top_height, :] = mask_color
    
    # Mask bottom strip (last 13% of image) - often contains patient info
    bottom_height = int(0.87 * H)
    pixels_masked[bottom_height:, :] = mask_color
    
    return pixels_masked


# ========================
# 4. Main processing function
# ========================

def process_dicom_image_with_deidentification(
    dcm_path: str,
    output_path: str,
    quality: int = 95
) -> bool:
    """
    Process a DICOM image: mask regions, save as JPG.
    """
    try:
        # Read DICOM
        dcm = dicom.dcmread(dcm_path)
        
        # Check if it's an image
        n_frames = int(dcm.NumberOfFrames) if "NumberOfFrames" in dcm else 1
        if n_frames > 1:
            print(f"Warning: {dcm_path} is a video ({n_frames} frames). Using middle frame.")
        
        # De-identify tags
        clean_tags(dcm)
        try:
            dcm.remove_private_tags()
        except Exception:
            pass
        
        # Get pixels
        pixels, is_grayscale = get_pixels(dcm)
        
        # If video, extract middle frame
        if len(pixels.shape) == 4:
            pixels = pixels[pixels.shape[0] // 2]
        
        # Mask common PHI regions
        pixels = simple_mask_regions(pixels)
        
        # Ensure uint8
        if pixels.dtype != np.uint8:
            if pixels.max() > 0:
                pixels = (pixels / pixels.max() * 255).astype(np.uint8)
            else:
                pixels = pixels.astype(np.uint8)
        
        # Save as JPG
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        PIL.Image.fromarray(pixels).save(output_path, format="JPEG", quality=quality)
        
        return True
        
    except Exception as e:
        print(f"Error processing {dcm_path}: {e}")
        import traceback
        traceback.print_exc()
        return False
