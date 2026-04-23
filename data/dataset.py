class EchoDataset:
    def __init__(self, data_root, transform=None):
        self.samples = self.load_metadata()
        self.transform = transform
    
    def __getitem__(self, idx):
        return {
            "image": image_tensor,
            "segmentation": seg_mask,
            "landmarks": landmark_coords,
            "metadata": patient_info
        }