import sys
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io
import torchvision.transforms as T
import numpy as np
import pandas as pd
from models.MesoNet4_forEnsemble import Meso4 as MesoNet

pretrained_size = 224
pretrained_means = [0.4489, 0.3352, 0.3106]  # [0.485, 0.456, 0.406]
pretrained_stds = [0.2380, 0.1965, 0.1962]  # [0.229, 0.224, 0.225]


class VideoChunkDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=25, resize_shape=(224, 224)):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = T.Compose([
            T.Resize((pretrained_size, pretrained_size)),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=pretrained_means, std=pretrained_stds)
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        # Load video: (T, H, W, C)
        vframes, _, _ = io.read_video(
            path, pts_unit='sec', output_format='TCHW'
        )
        total_frames = vframes.shape[0]

        # Uniformly sample indices
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = vframes[indices]  # Shape: (num_frames, C, H, W)

        # Apply transforms to each frame
        frames = self.transform(frames)

        return frames, path, self.labels[idx]


def run_predictions(model, dataloader, device):

    results = {'filename': [],
               'label': [],
               'logit': [],
               'confidence': [],
               'prediction': []}
    with torch.no_grad():
        for videos, paths, labels in tqdm(dataloader, desc="Inference", unit="batch", leave=True):
            B, T, C, H, W = videos.shape

            # Flatten B and T to feed to model
            inputs = videos.view(B * T, C, H, W).to(device)

            logits = model(inputs)   # (B * T, num_classes)

            # Reshape back to group by video: (B, T, num_classes)
            logits = logits.view(B, T, -1)

            # Aggregate the logits across the T frames
            avg_logits = torch.mean(logits, dim=1)  # Shape: (B, num_classes)
            probs = torch.softmax(avg_logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            # labels = torch.tensor(labels, dtype=torch.int16)

            results['filename'].extend(paths)
            results['label'].extend(labels.tolist())
            results['logit'].extend(avg_logits[:, 1].detach().cpu().numpy())
            results['confidence'].extend(probs[:, 1].detach().cpu().numpy())
            results['prediction'].extend(preds.detach().cpu().numpy())

            # for i, path in enumerate(paths):
            #     print(
            #         f"File: {path.split('/')[-1]} | Pred: {preds[i].item()} | Conf: {probs[i, 1].item():.4f}"
            #     )

    return pd.DataFrame(results)


def setup_inference(video_files, model_ckpt_file, output_dir, num_frames=25,
                    batch_size=64, num_workers=4):

    df = pd.read_csv(video_files)
    videos_list = df['filename'].tolist()
    labels = [1 if (v.lower() == 'fake') else 0 for v in df['label']]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VideoChunkDataset(videos_list, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model = MesoNet()
    ckpt = torch.load(model_ckpt_file)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    # Testing
    # videos, paths, labels = next(iter(dataloader))
    # B, T, C, H, W = videos.shape
    # inputs = videos.view(B * T, C, H, W).to(device)
    # logits = model(inputs)

    df_preds = run_predictions(model, dataloader, device)
    df_preds.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


def main():
    video_files = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    model_ckpt_file = "./Unimodal/weights/video/Meso4_realA_fakeC.pt"

    setup_inference(video_files, model_ckpt_file, output_dir,
                    num_frames=25, batch_size=32, num_workers=4)


if __name__ == "__main__":
    main()
