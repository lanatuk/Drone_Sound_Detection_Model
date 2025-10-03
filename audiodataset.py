from torch.utils.data import Dataset
import torchaudio.transforms as T
import torch.nn.functional as F
import torch

class AudioDataset(Dataset):
    """
    Custom PyTorch Dataset for drone audio classification.

    Loads audio clips from a Hugging Face dataset and converts them into
    spectrogram images ready for EfficientNet.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with audio indices and labels.
    dataset : datasets.Dataset
        Hugging Face dataset containing the raw audio samples.
    target_sr : int, default=16000
        Target sampling rate to resample all audio to.
    target_seconds : float, default=10
        Desired fixed audio length in seconds (clips are padded or cropped accordingly).
    n_mels : int, default=128
        Number of mel frequency bins for the spectrogram.
    """
    def __init__(self, df, dataset, target_sr: int = 16000, target_seconds: float = 10, n_mels: int = 128):
        self.df = df.reset_index(drop=True)
        self.dataset = dataset
        self.target_sr = target_sr
        self.target_length = int(round(target_sr * target_seconds))
        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sr,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            f_min=0.0,
            f_max=target_sr / 2,
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80)

    def __len__(self):
        """Returns number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Loads single audio sample and its label.

        Fetches the raw waveform, applies resampling, padding/cropping, mel-spectrogram
        transformation, normalization, and resizing to prepare the input for EfficientNet.

        Returns
        -------
        image_3ch : torch.Tensor
            Preprocessed spectrogram image with shape (3, 224, 224).
        label : int
            Ground truth label (0 = no drone, 1 = drone).
        """
        row = self.df.iloc[idx]
        audio_index = int(row['audio_index'])
        label = int(row['label'])
        
        # Load audio from Hugging Face 
        audio_data = self.dataset[audio_index]['audio']
        waveform = torch.tensor(audio_data['array'], dtype=torch.float32)
        sample_rate = int(audio_data['sampling_rate'])
        
        # Normalize amplitude to [-1,1] if needed
        if waveform.abs().max() > 1.0:
            waveform = waveform / waveform.abs().max()
        
        # Resample if needed
        if sample_rate != self.target_sr:
            resampler = T.Resample(sample_rate, self.target_sr)
            waveform = resampler(waveform)
        
        # Pad/Crop 
        if waveform.numel() > self.target_length:
            # Crop from the middle
            start = int((waveform.numel() - self.target_length) // 2)
            waveform = waveform[start:start + self.target_length]
        else:
            padding = int(self.target_length - waveform.numel())
            waveform = F.pad(waveform, (0, padding))
        
        # Mel-spectrogram (1, n_mels, time) 
        # unsqueeze(0) -> [samples] to [1, samples] (mono channel)
        mel = self.mel_transform(waveform.unsqueeze(0))

        # Log-scale (dB) 
        mel_db = self.amplitude_to_db(mel)

        # Normalization per sample
        mean = mel_db.mean()
        std = mel_db.std().clamp(min=1e-6)
        mel_db = (mel_db - mean) / std
        
        # Resize to 224x224 and expand to 3 channels
        # unsqueeze(0) -> add batch dim [1,1,H,W] for interpolate
        # squeeze(0) -> back to [1,224,224]
        mel_resized = F.interpolate(mel_db.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        image_3ch = mel_resized.repeat(3, 1, 1)
        
        return image_3ch, label