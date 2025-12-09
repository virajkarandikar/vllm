# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/vllm-project/vllm/blob/94d8ec8d2bcb4ec55e33022b313c7e978edf05e1/vllm/model_executor/models/bamba.py
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import tarfile

import torch
import torch.nn as nn
import librosa
import torch.nn.functional as F
import soundfile
import resampy

from typing import Union


class FilterbankFeatures(nn.Module):
    """Minimal Mel Spectrogram feature extractor."""

    def __init__(
        self,
        sample_rate=16000,
        n_window_size=400,
        n_window_stride=160,
        n_fft=512,
        nfilt=80,
        preemph=0.97,
        log_zero_guard_value=5.960464477539063e-08,
        mag_power=2.0,
    ):
        super().__init__()

        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft
        self.preemph = preemph
        self.log_zero_guard_value = log_zero_guard_value
        self.mag_power = mag_power

        # Hann window
        window_tensor = torch.hann_window(self.win_length, periodic=False)
        self.register_buffer("window", window_tensor)

        # Mel filterbank
        highfreq = sample_rate / 2
        filterbanks = torch.tensor(
            librosa.filters.mel(
                sr=sample_rate,
                n_fft=self.n_fft,
                n_mels=nfilt,
                fmin=0,
                fmax=highfreq,
                norm="slaney",
            ),
            dtype=torch.float,
        ).unsqueeze(0)
        self.register_buffer("fb", filterbanks)

    def get_left_context_size(self) -> int:
        """
        In order to run streaming inference, we need to overlap this many samples,
        with the previous input.
        """
        return self.n_fft - self.hop_length

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: Input waveform [batch_size, time]
            There are assumptions about x for now:
                * batch_size=1, since we plan to run preprocessing for each request independently.
                Based on this assumption we keep streaming buffer as (1, buf_size)
                * `time` is always bigger than buf_size. Typically we would be feeding
                n_window_stride * 8 or more.

        Returns:
            features: Mel spectrogram [batch_size, n_mels, time]
        """
        # Preemphasis: x[t] = x[t] - preemph * x[t-1]
        x = torch.cat(
            (x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1
        )

        # change `center=False` so there is no padding on the left and right,
        # so its possible to implement streaming stft extraction
        with torch.amp.autocast(x.device.type, enabled=False):
            x = torch.stft(
                x,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                center=False,
                window=self.window.to(dtype=torch.float, device=x.device),
                return_complex=True,
            )

        # Convert complex to magnitude
        x = torch.view_as_real(x)
        x = torch.sqrt(x.pow(2).sum(-1))

        # Apply power
        x = x.pow(self.mag_power)

        # Apply mel filterbank
        with torch.amp.autocast(x.device.type, enabled=False):
            x = torch.matmul(self.fb.to(x.dtype), x)

        # Apply log
        x = torch.log(x + self.log_zero_guard_value)

        return x


class CausalConv2D(nn.Conv2d):
    """
    A causal version of nn.Conv2d. It pads across frequency axis the same way the padding
    is implemented in nemo code, there is no padding across time axis.
    Instead we would feed extra left context that comes from the streaming buffer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        assert not padding, "padding should be set to 0 or None for CausalConv2D."
        self._left_padding = kernel_size - 1
        self._right_padding = stride - 1
        padding = 0
        super(CausalConv2D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    def forward(
        self,
        x,  # B x CH x T x F
    ):
        # pad only frequencies
        before = x.shape
        x = F.pad(x, pad=(self._left_padding, self._right_padding))
        print(f">>> causal conv2d padding {before} -> {x.shape}", flush=True)
        before = x.shape
        x = super().forward(x)
        print(f">>> causal conv2d conv {before} -> {x.shape}", flush=True)
        return x


class ConvSubsampling(nn.Module):
    """
    Minimal ConvSubsampling for dw_striding, causal mode.
    Configuration: subsampling_factor=8, feat_in=80, feat_out=512, conv_channels=256
    """

    def __init__(self, feat_in=80, feat_out=512, conv_channels=256):
        super(ConvSubsampling, self).__init__()

        # Fixed parameters for the specific configuration
        self.subsampling_factor = 8
        self._sampling_num = 3  # log2(8)
        self._stride = 2
        self._kernel_size = 3
        self._ceil_mode = False
        self._left_padding = self._kernel_size - 1  # 2
        self._right_padding = self._stride - 1  # 1
        self._feat_in = feat_in
        self._feat_out = feat_out
        self._conv_channels = conv_channels

        # Build conv layers
        layers = []
        activation = nn.ReLU(inplace=True)

        # Layer 0: First causal conv (1 -> 256 channels)
        layers.append(
            CausalConv2D(
                in_channels=1,
                out_channels=conv_channels,
                kernel_size=self._kernel_size,
                stride=self._stride,
                padding=None,
            )
        )
        layers.append(activation)

        # Layers 2-7: Two iterations of (depthwise + pointwise + activation)
        for _ in range(self._sampling_num - 1):
            # Depthwise conv
            layers.append(
                CausalConv2D(
                    in_channels=conv_channels,
                    out_channels=conv_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=None,
                    groups=conv_channels,
                )
            )
            # Pointwise conv
            layers.append(
                nn.Conv2d(
                    in_channels=conv_channels,
                    out_channels=conv_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                )
            )
            layers.append(activation)

        self.conv = nn.ModuleList(layers)

        # hard code the size across frequency axis after convolutions
        # this assumes `feat_in == 80`
        out_length = 11
        self.out = nn.Linear(conv_channels * out_length, feat_out)

    def get_left_context_size(self) -> int:
        """
        Computes how many frames of left context will be sliced off
        by the convolutional stack
        """
        resolution = 1
        total_left_context = 0
        for _ in range(self._sampling_num):
            total_left_context += (self._kernel_size - 1) * resolution
            resolution *= self._stride
        return total_left_context

    def forward(self, x):
        """
        Args:
            x: [B, F, T]
        Returns:
            [B, T, feat_out]
        """
        # Transpose and add channel dimension: [B, F, T] -> [B, 1, T, F]
        x = x.transpose(1, 2).unsqueeze(1)

        # Apply convolutions
        for conv in self.conv:
            x = conv(x)
        # Flatten and project: [B, C, T, F] -> [B, T, C*F] -> [B, T, feat_out]
        b, _, t, _ = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))

        return x


class FastConformerPreprocessor(nn.Module):
    """
    A preprocessor for the fastconformer model.
    It consists of a filterbank features extractor and a convolutional subsampler.
    """

    def __init__(self):
        super(FastConformerPreprocessor, self).__init__()
        self.filterbank_features = FilterbankFeatures()
        self.pre_encode = ConvSubsampling()

        # compute how much of the left context is needed to do streaming inference
        self.buffer_size = self.filterbank_features.get_left_context_size()
        self.buffer_size += (
            self.pre_encode.get_left_context_size()
            * self.filterbank_features.hop_length
        )
        # create a streaming buffer
        self.streaming_buffer = torch.zeros(
            1, self.buffer_size, device=torch.device("cuda")
        )

        # CUDA graph members (initialized when capture_cuda_graph is called)
        self.cuda_graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x, buffer):
        # Concatenate buffer to the left of input
        x = torch.cat([buffer, x], dim=1)

        mel = self.filterbank_features(x)
        feat = self.pre_encode(mel)

        # Update buffer with rightmost samples from the concatenated input
        # Store for next iteration BEFORE any processing
        buffer.copy_(x[:, -self.buffer_size :])

        return feat

    def capture_cuda_graph(self):
        """
        Captures a CUDA graph for inputs of shape (1, 160*8).
        This should be called after moving the model to CUDA.
        The captured graph can then be replayed using forward_cuda_graph().
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot capture CUDA graph.")

        # Create static tensors for the specific input shape
        input_size = 160 * 8  # 1280 samples, corresponds to single frame of output
        self.static_input = torch.zeros(1, input_size, device="cuda")

        # Create CUDA graph
        self.cuda_graph = torch.cuda.CUDAGraph()

        # Warm-up: run the model a few times before capturing the graph
        for _ in range(3):
            _ = self.forward(self.static_input, self.streaming_buffer)

        # Capture the graph
        with torch.cuda.graph(self.cuda_graph):
            self.static_output = self.forward(self.static_input, self.streaming_buffer)

        print("CUDA graph captured successfully for input shape (1, 1280)")

    def forward_cuda_graph(self, x):
        """
        Replays the captured CUDA graph for inputs of shape (1, 160*8).

        Args:
            x: Input tensor of shape (1, 1280)

        Returns:
            Output features from the preprocessor

        Note: The input must have the same shape (1, 1280) as used during capture.
        """
        if self.cuda_graph is None:
            raise RuntimeError(
                "CUDA graph not captured. Call capture_cuda_graph() first."
            )

        if x.shape != (1, 160 * 8):
            raise ValueError(f"Input shape must be (1, 1280), got {x.shape}")

        # Copy input data into the static input tensor
        self.static_input.copy_(x)

        # Replay the graph
        self.cuda_graph.replay()

        # Return the output directly, carefull its just an address to static tensor
        # copy if storing into a list
        return self.static_output

    def load_weights_from_nemo(self, model_path: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(model_path, "r:") as tar:
                tar.extractall(tmpdir)

            weights_path = os.path.join(tmpdir, "model_weights.ckpt")
            state_dict = torch.load(
                weights_path, map_location="cpu", weights_only=False
            )
            pre_enc_weights = {
                k[len("encoder.pre_encode.") :]: v
                for k, v in state_dict.items()
                if k.startswith("encoder.pre_encode.")
            }
            # load this weigths for pre encode,
            # filterbank weights are non-trainable and are re-created in the constructor
            self.pre_encode.load_state_dict(pre_enc_weights, strict=True)


def get_original_feats(model_path: str, audio: torch.Tensor) -> torch.Tensor:
    """
    Extracts preprocessed features using the nemo code.
    This is used to check that the standalone streaming implementation is actually correct.
    """
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

    orig_model = EncDecHybridRNNTCTCBPEModel.restore_from(model_path, strict=False)
    audio_len = torch.tensor([audio.shape[1]], device=audio.device)
    mel, mel_len = orig_model.preprocessor(input_signal=audio, length=audio_len)
    # mel has shape (B x F x T), need to transpose before pre encoder
    # this transposition is in conformer_encoder forward code
    mel = torch.transpose(mel, 1, 2)  # [B, T, F]
    emb, _ = orig_model.encoder.pre_encode(x=mel, lengths=mel_len)
    return emb


def main():
    """
    Here we show how use FastConformerPreprocessor and also compare
    it with original preprocessing from nemo.

    We run nemo code on entire chunk.
    Then run `FastConformerPreprocessor` incrementally, frame by frame.
    Results are compared to verify that streaming implementation of conformer
    preprocessor is correct.
    """
    model_path = "stt_en_fastconformer_hybrid_large_streaming_80ms.nemo"

    y, sr = soundfile.read("pred.wav", dtype="float32", always_2d=True)
    if sr != 16000:
        # Resample to 16000 Hz if needed
        y = resampy.resample(y.T, sr, 16000).T
        sr = 16000
    test_audio = torch.from_numpy(y.T).contiguous().cuda()  # shape [channels, samples]

    # run nemo code on the entire chunk
    orig_feats = get_original_feats(model_path, test_audio)
    print(
        f"Nemo fastconformer produced {orig_feats.shape} features from {test_audio.shape} test audio"
    )
    torch.save(orig_feats, "orig_feats.pt")

    # now run `FastConformerPreprocessor` incrementally
    preprocessor = FastConformerPreprocessor().cuda()
    preprocessor.load_weights_from_nemo(model_path)
    preprocessor.capture_cuda_graph()

    res = preprocessor(test_audio, preprocessor.streaming_buffer)
    torch.save(res, "orig_feats.pt")
    preprocessor.streaming_buffer.zero_()

    outputs = []
    start = 0
    prefill_steps = 15
    # run context phase (16 frames) in eager mode
    context_len = 160 * prefill_steps * 8
    chunk = test_audio[:, start : start + context_len]
    start += context_len
    feats = preprocessor(chunk, preprocessor.streaming_buffer)
    outputs.append(feats)
    # now generate frame by frame of 20 steps
    step = 160 * 8
    generations_steps = 30
    for _ in range(generations_steps):
        chunk = test_audio[:, start : start + step]
        start += step
        feats = preprocessor.forward_cuda_graph(chunk)
        outputs.append(feats.clone())
    res = torch.cat(outputs, dim=1)
    print(
        f"FastConformerPreprocessor produced {res.shape} features, expected ({prefill_steps} context + {generations_steps} incremental steps)"
    )
    torch.save(res, "fastconformer_feats.pt")


if __name__ == "__main__":
    main()
