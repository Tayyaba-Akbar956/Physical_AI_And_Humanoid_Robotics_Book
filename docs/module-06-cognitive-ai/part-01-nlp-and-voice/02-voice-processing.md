---
sidebar_position: 2
title: Voice Processing
---

# Voice Processing

This chapter explores the technical aspects of processing human voice for robotic applications. Voice processing encompasses converting acoustic signals to text (speech recognition), understanding the meaning of spoken language, and generating appropriate voice responses. This technology is essential for natural human-robot interaction in physical environments.

## Learning Objectives

- Understand the fundamentals of digital speech processing
- Implement voice recognition systems for robotics applications
- Apply techniques for robust voice processing in noisy environments
- Design voice response systems for humanoid robots
- Evaluate voice processing performance in real-world scenarios

## Introduction: Voice in Physical AI

Voice processing enables natural communication between humans and robots. For humanoid robots operating in human-centric environments, voice processing must handle challenges not typically present in traditional voice applications:

- **Variable acoustic conditions**: Background noise, room acoustics, and distance from speaker
- **Real-time requirements**: Robots must respond quickly to maintain natural conversation flow
- **Multimodal integration**: Voice input must be combined with visual, tactile, and other sensory data
- **Safety considerations**: Voice commands must be validated before triggering robotic actions

The voice processing pipeline typically involves several components:
1. **Audio capture**: Converting acoustic signals to digital format
2. **Preprocessing**: Filtering, normalization, and noise reduction
3. **Feature extraction**: Converting audio signals to meaningful representations
4. **Recognition**: Converting audio features to text or commands
5. **Natural language processing**: Understanding the meaning of recognized text
6. **Response generation**: Creating appropriate text responses
7. **Speech synthesis**: Converting text responses to audible speech

### Voice Processing Challenges in Robotics

**Environmental Noise**: Domestic and industrial environments often contain significant background noise that interferes with speech recognition.

**Acoustic Variations**: Room acoustics, reverberation, and microphone position affect audio quality.

**Speaker Variations**: Different accents, speaking styles, and voice characteristics require robust recognition systems.

**Real-time Constraints**: Robots must process speech and respond within natural conversation timing.

**Safety Requirements**: Voice-controlled actions must be verified to prevent dangerous robot behaviors.

## Core Concepts

### Digital Signal Processing Basics

**Sampling and Quantization**:
Voice signals are continuous analog signals that must be converted to digital form for computer processing. The sampling rate determines how often the analog signal is measured, with 16kHz being common for speech recognition and 44.1kHz for high-quality audio.

**Fourier Transform**:
Converts time-domain audio signals to frequency-domain representations, revealing the frequency components present in the signal.

**Feature Extraction**:
Speech signals are converted to feature vectors that capture important aspects of the speech that aid recognition:
- **Mel-frequency cepstral coefficients (MFCCs)**: Captures spectral characteristics of speech
- **Spectral features**: Frequency domain representations
- **Prosodic features**: Intonation, stress, and rhythm patterns

### Speech Recognition Models

**Acoustic Models**: 
Map audio features to phonemes (basic speech units). Modern systems often use deep neural networks trained to recognize patterns in speech features.

**Language Models**:
Model the probability of word sequences, helping resolve ambiguity in acoustic recognition. Can be n-gram models or neural language models.

**Pronunciation Models**:
Define how words are pronounced as sequences of phonemes, accounting for different pronunciations of the same word.

### Voice Activity Detection (VAD)

VAD systems detect speech segments in audio streams, distinguishing between speech and non-speech (silence, noise). This is critical for efficient processing and reducing false triggers.

## Mathematical Framework

### Audio Signal Representation

A digital audio signal can be represented as:

```
x[n] = x(nT), for n = 0, 1, 2, ..., N-1
```

Where:
- `x[n]` is the nth sample of the audio signal
- `T` is the sampling period (1/sampling_rate)
- `N` is the number of samples

### Mel-Frequency Cepstral Coefficients (MFCC)

The MFCC computation involves:

1. **Pre-emphasis**: Amplify high frequencies
   ```
   y[n] = x[n] - α * x[n-1]
   ```

2. **Framing**: Split signal into overlapping frames
   ```
   frame[i] = x[i*frame_shift : i*frame_shift + frame_length]
   ```

3. **Windowing**: Apply window function (e.g., Hamming)
   ```
   w_frame[i] = frame[i] * w[i]
   ```

4. **Fourier Transform**: Convert to frequency domain
   ```
   X[k] = Σ x[n] * e^(-j*2π*k*n/N)
   ```

5. **Mel filtering**: Apply triangular filters on the Mel scale
   ```
   M[i] = Σ |X[k]| * H[i,k]
   ```

6. **Logarithm and DCT**: Final coefficient calculation
   ```
   MFCC[i] = Σ log(M[k]) * cos(π*i*(k-0.5)/K)
   ```

## Practical Implementation

### Audio Preprocessing

```python
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import librosa
from typing import Optional

class AudioPreprocessor:
    def __init__(self, sample_rate: int = 16000, frame_size: float = 0.025, frame_step: float = 0.01):
        """
        Initialize audio preprocessor
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_size: Frame size in seconds
            frame_step: Frame step in seconds
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.frame_step = frame_step
        
        # Convert frame size and step from seconds to samples
        self.frame_size_samples = int(frame_size * sample_rate)
        self.frame_step_samples = int(frame_step * sample_rate)
    
    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            normalized = audio_data / max_val
        else:
            normalized = audio_data
        return normalized
    
    def pre_emphasis(self, audio_data: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        """Apply pre-emphasis filter to amplify high frequencies"""
        return np.append(audio_data[0], audio_data[1:] - alpha * audio_data[:-1])
    
    def framing(self, audio_data: np.ndarray) -> np.ndarray:
        """Convert audio to overlapping frames"""
        # Calculate number of frames
        num_frames = 1 + int(np.ceil((len(audio_data) - self.frame_size_samples) / self.frame_step_samples))
        
        # Pad audio if needed
        pad_length = (num_frames - 1) * self.frame_step_samples + self.frame_size_samples
        pad_width = pad_length - len(audio_data)
        if pad_width > 0:
            audio_data = np.append(audio_data, np.zeros(pad_width))
        
        # Create frames
        indices = np.tile(np.arange(self.frame_size_samples), (num_frames, 1))
        indices += np.tile(np.arange(num_frames) * self.frame_step_samples, (self.frame_size_samples, 1)).T
        
        frames = audio_data[indices]
        
        return frames
    
    def apply_window(self, frames: np.ndarray) -> np.ndarray:
        """Apply window function (Hamming) to frames"""
        window = np.hamming(self.frame_size_samples)
        return frames * window
    
    def compute_fft(self, framed_signal: np.ndarray, n_fft: int = 512) -> np.ndarray:
        """Compute FFT for each frame"""
        # Zero-pad if frame size is less than n_fft
        if self.frame_size_samples < n_fft:
            pad_width = n_fft - self.frame_size_samples
            padded_frames = np.pad(framed_signal, ((0, 0), (0, pad_width)), mode='constant')
        else:
            padded_frames = framed_signal[:, :n_fft]
        
        # Compute FFT
        fft_frames = np.fft.rfft(padded_frames, n_fft)
        return np.abs(fft_frames)
    
    def compute_mel_filterbank(self, n_fft: int = 512, n_mels: int = 26) -> np.ndarray:
        """Compute mel-scale filterbank"""
        # Convert Hz to mel
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700.0)
        
        # Convert mel to Hz
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595.0) - 1)
        
        # Create mel filterbank
        low_freq = 0
        high_freq = self.sample_rate / 2
        low_mel = hz_to_mel(low_freq)
        high_mel = hz_to_mel(high_freq)
        
        # Create equally spaced mel points
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Convert to bin numbers
        bin_points = np.floor((n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        # Create filterbank
        fbank = np.zeros((n_mels, int(np.floor(n_fft / 2 + 1))))
        
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            
            # Create triangular filter
            for j in range(left, center):
                fbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                fbank[i, j] = (right - j) / (right - center)
        
        return fbank
    
    def compute_mfcc(self, audio_data: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """Compute MFCC features from audio data"""
        # Normalize audio
        normalized_audio = self.normalize_audio(audio_data)
        
        # Apply pre-emphasis filter
        pre_emphasized = self.pre_emphasis(normalized_audio)
        
        # Frame the signal
        frames = self.framing(pre_emphasized)
        
        # Apply window function
        windowed_frames = self.apply_window(frames)
        
        # Compute FFT
        power_spectrum = (self.compute_fft(windowed_frames) / 512) ** 2
        
        # Compute mel filterbank
        fbank = self.compute_mel_filterbank()
        
        # Apply filterbank to power spectrum
        mel_spectrum = np.dot(power_spectrum, fbank.T)
        
        # Add small constant to avoid log(0)
        mel_spectrum = np.where(mel_spectrum == 0, np.finfo(float).eps, mel_spectrum)
        
        # Apply log
        log_mel_spectrum = np.log(mel_spectrum)
        
        # Apply DCT to get MFCCs
        mfcc = np.dot(log_mel_spectrum, fbank.T)
        
        # Keep only first n_mfcc coefficients
        return mfcc[:, :n_mfcc]

# Example usage
if __name__ == "__main__":
    # Create preprocessor
    preprocessor = AudioPreprocessor()
    
    # Simulate audio data (in practice, you'd load from file or capture from microphone)
    t = np.linspace(0, 1, 16000)  # 1 second of audio at 16kHz
    simulated_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    # Compute MFCC features
    mfcc_features = preprocessor.compute_mfcc(simulated_audio)
    print(f"MFCC features shape: {mfcc_features.shape}")
```

### Voice Activity Detection (VAD)

```python
import webrtcvad
import collections

class VoiceActivityDetector:
    def __init__(self, sample_rate=16000, frame_duration=30, aggressiveness=3):
        """
        Initialize Voice Activity Detector
        
        Args:
            sample_rate: Sample rate in Hz (must be 8000, 16000, 32000, or 48000)
            frame_duration: Frame duration in milliseconds (10, 20, or 30)
            aggressiveness: Aggressiveness level (0-3, higher = more aggressive)
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration  # in milliseconds
        self.frame_size = int(sample_rate * frame_duration / 1000)  # number of samples per frame
    
    def is_speech(self, audio_frame: bytes) -> bool:
        """
        Check if the frame contains speech
        
        Args:
            audio_frame: Audio data as bytes (16-bit PCM)
        
        Returns:
            True if speech detected, False otherwise
        """
        try:
            return self.vad.is_speech(audio_frame, self.sample_rate)
        except:
            return False  # In case of invalid frame
    
    def detect_voice_activity(self, audio_data: np.ndarray) -> list:
        """
        Detect voice activity throughout audio stream
        
        Args:
            audio_data: Audio signal as numpy array (assumed to be 16-bit PCM, mono)
        
        Returns:
            List of booleans indicating speech presence for each frame
        """
        # Convert to 16-bit int
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Ensure audio is properly shaped
        if audio_int16.ndim > 1:
            audio_int16 = audio_int16[:, 0]  # Take first channel if stereo
        
        # Pad audio if needed to make complete frames
        pad_samples = self.frame_size - (len(audio_int16) % self.frame_size)
        if pad_samples != self.frame_size:
            audio_int16 = np.pad(audio_int16, (0, pad_samples), mode='constant')
        
        # Split into frames
        num_frames = len(audio_int16) // self.frame_size
        frames = audio_int16[:num_frames * self.frame_size].reshape(num_frames, self.frame_size)
        
        # Convert each frame to bytes and detect speech
        speech_flags = []
        for frame in frames:
            frame_bytes = frame.tobytes()
            is_speech = self.is_speech(frame_bytes)
            speech_flags.append(is_speech)
        
        return speech_flags
    
    def extract_speech_segments(self, audio_data: np.ndarray, min_silence_duration: float = 0.5) -> list:
        """
        Extract speech segments with minimum silence filtering
        
        Args:
            audio_data: Audio signal as numpy array
            min_silence_duration: Minimum silence duration to separate speech segments (seconds)
        
        Returns:
            List of (start_time, end_time, audio_segment) tuples
        """
        # Detect voice activity
        activity_flags = self.detect_voice_activity(audio_data)
        
        # Convert frame-based flags to sample-based array
        step_size = self.frame_size
        full_flags = np.repeat(activity_flags, step_size)
        
        # Adjust for any padding
        if len(full_flags) > len(audio_data):
            full_flags = full_flags[:len(audio_data)]
        elif len(full_flags) < len(audio_data):
            full_flags = np.pad(full_flags, (0, len(audio_data) - len(full_flags)), mode='constant')
        
        # Find speech segment boundaries
        speech_segments = []
        in_speech = False
        current_start = 0
        
        for i, is_speech in enumerate(full_flags):
            if is_speech and not in_speech:
                # Start of speech segment
                current_start = i
                in_speech = True
            elif not is_speech and in_speech:
                # End of speech segment
                if i > current_start:  # Check for valid segment
                    start_time = current_start / self.sample_rate
                    end_time = i / self.sample_rate
                    speech_segment = audio_data[current_start:i]
                    
                    # Check if this segment is isolated (not part of a longer speech)
                    silence_before = (current_start > 0 and 
                                    len(audio_data[:current_start]) / self.sample_rate >= min_silence_duration)
                    silence_after = (i < len(audio_data) and 
                                   len(audio_data[i:]) / self.sample_rate >= min_silence_duration)
                    
                    speech_segments.append((start_time, end_time, speech_segment))
                in_speech = False
        
        # Handle case where audio ends with speech
        if in_speech:
            start_time = current_start / self.sample_rate
            end_time = len(audio_data) / self.sample_rate
            speech_segment = audio_data[current_start:]
            speech_segments.append((start_time, end_time, speech_segment))
        
        return speech_segments

# Example usage
if __name__ == "__main__":
    import scipy.io.wavfile as wavfile
    
    # Initialize VAD
    vad = VoiceActivityDetector(sample_rate=16000, aggressiveness=3)
    
    # Create or load sample audio data
    # For example, simulate audio with speech and silence
    t1 = np.linspace(0, 0.5, int(16000 * 0.5))  # 0.5s silence
    t2 = np.linspace(0, 0.5, int(16000 * 0.5))  # 0.5s tone (simulated speech)
    t3 = np.linspace(0, 1.0, int(16000 * 1.0))  # 1.0s silence
    t4 = np.linspace(0, 0.7, int(16000 * 0.7))  # 0.7s tone (simulated speech)
    
    audio_data = np.concatenate([
        np.zeros_like(t1),  # Silence
        np.sin(2 * np.pi * 440 * t2),  # Simulated speech
        np.zeros_like(t3),  # Silence
        np.sin(2 * np.pi * 523 * t4)  # Simulated speech
    ])
    
    # Normalize to avoid clipping
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Detect voice activity
    activity_flags = vad.detect_voice_activity(audio_data)
    print(f"Number of frames: {len(activity_flags)}")
    print(f"Speech frames: {sum(activity_flags)}")
    
    # Extract speech segments
    segments = vad.extract_speech_segments(audio_data)
    print(f"Number of speech segments found: {len(segments)}")
    for i, (start, end, seg) in enumerate(segments):
        print(f"Segment {i+1}: {start:.2f}s to {end:.2f}s (duration: {end-start:.2f}s)")
```

### Speech Recognition with Deep Learning

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataclasses import dataclass

@dataclass
class SpeechRecognitionConfig:
    n_mels: int = 80
    n_classes: int = 30  # Number of output classes (phonemes, subword units, etc.)
    hidden_size: int = 512
    n_layers: int = 3
    dropout: float = 0.2
    input_size: int = 1  # Size of each input token (1 for mel-scaled features)

class SpeechRecognitionModel(nn.Module):
    def __init__(self, config: SpeechRecognitionConfig):
        super(SpeechRecognitionModel, self).__init__()
        
        self.config = config
        
        # LSTM layers for sequence processing
        self.lstm = nn.LSTM(
            input_size=config.n_mels,  # mel features
            hidden_size=config.hidden_size,
            num_layers=config.n_layers,
            dropout=config.dropout if config.n_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Bidirectional LSTM for context in both directions
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
        # Linear layer to project to number of classes
        self.classifier = nn.Linear(config.hidden_size * 2, config.n_classes)  # *2 for bidirectional
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_mels)
        """
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Project to number of classes
        output = self.classifier(lstm_out)
        
        # Apply log softmax for numerical stability
        output = F.log_softmax(output, dim=-1)
        
        return output

class SpeechDataset(Dataset):
    def __init__(self, features_list, labels_list):
        """
        Args:
            features_list: List of feature tensors for each sample
            labels_list: List of label tensors for each sample
        """
        self.features = features_list
        self.labels = labels_list
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class SimpleSpeechRecognizer:
    def __init__(self, config: SpeechRecognitionConfig):
        self.config = config
        self.model = SpeechRecognitionModel(config)
        self.preprocessor = AudioPreprocessor()
        self.char_to_idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx_to_char = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.vocab_size = 4  # Start with basic tokens
        
        # For demonstration, this is a simplified approach
        # In practice, you'd use more sophisticated tokenization
    
    def add_to_vocab(self, text: str):
        """Add characters from text to vocabulary"""
        for char in text:
            if char not in self.char_to_idx:
                idx = len(self.char_to_idx)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
                self.vocab_size = len(self.char_to_idx)
    
    def text_to_indices(self, text: str) -> list:
        """Convert text to sequence of indices"""
        indices = [self.char_to_idx.get(c, self.char_to_idx['<unk>']) for c in text]
        return [self.char_to_idx['<sos>']] + indices + [self.char_to_idx['<eos>']]
    
    def indices_to_text(self, indices: list) -> str:
        """Convert sequence of indices back to text"""
        chars = []
        for idx in indices:
            if idx == self.char_to_idx['<eos>']:
                break
            if idx not in [self.char_to_idx['<sos>'], self.char_to_idx['<pad>']]:
                chars.append(self.idx_to_char.get(idx, '<unk>'))
        return ''.join(chars)
    
    def extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract features for speech recognition"""
        # In this simplified example, we'll use MFCC features
        # In a real system, you might use mel-scale spectrograms
        mfcc_features = self.preprocessor.compute_mfcc(audio_data, n_mfcc=self.config.n_mels)
        return mfcc_features.T  # Transpose to get (sequence_length, n_mels)
    
    def recognize_speech(self, audio_data: np.ndarray) -> str:
        """Simple speech recognition function"""
        # This is a placeholder implementation
        # In a real system, you would:
        # 1. Extract features from audio
        features = self.extract_features(audio_data)
        
        # 2. Pass features through the trained model
        # For this example, we'll simulate the recognition process
        # In practice, you'd need a trained model and appropriate decoding
        
        # Placeholder: return a simple response
        # This would be replaced with actual recognition logic
        if len(audio_data) > 8000:  # If there's enough audio data
            return "hello robot"  # Simulated recognition result
        else:
            return ""

# Example usage
if __name__ == "__main__":
    # Create configuration
    config = SpeechRecognitionConfig(n_mels=40, n_classes=29)  # 26 letters + space + special chars
    
    # Create recognizer
    recognizer = SimpleSpeechRecognizer(config)
    
    # Simulate audio data
    t = np.linspace(0, 1, 16000)  # 1 second at 16kHz
    simulated_audio = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)  # Multi-tone
    
    # Recognize speech
    result = recognizer.recognize_speech(simulated_audio)
    print(f"Recognized text: {result}")
```

### Speech Synthesis (Text-to-Speech)

```python
import pyttsx3
import numpy as np
from typing import Dict, List
import threading
import queue

class TextToSpeech:
    def __init__(self, voice_type='default'):
        """
        Initialize Text-to-Speech engine
        
        Args:
            voice_type: Type of voice to use ('male', 'female', 'default')
        """
        self.engine = pyttsx3.init()
        
        # Get and set voice properties
        voices = self.engine.getProperty('voices')
        
        if voice_type == 'male' and len(voices) > 0:
            self.engine.setProperty('voice', voices[0].id)
        elif voice_type == 'female' and len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)
        
        # Set default properties
        self.engine.setProperty('rate', 150)  # Speed of speech (words per minute)
        self.engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
        
        # Queues for managing speech
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.interrupt_flag = False
    
    def set_voice_properties(self, rate=None, volume=None, voice_id=None):
        """
        Adjust voice properties
        
        Args:
            rate: Speech rate in words per minute
            volume: Volume level (0.0 to 1.0)
            voice_id: Specific voice ID to use
        """
        if rate is not None:
            self.engine.setProperty('rate', rate)
        if volume is not None:
            self.engine.setProperty('volume', volume)
        if voice_id is not None:
            self.engine.setProperty('voice', voice_id)
    
    def speak(self, text: str, blocking: bool = True):
        """
        Speak the provided text
        
        Args:
            text: Text to speak
            blocking: If True, wait until speech is complete
        """
        if not text.strip():
            return
        
        def speak_thread():
            self.engine.say(text)
            if blocking:
                self.engine.runAndWait()
            else:
                self.engine.start()
        
        # Clear any pending speech if interrupting
        if self.is_speaking:
            self.interrupt_speech()
        
        thread = threading.Thread(target=speak_thread)
        thread.start()
        
        if blocking:
            thread.join()
    
    def interrupt_speech(self):
        """Stop current speech"""
        self.engine.stop()
        self.interrupt_flag = True
    
    def speak_with_emotion(self, text: str, emotion: str = 'neutral'):
        """
        Speak text with emotional intonation
        
        Args:
            text: Text to speak
            emotion: Emotion to convey ('happy', 'sad', 'angry', 'excited', 'neutral')
        """
        # Adjust rate and volume based on emotion
        original_rate = self.engine.getProperty('rate')
        original_volume = self.engine.getProperty('volume')
        
        if emotion == 'happy':
            self.engine.setProperty('rate', min(200, original_rate + 20))
            self.engine.setProperty('volume', min(1.0, original_volume + 0.1))
        elif emotion == 'sad':
            self.engine.setProperty('rate', max(100, original_rate - 20))
            self.engine.setProperty('volume', max(0.5, original_volume - 0.2))
        elif emotion == 'excited':
            self.engine.setProperty('rate', min(220, original_rate + 40))
            self.engine.setProperty('volume', min(1.0, original_volume + 0.2))
        elif emotion == 'angry':
            self.engine.setProperty('rate', max(180, original_rate + 10))
            self.engine.setProperty('volume', min(1.0, original_volume + 0.3))
        
        self.speak(text)
        
        # Restore original settings
        self.engine.setProperty('rate', original_rate)
        self.engine.setProperty('volume', original_volume)
    
    def speak_dialogue(self, text: str, context: str = 'neutral'):
        """
        Speak text with context-appropriate intonation
        
        Args:
            text: Text to speak
            context: Context for prosody ('question', 'statement', 'command', 'greeting')
        """
        original_rate = self.engine.getProperty('rate')
        original_volume = self.engine.getProperty('volume')
        
        # Modify intonation based on context
        if context == 'question':
            # Raise pitch at the end (simulated by speaking with slightly higher rate)
            self.engine.setProperty('rate', min(180, original_rate + 10))
        elif context == 'command':
            # More authoritative tone
            self.engine.setProperty('rate', max(120, original_rate - 10))
            self.engine.setProperty('volume', min(1.0, original_volume + 0.1))
        elif context == 'greeting':
            # Warm, welcoming tone
            self.engine.setProperty('rate', 140)
            self.engine.setProperty('volume', original_volume + 0.1)
        
        self.speak(text)
        
        # Restore original settings
        self.engine.setProperty('rate', original_rate)
        self.engine.setProperty('volume', original_volume)

# Example usage
if __name__ == "__main__":
    tts = TextToSpeech(voice_type='default')
    
    # Basic speaking
    tts.speak("Hello, I am a robotic assistant ready to help you.")
    
    # Speaking with emotion
    tts.speak_with_emotion("Great! I'm happy to assist you.", emotion='happy')
    
    # Speaking with context
    tts.speak_dialogue("How can I help you today?", context='greeting')
    tts.speak_dialogue("Please move to the kitchen.", context='command')
    tts.speak_dialogue("Is this your cup?", context='question')
```

### Integration: Complete Voice Processing Pipeline

```python
import threading
import time
import queue
from dataclasses import dataclass

@dataclass
class VoiceCommand:
    text: str
    confidence: float
    timestamp: float
    speaker_id: str = 'unknown'

class VoiceProcessingPipeline:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.is_running = False
        
        # Initialize components
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
        self.vad = VoiceActivityDetector(sample_rate=sample_rate)
        self.recognizer = SimpleSpeechRecognizer(SpeechRecognitionConfig())
        self.tts = TextToSpeech()
        
        # Processing queues
        self.audio_queue = queue.Queue()
        self.command_queue = queue.Queue()
        
        # Threading
        self.processing_thread = None
        self.command_thread = None
    
    def start_listening(self):
        """Start the voice processing pipeline"""
        self.is_running = True
        
        # Start processing threads
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.command_thread = threading.Thread(target=self._process_commands)
        
        self.processing_thread.daemon = True
        self.command_thread.daemon = True
        
        self.processing_thread.start()
        self.command_thread.start()
        
        print("Voice processing pipeline started")
    
    def stop_listening(self):
        """Stop the voice processing pipeline"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        if self.command_thread:
            self.command_thread.join(timeout=1.0)
        print("Voice processing pipeline stopped")
    
    def add_audio(self, audio_data: np.ndarray):
        """Add audio data to processing queue"""
        if not self.audio_queue.full():
            self.audio_queue.put(audio_data)
    
    def _process_audio(self):
        """Main audio processing loop"""
        while self.is_running:
            try:
                # Get audio from queue
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Detect voice activity
                activity_flags = self.vad.detect_voice_activity(audio_data)
                
                # If speech detected, process for recognition
                if any(activity_flags):
                    # Extract features
                    features = self.recognizer.extract_features(audio_data)
                    
                    # Recognize speech (this is simplified)
                    recognized_text = self.recognizer.recognize_speech(audio_data)
                    
                    if recognized_text.strip():
                        # Create voice command with confidence estimate
                        command = VoiceCommand(
                            text=recognized_text,
                            confidence=0.8,  # Placeholder confidence
                            timestamp=time.time()
                        )
                        
                        # Add to command queue
                        self.command_queue.put(command)
                        print(f"Recognized: {recognized_text}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing: {e}")
                continue
    
    def _process_commands(self):
        """Process recognized commands"""
        while self.is_running:
            try:
                command = self.command_queue.get(timeout=0.1)
                
                # Here you would implement command handling logic
                # For example, trigger robot actions based on recognized text
                response = self._handle_command(command.text)
                
                # Respond to command
                if response:
                    self.tts.speak(response)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in command processing: {e}")
                continue
    
    def _handle_command(self, text: str) -> str:
        """Handle recognized command and generate response"""
        text_lower = text.lower()
        
        # Simple command responses
        if any(word in text_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! How can I assist you today?"
        elif any(word in text_lower for word in ['how are you', 'hows it going']):
            return "I'm functioning well, thank you for asking!"
        elif any(word in text_lower for word in ['stop', 'halt', 'pause']):
            return "Stopping current operations."
        elif any(word in text_lower for word in ['help', 'what can you do']):
            return "I can respond to greetings, answer questions, and perform simple tasks."
        else:
            return f"I heard: {text}. Could you please repeat or rephrase?"
    
    def speak_response(self, text: str, emotion: str = 'neutral'):
        """Speak a response with appropriate emotion"""
        self.tts.speak_with_emotion(text, emotion)

# Example simulation
if __name__ == "__main__":
    pipeline = VoiceProcessingPipeline()
    
    # For demonstration, we'll simulate audio input
    # In a real robot, this would come from the microphone
    print("Starting voice processing pipeline simulation...")
    
    # Create simulated audio (in a real system, you'd capture from microphone)
    import scipy.io.wavfile as wavfile
    
    # Simulate "Hello robot" audio (this is just for demonstration)
    t = np.linspace(0, 1, 16000)
    simulated_audio = np.sin(2 * np.pi * 523 * t)  # Tone simulating "Hello"
    
    # Add to pipeline
    pipeline.add_audio(simulated_audio)
    
    # Start pipeline
    pipeline.start_listening()
    
    # Let it process for a few seconds
    time.sleep(3)
    
    # Stop pipeline
    pipeline.stop_listening()
```

## Advanced Voice Processing Techniques

### Noise Reduction and Audio Enhancement

```python
import scipy.signal as signal
from scipy import ndimage

class AudioEnhancer:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.noise_threshold = 0.01  # Threshold for noise detection
    
    def spectral_subtraction(self, audio_signal, noise_sample):
        """
        Reduce noise using spectral subtraction method
        """
        # Compute FFT of the signal and noise
        sig_fft = np.fft.fft(audio_signal)
        noise_fft = np.fft.fft(noise_sample)
        
        # Estimate noise spectrum
        noise_power = np.abs(noise_fft)**2
        signal_power = np.abs(sig_fft)**2
        
        # Subtract noise from signal
        enhanced_power = np.maximum(signal_power - noise_power, 0)
        
        # Reconstruct the signal
        enhanced_fft = np.sqrt(enhanced_power) * np.exp(1j * np.angle(sig_fft))
        enhanced_signal = np.real(np.fft.ifft(enhanced_fft))
        
        return enhanced_signal
    
    def wiener_filter(self, audio_signal, noise_variance=0.01):
        """
        Apply Wiener filter for noise reduction
        """
        # Compute FFT
        fft_signal = np.fft.fft(audio_signal)
        power_spectrum = np.abs(fft_signal)**2
        
        # Wiener filter
        wiener_filter = power_spectrum / (power_spectrum + noise_variance)
        
        # Apply filter
        enhanced_fft = fft_signal * wiener_filter
        enhanced_signal = np.real(np.fft.ifft(enhanced_fft))
        
        return enhanced_signal
```

### Speaker Recognition

```python
from sklearn.mixture import GaussianMixture
import librosa

class SpeakerRecognizer:
    def __init__(self, n_components=32):
        self.n_components = n_components
        self.models = {}  # Dictionary to store models for each speaker
        self.speakers = []  # List of known speakers
    
    def extract_speaker_features(self, audio_data, sample_rate=16000):
        """
        Extract speaker-discriminative features (MFCCs)
        """
        # Compute MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio_data, 
            sr=sample_rate, 
            n_mfcc=13,
            n_fft=2048,
            hop_length=512
        )
        
        # Transpose to get (n_frames, n_features)
        return mfccs.T
    
    def train_speaker_model(self, audio_data, speaker_id, sample_rate=16000):
        """
        Train a model for a specific speaker
        """
        # Extract features
        features = self.extract_speaker_features(audio_data, sample_rate)
        
        # Train Gaussian Mixture Model
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='diag',
            n_init=1
        )
        
        gmm.fit(features)
        
        # Store the model
        self.models[speaker_id] = gmm
        if speaker_id not in self.speakers:
            self.speakers.append(speaker_id)
    
    def identify_speaker(self, audio_data, sample_rate=16000):
        """
        Identify the speaker from the provided audio
        """
        if not self.models:
            return "unknown", 0.0
        
        # Extract features
        features = self.extract_speaker_features(audio_data, sample_rate)
        
        best_speaker = "unknown"
        best_score = float('-inf')
        
        # Compare against all known speakers
        for speaker_id, model in self.models.items():
            log_likelihood = model.score(features)
            if log_likelihood > best_score:
                best_score = log_likelihood
                best_speaker = speaker_id
        
        # Convert log likelihood to a probability-like score
        score = 1.0 / (1.0 + np.exp(-best_score)) if best_score > -100 else 0.0
        
        return best_speaker, min(score, 1.0)
```

## Troubleshooting Common Issues

### Audio Quality Problems

**Background Noise**:
- Implement noise reduction algorithms like spectral subtraction
- Use microphone arrays for beamforming
- Apply VAD to ignore non-speech segments

**Audio Clipping**:
- Monitor input levels and adjust gain
- Implement soft clipping algorithms
- Use automatic gain control

**Reverberation**:
- Apply dereverberation algorithms
- Use multiple microphones for spatial filtering
- Train models to be robust to reverberation

### Recognition Accuracy Issues

**Misrecognition**:
- Implement confidence scoring
- Use language models to improve context
- Add confirmation for critical commands

**Speaker Variability**:
- Use speaker adaptation techniques
- Include diverse voices in training data
- Implement speaker normalization

**Domain-Specific Terms**:
- Create custom language models
- Add domain-specific vocabulary
- Use phonetic dictionaries for proper names

## Best Practices

### Robust Design

- Implement multiple confirmation steps for critical commands
- Design graceful degradation when recognition confidence is low
- Provide alternative input methods for accessibility
- Consider privacy implications of voice data storage

### Performance Optimization

- Use lightweight models for real-time applications
- Optimize audio processing pipelines for minimal latency
- Implement caching for common responses
- Profile and optimize resource usage

### Safety Considerations

- Validate commands before execution
- Implement timeout mechanisms for unresponsive systems
- Design failsafes for misinterpreted commands
- Use multi-modal confirmation for critical actions

## Hands-on Exercise

1. **Audio Preprocessing Pipeline**: Implement a complete audio preprocessing pipeline that includes normalization, framing, windowing, and feature extraction.

2. **Voice Activity Detection**: Create a VAD system that can distinguish between speech and non-speech segments in noisy environments.

3. **Speaker Identification**: Implement a speaker recognition system that can identify different speakers and adapt its interaction style accordingly.

4. **Noise Reduction**: Apply noise reduction techniques to improve speech recognition performance in noisy environments.

5. **Response Generation**: Develop a system that generates appropriate speech responses based on context and robot state.

## Key Takeaways

- Voice processing in robotics must handle real-world acoustic challenges
- The pipeline includes multiple processing steps from audio capture to speech generation
- Robust performance requires addressing noise, speaker variation, and real-time constraints
- Safety and privacy considerations are paramount in voice-enabled robots
- Contextual understanding enhances the effectiveness of voice interactions

## Further Reading

- "Digital Processing of Speech Signals" by Rabiner and Schafer
- "Speech and Language Processing" by Jurafsky and Martin
- "Robust Automatic Speech Recognition" - Research papers
- "Voice User Interface Design for Robots" - HCI literature

## Next Steps

Continue to Chapter 3: Conversational Robotics to explore how voice processing integrates with broader conversational AI systems for natural human-robot interaction.