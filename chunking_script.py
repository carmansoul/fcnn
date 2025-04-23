import os
import numpy as np
import wave
import scipy.signal  # For resampling
import soundfile as sf

def normalize_energy(y, target_energy):
    current_energy = np.sum(y.astype(np.float64) ** 2)
    if current_energy < 1e-10:
        return y  # Avoid division by 0 for silent input
    scale = np.sqrt(target_energy / current_energy)
    return y * scale

def generate_wiener_noise(duration_ms, sample_rate, amplitude_db):
    """Generates a Wiener process (Brownian noise) and scales it to amplitude_db."""
    num_samples = (duration_ms * sample_rate) // 1000
    noise = np.cumsum(np.random.normal(0, 1, num_samples))  # Wiener process
    noise = noise - np.mean(noise)  # Center the noise
    amplitude = 10 ** (amplitude_db / 20)  # Convert dB to linear scale
    noise = (noise * amplitude).astype(np.float32)  # Scale without normalization
    return noise

def read_wav(file_path):
    """Reads a WAV file and returns the audio data as a numpy array."""
    with wave.open(file_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        audio_data = np.frombuffer(wav_file.readframes(num_frames), dtype=np.int16)
        return sample_rate, audio_data

def write_float_wav(file_path, sample_rate, audio_data):
    """Writes a 32-bit float WAV file."""
    sf.write(file_path, audio_data.astype(np.float32), sample_rate, subtype='FLOAT')
def resample_audio(audio_data, original_sr, target_sr):
    if original_sr == target_sr:
        return audio_data.astype(np.float32)
    return scipy.signal.resample_poly(audio_data, target_sr, original_sr).astype(np.float32)

def chunk_wav_files(input_directory, target_sample_rate=40000):
    """Chunks WAV files into 2-second segments from 10-second snippets, adds noise, resamples to 40 kHz, and saves them."""
    output_directory = "/home/cgreenway/FCNN/Chunked_Samples"
    os.makedirs(output_directory, exist_ok=True)

    pitch_labels = [
        "C1", "C#1", "D1", "D#1", "E1", "F1", "F#1", "G1", "G#1", "A1", "A#1", "B1",
        "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2", "A2", "A#2", "B2",
        "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3",
        "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
        "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5",
        "C6", "C#6", "D6", "D#6", "E6", "F6", "F#6", "G6", "G#6", "A6", "A#6", "B6",
        "C7", "C#7", "D7", "D#7", "E7", "F7", "F#7", "G7", "G#7", "A7", "A#7", "B7"
    ]

    for subdir in os.listdir(input_directory):
        if subdir == "Chunked_Samples":
            continue  # Skip already processed samples

        subdir_path = os.path.join(input_directory, subdir)
        if not os.path.isdir(subdir_path):
            continue

        for filename in sorted(os.listdir(subdir_path)):  # Process files in order
            if filename.lower().endswith(".wav"):
                file_path = os.path.join(subdir_path, filename)
                original_sample_rate, audio_data = read_wav(file_path)

                chunk_size_10s = original_sample_rate * 10  # 10 seconds worth of samples
                chunk_size_2s = original_sample_rate * 2   # 2-second chunks
                num_10s_chunks = len(audio_data) // chunk_size_10s  # Count full 10s snippets
                
                for i in range(min(num_10s_chunks, len(pitch_labels))):  # Up to 84 pitch-labeled 10s intervals
                    start_10s = i * chunk_size_10s
                    end_10s = start_10s + chunk_size_10s
                    chunk_10s = audio_data[start_10s:end_10s]

                    noise_10s = generate_wiener_noise(10_000, original_sample_rate, -18)

                    # Take only the first 2 seconds
                    chunk_2s = chunk_10s[:chunk_size_2s].astype(np.float32)
                    noise_2s = noise_10s[:chunk_size_2s].astype(np.float32)

                    chunk_2s = normalize_energy(chunk_2s, target_energy=0.5)
                    noise_2s = normalize_energy(noise_2s, target_energy=0.05)

                    chunk_with_noise = chunk_2s + noise_2s
                    chunk_resampled = resample_audio(chunk_with_noise, original_sample_rate, target_sample_rate)
                    # Save with part{i+1} to match 10s block
                    # Replace spaces with underscores in the base filename
                    safe_filename = filename.replace(" ", "_").replace("(", "").replace(")", "")
                    chunk_filename = f"{pitch_labels[i]}_{subdir}_{safe_filename[:-4]}_part{i+1}.wav"
                    chunk_path = os.path.join(output_directory, chunk_filename)
                    write_float_wav(chunk_path, target_sample_rate, chunk_resampled)

if __name__ == "__main__":
    input_dir = "/home/cgreenway/FCNN"
    chunk_wav_files(input_dir)
print('Chunking complete!')
