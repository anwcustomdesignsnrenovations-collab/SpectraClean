
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft


MAX_SECONDS = 120 * 60
CHUNK_SECONDS = 10


def clean_audio(input_file, output_file, strength=1.5):
    info = sf.info(input_file)

    if info.frames / info.samplerate > MAX_SECONDS:
        raise ValueError("Audio must be 120 minutes or shorter.")

    rate = info.samplerate
    channels = info.channels
    chunk_size = rate * CHUNK_SECONDS
    window_size = 2048 if rate >= 32000 else 1024
    overlap = window_size * 3 // 4

    # Estimate background noise from the first second.
    with sf.SoundFile(input_file) as source:
        noise_frames = min(rate, len(source))
        noise_sample = source.read(
            noise_frames, dtype="float32", always_2d=True
        )

        if len(noise_sample) == 0:
            raise ValueError("The input audio is empty.")

        noise_sample = noise_sample.T
        _, _, noise_spec = stft(
            noise_sample,
            fs=rate,
            nperseg=window_size,
            noverlap=overlap,
            axis=-1,
        )
        noise_level = np.mean(np.abs(noise_spec), axis=-1)
        noise_level = noise_level[:, :, None]

        with sf.SoundFile(
            output_file, mode="w", samplerate=rate,
            channels=channels, subtype="PCM_16"
        ) as target:
            source.seek(0)

            while True:
                audio = source.read(
                    chunk_size, dtype="float32", always_2d=True
                )
                if len(audio) == 0:
                    break

                block = audio.T
                _, times, spectrum = stft(
                    block,
                    fs=rate,
                    nperseg=window_size,
                    noverlap=overlap,
                    axis=-1,
                    boundary="zeros",
                )

                magnitude = np.abs(spectrum)
                phase = np.exp(1j * np.angle(spectrum))

                floor = noise_level * strength
                mask = np.clip(
                    (magnitude - floor) /
                    (magnitude + 1e-10),
                    0.08, 1.0
                )

                cleaned = spectrum * mask
                _, restored = istft(
                    cleaned,
                    fs=rate,
                    nperseg=window_size,
                    noverlap=overlap,
                    input_onesided=True,
                    boundary=True,
                    axis=-1,
                )

                restored = restored[:, :len(audio)].T
                target.write(restored)

    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="SpectraClean audio noise reduction"
    )
    parser.add_argument("input", help="Input WAV, FLAC, or supported audio file")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output audio filename"
    )
    parser.add_argument(
        "--strength", type=float, default=1.5,
        help="Noise reduction strength (default: 1.5)"
    )
    args = parser.parse_args()

    source = Path(args.input)
    output = args.output or str(
        source.with_name(source.stem + "_cleaned.wav")
    )

    clean_audio(source, output, args.strength)
    print(f"Cleaned audio saved to: {output}")


if __name__ == "__main__":
    main()
