#!/usr/bin/env python3
"""Test TTS backends"""

import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("TTS Backend Test")
    print("=" * 60)

    # Import TTS backends
    try:
        from src.audio.tts_backends import (
            get_tts_backend,
            list_available_backends,
            print_tts_status,
        )

        print("\n✓ TTS backends module imported successfully")
    except ImportError as e:
        print(f"\n✗ Failed to import TTS backends: {e}")
        sys.exit(1)

    # Print status
    print_tts_status()

    # List backends
    backends = list_available_backends()
    print(f"\nAvailable backends: {backends}")

    if not backends:
        print("\n⚠ No TTS backends available!")
        print("Make sure MeloTTS.cpp is built in vendor/MeloTTS.cpp/build/")
        sys.exit(1)

    # Test auto backend selection
    print("\n" + "-" * 40)
    print("Testing auto backend selection...")
    tts = get_tts_backend()
    if tts:
        print(f"  Selected: {tts}")
        if tts.load():
            print("  ✓ Backend loaded successfully")

            # Test synthesis
            print("\n  Testing synthesis...")
            try:
                result = tts.synthesize(
                    "Hello! This is a test of the MeloTTS text to speech system.",
                    output_path=Path("/tmp/tts_test_output.wav"),
                )
                print(f"  ✓ Audio generated: {result.audio_path}")
                print(f"    Voice: {result.voice}")
                print(f"    Sample rate: {result.sample_rate}")

                # Check file exists
                if result.audio_path and result.audio_path.exists():
                    size = result.audio_path.stat().st_size
                    print(f"    File size: {size:,} bytes")
                    print("\n  ✓ TTS test PASSED!")
                else:
                    print("  ✗ Output file not found")
            except Exception as e:
                print(f"  ✗ Synthesis failed: {e}")
        else:
            print("  ✗ Backend failed to load")
    else:
        print("  ✗ No backend available")

    # Test specific voices
    print("\n" + "-" * 40)
    print("Available voices:")
    if tts:
        for voice in tts.get_available_voices():
            print(f"  - {voice}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

