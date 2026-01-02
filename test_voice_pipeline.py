#!/usr/bin/env python3
"""Test the complete voice pipeline: STT → LLM → TTS"""

import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Test voice pipeline")
    parser.add_argument(
        "--text",
        default="What time is it?",
        help="Text to process (skip STT)",
    )
    parser.add_argument(
        "--audio",
        help="Audio file to process (use STT)",
    )
    parser.add_argument(
        "--output",
        default="/tmp/pipeline_output.wav",
        help="Output audio file path",
    )
    parser.add_argument(
        "--llm-model",
        default="qwen2.5-1.5b",
        help="LLM model to use",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play output audio",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Voice Pipeline Test")
    print("=" * 60)

    # Import pipeline
    try:
        from src.audio.voice_pipeline import VoicePipeline

        print("\n✓ Voice pipeline module imported")
    except ImportError as e:
        print(f"\n✗ Failed to import voice pipeline: {e}")
        sys.exit(1)

    # Create pipeline
    print("\n--- Creating Pipeline ---")
    pipeline = VoicePipeline(
        stt_device="auto",
        llm_model=args.llm_model,
        llm_device="auto",
        tts_voice="EN-Default",
    )

    # Load pipeline
    print("\n--- Loading Pipeline ---")
    if not pipeline.load():
        print("✗ Failed to load pipeline")
        sys.exit(1)

    # Get status
    status = pipeline.get_status()
    print(f"\nPipeline Status:")
    print(f"  STT: {status['stt']}")
    print(f"  LLM: {status['llm']}")
    print(f"  TTS: {status['tts']}")

    # Process
    output_path = Path(args.output)

    if args.audio:
        # Process audio file
        print(f"\n--- Processing Audio: {args.audio} ---")
        import numpy as np
        import wave

        with wave.open(args.audio, "rb") as wf:
            audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0
            sample_rate = wf.getframerate()

        result = pipeline.process(audio_data, sample_rate, output_path)
    else:
        # Process text (skip STT)
        print(f"\n--- Processing Text: '{args.text}' ---")
        result = pipeline.process_text(args.text, output_path)

    # Show results
    print("\n--- Results ---")
    print(f"  Success: {result.success}")
    if result.error:
        print(f"  Error: {result.error}")
    print(f"  Input: '{result.transcribed_text}'")
    print(f"  Response: '{result.llm_response}'")
    print(f"  Audio: {result.audio_path}")
    print(f"\n  Timing:")
    print(f"    STT: {result.stt_time:.2f}s")
    print(f"    LLM: {result.llm_time:.2f}s")
    print(f"    TTS: {result.tts_time:.2f}s")
    print(f"    Total: {result.total_time:.2f}s")

    if result.success:
        print("\n✓ Pipeline test PASSED!")

        # Play audio if requested
        if args.play and result.audio_path and result.audio_path.exists():
            print(f"\nPlaying audio: {result.audio_path}")
            import subprocess

            subprocess.run(["paplay", str(result.audio_path)], check=False)
    else:
        print("\n✗ Pipeline test FAILED")
        sys.exit(1)

    # Cleanup
    pipeline.unload()

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

