#!/usr/bin/env python3
"""Test streaming voice pipeline

Compares:
- Sequential processing (old) - wait for full response
- Streaming with sentence buffering (new) - hear first sentence immediately

Optimizations:
- LLM token streaming with sentence buffering
- Parallel TTS synthesis for multiple sentences
- Single-speaker TTS (5x faster than all speakers)
- Real-time audio playback
"""

import sys
import argparse
import time


def main():
    parser = argparse.ArgumentParser(description="Test streaming pipeline")
    parser.add_argument(
        "--prompt",
        default="Tell me three interesting facts about space.",
        help="Prompt to test",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare streaming vs sequential",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-1.5b",
        help="LLM model",
    )
    parser.add_argument(
        "--device",
        default="CPU",
        help="LLM device (CPU, NPU, GPU)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Streaming Pipeline Test")
    print("=" * 70)

    from src.audio.streaming_pipeline import StreamingPipeline, StreamingConfig

    config = StreamingConfig(
        llm_model=args.model,
        llm_device=args.device,
        use_ram_disk=True,
    )

    pipeline = StreamingPipeline(config)

    print(f"\nPrompt: '{args.prompt}'")
    print("-" * 70)

    # Callbacks for progress
    tokens = []

    def on_token(token):
        tokens.append(token)
        print(token, end="", flush=True)

    def on_sentence(sentence):
        print(
            f"\n  [TTS queued: '{sentence[:50]}...']"
            if len(sentence) > 50
            else f"\n  [TTS queued: '{sentence}']"
        )

    first_audio_time = [None]  # Use list to allow closure modification

    def on_audio_start():
        first_audio_time[0] = time.time()
        print("\n  🔊 First audio playing!")

    if args.compare:
        # Test sequential first
        print("\n--- Sequential Mode ---")
        seq_start = time.time()
        seq_result = pipeline.process_simple(args.prompt)
        seq_total = time.time() - seq_start
        print(f"\nResponse: {seq_result['response']}")
        print(f"\nSequential timing:")
        print(f"  LLM: {seq_result['timing']['llm']:.2f}s")
        print(f"  TTS: {seq_result['timing']['tts']:.2f}s")
        print(f"  Total: {seq_total:.2f}s")

        # Wait for audio to finish
        print("\n[Waiting for audio playback...]")
        time.sleep(5)

        print("\n--- Streaming Mode ---")

    # Test streaming
    stream_start = time.time()
    print("\nLLM streaming: ", end="")
    result = pipeline.process_streaming(
        args.prompt,
        on_token=on_token,
        on_sentence=on_sentence,
        on_audio_start=on_audio_start,
    )
    stream_total = time.time() - stream_start

    print(f"\n\n--- Results ---")
    print(f"Response: {result['response']}")
    print(f"Sentences: {len(result['sentences'])}")
    time_to_first_audio = (
        first_audio_time[0] - stream_start if first_audio_time[0] else None
    )

    print(f"\nStreaming timing:")
    print(
        f"  First token: {result['timing']['first_token']:.2f}s"
        if result["timing"]["first_token"]
        else "  First token: N/A"
    )
    if time_to_first_audio:
        print(f"  ⚡ First audio: {time_to_first_audio:.2f}s  ← KEY METRIC")
    print(f"  LLM: {result['timing']['llm']:.2f}s")
    print(f"  TTS: {result['timing']['tts']:.2f}s")
    print(f"  Total: {stream_total:.2f}s")

    if args.compare:
        print(f"\n--- Comparison ---")
        print(f"Sequential total: {seq_total:.2f}s (user waits this long)")
        if time_to_first_audio:
            print(f"Streaming first audio: {time_to_first_audio:.2f}s (user hears response)")
            perceived_improvement = (seq_total - time_to_first_audio) / seq_total * 100
            print(f"Perceived latency improvement: {perceived_improvement:.1f}%")

    # Wait for audio to finish
    print("\n[Waiting for audio playback to complete...]")
    time.sleep(10)

    pipeline.unload()

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
