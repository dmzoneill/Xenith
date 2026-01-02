#!/usr/bin/env python3
"""Test LLM backends"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Test LLM backends")
    parser.add_argument(
        "--model",
        default="qwen2.5-1.5b",
        help="Model to test (default: qwen2.5-1.5b)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu, gpu, npu (default: auto)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the model without testing",
    )
    parser.add_argument(
        "--prompt",
        default="What is 2 + 2?",
        help="Test prompt",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("LLM Backend Test")
    print("=" * 60)

    # Import LLM backends
    try:
        from src.audio.llm_backends import (
            get_llm_backend,
            list_available_backends,
            list_available_models,
            print_llm_status,
        )

        print("\n✓ LLM backends module imported successfully")
    except ImportError as e:
        print(f"\n✗ Failed to import LLM backends: {e}")
        sys.exit(1)

    # Print status
    print_llm_status()

    # List backends
    backends = list_available_backends()
    print(f"\nAvailable backends: {backends}")

    if not backends:
        print("\n⚠ No LLM backends available!")
        print("Install openvino-genai: pip install openvino-genai")
        sys.exit(1)

    # Download only mode
    if args.download:
        print(f"\n--- Downloading {args.model} ---")
        from src.audio.llm_backends.openvino_backend import download_model

        path = download_model(args.model)
        print(f"✓ Model downloaded to: {path}")
        return

    # Test model
    print(f"\n--- Testing {args.model} on {args.device} ---")
    llm = get_llm_backend(model=args.model, device=args.device)

    if not llm:
        print("✗ Failed to create backend")
        sys.exit(1)

    print(f"Backend: {llm}")

    print("\nLoading model (this may take a while on first run)...")
    if not llm.load():
        print("✗ Failed to load model")
        sys.exit(1)

    print("✓ Model loaded successfully")

    # Test generation
    print(f"\nTest prompt: '{args.prompt}'")
    print("-" * 40)

    result = llm.generate(args.prompt)

    print(f"Response: {result.text}")
    print("-" * 40)
    print(f"Finish reason: {result.finish_reason}")
    print(f"Model: {result.model}")

    if result.finish_reason == "error":
        print("\n✗ Generation failed")
        sys.exit(1)

    print("\n✓ LLM test PASSED!")

    # Cleanup
    llm.unload()

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

