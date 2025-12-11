#!/usr/bin/env python3
"""Test script to list and test audio input devices"""

import sys
from src.audio.voice_input import VoiceInput

def test_device_listing():
    """Test listing audio devices"""
    print("Testing audio device listing...")
    devices = VoiceInput.list_input_devices()
    
    if not devices:
        print("No input devices found.")
        return
    
    print(f"\nFound {len(devices)} input device(s):")
    for device in devices:
        print(f"  [{device['index']}] {device['name']}")
        print(f"      Channels: {device['channels']}, Sample Rate: {int(device['sample_rate'])} Hz")
    
    return devices

def test_device_selection():
    """Test interactive device selection"""
    print("\nTesting interactive device selection...")
    device = VoiceInput.select_device_interactive()
    
    if device is not None:
        print(f"\nSelected device index: {device}")
        return device
    else:
        print("\nNo device selected.")
        return None

def test_audio_input(device_index: int):
    """Test audio input from selected device"""
    print(f"\nTesting audio input from device [{device_index}]...")
    print("Speak into your microphone. Press Ctrl+C to stop.")
    
    try:
        import numpy as np
        import time as time_module
        
        voice_input = VoiceInput(device=device_index)
        
        # Track audio levels
        last_print_time = [time_module.time()]
        
        # Override the callback to show audio levels
        original_callback = voice_input._audio_callback
        
        def test_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio callback status: {status}")
            
            audio_data = indata[:, 0]
            energy = np.abs(audio_data).mean()
            
            # Print energy level every 0.5 seconds
            current_time = time_module.time()
            if current_time - last_print_time[0] >= 0.5:
                if energy > 0.001:
                    print(f"Audio level: {energy:.4f} {'█' * int(energy * 100)}")
                last_print_time[0] = current_time
            
            # Call original callback
            original_callback(indata, frames, time_info, status)
        
        voice_input._audio_callback = test_callback
        voice_input.start_listening()
        
        # Keep running
        try:
            while True:
                time_module.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            voice_input.cleanup()
            
    except Exception as e:
        print(f"Error testing audio input: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("=" * 70)
    print("Audio Device Test")
    print("=" * 70)
    
    # Test 1: List devices
    devices = test_device_listing()
    
    if not devices:
        print("\nNo devices available. Exiting.")
        return 1
    
    # Test 2: Interactive selection
    selected = test_device_selection()
    
    if selected is not None:
        # Test 3: Test audio input
        response = input("\nTest audio input with selected device? (y/n): ").strip().lower()
        if response == 'y':
            test_audio_input(selected)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

