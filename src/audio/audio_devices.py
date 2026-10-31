"""Audio input device discovery and interactive selection"""

from .audio_deps import AUDIO_AVAILABLE, AUDIO_ERRORS, sd


def list_input_devices() -> list[dict]:
    """List all available input devices"""
    if not AUDIO_AVAILABLE:
        print("Audio libraries not available.")
        return []
    try:
        devices = sd.query_devices()
    except AUDIO_ERRORS as e:
        print(f"Error listing devices: {e}")
        return []
    return [
        {
            "index": i,
            "name": device["name"],
            "channels": device["max_input_channels"],
            "sample_rate": device["default_samplerate"],
        }
        for i, device in enumerate(devices)
        if device["max_input_channels"] > 0
    ]


def print_input_devices() -> list[dict] | None:
    """Print all available input devices in a formatted way"""
    devices = list_input_devices()
    if not devices:
        print("No input devices found.")
        return None

    print("\nAvailable audio input devices:")
    print("-" * 70)
    for device in devices:
        print(f"  [{device['index']}] {device['name']}")
        print(f"      Channels: {device['channels']}, Sample Rate: {int(device['sample_rate'])} Hz")
    print("-" * 70)
    return devices


def _parse_choice(choice: str, devices: list[dict], default_device) -> tuple[bool, int | None]:
    """Interpret one answer; returns (done, selected index)"""
    if choice.lower() == "q":
        return True, None
    if not choice:
        return True, default_device
    try:
        device_index = int(choice)
    except ValueError:
        print("Invalid input. Please enter a number or press Enter for default.")
        return False, None
    if 0 <= device_index < len(devices):
        selected = devices[device_index]
        print(f"Selected: [{selected['index']}] {selected['name']}")
        return True, selected["index"]
    print(f"Invalid choice. Please enter a number between 0 and {len(devices) - 1}.")
    return False, None


def select_device_interactive() -> int | None:
    """Interactively select an input device"""
    devices = print_input_devices()
    if not devices:
        return None

    default_device = sd.default.device[0] if AUDIO_AVAILABLE else None
    if default_device is not None:
        print(f"\nDefault device: [{default_device}]")

    while True:
        try:
            choice = input(f"\nSelect input device [0-{len(devices) - 1}] (Enter for default, 'q' to quit): ").strip()
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None
        done, selected = _parse_choice(choice, devices, default_device)
        if done:
            return selected
