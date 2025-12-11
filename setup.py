from setuptools import setup, find_packages

setup(
    name="xenith",
    version="0.1.0",
    description="Highly interoperable Linux desktop AI agent with plugin architecture",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "PyGObject>=3.42.0",
        "cairo>=1.21.0",
        "openai-whisper>=20231117",  # Local voice-to-text (default)
        "torch>=2.0.0",  # Required by Whisper
        "torchaudio>=2.0.0",  # Required by Whisper
        # "pyaudio>=0.2.14",  # Optional - requires portaudio-devel. We use sounddevice instead.
        "sounddevice>=0.4.6",
        "numpy>=1.24.0",
        "openai>=1.3.0",
        "anthropic>=0.7.0",
        "google-generativeai>=0.3.0",
        "pyttsx3>=2.90",
        "gTTS>=2.4.0",
        "PyYAML>=6.0",
        "jsonschema>=4.19.0",
        # "python-dbus>=1.3.2",  # Not available for Python 3.13 - install via system package
        "pyatspi>=2.46.0",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "xenith=main:main",
        ],
    },
)



