"""Voice input handler with wake word detection"""

try:
    import numpy as np
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    np = None
    sd = None
    print("Warning: sounddevice or numpy not available. Voice input will be disabled.")

from typing import Optional, Callable
import threading
import queue
import time as time_module


class VoiceInput:
    """Handles voice input with wake word detection"""
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024, wake_word: str = "hi", device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.wake_word = wake_word.lower()
        self.device = device  # Audio input device index
        self.is_listening_for_wake_word = False
        self.is_listening_for_command = False
        self.is_processing = False
        self._audio_queue = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._wake_word_thread: Optional[threading.Thread] = None
        
        # Callbacks for state changes
        self.on_state_change: Optional[Callable[[str], None]] = None
        self.on_transcript: Optional[Callable[[str], None]] = None
        self.on_wake_word_detected: Optional[Callable[[], None]] = None
        
        # Voice activity detection
        self.energy_threshold = 0.01  # Adjust based on your microphone
        self.silence_duration = 1.0  # Seconds of silence before stopping
        self._last_voice_time = 0.0
        self._command_listening_start_time = 0.0  # Track when we started listening for command
        self._audio_buffer = []
        
        # Initialize time tracking
        self._last_voice_time = time_module.time()
        
        # Wake word detection buffer (keep last few seconds for analysis)
        self._wake_word_buffer = []
        self._wake_word_buffer_duration = 3.0  # Keep 3 seconds of audio
        self._wake_word_buffer_max_samples = int(sample_rate * self._wake_word_buffer_duration)
        
        # Wake word detection using Whisper (if available)
        self._whisper_model = None
        self._whisper_device = "cpu"  # Device for Whisper (cpu or cuda)
        self._wake_word_check_interval = 2.0  # Check for wake word every 2 seconds
        self._last_wake_word_check = 0.0
    
    @staticmethod
    def list_input_devices():
        """List all available input devices"""
        if not AUDIO_AVAILABLE:
            print("Audio libraries not available.")
            return []
        
        try:
            devices = sd.query_devices()
            input_devices = []
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append({
                        'index': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate']
                    })
            return input_devices
        except Exception as e:
            print(f"Error listing devices: {e}")
            return []
    
    @staticmethod
    def print_input_devices():
        """Print all available input devices in a formatted way"""
        devices = VoiceInput.list_input_devices()
        if not devices:
            print("No input devices found.")
            return
        
        print("\nAvailable audio input devices:")
        print("-" * 70)
        for device in devices:
            print(f"  [{device['index']}] {device['name']}")
            print(f"      Channels: {device['channels']}, Sample Rate: {int(device['sample_rate'])} Hz")
        print("-" * 70)
        return devices
    
    @staticmethod
    def select_device_interactive() -> Optional[int]:
        """Interactively select an input device"""
        devices = VoiceInput.print_input_devices()
        if not devices:
            return None
        
        default_device = sd.default.device[0] if AUDIO_AVAILABLE else None
        if default_device is not None:
            print(f"\nDefault device: [{default_device}]")
        
        while True:
            try:
                choice = input(f"\nSelect input device [0-{len(devices)-1}] (Enter for default, 'q' to quit): ").strip()
                if choice.lower() == 'q':
                    return None
                if not choice:
                    return default_device
                device_index = int(choice)
                if 0 <= device_index < len(devices):
                    selected = devices[device_index]
                    print(f"Selected: [{selected['index']}] {selected['name']}")
                    return selected['index']
                else:
                    print(f"Invalid choice. Please enter a number between 0 and {len(devices)-1}.")
            except ValueError:
                print("Invalid input. Please enter a number or press Enter for default.")
            except KeyboardInterrupt:
                print("\nCancelled.")
                return None
    
    def _load_whisper(self):
        """Load Whisper model for wake word detection with GPU acceleration if available"""
        try:
            import whisper
            import torch
            
            # Check for CUDA/GPU availability
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
                print(f"[WHISPER] GPU acceleration available: {torch.cuda.get_device_name(0)}")
                print(f"[WHISPER] CUDA version: {torch.version.cuda}")
            else:
                print("[WHISPER] GPU not available, using CPU")
            
            # Load base model (smaller, faster for wake word detection)
            # Whisper will automatically use GPU if available
            self._whisper_model = whisper.load_model("base", device=device)
            self._whisper_device = device
            
            if device == "cuda":
                print(f"[WHISPER] Model loaded on GPU for faster transcription")
            else:
                print(f"[WHISPER] Model loaded on CPU")
            
            return True
        except ImportError:
            print("Warning: OpenAI Whisper not available. Wake word detection will use simple pattern matching.")
            return False
        except Exception as e:
            print(f"Warning: Could not load Whisper model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_listening(self):
        """Start listening for wake word (always-on mode)"""
        if not AUDIO_AVAILABLE:
            print("Audio libraries not available. Cannot start voice input.")
            return
        
        if self._stream is not None:
            return  # Already listening
        
        self.is_listening_for_wake_word = True
        self.is_listening_for_command = False
        self._audio_buffer = []
        self._wake_word_buffer = []
        
        # Load Whisper if available
        self._load_whisper()
        
        # Start in idle state - waiting for wake word
        if self.on_state_change:
            self.on_state_change('idle')
        
        # Start audio stream for wake word detection
        try:
            stream_kwargs = {
                'samplerate': self.sample_rate,
                'channels': 1,
                'dtype': 'float32',
                'blocksize': self.chunk_size,
                'callback': self._audio_callback
            }
            
            # Add device if specified
            if self.device is not None:
                stream_kwargs['device'] = self.device
                try:
                    device_info = sd.query_devices(self.device)
                    print(f"Using input device: [{self.device}] {device_info['name']}")
                except Exception as e:
                    print(f"Warning: Could not get device info: {e}")
            
            self._stream = sd.InputStream(**stream_kwargs)
            self._stream.start()
            print(f"[INIT] ✓ Audio stream started")
            print(f"[INIT] Listening for wake word: '{self.wake_word}'")
            print(f"[INIT] Sample rate: {self.sample_rate} Hz, Chunk size: {self.chunk_size}")
            print(f"[INIT] Whisper model: {'Loaded' if self._whisper_model else 'Not available'}")
            print(f"[INIT] Audio callback active - you should see [AUDIO] messages every 2 seconds")
            
            # Start wake word detection thread
            self._wake_word_thread = threading.Thread(target=self._wake_word_detection_loop, daemon=True)
            self._wake_word_thread.start()
            print(f"[INIT] ✓ Wake word detection thread started")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self.is_listening_for_wake_word = False
            if self.on_state_change:
                self.on_state_change('idle')
    
    def stop_listening(self):
        """Stop listening for voice input"""
        self.is_listening_for_wake_word = False
        self.is_listening_for_command = False
        
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except:
                pass
            self._stream = None
        
        # Notify state change
        if self.on_state_change:
            self.on_state_change('idle')
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input"""
        if status:
            print(f"[AUDIO] Callback status: {status}")
        
        if not AUDIO_AVAILABLE:
            return
        
        audio_data = indata[:, 0]
        energy = np.abs(audio_data).mean()
        current_time = time_module.time()
        
        # Debug: Print audio levels periodically (every 10 seconds, only if significant activity)
        if not hasattr(self, '_last_debug_print'):
            self._last_debug_print = current_time
            self._debug_counter = 0
        
        self._debug_counter += 1
        if current_time - self._last_debug_print >= 10.0:  # Every 10 seconds
            # Only print if there's significant audio activity
            if energy > 0.001:
                buffer_info = ""
                if self.is_listening_for_wake_word:
                    buffer_duration = len(self._wake_word_buffer) / self.sample_rate
                    buffer_info = f" | Buffer: {buffer_duration:.1f}s"
                if self.is_listening_for_command:
                    buffer_duration = len(self._audio_buffer) / self.sample_rate
                    buffer_info = f" | Recording: {buffer_duration:.1f}s"
                
                print(f"[AUDIO] Level: {energy:.4f}{buffer_info}")
            self._last_debug_print = current_time
            self._debug_counter = 0
        
        # Always maintain wake word buffer (for wake word detection)
        if self.is_listening_for_wake_word:
            self._wake_word_buffer.extend(audio_data.tolist())
            # Keep buffer size limited
            if len(self._wake_word_buffer) > self._wake_word_buffer_max_samples:
                # Remove oldest samples
                excess = len(self._wake_word_buffer) - self._wake_word_buffer_max_samples
                self._wake_word_buffer = self._wake_word_buffer[excess:]
        
        # If listening for command (after wake word detected)
        if self.is_listening_for_command:
            # Ignore audio for a brief moment after wake word detection
            # This prevents the wake word itself from being processed as a command
            if hasattr(self, '_ignore_audio_until') and current_time < self._ignore_audio_until:
                return  # Skip this audio chunk
            
            if energy > self.energy_threshold:
                self._last_voice_time = current_time
                self._audio_buffer.extend(audio_data.tolist())
                # Only print when starting to capture
                if len(self._audio_buffer) == len(audio_data):
                    print(f"[COMMAND] Listening...")
            else:
                # Check if we've had enough silence
                silence_duration = current_time - self._last_voice_time
                if silence_duration > self.silence_duration and len(self._audio_buffer) > 0:
                    # Prevent multiple rapid processing calls
                    if not self.is_processing:
                        duration = len(self._audio_buffer) / self.sample_rate
                        # Only process if we have meaningful audio (at least 0.5 seconds)
                        # This prevents processing very short audio clips that might be noise or wake word echo
                        if duration >= 0.5:
                            print(f"[COMMAND] Processing {duration:.1f}s of audio...")
                            # Process the audio command
                            self._process_audio()
                        else:
                            # Audio too short - but don't give up immediately
                            # Wait at least 5 seconds after wake word before giving up on short audio
                            # This gives user time to actually speak
                            time_since_listening_start = current_time - self._command_listening_start_time
                            
                            if time_since_listening_start < 5.0:
                                # Too early to give up - user might still be speaking
                                # Clear the buffer but keep listening
                                print(f"[COMMAND] Audio too short ({duration:.1f}s), waiting for more...")
                                self._audio_buffer = []
                                # Reset voice time to continue listening
                                self._last_voice_time = current_time
                            else:
                                # Been listening for 5+ seconds with only short audio - give up
                                print(f"[COMMAND] Audio too short ({duration:.1f}s) after {time_since_listening_start:.1f}s, returning to idle...")
                                self._audio_buffer = []
                                self.is_listening_for_command = False
                                # Return to wake word listening
                                self.is_listening_for_wake_word = True
                                if self.on_state_change:
                                    self.on_state_change('idle')
    
    def _wake_word_detection_loop(self):
        """Continuously check for wake word in background thread"""
        print("[WAKE WORD] Detection active - listening for '{}'".format(self.wake_word))
        check_count = 0
        while self.is_listening_for_wake_word and self._stream is not None:
            try:
                current_time = time_module.time()
                
                # Check for wake word periodically
                if current_time - self._last_wake_word_check >= self._wake_word_check_interval:
                    self._last_wake_word_check = current_time
                    check_count += 1
                    
                    buffer_size = len(self._wake_word_buffer)
                    buffer_duration = buffer_size / self.sample_rate if self.sample_rate > 0 else 0
                    
                    # Only print status every 10 checks (every ~20 seconds) or on first check
                    if check_count == 1 or check_count % 10 == 0:
                        print(f"[WAKE WORD] Status: {buffer_duration:.1f}s buffer ready")
                    
                    if len(self._wake_word_buffer) > self.sample_rate:  # At least 1 second of audio
                        # Check for wake word in background (non-blocking)
                        # This runs in a separate thread, so it won't block UI
                        if self._check_wake_word():
                            self._on_wake_word_detected()
                    elif check_count == 1:
                        # Only print "not enough audio" on first check
                        print(f"[WAKE WORD] Waiting for audio... (need 1.0s, have {buffer_duration:.1f}s)")
                
                time_module.sleep(1.0)  # Check every 1 second (reduced from 0.5s to save CPU)
            except Exception as e:
                print(f"[WAKE WORD] Error: {e}")
                import traceback
                traceback.print_exc()
                time_module.sleep(1.0)
    
    def _check_wake_word(self) -> bool:
        """Check if wake word is present in buffer"""
        if not AUDIO_AVAILABLE or len(self._wake_word_buffer) < self.sample_rate:
            return False
        
        # Use Whisper if available
        if self._whisper_model:
            try:
                # Get recent audio (last 2 seconds)
                recent_samples = min(int(self.sample_rate * 2), len(self._wake_word_buffer))
                audio_array = np.array(self._wake_word_buffer[-recent_samples:], dtype=np.float32)
                
                # Check audio level
                audio_level = np.abs(audio_array).mean()
                
                # Only transcribe if audio level is significant (increased threshold to save CPU/GPU)
                # Skip Whisper transcription if audio is too quiet - saves significant resources
                if audio_level < 0.001:  # Increased from 0.0005 to reduce unnecessary Whisper calls
                    return False  # Too quiet, skip transcription
                
                # Normalize audio
                max_abs = np.max(np.abs(audio_array))
                if max_abs > 0:
                    audio_array = audio_array / max_abs
                else:
                    return False
                
                # Transcribe with Whisper
                result = self._whisper_model.transcribe(audio_array, language="en", fp16=False)
                transcript = result["text"].strip()
                transcript_lower = transcript.lower().strip()
                
                # Always print what was heard (for debugging)
                if transcript and len(transcript.strip()) > 0:
                    # Highlight if wake word is found
                    if self.wake_word in transcript_lower:
                        print(f"[WAKE WORD] ✓ Heard: '{transcript}' ← WAKE WORD DETECTED!")
                    else:
                        print(f"[WAKE WORD] Heard: '{transcript}'")
                else:
                    print(f"[WAKE WORD] (silence or unclear)")
                
                # Check if wake word is in transcript
                if self.wake_word in transcript_lower:
                    return True
            except Exception as e:
                print(f"[WAKE WORD] Transcription error: {e}")
                import traceback
                traceback.print_exc()
        else:
            # Fallback: simple energy-based detection (less reliable)
            # Only print warning once
            if not hasattr(self, '_whisper_warning_printed'):
                print("[WAKE WORD] ⚠ Whisper not available - wake word detection disabled")
                print("[WAKE WORD] Install with: pip3 install openai-whisper")
                self._whisper_warning_printed = True
            return False
        
        return False
    
    def _on_wake_word_detected(self):
        """Handle wake word detection"""
        # Prevent multiple rapid detections (debounce)
        current_time = time_module.time()
        if hasattr(self, '_last_wake_word_detection_time'):
            if current_time - self._last_wake_word_detection_time < 3.0:  # 3 second debounce
                return  # Ignore rapid repeated detections
        self._last_wake_word_detection_time = current_time
        
        # Don't process if we're already listening for a command
        if self.is_listening_for_command:
            return
        
        print(f"Wake word '{self.wake_word}' detected!")
        
        # Switch from wake word listening to command listening
        self.is_listening_for_wake_word = False
        self.is_listening_for_command = True
        
        # CRITICAL: Clear audio buffer and reset voice time to prevent processing wake word as command
        # The buffer might contain the "hi" or other audio that triggered the wake word
        self._audio_buffer = []
        self._last_voice_time = time_module.time()
        self._command_listening_start_time = time_module.time()  # Track when we started listening
        
        # Add a small delay before starting to listen for commands
        # This prevents the wake word itself from being processed as a command
        # We'll set a flag to ignore audio for a brief moment
        self._ignore_audio_until = time_module.time() + 1.0  # Ignore audio for 1 second after wake word
        
        # CRITICAL: Clear the wake word buffer to prevent immediate re-detection
        # The buffer still contains the "Hi" that triggered this detection
        self._wake_word_buffer = []
        print(f"[WAKE WORD] Buffer cleared to prevent re-detection")
        
        # Notify wake word detected
        if self.on_wake_word_detected:
            self.on_wake_word_detected()
        
        # Change state to listening
        if self.on_state_change:
            self.on_state_change('listening')
        
        # After command is processed, return to wake word listening
        # Only start this if we're not already listening for a command
        def return_to_wake_word_listening():
            # Wait 10 seconds, but check periodically if user is actually speaking
            wait_time = 10.0
            check_interval = 1.0  # Check every second
            elapsed = 0.0
            
            while elapsed < wait_time:
                time_module.sleep(check_interval)
                elapsed += check_interval
                
                # Check if user is actually speaking (has recent voice activity)
                # Don't just check is_listening_for_command - check if there's actual audio
                current_time = time_module.time()
                time_since_last_voice = current_time - self._last_voice_time
                
                # If user spoke recently (within last 3 seconds), reset the timer
                # Check both audio buffer and recent voice activity
                if time_since_last_voice < 3.0:
                    # User is actually speaking (recent voice activity), reset the timer
                    elapsed = 0.0
                    print(f"[WAKE WORD] User still speaking, extending wait time...")
                elif not self.is_listening_for_command and not self.is_processing:
                    # Command processing is done and we're not listening - check if user is quiet
                    if time_since_last_voice > 3.0:
                        # User has been quiet for 3+ seconds, safe to return
                        break
                    # Otherwise, keep waiting (user might still be speaking)
            
            # Double-check we're still not listening for a command and user is quiet
            current_time = time_module.time()
            time_since_last_voice = current_time - self._last_voice_time
            if not self.is_processing and (not self.is_listening_for_command or time_since_last_voice > 3.0):
                self.is_listening_for_wake_word = True
                
                # Restart wake word detection thread if it's not running
                if self._wake_word_thread is None or not self._wake_word_thread.is_alive():
                    print(f"[WAKE WORD] Restarting detection thread...")
                    self._wake_word_thread = threading.Thread(target=self._wake_word_detection_loop, daemon=True)
                    self._wake_word_thread.start()
                
                # CRITICAL: Clear wake word buffer when returning to prevent old audio from triggering
                # The buffer might still contain the "Hi" that triggered the previous detection
                self._wake_word_buffer = []
                print(f"[WAKE WORD] Buffer cleared when returning to wake word listening")
                
                # Reset wake word check timer
                self._last_wake_word_check = time_module.time()
                
                if self.on_state_change:
                    self.on_state_change('idle')
                print(f"Returned to listening for wake word: '{self.wake_word}'")
            else:
                print(f"[WAKE WORD] User still active, not returning to idle yet")
        
        threading.Thread(target=return_to_wake_word_listening, daemon=True).start()
    
    def _process_audio(self):
        """Process captured audio command"""
        if self.is_processing or len(self._audio_buffer) == 0:
            return
        
        self.is_processing = True
        
        # Notify state change to processing
        if self.on_state_change:
            self.on_state_change('processing')
        
        # Process audio in a separate thread
        def process():
            try:
                # Convert buffer to numpy array
                audio_array = np.array(self._audio_buffer, dtype=np.float32)
                
                # Normalize audio
                audio_array = audio_array / np.max(np.abs(audio_array)) if np.max(np.abs(audio_array)) > 0 else audio_array
                
                # Use Whisper for transcription if available
                transcript = None
                duration = len(audio_array) / self.sample_rate
                
                if self._whisper_model:
                    try:
                        # Whisper transcription runs in background thread (won't block UI)
                        # This is already in a separate thread from process()
                        result = self._whisper_model.transcribe(audio_array, language="en", fp16=False)
                        transcript = result["text"].strip()
                        if transcript:
                            print(f"[TRANSCRIBE] '{transcript}'")
                        else:
                            print(f"[TRANSCRIBE] (no speech detected)")
                    except Exception as e:
                        print(f"[TRANSCRIBE] Error: {e}")
                        import traceback
                        traceback.print_exc()
                        transcript = "Voice input detected (transcription error)"
                else:
                    # Fallback: simulate processing
                    time_module.sleep(0.5)
                    transcript = "Voice input detected (Whisper not available)"
                
                # Notify state change to responding
                if self.on_state_change:
                    self.on_state_change('responding')
                
                # Simulate response time
                time_module.sleep(1.0)
                
                # Notify transcript
                if self.on_transcript:
                    self.on_transcript(transcript)
                
                # Mark processing as complete
                self.is_processing = False
                
                # DON'T immediately return to wake word listening here
                # Let the background return_to_wake_word_listening thread handle it
                # This allows the user to continue speaking if they want
                # The background thread will wait 10 seconds and check if user is still speaking
                
                # Just clear the audio buffer to prepare for next command
                self._audio_buffer = []
                
            except Exception as e:
                print(f"Error processing audio: {e}")
                if self.on_state_change:
                    self.on_state_change('idle')
                # Still return to wake word listening even on error
                self.is_listening_for_command = False
                self.is_listening_for_wake_word = True
                if self._wake_word_thread is None or not self._wake_word_thread.is_alive():
                    self._wake_word_thread = threading.Thread(target=self._wake_word_detection_loop, daemon=True)
                    self._wake_word_thread.start()
                self._last_wake_word_check = time_module.time()
            finally:
                self.is_processing = False
                self._audio_buffer = []
        
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_listening()
        if self._wake_word_thread and self._wake_word_thread.is_alive():
            # Thread will exit when is_listening_for_wake_word becomes False
            pass
