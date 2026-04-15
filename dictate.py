import os
import subprocess
import tempfile
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import signal
import sys
import time
import threading
from queue import Queue
from evdev import UInput, ecodes as e

# --- 1. CONFIGURATION & CONSTANTS (No Magic Numbers) ---

# Audio capture settings
SAMPLE_RATE = 16000            # Standard frequency for Whisper AI
CHANNELS = 1                    # Mono audio is enough for dictation
INT16_MAX_ABS = 32768.0         # Max absolute value for 16-bit signed audio

# Timing & VAD (Voice Activity Detection) parameters
MAX_CHUNK_SECONDS = 12.0       # Hard limit to prevent memory overflow
MIN_SILENCE_SECONDS = 1.2      # 1.2s pause triggers a 'flush' (natural pause)

# [CONTEXT_HISTORY_LENGTH]: Range 0 to 200. 
# Lower (50-100) is safer against loops. Higher (200+) helps with long-range grammar.
CONTEXT_HISTORY_LENGTH = 100    # How many characters of previous text to feed the AI

# Priming prompt to encourage punctuation and capitalization style in both English and Spanish.
PUNCTUATION_PROMPT = "Hello! This is a punctuated sentence in English. ¡Hola! Esta es una frase con puntuación en español. I am a bilingual speaker. Soy una persona bilingüe."

# Sensitivity & Quality thresholds
SILENCE_THRESHOLD = 0.03       # Average energy below this is 'room noise'
MIN_RECORDING_SECONDS = 0.5    # Ignore accidental clicks/noises shorter than this
NO_SPEECH_THRESHOLD = 0.6      # Reject if AI is >60% sure it's just noise
LOGPROB_THRESHOLD = -1.0       # Reject if AI confidence is too low (hallucination)

# [COMPRESSION_RATIO_THRESHOLD]: Range 1.0 to 3.0. 
# LOWER (2.0-2.4) is stricter (rejects repetitive "looping" hallucinations).
COMPRESSION_RATIO_THRESHOLD = 2.2

# Internal calculation windows (Calculated from SAMPLE_RATE)
# We look at the last 200ms of audio to check for silence
ENERGY_WINDOW_SAMPLES = int(SAMPLE_RATE * 0.2) # Default: 3200 samples

# AI Hardware acceleration
MODEL_SIZE = "large-v3-turbo" 
DEVICE = "cuda"                # Use RTX 3050 Ti
COMPUTE_TYPE = "float16"       # Fast performance on 30-series GPUs

# --- 2. INTERNAL STATE ---
TEMP_DIR = tempfile.gettempdir()
ui = UInput()      # Virtual Keyboard via evdev
audio_queue = Queue() # Thread-safe bucket to pass audio from Mic -> AI
is_running = True
last_text_context = "" 

# --- 3. CORE FUNCTIONS ---

def paste_text(text):
    """
    Simulates a physical Paste (Shift+Insert).
    Backs up and restores the user's clipboards so dictation doesn't overwrite them!
    """
    if not text.strip():
        return
        
    text_to_paste = (text + " ").encode('utf-8')

    # 1. Back up current clipboards
    try:
        old_clipboard = subprocess.check_output(["wl-paste", "-n"])
    except subprocess.CalledProcessError:
        old_clipboard = None
        
    try:
        old_primary = subprocess.check_output(["wl-paste", "-p", "-n"])
    except subprocess.CalledProcessError:
        old_primary = None

    # 2. Set clipboards to dictation text
    # Add tiny delays between wl-copy calls to prevent Mutter (GNOME) gear icon race condition.
    subprocess.run(["wl-copy"], input=text_to_paste, check=True)
    time.sleep(0.15)
    subprocess.run(["wl-copy", "-p"], input=text_to_paste, check=True)
    time.sleep(0.15)
    
    # 3. Paste
    ui.write(e.EV_KEY, e.KEY_LEFTSHIFT, 1) # 1 = Press
    ui.write(e.EV_KEY, e.KEY_INSERT, 1)
    ui.syn()
    time.sleep(0.05)                      # Hold for 50ms so apps see it
    ui.write(e.EV_KEY, e.KEY_INSERT, 0)   # 0 = Release
    ui.write(e.EV_KEY, e.KEY_LEFTSHIFT, 0)
    ui.syn()

    # 4. Wait for paste to finish, then restore original clipboards!
    time.sleep(0.15)
    if old_clipboard is not None:
        subprocess.run(["wl-copy"], input=old_clipboard)
    else:
        subprocess.run(["wl-copy", "-c"]) # Clear it
        
    if old_primary is not None:
        subprocess.run(["wl-copy", "-p"], input=old_primary)
    else:
        subprocess.run(["wl-copy", "-p", "-c"])

def transcription_worker():
    """
    THREAD 2: THE TRANSLATOR
    This thread waits for 'audio_chunk' tasks to appear in the Queue.
    It runs in the background so the microphone never has to stop.
    """
    global is_running, last_text_context
    from faster_whisper import WhisperModel
    
    print(f"AI Model is loading ({MODEL_SIZE})...")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("AI Model READY.")

    while is_running or not audio_queue.empty():
        if audio_queue.empty():
            time.sleep(0.1) # Don't burn CPU while waiting for speech
            continue
            
        # Pull the oldest audio piece from the bucket
        audio_chunk = audio_queue.get()
        
        # Save to a temporary file for Whisper to read
        temp_wav = os.path.join(TEMP_DIR, f"chunk_{int(time.time()*1000)}.wav")
        wavfile.write(temp_wav, SAMPLE_RATE, audio_chunk)
        
        # Build the 'Memory': show the AI the style we want and the last 100 characters we wrote
        current_prompt = PUNCTUATION_PROMPT + (last_text_context[-CONTEXT_HISTORY_LENGTH:] if last_text_context else "")
        
        segments, _ = model.transcribe(
            temp_wav, 
            beam_size=3, 
            initial_prompt=current_prompt,
            task="transcribe",
            # LOOP PREVENTION & NOISE FILTERING:
            condition_on_previous_text=True, 
            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
            vad_filter=True, # New: Ignores non-speech sounds like whistling
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        chunk_text = []
        for s in segments:
            # QUALITY FILTER: Only keep words if they are high-confidence
            if s.no_speech_prob < NO_SPEECH_THRESHOLD and s.avg_logprob > LOGPROB_THRESHOLD:
                chunk_text.append(s.text)
            else:
                print(f"\n[FILTERED]: AI was not confident in: '{s.text}'")
        
        text = "".join(chunk_text).strip()
        if text:
            print(f"\r[PASTING]: {text}")
            paste_text(text)
            last_text_context += " " + text
            
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        audio_queue.task_done() # Tell the Queue this chunk is finished

def record_audio():
    """
    THREAD 1: THE CUTTER (Main Thread)
    Captures live audio from the mic and decides when a sentence ends.
    """
    global is_running
    buffer = []
    
    # This function is called by the hardware every time new audio arrives
    def callback(indata, frames, time_info, status):
        if is_running:
            buffer.append(indata.copy())

    # START THE TRANSLATOR THREAD
    # It will run 'transcription_worker' parallel to this recording loop
    worker_thread = threading.Thread(target=transcription_worker)
    worker_thread.start()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', callback=callback):
        # print("Mics are HOT - Recording started.")
        last_speech_time = time.time()
        start_time = time.time()
        was_speech_detected_in_chunk = False
        
        while is_running:
            time.sleep(0.1)
            if not buffer: continue

            # Analyze current audio in the buffer
            current_audio = np.concatenate(buffer, axis=0)
            
            # Look at the 'Energy' (Volume) of the most recent slice
            # We divide by INT16_MAX_ABS to get a normalized 0.0 - 1.0 range
            energy = np.abs(current_audio[-ENERGY_WINDOW_SAMPLES:]).mean() / INT16_MAX_ABS 
            
            is_silent = energy < SILENCE_THRESHOLD
            now = time.time()

            if not is_silent:
                last_speech_time = now 
                was_speech_detected_in_chunk = True

            silence_duration = now - last_speech_time
            total_duration = now - start_time
            
            # Terminal status update (5 times per second)
            # if int(now * 5) % 5 == 0:
            #     status = "SPEAKING" if not is_silent else "..."
            #     print(f"\r[ENERGY: {energy:.4f}] {status} (Buffer: {total_duration:.1f}s)", end="")

            # FLUSH CONDITIONS
            # 1. Pause detected (0.7s) 
            # 2. OR Safety limit reached (12s)
            should_flush = (silence_duration >= MIN_SILENCE_SECONDS and total_duration > MIN_RECORDING_SECONDS) or \
                           (total_duration >= MAX_CHUNK_SECONDS)

            if should_flush:
                # If we actually heard voice, hand it over to the Translator thread via the Queue
                if was_speech_detected_in_chunk and len(current_audio) > (SAMPLE_RATE * MIN_RECORDING_SECONDS):
                    audio_queue.put(current_audio)
                
                # Reset for the next sentence
                buffer = []
                start_time = time.time()
                was_speech_detected_in_chunk = False 

    audio_queue.join() # Wait for the bucket to be empty before closing
    is_running = False
    worker_thread.join()

if __name__ == "__main__":
    # Handle the 'Win + E' toggle by exiting cleanly on SIGTERM/SIGINT
    signal.signal(signal.SIGINT, lambda s,f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s,f: sys.exit(0))
    try:
        record_audio()
    except SystemExit:
        is_running = False
unning = False
