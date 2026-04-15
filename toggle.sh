#!/bin/bash

BASE_DIR="/home/jantofp/whisper-dictation"
PID_FILE="/tmp/whisper_dictate.pid"

if [ -f "$PID_FILE" ]; then
    # Stop everything
    PID=$(cat "$PID_FILE")
    kill -SIGTERM "$PID"
    rm "$PID_FILE"
    notify-send -r 1234 "Whisper" "Stopped Streaming." -i audio-input-microphone -t 2000
else
    # Start the streaming dictation
    notify-send -r 1234 "Whisper" "Streaming... (Auto-Paste)" -i audio-input-microphone -t 2000
    
    # Run the python script with CUDA libraries in path
    LD_LIBRARY_PATH="$BASE_DIR/venv/lib/python3.12/site-packages/nvidia/cublas/lib:$BASE_DIR/venv/lib/python3.12/site-packages/nvidia/cudnn/lib" \
    "$BASE_DIR/venv/bin/python3" "$BASE_DIR/dictate.py" &
    echo $! > "$PID_FILE"
fi
