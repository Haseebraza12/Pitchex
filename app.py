# requirements.txt
# gradio>=4.0.0
# transformers>=4.39.0
# torch
# soundfile
# numpy
# scipy
# gtts
# pydub
# ffmpeg-python
# subprocess # Standard library, no need to install

import os
import tempfile
import gradio as gr
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import soundfile as sf
import numpy as np
import subprocess
import time
import io
from gtts import gTTS
from pydub import AudioSegment
import wave

# --- Configuration ---
# Adjust the gemini CLI command if necessary
GEMINI_CLI_COMMAND = ["gemini", "chat"]  # <-- Adjust this command based on your CLI setup

# --- Global Variables ---
# Placeholder for loaded models - ideally load once at startup
stt_model = None
stt_processor = None
stt_pipe = None

# --- Utility Functions ---
def load_models():
    """Loads the required models."""
    global stt_model, stt_processor, stt_pipe
    try:
        print("Loading STT model...")
        # Load STT model (distil-large-v3.5) for ASR
        stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "distil-whisper/distil-large-v3.5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        stt_processor = AutoProcessor.from_pretrained("distil-whisper/distil-large-v3.5")
        
        # Create the pipeline once to avoid reloading for each request
        stt_pipe = pipeline(
            "automatic-speech-recognition",
            model=stt_model,
            tokenizer=stt_processor.tokenizer,
            feature_extractor=stt_processor.feature_extractor,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )

        print("STT model loaded successfully.")
        print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'} for inference")

        # --- LLM Model (Gemini CLI) ---
        print("LLM model will be accessed via CLI.")

        print("All models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

# --- Core Processing Function ---
def process_audio(audio_input_path):
    """
    Processes the audio input through STT -> LLM (CLI) -> TTS (Simulated) -> Returns audio.
    Args:
        audio_input_path (str): Path to the audio file from Gradio.
    Returns:
        tuple: (sample_rate, audio_array) of the AI response.
    """
    # --- Step 0: Initial validation ---
    if audio_input_path is None:
        print("Error: No audio input received")
        # Return a short error beep
        sample_rate = 16000
        duration = 1.0
        frequency = 880
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        error_audio = np.sin(2 * np.pi * frequency * t) * 0.3
        error_audio = error_audio / np.max(np.abs(error_audio))
        return (sample_rate, error_audio.astype(np.float32))
    
    # --- Step 1: STT (Speech-to-Text) ---
    print("Starting STT processing...")
    try:
        # Load audio file
        audio_array, sample_rate = sf.read(audio_input_path)
        print(f"Received audio: {len(audio_array)} samples at {sample_rate} Hz")

        # Perform STT using the pre-loaded pipeline
        print("Transcribing audio...")
        start_time = time.time()
        
        # For very long audio files, we might need to use chunking
        result_stt = stt_pipe(audio_input_path)
        stt_time = time.time() - start_time
        print(f"STT completed in {stt_time:.2f} seconds")
        
        text_input = result_stt.get("text", "").strip()
        print(f"STT Output: '{text_input}'")

        if not text_input:
            print("Warning: STT returned empty text.")
            # Return error sound
            duration = 1.0
            frequency = 880
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            error_audio = np.sin(2 * np.pi * frequency * t) * 0.3
            error_audio = error_audio / np.max(np.abs(error_audio))
            return (sample_rate, error_audio.astype(np.float32))

    except Exception as e:
        print(f"Error in STT processing: {e}")
        # Return a silent audio or error audio
        sample_rate = 16000 if 'sample_rate' not in locals() else sample_rate
        silent_audio = np.zeros((1000,), dtype=np.float32)
        return (sample_rate, silent_audio)

    # --- Step 2: LLM (Language Model) via CLI ---
    print("Starting LLM processing via CLI...")
    ai_response_text = ""
    try:
        # Define prompt for the LLM
        prompt = f"""
        You are an AI assistant acting as an investor in a startup fundraising scenario.
        The user has just said: '{text_input}'
        Respond appropriately as an investor in a professional and engaging way.
        Keep your response concise but informative.
        """
        print(f"Sending prompt to Gemini CLI:\n{prompt}")

        # --- Using subprocess to call gemini CLI ---
        # First, check if the CLI exists
        try:
            subprocess.run(["which", "gemini"], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            print("Warning: 'gemini' CLI not found. Using simulated response.")
            ai_response_text = f"Thanks for your input about '{text_input}'. As an investor, I appreciate the detail and clarity. Let's explore the market fit further."
            print(f"Simulated LLM Output: {ai_response_text}")
            return generate_tts_response(ai_response_text, sample_rate)
        
        # Try to get the version to confirm it's working
        try:
            version_check = subprocess.run(["gemini", "--version"], capture_output=True, text=True, timeout=5)
            print(f"Gemini CLI version: {version_check.stdout.strip()}")
        except Exception as e:
            print(f"Could not get Gemini CLI version: {e}")

        # Execute the command - try different approaches based on CLI behavior
        try:
            # Approach 1: Try with -q flag (common pattern)
            cmd = GEMINI_CLI_COMMAND + ["-q", prompt]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            ai_response_text = result.stdout.strip()
            print(f"CLI Output (with -q flag): {ai_response_text}")
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"Approach 1 failed: {e}")
            try:
                # Approach 2: Try with --prompt flag
                cmd = GEMINI_CLI_COMMAND + ["--prompt", prompt]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30
                )
                ai_response_text = result.stdout.strip()
                print(f"CLI Output (with --prompt flag): {ai_response_text}")
                
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"Approach 2 failed: {e}")
                try:
                    # Approach 3: Pass prompt via stdin
                    result = subprocess.run(
                        GEMINI_CLI_COMMAND,
                        input=prompt,
                        text=True,
                        capture_output=True,
                        check=True,
                        timeout=30
                    )
                    ai_response_text = result.stdout.strip()
                    print(f"CLI Output (via stdin): {ai_response_text}")
                    
                except Exception as e:
                    print(f"All CLI approaches failed: {e}")
                    print("Falling back to simulated response")
                    ai_response_text = f"Thanks for sharing your thoughts about '{text_input}'. As an investor, I find this promising. Could you elaborate on your go-to-market strategy?"
        
        # Ensure we have a valid response
        if not ai_response_text or len(ai_response_text) < 10:
            print("Warning: LLM returned invalid response, using fallback")
            ai_response_text = f"Your point about '{text_input[:30]}...' is interesting. Let's discuss the scalability of your solution."

    except Exception as e:
        print(f"Unexpected error in LLM processing: {e}")
        ai_response_text = "I'm having trouble processing your request. Please try again."

    print(f"Final LLM Output: {ai_response_text}")

    # --- Step 3: TTS (Text-to-Speech) ---
    return generate_tts_response(ai_response_text, sample_rate)

def generate_tts_response(text, sample_rate=16000):
    """Generate TTS response using gTTS with proper audio format handling"""
    print("Generating TTS response with gTTS...")
    
    try:
        # Create a gTTS object
        tts = gTTS(text=text, lang='en')
        
        # Save to a file-like object
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        # Convert MP3 to WAV with the desired sample rate
        audio = AudioSegment.from_file(mp3_fp, format="mp3")
        audio = audio.set_frame_rate(sample_rate)
        
        # Export as WAV to a file-like object
        wav_fp = io.BytesIO()
        audio.export(wav_fp, format="wav")
        wav_fp.seek(0)
        
        # Read the WAV data
        with wave.open(wav_fp, 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            # Convert to numpy array based on sample width
            if sample_width == 1:
                dtype = np.uint8
            elif sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
                
            audio_data = np.frombuffer(frames, dtype=dtype)
            
            # Convert to float32 in the range [-1, 1]
            if dtype == np.uint8:
                audio_data = (audio_data.astype(np.float32) - 128) / 128.0
            elif dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
                
            # Reshape if stereo
            if n_channels == 2:
                audio_data = audio_data.reshape(-1, 2)
                audio_data = np.mean(audio_data, axis=1)  # Convert to mono
            
            # Ensure the audio is in the expected range [-1, 1]
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val
                
            print(f"TTS generated {len(audio_data)} samples at {sample_rate} Hz")
            return (sample_rate, audio_data.astype(np.float32))
            
    except Exception as e:
        print(f"Error in TTS generation: {e}")
        # Fallback to a simple beep if TTS fails
        duration = 1.0
        frequency = 880
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        error_audio = np.sin(2 * np.pi * frequency * t) * 0.3
        error_audio = error_audio / np.max(np.abs(error_audio))
        print(f"TTS fallback generated {len(error_audio)} samples at {sample_rate} Hz")
        return (sample_rate, error_audio.astype(np.float32))

# --- Gradio Interface ---
def main():
    """Main function to launch the Gradio app."""
    print("Loading models...")
    load_models()
    print("Models loaded.")
    
    # Create a temporary directory for audio files if needed
    os.makedirs("temp_audio", exist_ok=True)

    # Define the Gradio interface
    with gr.Blocks(title="AI Pitching & Negotiation Coach", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎤 AI Pitching & Negotiation Coach
        Practice your startup fundraising pitches with our AI coach. Record your pitch and get instant feedback!
        """)
        
        gr.Markdown("""
        **How it works:**
        1. Click the microphone button below to record your pitch
        2. Click "Submit and Process" to analyze your pitch
        3. The AI will respond as an investor with feedback
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Your Input")
                # In Gradio 4.x, use type="filepath" for microphone input
                audio_input = gr.Audio(
                    label="Record your pitch", 
                    type="filepath",
                    show_download_button=False,
                    container=True,
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#00b894",
                        waveform_progress_color="#fdcb6e",
                        skip_length=2
                    )
                )
                submit_btn = gr.Button("Submit and Process", variant="primary")
                
                # Status indicator
                status = gr.Textbox(label="Status", value="Ready to record", interactive=False)
            
            with gr.Column():
                gr.Markdown("### AI Investor Response")
                audio_output = gr.Audio(
                    label="AI Response", 
                    type="numpy",
                    show_download_button=True,
                    container=True,
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#0984e3",
                        waveform_progress_color="#e84393",
                        skip_length=2
                    )
                )
                response_text = gr.Textbox(label="AI Response Text", lines=5, interactive=False)
        
        # Update status when recording starts
        def update_status(recording):
            if recording is not None:
                return "Audio received. Click 'Submit and Process' to analyze."
            return "Ready to record"
        
        audio_input.change(
            fn=update_status,
            inputs=audio_input,
            outputs=status
        )
        
        # Event listener for processing - CORRECTED VERSION
        def process_and_update(audio_path):
            if audio_path is None:
                return (
                    None,
                    "Please record audio first",
                    "Please record audio first"
                )
                
            # First, indicate we're processing (return the processing status)
            # This is the correct way to update components in Gradio
            yield (
                None,  # audio_output
                "Processing your pitch...",  # status
                "Processing your pitch..."  # response_text
            )
            
            try:
                # Process the audio
                output_audio = process_audio(audio_path)
                
                # For demonstration, we'll extract the response text from the audio processing
                # In a real implementation, you'd have the actual text from the LLM
                response_text_content = "Thanks for your input. As an investor, I'd like to know more about your go-to-market strategy."
                
                # Return the final results
                yield (
                    output_audio,
                    "Processing complete!",
                    response_text_content
                )
            except Exception as e:
                print(f"Error in processing: {e}")
                yield (
                    None,
                    f"Error: {str(e)}",
                    f"Error processing your pitch: {str(e)}"
                )
        
        submit_btn.click(
            fn=process_and_update,
            inputs=audio_input,
            outputs=[audio_output, status, response_text]
        )

    # Launch the app
    print("Launching Gradio interface...")
    demo.launch(
       debug=True,
       share=True,  # Set to True for public sharing
    )

if __name__ == "__main__":
    main()