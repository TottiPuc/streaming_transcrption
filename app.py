import streamlit as st
import pyaudio
import websocket
import json
import threading
import time
import wave
from urllib.parse import urlencode
from datetime import datetime
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# --- Configuración Original ---
YOUR_API_KEY = os.getenv("API_KEY") 

CONNECTION_PARAMS = {
    "sample_rate": 16000,
    "format_turns": True,
    "speech_model": "universal-streaming-multilingual",
    "language_detection": True
}
API_ENDPOINT_BASE_URL = "wss://streaming.assemblyai.com/v3/ws"
API_ENDPOINT = f"{API_ENDPOINT_BASE_URL}?{urlencode(CONNECTION_PARAMS)}"

# Configuración de Audio
FRAMES_PER_BUFFER = 1024
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16

# --- Clase Gestora de Estado (Para persistencia en Streamlit) ---
class TranscriptionManager:
    def __init__(self):
        self.audio = None
        self.stream = None
        self.ws_app = None
        self.audio_thread = None
        self.ws_thread = None
        self.stop_event = threading.Event()
        self.recorded_frames = []
        self.recording_lock = threading.Lock()
        
        # Variables para la UI
        self.committed_text = ""     # Texto confirmado acumulado (string único)
        self.partial_text = ""       # Variable para la frase actual (borrador)
        self.transcript_text = ""    # Texto combinado para mostrar
        self.last_final_time = 0     # Timestamp de la última frase confirmada
        
        self.status = "Detenido"
        self.is_running = False

    def on_open(self, ws):
        """Called when the WebSocket connection is established."""
        print("WebSocket connection opened.")
        self.status = "Conectado. Grabando..."
        
        # Start sending audio data in a separate thread
        def stream_audio():
            print("Starting audio streaming...")
            while not self.stop_event.is_set():
                try:
                    if self.stream:
                        audio_data = self.stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)

                        # Store audio data for WAV recording
                        with self.recording_lock:
                            self.recorded_frames.append(audio_data)

                        # Send audio data as binary message
                        ws.send(audio_data, websocket.ABNF.OPCODE_BINARY)
                except Exception as e:
                    print(f"Error streaming audio: {e}")
                    break
            print("Audio streaming stopped.")

        self.audio_thread = threading.Thread(target=stream_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            msg_type = data.get('type')

            if msg_type == "Begin":
                session_id = data.get('id')
                print(f"Session began: ID={session_id}")
                self.last_final_time = time.time()
            
            elif msg_type == "Turn":
                transcript = data.get('transcript', '')
                is_final = data.get('turn_is_formatted', False)
                
                if is_final:
                    if transcript:
                        current_time = time.time()
                        # Si han pasado más de 2 segundos desde la última frase, hacemos un salto de párrafo
                        silence_duration = current_time - self.last_final_time
                        
                        separator = " " # Por defecto, espacio (mismo párrafo)
                        
                        if not self.committed_text:
                            separator = "" # Sin separador al inicio
                        elif silence_duration > 2.0:
                            separator = "\n\n" # Salto de párrafo tras pausa larga
                        
                        self.committed_text += separator + transcript
                        self.last_final_time = current_time
                    
                    self.partial_text = ""
                else:
                    self.partial_text = transcript
                
                # Reconstruimos el texto completo para la UI
                separator_partial = " " if self.committed_text and self.partial_text else ""
                self.transcript_text = self.committed_text + separator_partial + self.partial_text
                    
            elif msg_type == "Termination":
                print("Session Terminated")
                
        except Exception as e:
            print(f"Error handling message: {e}")

    def on_error(self, ws, error):
        self.status = f"Error: {error}"
        self.stop_event.set()

    def on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket Disconnected: {close_status_code}")
        self.save_wav_file()
        self.is_running = False
        self.status = "Desconectado. Archivo guardado."

    def save_wav_file(self):
        if not self.recorded_frames:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recorded_audio_{timestamp}.wav"

        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                with self.recording_lock:
                    wf.writeframes(b''.join(self.recorded_frames))
            
            print(f"Audio saved to: {filename}")
            # Reset buffers
            with self.recording_lock:
                self.recorded_frames = []
        except Exception as e:
            print(f"Error saving WAV: {e}")

    def start_transcription(self):
        if self.is_running:
            return

        self.stop_event.clear()
        self.committed_text = ""  # Limpiar texto confirmado
        self.partial_text = ""    # Limpiar parcial
        self.transcript_text = "" 
        self.recorded_frames = [] # Limpiar buffer de audio
        self.last_final_time = time.time() # Resetear tiempo
        self.is_running = True
        self.status = "Iniciando conexión..."

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        try:
            self.stream = self.audio.open(
                input=True,
                frames_per_buffer=FRAMES_PER_BUFFER,
                channels=CHANNELS,
                format=FORMAT,
                rate=SAMPLE_RATE,
            )
        except Exception as e:
            self.status = f"Error micrófono: {e}"
            self.is_running = False
            return

        # Create WebSocketApp
        websocket.enableTrace(False)
        self.ws_app = websocket.WebSocketApp(
            API_ENDPOINT,
            header={"Authorization": YOUR_API_KEY},
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )

        self.ws_thread = threading.Thread(target=self.ws_app.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def stop_transcription(self):
        if not self.is_running:
            return

        self.status = "Deteniendo..."
        self.stop_event.set()

        # Send termination message
        if self.ws_app and self.ws_app.sock and self.ws_app.sock.connected:
            try:
                terminate_message = {"type": "Terminate"}
                self.ws_app.send(json.dumps(terminate_message))
            except Exception as e:
                print(f"Error sending termination: {e}")

        # Close WebSocket
        if self.ws_app:
            self.ws_app.close()
        
        # Cleanup Audio
        if self.stream:
            if self.stream.is_active():
                self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.audio:
            self.audio.terminate()
            self.audio = None

        self.is_running = False


# --- Interfaz de Streamlit ---

st.set_page_config(page_title="Transcriptor AssemblyAI", layout="centered")
st.title("🎙️ Transcripción en Tiempo Real")

# Inicializar el Manager en session_state para que persista entre recargas
if 'manager' not in st.session_state:
    st.session_state.manager = TranscriptionManager()

manager = st.session_state.manager

# Controles
col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Iniciar Transcripción", disabled=manager.is_running, type="primary"):
        manager.start_transcription()
        st.rerun()

with col2:
    if st.button("⏹️ Detener Transcripción", disabled=not manager.is_running):
        manager.stop_transcription()
        st.rerun()

# Estado
st.info(f"Estado: **{manager.status}**")

# Área de Texto
st.subheader("Transcripción:")
text_area = st.empty()

# Bucle de actualización automática
if manager.is_running:
    # Mostramos el texto acumulado + parcial actual
    text_area.text_area("En vivo", value=manager.transcript_text, height=300)
    
    # Pausa para refresco
    time.sleep(0.5) 
    st.rerun()
else:
    # Resultado final estático
    text_area.text_area("Resultado Final", value=manager.transcript_text, height=300)