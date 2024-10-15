import asyncio
import pyaudio
import websockets
import json
import base64
import os
from dotenv import load_dotenv

# Load environment variables from the parent directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
BUFFER_SIZE = RATE // 10  # 100ms buffer

# OpenAI API settings
API_KEY = os.getenv("OPENAI_API_KEY")
URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

class AsyncMicrophone:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.buffer = b""
        self.is_recording = False

    def callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.buffer += in_data
        return (None, pyaudio.paContinue)

    def start_recording(self):
        if not self.stream:
            self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                      input=True, frames_per_buffer=CHUNK,
                                      stream_callback=self.callback)
        self.is_recording = True

    def stop_recording(self):
        self.is_recording = False

    async def get_audio_data(self):
        while len(self.buffer) < BUFFER_SIZE:
            await asyncio.sleep(0.01)
        data, self.buffer = self.buffer[:BUFFER_SIZE], self.buffer[BUFFER_SIZE:]
        return data

    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

class AudioPlayer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)

    def play(self, audio_data):
        self.stream.write(audio_data)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class VoiceInterface:
    def __init__(self):
        self.mic = AsyncMicrophone()
        self.player = AudioPlayer()
        self.websocket = None
        self.response_active = False

    async def connect(self):
        headers = {"Authorization": f"Bearer {API_KEY}", "OpenAI-Beta": "realtime=v1"}
        self.websocket = await websockets.connect(URL, extra_headers=headers)

    async def initialize_session(self):
        await self.websocket.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": "You are Sarah, a helpful assistant. Always start the conversation with a greeting.",
                "voice": "shimmer",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16"
            }
        }))

    async def send_initial_greeting(self):
        await self.websocket.send(json.dumps({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Start the conversation"}]}
        }))
        await self.websocket.send(json.dumps({"type": "response.create"}))

    async def process_audio(self):
        while True:
            audio_data = await self.mic.get_audio_data()
            base64_audio = base64.b64encode(audio_data).decode('utf-8')
            await self.websocket.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64_audio}))
            await asyncio.sleep(0.05)

    async def handle_event(self, event):
        event_type = event["type"]
        if event_type == "response.text.delta":
            print(event.get("delta", ""), end="", flush=True)
        elif event_type == "response.audio.delta":
            self.player.play(base64.b64decode(event["delta"]))
        elif event_type == "response.done":
            self.response_active = False
        elif event_type == "error":
            print(f"Error: {event.get('error', {}).get('message', 'Unknown error')}")
        elif event_type == "input_audio_buffer.speech_stopped":
            await self.handle_speech_stopped()

    async def handle_speech_stopped(self):
        if len(self.mic.buffer) > 0:
            await self.websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
            if not self.response_active:
                await self.websocket.send(json.dumps({"type": "response.create"}))
                self.response_active = True

    async def run(self):
        if not API_KEY:
            print("Error: OPENAI_API_KEY not found in environment variables")
            return

        try:
            await self.connect()
            await self.initialize_session()
            await self.send_initial_greeting()

            audio_task = asyncio.create_task(self.process_audio())
            self.mic.start_recording()

            while True:
                message = await self.websocket.recv()
                await self.handle_event(json.loads(message))

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.mic.stop_recording()
            self.mic.close()
            self.player.close()
            if 'audio_task' in locals():
                audio_task.cancel()

async def main():
    voice_interface = VoiceInterface()
    await voice_interface.run()

if __name__ == "__main__":
    asyncio.run(main())
