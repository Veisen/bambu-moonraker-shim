import asyncio
import ssl
import struct
import time
from typing import Optional

from bambu_moonraker_shim.config import Config


def build_auth_packet(access_code: str, username: str = "bblp") -> bytes:
    if len(username.encode("utf-8")) > 32:
        raise ValueError("Camera username must be 32 bytes or fewer")
    if len(access_code.encode("utf-8")) > 32:
        raise ValueError("Camera access code must be 32 bytes or fewer")

    packet = bytearray(struct.pack("<IIII", 0x40, 0x3000, 0x0, 0x0))
    packet.extend(username.encode("utf-8"))
    packet.extend(b"\x00" * (32 - len(username.encode("utf-8"))))
    packet.extend(access_code.encode("utf-8"))
    packet.extend(b"\x00" * (32 - len(access_code.encode("utf-8"))))
    return bytes(packet)


def is_jpeg_frame(payload: bytes) -> bool:
    return (
        len(payload) >= 6
        and payload.startswith(b"\xff\xd8\xff")
        and payload.endswith(b"\xff\xd9")
    )


class P1CameraManager:
    BUILTIN_UID = "bambu-p1-camera"
    MJPEG_BOUNDARY = "frame"
    _PACKET_SUFFIX = b"\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00"
    _MAX_FRAME_BYTES = 10 * 1024 * 1024

    def __init__(self):
        self.host = Config.BAMBU_HOST
        self.access_code = Config.BAMBU_ACCESS_CODE
        self.port = int(getattr(Config, "BAMBU_CAMERA_PORT", 6000))
        self.username = str(getattr(Config, "BAMBU_CAMERA_USERNAME", "bblp") or "bblp")
        self.enabled = str(getattr(Config, "BAMBU_CAMERA_ENABLED", "true")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._latest_frame: Optional[bytes] = None
        self._latest_frame_at = 0.0
        self._frame_available = asyncio.Event()
        self._subscribers: set[asyncio.Queue[bytes]] = set()
        self._task: Optional[asyncio.Task] = None
        self._connected = False

    @property
    def is_configured(self) -> bool:
        return self.enabled and bool(self.host and self.access_code)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def has_frame(self) -> bool:
        return self._latest_frame is not None

    def get_builtin_webcam(self) -> Optional[dict]:
        if not self.is_configured:
            return None
        return {
            "name": "Bambu Camera",
            "location": "printer",
            "service": "mjpegstreamer",
            "target_fps": 2,
            "stream_url": "/webcam?action=stream",
            "snapshot_url": "/webcam?action=snapshot",
            "flip_horizontal": False,
            "flip_vertical": False,
            "rotation": 0,
            "source": "system",
            "uid": self.BUILTIN_UID,
            "enabled": True,
        }

    async def start(self):
        if not self.is_configured:
            print("P1 camera streaming disabled: missing host/access code or explicitly disabled.")
            return
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="bambu-p1-camera")

    async def stop(self):
        if not self._task:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
            self._connected = False

    async def wait_for_frame(self, timeout: float = 10.0) -> Optional[bytes]:
        if self._latest_frame is not None:
            return self._latest_frame
        try:
            await asyncio.wait_for(self._frame_available.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        return self._latest_frame

    async def subscribe(self) -> asyncio.Queue[bytes]:
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
        self._subscribers.add(queue)
        if self._latest_frame is not None:
            queue.put_nowait(self._latest_frame)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[bytes]):
        self._subscribers.discard(queue)

    async def _run(self):
        while True:
            try:
                await self._capture_loop()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._connected = False
                print(f"P1 camera stream error: {exc}")
            await asyncio.sleep(2)

    async def _capture_loop(self):
        tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        tls_context.check_hostname = False
        tls_context.verify_mode = ssl.CERT_NONE

        reader, writer = await asyncio.open_connection(
            self.host,
            self.port,
            ssl=tls_context,
            server_hostname="192.168.1.27:8181",
        )

        try:
            writer.write(build_auth_packet(self.access_code, self.username))
            await writer.drain()
            self._connected = True
            print(f"Connected to Bambu camera stream at {self.host}:{self.port}")

            while True:
                header = await reader.readexactly(16)
                frame_length = struct.unpack("<I", header[:4])[0]
                if frame_length <= 0 or frame_length > self._MAX_FRAME_BYTES:
                    raise ValueError(f"Unexpected camera frame length: {frame_length}")

                payload = await reader.readexactly(frame_length)
                if header[4:] != self._PACKET_SUFFIX:
                    continue
                if not is_jpeg_frame(payload):
                    continue

                self._latest_frame = payload
                self._latest_frame_at = time.time()
                self._frame_available.set()
                self._publish_frame(payload)
        finally:
            self._connected = False
            writer.close()
            await writer.wait_closed()

    def _publish_frame(self, payload: bytes):
        for queue in tuple(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass


camera_manager = P1CameraManager()
