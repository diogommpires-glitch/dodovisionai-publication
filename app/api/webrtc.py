import json
import asyncio
from fastapi import APIRouter, Request, HTTPException
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay

router = APIRouter()
relay = MediaRelay()

# Global dictionary to track active peer connections
pcs = set()

@router.post("/rtc/offer")
async def rtc_offer(request: Request):
    """
    Handle WebRTC offer from the frontend and return an answer.
    """
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        if pc.iceConnectionState == "failed" or pc.iceConnectionState == "closed":
            pcs.discard(pc)

    # WebRTC DataChannel for sending inference results
    data_channel = None

    @pc.on("datachannel")
    def on_datachannel(channel):
        nonlocal data_channel
        data_channel = channel

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            async def process_track():
                from main import _process_frame_fast, _onnx_pool
                import asyncio
                import json
                loop = asyncio.get_running_loop()
                while True:
                    try:
                        frame = await track.recv()
                        img = frame.to_ndarray(format="bgr24")
                        
                        # Process frame using the ML thread pool
                        result = await loop.run_in_executor(
                            _onnx_pool, _process_frame_fast, img
                        )
                        
                        # Send result back via DataChannel if available
                        if data_channel and data_channel.readyState == "open":
                            from app.models.schemas import WebSocketEnvelope
                            import time
                            result_env = dict(result)
                            result_env.setdefault("t", "frame")
                            result_env["ts_server_mono"] = time.monotonic()
                            
                            # Strict Pydantic Serialization
                            safe_env = WebSocketEnvelope.model_validate(result_env).model_dump(exclude_none=True)
                            data_channel.send(json.dumps(safe_env))
                            
                    except Exception as e:
                        print(f"[WebRTC] track error: {e}")
                        break
            asyncio.create_task(process_track())

    # Handle the offer
    await pc.setRemoteDescription(offer)

    # Generate answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }

