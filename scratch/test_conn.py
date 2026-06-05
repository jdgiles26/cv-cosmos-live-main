import asyncio
import logging
import sys
from livekit import rtc
from cosmos_utils import generate_livekit_token
from cosmos_live.config import LiveKitConfig

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger("test_conn")

async def main():
    config = LiveKitConfig(
        url="ws://127.0.0.1:7880",
        api_key="devkey",
        api_secret="secret",
        room="cosmos",
    )
    token = generate_livekit_token(config, "test-client", "cosmos")
    room = rtc.Room()
    
    logger.info("Connecting to Room...")
    try:
        options = rtc.RoomOptions(
            rtc_config=rtc.RtcConfiguration(
                ice_transport_type=rtc.IceTransportType.TRANSPORT_NOHOST
            )
        )
        await room.connect(config.url, token, options=options)
        logger.info("SUCCESSFULLY CONNECTED!")
        await room.disconnect()
    except Exception as e:
        logger.error(f"FAILED TO CONNECT: {e}")

if __name__ == "__main__":
    asyncio.run(main())
