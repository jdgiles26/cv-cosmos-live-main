"""Entry point for the Cosmos LiveKit voice agent.

Connects to a LiveKit room, starts the stream operator, and runs
a voice agent that can query feed content and control the stream.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from livekit.agents import (
    AgentSession,
    AgentServer,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    cli,
    inference,
    metrics,
)
from livekit.plugins import silero

from cosmos_control.agent import CosmosAgent
from cosmos_control.operator import CVDisplayOperator
from cosmos_live.config import Config
from cosmos_storage import QdrantVectorStore

logger = logging.getLogger("cosmos_live.agent")

from typing import Any
from livekit.agents import stt, tts, llm, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

class SafeLLM(inference.LLM):
    def chat(self, *args, **kwargs) -> llm.LLMStream:
        if not self.model:
            from livekit.agents.llm import ChatChunk, ChoiceDelta
            import uuid
            
            chat_ctx = kwargs.get("chat_ctx") or (args[0] if args else None)
            tools = kwargs.get("tools") or (args[1] if len(args) > 1 else [])
            conn_options = kwargs.get("conn_options") or (args[2] if len(args) > 2 else DEFAULT_API_CONNECT_OPTIONS)
            
            class OfflineLLMStream(llm.LLMStream):
                async def _run(self) -> None:
                    chunk_id = str(uuid.uuid4())
                    self._event_ch.send(
                        ChatChunk(
                            id=chunk_id,
                            delta=ChoiceDelta(
                                role="assistant",
                                content="Welcome! No Retrieval LLM model has been configured yet. Please open the Operator Console at localhost:9090, choose a retrieval LLM model under the 'AI Engines & Models' panel, and talk to me again!"
                            )
                        )
                    )
            
            return OfflineLLMStream(
                self,
                chat_ctx=chat_ctx,
                tools=tools or [],
                conn_options=conn_options,
            )
            
        return super().chat(*args, **kwargs)

class OfflineSTT(stt.STT):
    def __init__(self) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming_supported=True)
        )

    @property
    def model(self) -> str:
        return "offline-stt"

    @property
    def provider(self) -> str:
        return "offline"

    async def _recognize_impl(
        self,
        buffer: stt.AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechAlternative(text="", language="")],
        )

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> stt.RecognizeStream:
        class OfflineRecognizeStream(stt.RecognizeStream):
            async def _run(self) -> None:
                async for _ in self._input_ch:
                    pass
        return OfflineRecognizeStream(
            stt=self,
            conn_options=conn_options or DEFAULT_API_CONNECT_OPTIONS,
        )


class OfflineTTS(tts.TTS):
    def __init__(self) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming_supported=True),
            sample_rate=24000,
            num_channels=1,
        )

    @property
    def model(self) -> str:
        return "offline-tts"

    @property
    def provider(self) -> str:
        return "offline"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.SynthesizeStream:
        class OfflineSynthesizeStream(tts.SynthesizeStream):
            async def _run(self, output_emitter: tts.AudioEmitter) -> None:
                output_emitter.initialize(
                    request_id="dummy",
                    sample_rate=24000,
                    num_channels=1,
                    mime_type="audio/pcm",
                    stream=True,
                )
                async for item in self._input_ch:
                    if isinstance(item, self._FlushSentinel):
                        output_emitter.flush()
                        continue
                    
                    logger.info("[OFFLINE TTS SPEAKING]: %s", item)
                    char_count = len(item)
                    duration_s = max(0.5, char_count * 0.05)
                    byte_count = int(duration_s * 24000 * 2)
                    byte_count = (byte_count // 2) * 2
                    
                    output_emitter.start_segment(segment_id="")
                    from livekit.agents.voice.io import TimedString
                    output_emitter.push_timed_transcript(TimedString(text=item, start_time=0.0, end_time=duration_s))
                    output_emitter.push(b"\0" * byte_count)
                    output_emitter.end_segment()
                    
        return OfflineSynthesizeStream(
            tts=self,
            conn_options=conn_options,
        )



def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


_setup_logging()

config = Config.from_yaml()
logger.info("Loaded config for room=%s", config.livekit.room)

vector_store = QdrantVectorStore(
    url=config.qdrant.url,
    collection_name=config.qdrant.collection_name,
    embedding_model=config.qdrant.embedding_model,
)

vector_store._ensure_model()

server = AgentServer(
    ws_url=config.livekit.url,
    api_key=config.livekit.api_key,
    api_secret=config.livekit.api_secret.get_secret_value(),
)

operator: CVDisplayOperator | None = None
if config.operator is not None:
    operator = CVDisplayOperator(config.operator, config.livekit)


@server.on("worker_started")
def on_worker_started():
    if operator is not None:
        async def start_operator():
            logger.info("Starting operator at worker startup...")
            try:
                await operator.start()
                logger.info("Operator started successfully at worker startup")
            except Exception as e:
                logger.exception("Failed to start operator at worker startup: %s", e)

        asyncio.create_task(start_operator())


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    if operator is not None:
        async def start_operator():
            logger.info("Starting operator...")
            await operator.start()
            logger.info("Operator started successfully")

        asyncio.create_task(start_operator())

    agent = CosmosAgent(
        vector_store=vector_store,
        operator=operator,
        instructions=config.agent.instructions,
    )

    stt_impl = (
        OfflineSTT()
        if config.agent.stt == "offline"
        else inference.STT(config.agent.stt)
    )

    tts_impl = (
        OfflineTTS()
        if config.agent.tts == "offline"
        else inference.TTS(config.agent.tts)
    )

    llm_impl = SafeLLM(
        model=config.agent.llm,
        base_url=config.vllm.base_url + "/v1",
        api_key=config.livekit.api_key,
        api_secret=config.livekit.api_secret.get_secret_value(),
    )

    if operator is not None:
        operator.llm_impl = llm_impl

    session = AgentSession(
        stt=stt_impl,
        llm=llm_impl,
        tts=tts_impl,
        vad=ctx.proc.userdata["vad"],
    )

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        room=ctx.room,
        agent=agent,
    )

    await session.generate_reply(
        instructions="Greet the user briefly and let them know you can help monitor feeds and control the stream."
    )


if __name__ == "__main__":
    cli.run_app(server)
