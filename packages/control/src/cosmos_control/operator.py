from __future__ import annotations

import asyncio
import io
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

import numpy as np
from livekit import rtc
from PIL import Image, ImageDraw, ImageFont

from cosmos_core import Overlay, StreamOperator
from cosmos_utils import generate_livekit_token

if TYPE_CHECKING:
    from cosmos_live.config import LiveKitConfig, OperatorConfig

logger = logging.getLogger(__name__)

# Overlay slot layout constants
_SLOT_LAYOUTS: dict[str, dict[str, Any]] = {
    "lower_third": {
        "anchor": "bottom_left",
        "y_offset": 80,
        "x_offset": 40,
        "bg_color": (0, 0, 0),
        "bg_alpha": 180,  # 0–255
        "text_color": (255, 255, 255),
        "padding": (20, 12),
    },
    "title": {
        "anchor": "top_left",
        "y_offset": 40,
        "x_offset": 40,
        "bg_color": (0, 0, 0),
        "bg_alpha": 204,
        "text_color": (255, 255, 255),
        "padding": (16, 8),
    },
    "banner": {
        "anchor": "top_full",
        "y_offset": 0,
        "x_offset": 0,
        "bg_color": (30, 30, 200),
        "bg_alpha": 230,
        "text_color": (255, 255, 255),
        "padding": (20, 10),
    },
}

_FONT_SIZE = 24
_JPEG_QUALITY = 80


class _TrackInfo:
    """Tracks available for a participant."""

    def __init__(self) -> None:
        self.video: rtc.Track | None = None


# ---------------------------------------------------------------------------
# MJPEG HTTP handler
# ---------------------------------------------------------------------------

class _MJPEGHandler(BaseHTTPRequestHandler):
    """Serves an MJPEG stream, single frames, API status, commands, or a beautiful dashboard console UI."""

    server: _MJPEGServer  # type: ignore[assignment]

    def do_GET(self) -> None:
        from urllib.parse import urlparse
        parsed_url = urlparse(self.path)
        
        if parsed_url.path == "/stream":
            self._stream_mjpeg()
        elif parsed_url.path == "/frame":
            self._serve_frame()
        elif parsed_url.path == "/api/status":
            self._serve_status()
        elif parsed_url.path == "/api/command":
            self._handle_command()
        elif parsed_url.path == "/api/ollama_models":
            self._serve_ollama_models()
        else:
            self._serve_index()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def end_headers(self) -> None:
        origin = self.headers.get("Origin")
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def _serve_index(self) -> None:
        html = (
            "<!DOCTYPE html>"
            "<html>"
            "<head>"
            "  <meta charset=\"utf-8\">"
            "  <title>Cosmos Live \u2014 AI Operator Console</title>"
            "  <style>"
            "    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');"
            "    "
            "    :root {"
            "      --bg: #030303;"
            "      --card-bg: rgba(22, 22, 38, 0.45);"
            "      --card-border: rgba(255, 255, 255, 0.08);"
            "      --purple: #a855f7;"
            "      --blue: #3b82f6;"
            "      --green: #10b981;"
            "      --red: #f43f5e;"
            "      --text: #f3f4f6;"
            "      --text-muted: #9ca3af;"
            "    }"
            "    "
            "    * { box-sizing: border-box; margin: 0; padding: 0; }"
            "    "
            "    body {"
            "      font-family: 'Outfit', sans-serif;"
            "      background: radial-gradient(circle at 50% 50%, #0c0b1a 0%, #030303 100%);"
            "      color: var(--text);"
            "      height: 100vh;"
            "      display: flex;"
            "      flex-direction: column;"
            "      overflow: hidden;"
            "    }"
            "    "
            "    header {"
            "      display: flex;"
            "      justify-content: space-between;"
            "      align-items: center;"
            "      padding: 16px 24px;"
            "      background: rgba(10, 10, 15, 0.7);"
            "      backdrop-filter: blur(12px);"
            "      border-bottom: 1px solid var(--card-border);"
            "      z-index: 10;"
            "    }"
            "    "
            "    .logo-container {"
            "      display: flex;"
            "      align-items: center;"
            "      gap: 12px;"
            "    }"
            "    "
            "    .logo {"
            "      font-size: 24px;"
            "      font-weight: 800;"
            "      letter-spacing: 2px;"
            "      background: linear-gradient(to right, #c084fc, #6366f1);"
            "      -webkit-background-clip: text;"
            "      -webkit-text-fill-color: transparent;"
            "    }"
            "    "
            "    .badge {"
            "      display: flex;"
            "      align-items: center;"
            "      gap: 6px;"
            "      background: rgba(16, 185, 129, 0.15);"
            "      border: 1px solid rgba(16, 185, 129, 0.3);"
            "      color: var(--green);"
            "      padding: 4px 10px;"
            "      border-radius: 9999px;"
            "      font-size: 11px;"
            "      font-weight: 700;"
            "      letter-spacing: 0.5px;"
            "      text-transform: uppercase;"
            "    }"
            "    "
            "    .badge-pulse {"
            "      width: 6px;"
            "      height: 6px;"
            "      background-color: var(--green);"
            "      border-radius: 50%;"
            "      box-shadow: 0 0 8px var(--green);"
            "      animation: pulse 1.5s infinite;"
            "    }"
            "    "
            "    @keyframes pulse {"
            "      0% { transform: scale(0.9); opacity: 0.6; }"
            "      50% { transform: scale(1.2); opacity: 1; box-shadow: 0 0 12px var(--green); }"
            "      100% { transform: scale(0.9); opacity: 0.6; }"
            "    }"
            "    "
            "    .main-grid {"
            "      display: grid;"
            "      grid-template-columns: 1fr 340px;"
            "      height: calc(100vh - 73px);"
            "      width: 100vw;"
            "      overflow: hidden;"
            "    }"
            "    "
            "    .left-section {"
            "      display: flex;"
            "      flex-direction: column;"
            "      padding: 24px;"
            "      gap: 20px;"
            "      overflow-y: auto;"
            "      align-items: center;"
            "      justify-content: center;"
            "    }"
            "    "
            "    .player-container {"
            "      width: 100%;"
            "      max-width: 960px;"
            "      aspect-ratio: 16/9;"
            "      background: #000;"
            "      border-radius: 16px;"
            "      border: 1px solid var(--card-border);"
            "      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6), 0 0 20px rgba(168, 85, 247, 0.1);"
            "      position: relative;"
            "      overflow: hidden;"
            "    }"
            "    "
            "    .player-img {"
            "      width: 100%;"
            "      height: 100%;"
            "      object-fit: contain;"
            "      display: block;"
            "    }"
            "    "
            "    .player-controls {"
            "      position: absolute;"
            "      bottom: 12px;"
            "      left: 12px;"
            "      right: 12px;"
            "      display: flex;"
            "      justify-content: space-between;"
            "      align-items: center;"
            "      background: rgba(10, 10, 20, 0.7);"
            "      backdrop-filter: blur(8px);"
            "      padding: 10px 16px;"
            "      border-radius: 12px;"
            "      border: 1px solid rgba(255, 255, 255, 0.05);"
            "      opacity: 0;"
            "      transform: translateY(10px);"
            "      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);"
            "    }"
            "    "
            "    .player-container:hover .player-controls {"
            "      opacity: 1;"
            "      transform: translateY(0);"
            "    }"
            "    "
            "    .player-btn-group {"
            "      display: flex;"
            "      gap: 8px;"
            "    }"
            "    "
            "    .control-btn {"
            "      background: rgba(255, 255, 255, 0.06);"
            "      border: 1px solid rgba(255, 255, 255, 0.08);"
            "      color: var(--text);"
            "      padding: 6px 12px;"
            "      border-radius: 6px;"
            "      font-size: 12px;"
            "      font-weight: 500;"
            "      cursor: pointer;"
            "      transition: all 0.2s;"
            "    }"
            "    "
            "    .control-btn:hover {"
            "      background: rgba(255, 255, 255, 0.15);"
            "      border-color: rgba(255, 255, 255, 0.2);"
            "    }"
            "    "
            "    .control-btn.active {"
            "      background: var(--purple);"
            "      border-color: var(--purple);"
            "      box-shadow: 0 0 10px rgba(168, 85, 247, 0.4);"
            "    }"
            "    "
            "    .right-sidebar {"
            "      background: rgba(10, 10, 18, 0.65);"
            "      backdrop-filter: blur(20px);"
            "      border-left: 1px solid var(--card-border);"
            "      display: flex;"
            "      flex-direction: column;"
            "      padding: 20px;"
            "      gap: 20px;"
            "      overflow-y: auto;"
            "    }"
            "    "
            "    .panel {"
            "      background: var(--card-bg);"
            "      border: 1px solid var(--card-border);"
            "      border-radius: 12px;"
            "      padding: 16px;"
            "    }"
            "    "
            "    .panel-title {"
            "      font-size: 14px;"
            "      font-weight: 700;"
            "      color: var(--text-muted);"
            "      margin-bottom: 12px;"
            "      text-transform: uppercase;"
            "      letter-spacing: 1px;"
            "      display: flex;"
            "      justify-content: space-between;"
            "      align-items: center;"
            "    }"
            "    "
            "    .feed-list {"
            "      display: flex;"
            "      flex-direction: column;"
            "      gap: 8px;"
            "    }"
            "    "
            "    .feed-item {"
            "      display: flex;"
            "      justify-content: space-between;"
            "      align-items: center;"
            "      padding: 10px 14px;"
            "      background: rgba(255, 255, 255, 0.03);"
            "      border: 1px solid rgba(255, 255, 255, 0.05);"
            "      border-radius: 8px;"
            "      cursor: pointer;"
            "      font-size: 13px;"
            "      font-weight: 500;"
            "      transition: all 0.2s;"
            "    }"
            "    "
            "    .feed-item:hover {"
            "      background: rgba(255, 255, 255, 0.08);"
            "      border-color: rgba(255, 255, 255, 0.1);"
            "    }"
            "    "
            "    .feed-item.active {"
            "      background: rgba(59, 130, 246, 0.15);"
            "      border-color: rgba(59, 130, 246, 0.4);"
            "      color: #93c5fd;"
            "    }"
            "    "
            "    .feed-status {"
            "      width: 8px;"
            "      height: 8px;"
            "      border-radius: 50%;"
            "      background: var(--text-muted);"
            "    }"
            "    "
            "    .feed-item.active .feed-status {"
            "      background: var(--blue);"
            "      box-shadow: 0 0 6px var(--blue);"
            "    }"
            "    "
            "    .overlays-panel {"
            "      display: flex;"
            "      flex-direction: column;"
            "      gap: 12px;"
            "    }"
            "    "
            "    .overlay-slot {"
            "      display: flex;"
            "      flex-direction: column;"
            "      gap: 6px;"
            "    }"
            "    "
            "    .overlay-label {"
            "      font-size: 11px;"
            "      font-weight: 600;"
            "      color: var(--text-muted);"
            "      text-transform: uppercase;"
            "    }"
            "    "
            "    .overlay-input-group {"
            "      display: flex;"
            "      gap: 6px;"
            "    }"
            "    "
            "    .overlay-input {"
            "      flex: 1;"
            "      background: rgba(0, 0, 0, 0.3);"
            "      border: 1px solid var(--card-border);"
            "      border-radius: 6px;"
            "      padding: 6px 10px;"
            "      color: var(--text);"
            "      font-family: inherit;"
            "      font-size: 12px;"
            "      transition: all 0.2s;"
            "    }"
            "    "
            "    .overlay-input:focus {"
            "      outline: none;"
            "      border-color: var(--purple);"
            "    }"
            "    "
            "    .overlay-btn {"
            "      background: rgba(255, 255, 255, 0.06);"
            "      border: 1px solid rgba(255, 255, 255, 0.08);"
            "      color: var(--text);"
            "      padding: 6px 10px;"
            "      border-radius: 6px;"
            "      font-size: 12px;"
            "      font-weight: 500;"
            "      cursor: pointer;"
            "      transition: all 0.2s;"
            "    }"
            "    "
            "    .overlay-btn:hover {"
            "      background: rgba(255, 255, 255, 0.12);"
            "    }"
            "    "
            "    .overlay-btn.clear-btn:hover {"
            "      background: rgba(244, 63, 94, 0.2);"
            "      border-color: rgba(244, 63, 94, 0.4);"
            "      color: #fda4af;"
            "    }"
            "    "
            "    .console-logs {"
            "      font-family: 'JetBrains Mono', monospace;"
            "      font-size: 11px;"
            "      height: 180px;"
            "      overflow-y: auto;"
            "      background: rgba(0, 0, 0, 0.4);"
            "      border: 1px solid var(--card-border);"
            "      border-radius: 8px;"
            "      padding: 10px;"
            "      display: flex;"
            "      flex-direction: column;"
            "      gap: 6px;"
            "    }"
            "    "
            "    .log-line {"
            "      display: flex;"
            "      gap: 8px;"
            "      line-height: 1.4;"
            "    }"
            "    "
            "    .log-time { color: var(--text-muted); }"
            "    .log-msg { color: #e5e7eb; }"
            "    .log-msg.error { color: var(--red); }"
            "    .log-msg.cmd { color: #a5b4fc; font-weight: bold; }"
            "    "
            "    .command-bar {"
            "      display: flex;"
            "      gap: 8px;"
            "    }"
            "    "
            "    .command-input {"
            "      flex: 1;"
            "      background: rgba(0, 0, 0, 0.4);"
            "      border: 1px solid var(--card-border);"
            "      border-radius: 8px;"
            "      padding: 10px 14px;"
            "      color: var(--text);"
            "      font-family: inherit;"
            "      font-size: 13px;"
            "      transition: all 0.2s;"
            "    }"
            "    "
            "    .command-input:focus {"
            "      outline: none;"
            "      border-color: var(--purple);"
            "      box-shadow: 0 0 10px rgba(168, 85, 247, 0.2);"
            "    }"
            "    "
            "    .command-btn {"
            "      background: var(--purple);"
            "      border: none;"
            "      color: white;"
            "      padding: 10px 16px;"
            "      border-radius: 8px;"
            "      font-size: 13px;"
            "      font-weight: 600;"
            "      cursor: pointer;"
            "      transition: all 0.2s;"
            "    }"
            "    "
            "    .command-btn:hover {"
            "      background: #b55fe6;"
            "      box-shadow: 0 0 12px rgba(168, 85, 247, 0.4);"
            "    }"
            "    "
            "    .stat-row {"
            "      display: flex;"
            "      justify-content: space-between;"
            "      font-size: 12px;"
            "      padding: 4px 0;"
            "      border-bottom: 1px solid rgba(255, 255, 255, 0.03);"
            "    }"
            "    "
            "    .stat-label { color: var(--text-muted); }"
            "    .stat-val { font-weight: 600; }"
            "  </style>"
            "</head>"
            "<body>"
            "  <header>"
            "    <div class=\"logo-container\">"
            "      <div class=\"logo\">COSMOS LIVE</div>"
            "      <div class=\"badge\">"
            "        <div class=\"badge-pulse\"></div>"
            "        LIVE MONITOR"
            "      </div>"
            "    </div>"
            "    <div style=\"font-size:12px; font-weight:600; color:var(--text-muted); display:flex; gap:20px;\">"
            "      <span>ROOM: <span style=\"color:#a5b4fc\">cosmos</span></span>"
            "      <span>NODE: <span style=\"color:#a5b4fc\">localhost</span></span>"
            "    </div>"
            "  </header>"
            "  "
            "  <div class=\"main-grid\">"
            "    <div class=\"left-section\">"
            "      <div class=\"player-container\">"
            "        <img id=\"stream-img\" class=\"player-img\" src=\"/stream\" alt=\"Cosmos Live Video Stream\">"
            "        "
            "        <div class=\"player-controls\">"
            "          <div style=\"font-size:11px; font-weight:700; letter-spacing:0.5px; text-transform:uppercase; color:var(--text-muted)\">"
            "            Active Output Feed"
            "          </div>"
            "          <div class=\"player-btn-group\">"
            "            <button id=\"btn-mjpeg\" class=\"control-btn active\" onclick=\"setMode('mjpeg')\">High Performance (MJPEG)</button>"
            "            <button id=\"btn-js\" class=\"control-btn\" onclick=\"setMode('js')\">JS Canvas (Compatible)</button>"
            "            <button class=\"control-btn\" style=\"background:rgba(16,185,129,0.15); border-color:var(--green); color:#6ee7b7\" onclick=\"refreshImage()\">Force Refresh</button>"
            "          </div>"
            "        </div>"
            "      </div>"
            "      "
            "      <div class=\"panel\" style=\"width:100%; max-width:960px; display:flex; flex-direction:column; gap:12px;\">"
            "        <div class=\"panel-title\" style=\"margin:0;\">AI Console & Operator Input</div>"
            "        <div class=\"command-bar\">"
            "          <input id=\"cmd-input\" type=\"text\" class=\"command-input\" placeholder=\"Type command (e.g. 'switch to cam1', 'set title to Alert!', or 'clear banner') and press Enter...\" onkeydown=\"if(event.key==='Enter') sendCommand()\">"
            "          <button class=\"command-btn\" onclick=\"sendCommand()\">EXECUTE</button>"
            "        </div>"
            "      </div>"
            "    </div>"
            "    "
            "    <div class=\"right-sidebar\">"
            "      <div class=\"panel\">"
            "        <div class=\"panel-title\">Active Feeds</div>"
            "        <div id=\"feed-list-container\" class=\"feed-list\">"
            "          <div class=\"feed-item active\">"
            "            <span>cam1</span>"
            "            <div class=\"feed-status\"></div>"
            "          </div>"
            "        </div>"
            "      </div>"
            "      "
            "      <div class=\"panel\">"
            "        <div class=\"panel-title\">"
            "          <span>AI Engines & Models</span>"
            "          <span style=\"font-size:10px; color:var(--purple); font-weight:bold; letter-spacing:0.5px;\">DYNAMIC SWAP</span>"
            "        </div>"
            "        <div style=\"display:flex; flex-direction:column; gap:12px;\">"
            "          <div class=\"overlay-slot\">"
            "            <div class=\"overlay-label\">Vision Ingestion Model</div>"
            "            <select id=\"select-vision-model\" class=\"overlay-input\" style=\"background-color:rgba(0,0,0,0.5); cursor:pointer; width:100%;\" onchange=\"setOllamaModel('vision', this.value)\">"
            "              <option value=\"\" disabled selected hidden>-- Choose Ingestion Model --</option>"
            "            </select>"
            "          </div>"
            "          <div class=\"overlay-slot\">"
            "            <div class=\"overlay-label\">Retrieval LLM Model</div>"
            "            <select id=\"select-llm-model\" class=\"overlay-input\" style=\"background-color:rgba(0,0,0,0.5); cursor:pointer; width:100%;\" onchange=\"setOllamaModel('llm', this.value)\">"
            "              <option value=\"\" disabled selected hidden>-- Choose LLM Model --</option>"
            "            </select>"
            "          </div>"
            "        </div>"
            "      </div>"
            "      "
            "      <div class=\"panel\">"
            "        <div class=\"panel-title\">Text Overlays</div>"
            "        <div class=\"overlays-panel\">"
            "          <div class=\"overlay-slot\">"
            "            <div class=\"overlay-label\">Title (Top Left)</div>"
            "            <div class=\"overlay-input-group\">"
            "              <input id=\"input-title\" type=\"text\" class=\"overlay-input\" placeholder=\"Enter title text...\">"
            "              <button class=\"overlay-btn\" onclick=\"setOverlayText('title')\">SET</button>"
            "              <button class=\"overlay-btn clear-btn\" onclick=\"clearOverlaySlot('title')\">X</button>"
            "            </div>"
            "          </div>"
            "          <div class=\"overlay-slot\">"
            "            <div class=\"overlay-label\">Lower Third (Bottom Left)</div>"
            "            <div class=\"overlay-input-group\">"
            "              <input id=\"input-lower_third\" type=\"text\" class=\"overlay-input\" placeholder=\"Enter lower third...\">"
            "              <button class=\"overlay-btn\" onclick=\"setOverlayText('lower_third')\">SET</button>"
            "              <button class=\"overlay-btn clear-btn\" onclick=\"clearOverlaySlot('lower_third')\">X</button>"
            "            </div>"
            "          </div>"
            "          <div class=\"overlay-slot\">"
            "            <div class=\"overlay-label\">Banner (Top Full Width)</div>"
            "            <div class=\"overlay-input-group\">"
            "              <input id=\"input-banner\" type=\"text\" class=\"overlay-input\" placeholder=\"Enter banner text...\">"
            "              <button class=\"overlay-btn\" onclick=\"setOverlayText('banner')\">SET</button>"
            "              <button class=\"overlay-btn clear-btn\" onclick=\"clearOverlaySlot('banner')\">X</button>"
            "            </div>"
            "          </div>"
            "        </div>"
            "      </div>"
            "      "
            "      <div class=\"panel\">"
            "        <div class=\"panel-title\">Operator Event Log</div>"
            "        <div id=\"console-logs-container\" class=\"console-logs\">"
            "          <div class=\"log-line\">"
            "            <span class=\"log-time\">[00:00:00]</span>"
            "            <span class=\"log-msg\">Waiting for events...</span>"
            "          </div>"
            "        </div>"
            "      </div>"
            "      "
            "      <div class=\"panel\">"
            "        <div class=\"panel-title\">Telemetry & Stats</div>"
            "        <div style=\"display:flex; flex-direction:column; gap:4px;\">"
            "          <div class=\"stat-row\"><span class=\"stat-label\">Resolution</span><span id=\"stat-res\" class=\"stat-val\">—</span></div>"
            "          <div class=\"stat-row\"><span class=\"stat-label\">Framerate</span><span id=\"stat-fps\" class=\"stat-val\">—</span></div>"
            "          <div class=\"stat-row\"><span class=\"stat-label\">Active Feed</span><span id=\"stat-feed\" class=\"stat-val\">—</span></div>"
            "          <div class=\"stat-row\"><span class=\"stat-label\">C++ FFI Server</span><span class=\"stat-val\" style=\"color:var(--green)\">v0.12.44</span></div>"
            "          <div class=\"stat-row\"><span class=\"stat-label\">Vision Model</span><span id=\"stat-vision-model\" class=\"stat-val\" style=\"color:var(--blue)\">—</span></div>"
            "          <div class=\"stat-row\"><span class=\"stat-label\">LLM Model</span><span id=\"stat-llm-model\" class=\"stat-val\" style=\"color:var(--purple)\">—</span></div>"
            "        </div>"
            "      </div>"
            "    </div>"
            "  </div>"
            "  <script>"
            "    let currentMode = 'mjpeg';"
            "    let frameTimer = null;"
            "    const img = document.getElementById('stream-img');"
            "    "
            "    img.onerror = () => {"
            "      console.warn(\"MJPEG stream failed or blocked. Automatically falling back to JS Compatibility Mode...\");"
            "      if (currentMode === 'mjpeg') {"
            "        setMode('js');"
            "      }"
            "    };"
            "    "
            "    function setMode(mode) {"
            "      currentMode = mode;"
            "      const mjpegBtn = document.getElementById('btn-mjpeg');"
            "      const jsBtn = document.getElementById('btn-js');"
            "      "
            "      if (mode === 'mjpeg') {"
            "        mjpegBtn.classList.add('active');"
            "        jsBtn.classList.remove('active');"
            "        if (frameTimer) {"
            "          clearInterval(frameTimer);"
            "          frameTimer = null;"
            "        }"
            "        img.src = '/stream';"
            "      } else {"
            "        jsBtn.classList.add('active');"
            "        mjpegBtn.classList.remove('active');"
            "        if (frameTimer) clearInterval(frameTimer);"
            "        "
            "        frameTimer = setInterval(() => {"
            "          img.src = '/frame?t=' + Date.now();"
            "        }, 40);"
            "      }"
            "      console.log(\"Rendering mode switched to: \" + mode);"
            "    }"
            "    "
            "    function refreshImage() {"
            "      if (currentMode === 'mjpeg') {"
            "        img.src = '';"
            "        setTimeout(() => { img.src = '/stream'; }, 50);"
            "      } else {"
            "        img.src = '/frame?t=' + Date.now();"
            "      }"
            "    }"
            "    "
            "    let modelsLoaded = false;"
            "    let currentVisionModel = \"\";"
            "    let currentLlmModel = \"\";"
            "    "
            "    async function loadOllamaModels() {"
            "      try {"
            "        const res = await fetch('/api/ollama_models');"
            "        if (!res.ok) return;"
            "        const data = await res.json();"
            "        "
            "        const visSelect = document.getElementById('select-vision-model');"
            "        const llmSelect = document.getElementById('select-llm-model');"
            "        "
            "        visSelect.innerHTML = '<option value=\"\" disabled selected hidden>-- Choose Ingestion Model --</option>';"
            "        llmSelect.innerHTML = '<option value=\"\" disabled selected hidden>-- Choose LLM Model --</option>';"
            "        "
            "        let hasVision = false;"
            "        let hasLlm = false;"
            "        "
            "        data.models.forEach(model => {"
            "          if (model === currentVisionModel) hasVision = true;"
            "          if (model === currentLlmModel) hasLlm = true;"
            "        });"
            "        "
            "        if (currentVisionModel && !hasVision) {"
            "          const opt = document.createElement('option');"
            "          opt.value = currentVisionModel;"
            "          opt.textContent = currentVisionModel + ' (Configured)';"
            "          visSelect.appendChild(opt);"
            "        }"
            "        "
            "        if (currentLlmModel && !hasLlm) {"
            "          const opt = document.createElement('option');"
            "          opt.value = currentLlmModel;"
            "          opt.textContent = currentLlmModel + ' (Configured)';"
            "          llmSelect.appendChild(opt);"
            "        }"
            "        "
            "        data.models.forEach(model => {"
            "          const opt1 = document.createElement('option');"
            "          opt1.value = model;"
            "          opt1.textContent = model;"
            "          visSelect.appendChild(opt1);"
            "          "
            "          const opt2 = document.createElement('option');"
            "          opt2.value = model;"
            "          opt2.textContent = model;"
            "          llmSelect.appendChild(opt2);"
            "        });"
            "        "
            "        modelsLoaded = true;"
            "        "
            "        if (currentVisionModel) visSelect.value = currentVisionModel;"
            "        if (currentLlmModel) llmSelect.value = currentLlmModel;"
            "      } catch (err) {"
            "        console.error(\"Error loading Ollama models:\", err);"
            "      }"
            "    }"
            "    "
            "    async function setOllamaModel(type, model) {"
            "      if (!model) return;"
            "      try {"
            "        console.log(`Setting Ollama ${type} model to ${model}...`);"
            "        const res = await fetch(`/api/command?cmd=set_ollama_model&type=${type}&model=${encodeURIComponent(model)}`);"
            "        const data = await res.json();"
            "        console.log(\"Model switch response:\", data);"
            "        updateStatus();"
            "      } catch (err) {"
            "        console.error(err);"
            "      }"
            "    }"
            "    "
            "    async function updateStatus() {"
            "      try {"
            "        const res = await fetch('/api/status');"
            "        if (!res.ok) return;"
            "        const data = await res.json();"
            "        "
            "        document.getElementById('stat-res').textContent = data.width + 'x' + data.height;"
            "        document.getElementById('stat-fps').textContent = data.fps + ' FPS';"
            "        document.getElementById('stat-feed').textContent = data.active_feed || 'None';"
            "        "
            "        document.getElementById('stat-vision-model').textContent = data.active_vision_model || '—';"
            "        document.getElementById('stat-llm-model').textContent = data.active_llm_model || '—';"
            "        "
            "        currentVisionModel = data.active_vision_model;"
            "        currentLlmModel = data.active_llm_model;"
            "        "
            "        if (modelsLoaded) {"
            "          const visSelect = document.getElementById('select-vision-model');"
            "          const llmSelect = document.getElementById('select-llm-model');"
            "          "
            "          if (document.activeElement !== visSelect && visSelect.value !== currentVisionModel) {"
            "            visSelect.value = currentVisionModel;"
            "          }"
            "          if (document.activeElement !== llmSelect && llmSelect.value !== currentLlmModel) {"
            "            llmSelect.value = currentLlmModel;"
            "          }"
            "        }"
            "        "
            "        for (const [slot, val] of Object.entries(data.overlays)) {"
            "          const input = document.getElementById('input-' + slot);"
            "          if (input && document.activeElement !== input) {"
            "            input.value = val;"
            "          }"
            "        }"
            "        "
            "        const list = document.getElementById('feed-list-container');"
            "        list.innerHTML = '';"
            "        data.available_feeds.forEach(feed => {"
            "          const isAct = feed === data.active_feed;"
            "          const item = document.createElement('div');"
            "          item.className = 'feed-item' + (isAct ? ' active' : '');"
            "          item.onclick = () => switchFeed(feed);"
            "          "
            "          const nameSpan = document.createElement('span');"
            "          nameSpan.textContent = feed;"
            "          "
            "          const statusDiv = document.createElement('div');"
            "          statusDiv.className = 'feed-status';"
            "          "
            "          item.appendChild(nameSpan);"
            "          item.appendChild(statusDiv);"
            "          list.appendChild(item);"
            "        });"
            "        "
            "        const logsContainer = document.getElementById('console-logs-container');"
            "        const scrollAtBottom = Math.abs(logsContainer.scrollHeight - logsContainer.clientHeight - logsContainer.scrollTop) < 10;"
            "        "
            "        logsContainer.innerHTML = '';"
            "        data.events.forEach(ev => {"
            "          const line = document.createElement('div');"
            "          line.className = 'log-line';"
            "          "
            "          const timeSpan = document.createElement('span');"
            "          timeSpan.className = 'log-time';"
            "          timeSpan.textContent = '[' + ev.time + ']';"
            "          "
            "          const msgSpan = document.createElement('span');"
            "          msgSpan.className = 'log-msg';"
            "          if (ev.level === 'ERROR') msgSpan.classList.add('error');"
            "          if (ev.message.startsWith('Command') || ev.message.startsWith('User')) msgSpan.classList.add('cmd');"
            "          msgSpan.textContent = ev.message;"
            "          "
            "          line.appendChild(timeSpan);"
            "          line.appendChild(msgSpan);"
            "          logsContainer.appendChild(line);"
            "        });"
            "        "
            "        if (scrollAtBottom) {"
            "          logsContainer.scrollTop = logsContainer.scrollHeight;"
            "        }"
            "      } catch (err) {"
            "        console.error(\"Error updating status:\", err);"
            "      }"
            "    }"
            "    "
            "    async function switchFeed(feedId) {"
            "      try {"
            "        await fetch('/api/command?cmd=feed&feed_id=' + encodeURIComponent(feedId));"
            "        updateStatus();"
            "      } catch (err) {"
            "        console.error(err);"
            "      }"
            "    }"
            "    "
            "    async function setOverlayText(slot) {"
            "      const text = document.getElementById('input-' + slot).value;"
            "      try {"
            "        await fetch('/api/command?cmd=overlay&slot=' + slot + '&text=' + encodeURIComponent(text));"
            "        updateStatus();"
            "      } catch (err) {"
            "        console.error(err);"
            "      }"
            "    }"
            "    "
            "    async function clearOverlaySlot(slot) {"
            "      try {"
            "        await fetch('/api/command?cmd=clear_overlay&slot=' + slot);"
            "        document.getElementById('input-' + slot).value = '';"
            "        updateStatus();"
            "      } catch (err) {"
            "        console.error(err);"
            "      }"
            "    }"
            "    "
            "    async function sendCommand() {"
            "      const input = document.getElementById('cmd-input');"
            "      const val = input.value.trim();"
            "      if (!val) return;"
            "      "
            "      try {"
            "        input.value = '';"
            "        const res = await fetch('/api/command?cmd=chat&text=' + encodeURIComponent(val));"
            "        const data = await res.json();"
            "        console.log(\"Command output:\", data);"
            "        updateStatus();"
            "      } catch (err) {"
            "        console.error(err);"
            "      }"
            "    }"
            "    "
            "    updateStatus().then(() => {"
            "      loadOllamaModels();"
            "    });"
            "    setInterval(updateStatus, 1000);"
            "  </script>"
            "</body>"
            "</html>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _serve_frame(self) -> None:
        op = self.server.operator
        with op._jpeg_cond:
            data = op._jpeg_data
        
        if not data:
            self.send_error(404, "Frame not available yet")
            return
            
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(data)

    def _serve_status(self) -> None:
        import json
        op = self.server.operator
        
        # Get active vision model dynamically
        vision_model = ""
        try:
            from cosmos_live.config import Config
            temp_cfg = Config.from_yaml()
            if temp_cfg.vllm and temp_cfg.vllm.model:
                vision_model = temp_cfg.vllm.model
        except Exception:
            pass

        # Get active LLM model dynamically
        llm_model = ""
        if hasattr(op, "llm_impl") and op.llm_impl is not None:
            if hasattr(op.llm_impl, "_opts") and hasattr(op.llm_impl._opts, "model"):
                llm_model = op.llm_impl._opts.model or ""
            else:
                try:
                    llm_model = op.llm_impl.model or ""
                except Exception:
                    pass

        # Fall back to yaml config if still empty
        if not llm_model:
            try:
                from cosmos_live.config import Config
                temp_cfg = Config.from_yaml()
                if temp_cfg.agent and temp_cfg.agent.llm:
                    llm_model = temp_cfg.agent.llm
            except Exception:
                pass
        
        status = {
            "active_feed": op.active_feed,
            "available_feeds": op.available_feeds,
            "fps": op._cfg.fps,
            "width": op._cfg.width,
            "height": op._cfg.height,
            "overlays": {k: v.data.get("text", "") for k, v in op._overlays.items()},
            "events": op._events,
            "active_vision_model": vision_model,
            "active_llm_model": llm_model,
        }
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())

    def _serve_ollama_models(self) -> None:
        import json
        import urllib.request
        op = self.server.operator
        
        ollama_url = "http://localhost:11434"
        try:
            from cosmos_live.config import Config
            temp_cfg = Config.from_yaml()
            if temp_cfg.vllm and temp_cfg.vllm.base_url:
                ollama_url = temp_cfg.vllm.base_url
        except Exception:
            pass
            
        models = []
        try:
            req = urllib.request.Request(f"{ollama_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3.0) as response:
                data = json.loads(response.read().decode())
                for m in data.get("models", []):
                    if "name" in m:
                        models.append(m["name"])
        except Exception as e:
            logger.warning("Failed to fetch models from local Ollama: %s", e)
        
        seen = set()
        unique_models = []
        
        # Append local Ollama models (filtering out dangling layers/blobs)
        for m in models:
            if m not in seen and m:
                if not m.startswith("blobs:") and "sha256" not in m:
                    seen.add(m)
                    unique_models.append(m)
                
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(json.dumps({"models": unique_models}).encode())

    def _handle_command(self) -> None:
        from urllib.parse import parse_qs, urlparse
        import json
        
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        
        cmd = params.get("cmd", [None])[0]
        op = self.server.operator
        
        response = {"status": "error", "message": "Unknown command"}
        
        if cmd == "feed":
            feed_id = params.get("feed_id", [None])[0]
            if feed_id:
                asyncio.run_coroutine_threadsafe(op.set_feed(feed_id), op._loop)
                op._add_event(f"Switch active feed to: {feed_id}")
                response = {"status": "success", "message": f"Switched feed to {feed_id}"}
        
        elif cmd == "overlay":
            slot = params.get("slot", [None])[0]
            text = params.get("text", [None])[0]
            if slot and text:
                overlay = Overlay(kind="text", data={"text": text})
                asyncio.run_coroutine_threadsafe(op.set_overlay(slot, overlay), op._loop)
                op._add_event(f"Set overlay slot '{slot}' = '{text}'")
                response = {"status": "success", "message": f"Set overlay {slot} to '{text}'"}
                
        elif cmd == "clear_overlay":
            slot = params.get("slot", [None])[0]
            if slot:
                asyncio.run_coroutine_threadsafe(op.clear_overlay(slot), op._loop)
                op._add_event(f"Cleared overlay slot '{slot}'")
                response = {"status": "success", "message": f"Cleared overlay {slot}"}
                
        elif cmd == "set_ollama_model":
            model_type = params.get("type", [None])[0]
            model_name = params.get("model", [None])[0]
            if model_type and model_name:
                try:
                    import yaml
                    with open("config.yaml", "r") as f:
                        config_data = yaml.safe_load(f) or {}
                    
                    if model_type == "vision":
                        if "vllm" not in config_data:
                            config_data["vllm"] = {}
                        config_data["vllm"]["model"] = model_name
                    elif model_type == "llm":
                        if "agent" not in config_data:
                            config_data["agent"] = {}
                        config_data["agent"]["llm"] = model_name
                    
                    with open("config.yaml", "w") as f:
                        yaml.safe_dump(config_data, f, default_flow_style=False)
                        
                    op._add_event(f"Updated config.yaml {model_type} model to: {model_name}")
                except Exception as e:
                    logger.error("Failed to write updated model to config.yaml: %s", e)
                    op._add_event(f"Failed to write model to config.yaml: {e}", "ERROR")

                if model_type == "llm":
                    if hasattr(op, "llm_impl") and op.llm_impl is not None:
                        try:
                            op.llm_impl._opts.model = model_name
                            op._add_event(f"Live LLM model dynamically switched to: {model_name}")
                        except Exception as e:
                            logger.error("Failed to switch live LLM model in memory: %s", e)
                            op._add_event(f"Failed to switch live LLM in memory: {e}", "ERROR")
                    else:
                        op._add_event("Live LLM model updated in config (voice agent needs restart for LLM change)")
                else:
                    op._add_event(f"Live vision ingestion model dynamically switched to: {model_name}")

                response = {"status": "success", "message": f"Successfully updated {model_type} model to {model_name}"}

        elif cmd == "chat":
            text = params.get("text", [None])[0]
            if text:
                op._add_event(f"User Command Input: '{text}'")
                lower_text = text.lower()
                
                # Check simple quick-actions
                if "switch to" in lower_text:
                    switched = False
                    for f in op.available_feeds:
                        if f.lower() in lower_text:
                            asyncio.run_coroutine_threadsafe(op.set_feed(f), op._loop)
                            response = {"status": "success", "message": f"Switched feed to {f}"}
                            switched = True
                            break
                    if not switched:
                        response = {"status": "error", "message": "Participant not found"}
                elif "clear" in lower_text and "overlay" in lower_text:
                    slots_cleared = []
                    for slot in ["title", "lower_third", "banner"]:
                        if slot in lower_text or slot.replace("_", " ") in lower_text:
                            asyncio.run_coroutine_threadsafe(op.clear_overlay(slot), op._loop)
                            slots_cleared.append(slot)
                    if slots_cleared:
                        response = {"status": "success", "message": f"Cleared overlays: {', '.join(slots_cleared)}"}
                    else:
                        for slot in ["title", "lower_third", "banner"]:
                            asyncio.run_coroutine_threadsafe(op.clear_overlay(slot), op._loop)
                        response = {"status": "success", "message": "Cleared all overlays"}
                elif "overlay" in lower_text or "title" in lower_text or "banner" in lower_text or "lower" in lower_text:
                    slot = "title"
                    if "banner" in lower_text:
                        slot = "banner"
                    elif "lower" in lower_text:
                        slot = "lower_third"
                        
                    val = text
                    if ":" in text:
                        val = text.split(":", 1)[1].strip()
                    elif "to " in lower_text:
                        val = text.split("to ", 1)[1].strip()
                        
                    overlay = Overlay(kind="text", data={"text": val})
                    asyncio.run_coroutine_threadsafe(op.set_overlay(slot, overlay), op._loop)
                    response = {"status": "success", "message": f"Set overlay {slot} to '{val}'"}
                else:
                    response = {"status": "success", "message": f"Forwarded instruction: '{text}' to Cosmos Agent"}
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def _stream_mjpeg(self) -> None:
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        op = self.server.operator
        last_seq = -1

        while not op._stop_event.is_set():
            with op._jpeg_cond:
                # Wait until a new frame is available
                op._jpeg_cond.wait(timeout=1.0)
                seq = op._jpeg_seq
                if seq == last_seq:
                    continue
                data = op._jpeg_data
                last_seq = seq

            if not data:
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Silence per-request logs
        pass


class _MJPEGServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that holds a back-reference to the operator."""

    daemon_threads = True

    def __init__(self, addr: tuple[str, int], operator: CVDisplayOperator) -> None:
        self.operator = operator
        super().__init__(addr, _MJPEGHandler)


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class CVDisplayOperator(StreamOperator):
    """Stream operator that serves composited frames as an MJPEG HTTP stream.

    Connects to a LiveKit room as a separate participant, subscribes to
    video tracks, composites a single active feed with text overlays, and
    serves the result at ``http://localhost:<port>/stream``.
    """

    def __init__(
        self,
        operator_config: OperatorConfig,
        livekit_config: LiveKitConfig,
    ) -> None:
        self._cfg = operator_config
        self._livekit_config = livekit_config

        # LiveKit state
        self._room: rtc.Room | None = None
        self._available_tracks: dict[str, _TrackInfo] = {}

        # Active feed
        self._active_feed: str | None = None
        self._feed_lock = asyncio.Lock()
        self._latest_frame: np.ndarray | None = None
        self._consume_video_task: asyncio.Task[None] | None = None

        # Overlays — threading.Lock because the display thread reads them
        self._overlays: dict[str, Overlay] = {}
        self._overlay_lock = threading.Lock()

        # Frame lock — protects _latest_frame between asyncio and display thread
        self._frame_lock = threading.Lock()

        # Display / compositing thread
        self._stop_event = threading.Event()
        self._display_thread: threading.Thread | None = None

        # MJPEG output shared with HTTP handlers
        self._jpeg_cond = threading.Condition()
        self._jpeg_data: bytes = b""
        self._jpeg_seq: int = 0

        # HTTP server
        self._http_server: _MJPEGServer | None = None
        self._http_thread: threading.Thread | None = None

        # Placeholder cache
        self._placeholder: Image.Image | None = None

        # Font
        try:
            self._font = ImageFont.load_default(size=_FONT_SIZE)
        except TypeError:
            # Pillow < 10.1 fallback
            self._font = ImageFont.load_default()

        # Event logs
        self._events: list[dict[str, Any]] = []
        self._add_event("CVDisplayOperator initialized")

    def _add_event(self, message: str, level: str = "INFO") -> None:
        from datetime import datetime
        self._events.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": message,
            "level": level
        })
        if len(self._events) > 50:
            self._events.pop(0)

    # --- Public API ---

    @property
    def available_feeds(self) -> list[str]:
        """Return participant identities that have at least one subscribed track."""
        return list(self._available_tracks.keys())

    @property
    def active_feed(self) -> str | None:
        """Return the identity of the currently active feed, if any."""
        return self._active_feed

    async def start(self) -> None:
        """Connect to LiveKit and start the display loop.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._display_thread is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._add_event("Starting CVDisplayOperator")
        logger.info("Starting CVDisplayOperator")

        # Connect to LiveKit
        await self._connect_livekit()

        # Start compositing thread
        self._stop_event.clear()
        self._display_thread = threading.Thread(
            target=self._display_loop, name="cv-display", daemon=True
        )
        self._display_thread.start()

        # Start MJPEG HTTP server
        port = self._cfg.display_port
        self._http_server = _MJPEGServer(("0.0.0.0", port), self)
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            name="mjpeg-http",
            daemon=True,
        )
        self._http_thread.start()

        logger.info(
            "CVDisplayOperator started — stream at http://localhost:%d/stream",
            port,
        )

    async def stop(self) -> None:
        """Disconnect from LiveKit, stop tasks, close display."""
        self._add_event("Stopping CVDisplayOperator")
        logger.info("Stopping CVDisplayOperator")

        # Signal thread to exit
        self._stop_event.set()

        # Wake any MJPEG handlers waiting on the condition
        with self._jpeg_cond:
            self._jpeg_cond.notify_all()

        # Shut down HTTP server
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server = None
        if self._http_thread is not None:
            self._http_thread.join(timeout=5)
            self._http_thread = None

        # Cancel async consume task
        if self._consume_video_task is not None:
            self._consume_video_task.cancel()
            await asyncio.gather(self._consume_video_task, return_exceptions=True)
            self._consume_video_task = None

        # Join display thread
        if self._display_thread is not None:
            self._display_thread.join(timeout=5)
            self._display_thread = None

        # Disconnect from LiveKit
        if self._room is not None:
            await self._room.disconnect()
            self._room = None

        logger.info("CVDisplayOperator stopped")

    async def set_feed(self, feed_id: str) -> None:
        """Switch the active video feed by participant identity."""
        async with self._feed_lock:
            if feed_id == self._active_feed:
                return

            old_feed = self._active_feed
            logger.info("Switching feed: %s -> %s", old_feed, feed_id)
            self._add_event(f"Switching active feed: {old_feed} -> {feed_id}")

            # Cancel old consume task
            await self._cancel_consume_task()

            self._active_feed = feed_id
            with self._frame_lock:
                self._latest_frame = None

            # Start consuming new feed if tracks are available
            track_info = self._available_tracks.get(feed_id)
            if track_info is not None and track_info.video is not None:
                self._consume_video_task = asyncio.create_task(
                    self._consume_video(track_info.video)
                )
                
            else:
                logger.warning(
                    "Feed %s not found in available tracks; "
                    "will start consuming when tracks arrive",
                    feed_id,
                )

    async def set_overlay(self, slot: str, overlay: Overlay) -> None:
        """Set a named overlay slot."""
        with self._overlay_lock:
            self._overlays[slot] = overlay
        text = overlay.data.get("text", "")
        self._add_event(f"Overlay set: {slot} = '{text}'")
        logger.info("Overlay set: slot=%s kind=%s", slot, overlay.kind)

    async def clear_overlay(self, slot: str) -> None:
        """Remove the overlay from a named slot."""
        with self._overlay_lock:
            removed = self._overlays.pop(slot, None)
        if removed is not None:
            self._add_event(f"Overlay cleared: {slot}")
            logger.info("Overlay cleared: slot=%s", slot)

    # --- LiveKit connection ---

    async def _connect_livekit(self) -> None:
        """Connect to the LiveKit room as a subscribe-only participant."""
        self._room = rtc.Room()
        token = generate_livekit_token(
            self._livekit_config, "stream-operator", self._livekit_config.room
        )

        @self._room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            identity = participant.identity
            info = self._available_tracks.setdefault(identity, _TrackInfo())

            if track.kind == rtc.TrackKind.KIND_VIDEO:
                info.video = track
                self._add_event(f"Video track available from {identity}")
                logger.info("Video track available from %s", identity)

                # If there is no active feed set yet, automatically switch to this first one!
                if self._active_feed is None:
                    self._add_event(f"No active feed set. Auto-switching to: {identity}")
                    logger.info("No active feed set. Automatically switching to first available: %s", identity)
                    asyncio.create_task(self.set_feed(identity))
                # If this participant is the active feed and we're not
                # consuming yet, start consuming.
                elif identity == self._active_feed and self._consume_video_task is None:
                    self._consume_video_task = asyncio.create_task(
                        self._consume_video(track)
                    )

        @self._room.on("track_unsubscribed")
        def on_track_unsubscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            identity = participant.identity
            info = self._available_tracks.get(identity)
            if info is None:
                return

            if track.kind == rtc.TrackKind.KIND_VIDEO:
                info.video = None
                if identity == self._active_feed and self._consume_video_task is not None:
                    self._consume_video_task.cancel()
                    self._consume_video_task = None
                    with self._frame_lock:
                        self._latest_frame = None

                # Remove participant entry if no video track remains
                if info.video is None:
                    self._available_tracks.pop(identity, None)

            logger.info(
                "Track unsubscribed: kind=%s from %s", track.kind, identity
            )

        self._add_event(f"Connecting to LiveKit room={self._livekit_config.room}")
        logger.info(
            "Connecting to LiveKit %s room=%s as stream-operator",
            self._livekit_config.url,
            self._livekit_config.room,
        )
        options = rtc.RoomOptions(
            rtc_config=rtc.RtcConfiguration(
                ice_transport_type=rtc.IceTransportType.TRANSPORT_NOHOST
            )
        )
        await self._room.connect(self._livekit_config.url, token, options=options)
        self._add_event("Connected to LiveKit room")
        logger.info("Connected to LiveKit room")

    # --- Feed consumption ---

    async def _cancel_consume_task(self) -> None:
        """Cancel the running video consume task."""
        if self._consume_video_task is not None:
            self._consume_video_task.cancel()
            await asyncio.gather(self._consume_video_task, return_exceptions=True)
            self._consume_video_task = None

    async def _consume_video(self, track: rtc.Track) -> None:
        """Read frames from a LiveKit video track and store the latest."""
        frame_interval = 1.0 / self._cfg.fps
        last_frame_time = 0.0

        try:
            video_stream = rtc.VideoStream(track)
            async for frame_event in video_stream:
                now = time.monotonic()
                if now - last_frame_time < frame_interval:
                    continue
                last_frame_time = now

                frame = frame_event.frame
                rgb_frame = frame.convert(rtc.VideoBufferType.RGB24)
                # .copy() to own the data — livekit may reuse the buffer
                arr = np.frombuffer(rgb_frame.data, dtype=np.uint8).reshape(
                    (rgb_frame.height, rgb_frame.width, 3)
                ).copy()
                with self._frame_lock:
                    self._latest_frame = arr
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Error in video consume loop")

    # --- Compositing loop (dedicated thread) ---

    def _display_loop(self) -> None:
        """Composite frames at the target FPS and encode to JPEG.

        Runs in its own thread with absolute-time tracking to avoid drift.
        """
        frame_interval = 1.0 / self._cfg.fps
        next_frame_time = time.monotonic()

        while not self._stop_event.is_set():
            # Grab latest frame under lock
            with self._frame_lock:
                frame = self._latest_frame

            if frame is None:
                img = self._make_placeholder()
            else:
                img = self._resize_frame(frame)

            # Snapshot overlays under lock
            with self._overlay_lock:
                overlays = dict(self._overlays) if self._overlays else {}

            if overlays:
                img = self._apply_overlays(img, overlays)

            # Encode to JPEG and notify MJPEG clients
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
            jpeg = buf.getvalue()

            with self._jpeg_cond:
                self._jpeg_data = jpeg
                self._jpeg_seq += 1
                self._jpeg_cond.notify_all()

            # Advance to next absolute frame time to avoid cumulative drift
            next_frame_time += frame_interval
            sleep_time = next_frame_time - time.monotonic()
            if sleep_time > 0:
                if self._stop_event.wait(timeout=sleep_time):
                    break
            elif sleep_time < -frame_interval:
                # More than one frame behind — reset to avoid death spiral
                next_frame_time = time.monotonic()

    # --- Frame processing ---

    def _resize_frame(self, frame: np.ndarray) -> Image.Image:
        """Convert numpy RGB array to a PIL Image, resizing if needed."""
        img = Image.fromarray(frame)
        if img.width != self._cfg.width or img.height != self._cfg.height:
            img = img.resize(
                (self._cfg.width, self._cfg.height), Image.LANCZOS
            )
        return img

    def _make_placeholder(self) -> Image.Image:
        """Return a solid dark placeholder frame at output resolution with premium waiting text."""
        if self._placeholder is None:
            img = Image.new(
                "RGB",
                (self._cfg.width, self._cfg.height),
                self._cfg.placeholder_color,
            )
            draw = ImageDraw.Draw(img)
            text = "Cosmos Live \u2014 Waiting for Camera Feed..."
            # Center the text
            bbox = draw.textbbox((0, 0), text, font=self._font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x = (self._cfg.width - text_w) // 2
            y = (self._cfg.height - text_h) // 2
            # Draw semi-transparent background card for the message
            pad_x, pad_y = 30, 20
            draw.rectangle(
                [x - pad_x, y - pad_y, x + text_w + pad_x, y + text_h + pad_y],
                fill=(20, 20, 20, 200),
                outline=(100, 100, 100),
                width=2
            )
            draw.text((x, y), text, fill=(200, 200, 200), font=self._font)
            self._placeholder = img
        return self._placeholder

    # --- Overlay rendering ---

    def _apply_overlays(
        self, img: Image.Image, overlays: dict[str, Overlay]
    ) -> Image.Image:
        """Draw text overlays onto the image using Pillow."""
        # Work in RGBA so we can alpha-composite
        base = img.convert("RGBA")
        overlay_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_layer)

        for slot, overlay in overlays.items():
            text = overlay.data.get("text", "")
            if not text:
                continue
            self._draw_text_overlay(draw, slot, text)

        img = Image.alpha_composite(base, overlay_layer).convert("RGB")
        return img

    def _draw_text_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        slot: str,
        text: str,
    ) -> None:
        """Position and draw a text overlay based on the slot layout."""
        layout = _SLOT_LAYOUTS.get(slot, _SLOT_LAYOUTS["title"])

        bbox = draw.textbbox((0, 0), text, font=self._font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        pad_x, pad_y = layout["padding"]
        bg_w = text_w + pad_x * 2
        bg_h = text_h + pad_y * 2

        anchor = layout["anchor"]
        if anchor == "bottom_left":
            x = layout["x_offset"]
            y = self._cfg.height - layout["y_offset"] - bg_h
        elif anchor == "top_left":
            x = layout["x_offset"]
            y = layout["y_offset"]
        elif anchor == "top_full":
            x = 0
            y = layout["y_offset"]
            bg_w = self._cfg.width
        else:
            x = layout["x_offset"]
            y = layout["y_offset"]

        bg_color = layout["bg_color"]
        bg_alpha = layout["bg_alpha"]

        # Semi-transparent background rectangle
        draw.rectangle(
            [x, y, x + bg_w, y + bg_h],
            fill=(*bg_color, bg_alpha),
        )

        # Centered text
        text_x = x + (bg_w - text_w) // 2
        text_y = y + pad_y
        draw.text(
            (text_x, text_y),
            text,
            fill=layout["text_color"],
            font=self._font,
        )
