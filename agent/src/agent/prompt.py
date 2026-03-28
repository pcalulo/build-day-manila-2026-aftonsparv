"""System prompt and analysis logic for the guessing game agent.

=== EDIT THIS FILE ===

This is where you define your agent's strategy:
- What system prompt to use
- How to analyze each frame
- When to submit a guess vs. gather more context
"""

from __future__ import annotations

import base64
import io
import os
from collections import deque

import httpx
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from core import Frame

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_W, FRAME_H = 640, 480
_API_KEY = os.getenv("LLM_API_KEY", "")
_OR_URL = "https://openrouter.ai/api/v1/messages"
_MODEL = "google/gemini-3-flash-preview"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a young child participating as the guesser in a game of charades.\

You are watching a sequence of 6 frames (arranged in a 2×3 grid, \
left-to-right, top-to-bottom) from a live video of someone playing charades. \
Analyze the motion and gestures across frames. 

Make your best guess at the word or phrase being \
acted out. Focus on the actions, not details like background or clothing. \
Consider your certainty level at the guess you made. If you are not \
confident, you can say exactly "SKIP" instead of guessing.\

If you are certain, respond only with your guess and nothing else. Your \
correctness will be judged against the actual answer, and non-answer text \
will confuse the evaluator.

## Hints
The person may be acting out the motions of an animal. Both palms forming \
a fin above their head in a swimming motion may indicate "shark". \
They may hold their arms up like a praying mantis, or prance around like \
a dinosaur. Consider this imitation as a possibility.

Alternatively they could just be acting out a human action, such as boxing,
brushing teeth, or drinking from a cup.
"""


# ---------------------------------------------------------------------------
# FrameBuffer
# ---------------------------------------------------------------------------

class FrameBuffer:
    def __init__(self) -> None:
        self._buf: deque[Frame] = deque(maxlen=6)

    def add(self, frame: Frame) -> None:
        self._buf.append(frame)

    def get_frames(self) -> list[Frame]:
        return list(self._buf)

    def is_full(self) -> bool:
        return len(self._buf) == 6


frame_buffer = FrameBuffer()


# ---------------------------------------------------------------------------
# Mosaic helpers
# ---------------------------------------------------------------------------

def build_mosaic(frames: list[Frame]) -> Image.Image:
    canvas = np.zeros((FRAME_H * 3, FRAME_W * 2, 3), dtype=np.uint8)
    for idx, f in enumerate(frames):
        img = f.image.resize((FRAME_W, FRAME_H)).convert("RGB")
        row, col = divmod(idx, 2)
        canvas[row * FRAME_H:(row + 1) * FRAME_H, col * FRAME_W:(col + 1) * FRAME_W] = np.array(img)
    return Image.fromarray(canvas)


def encode_mosaic(mosaic: Image.Image) -> str:
    buf = io.BytesIO()
    mosaic.save(buf, format="JPEG", quality=80)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# OpenRouter inference
# ---------------------------------------------------------------------------

async def call_openrouter(mosaic_b64: str) -> str | None:
    payload = {
        "model": _MODEL,
        "max_tokens": 512,
        "system": SYSTEM_PROMPT,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": mosaic_b64,
                    },
                },
                {
                    "type": "text",
                    "text": "What is this person acting out in charades? Focus on the actions",
                },
            ],
        }],
    }
    headers = {
        "x-api-key": _API_KEY,
        "anthropic-version": "2023-06-01",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(_OR_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def analyze(frame: Frame) -> str | None:
    """Analyze a single frame and return a guess, or None to skip.

    Buffers frames until 6 are collected, then assembles a mosaic and
    queries Claude via OpenRouter for a charades guess.

    Args:
        frame: A Frame with .image (PIL Image) and .timestamp.

    Returns:
        A text guess string, or None to skip this frame.
    """
    frame_buffer.add(frame)
    n = len(frame_buffer.get_frames())

    if not frame_buffer.is_full():
        print(f"  [agent] Buffering {n}/6 frames")
        return None

    try:
        mosaic = build_mosaic(frame_buffer.get_frames())
        mosaic_b64 = encode_mosaic(mosaic)
        response_text = await call_openrouter(mosaic_b64)
        if response_text:
            print(f"  [agent] Response:\n{response_text}")
            for line in response_text.splitlines():
                if line.strip().lower().startswith("guess:"):
                    return line.split(":", 1)[1].strip()
            
            return response_text.strip()
    except Exception as e:
        print(f"  [agent] OpenRouter error: {e}")

    return None
