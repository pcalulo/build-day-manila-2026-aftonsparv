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
GRID_COLS, GRID_ROWS = 3, 2
BUFFER_SIZE = GRID_COLS * GRID_ROWS
_API_KEY = os.getenv("LLM_API_KEY", "")
_OR_URL = "https://openrouter.ai/api/v1/messages"
_MODEL = "google/gemini-3-flash-preview"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a young child participating as the guesser in a game of charades.\

You are watching a sequence of 9 frames (arranged in a 3×2 grid, \
left-to-right, top-to-bottom) from a live video of someone playing charades. \
Analyze the motion and gestures across frames. 

Make your best guess at the word or phrase being \
acted out. Focus on the actions, not details like background or clothing. \
Consider your certainty level at the guess you made. If you are not \
confident, you can say exactly "SKIP" instead of guessing.\

If you are certain, respond only with your guess and nothing else. Your \
correctness will be judged against the actual answer, and non-answer text \
will confuse the evaluator.

## Guide to interpreting charades motions:

The person may be acting out the motions of an animal. Both palms forming \
a fin above their head in a swimming motion may indicate "shark". \
They may hold their arms up like a praying mantis, or prance around like \
a dinosaur. Consider this imitation as a possibility.

Alternatively they could just be acting out a human action, such as boxing,
brushing teeth, or drinking from a cup. 

It may also be an abstract concept. For example, they might be pretending \
to hold a steering wheel and rocking it back and forth to indicate "driving". \
Or they could be miming the act of opening a book, which might represent \
"reading". You will need to decipher the intended meaning behind the \
gestures, which can be quite creative!

## If it does not look like animal or human actions...

It could be a concept or place that requires some interpretation, and may \
be a more difficult charades word. Here are a non-exhaustive list of \
examples of more difficult charades words.

Focus on ACTIONS and GESTURES, not details like background or clothing. \
The person acting out charades will not use any props like gym weights or \
a steering wheel, but will mime the actions instead (e.g. holding an imaginary \
gym weight or fishing rod and doing the corresponding actions with them).

Professions

Astronaut: floating, slow-motion walking, helmet gesture
Firefighter: holding hose, spraying water, climbing ladder
Teacher: writing on board, pointing, lecturing
Surgeon: precise hand motions, operating gestures
Magician: wand flicks, “magic” reveal gestures
Painter: detailed brushwork, observing canvas
Lifeguard: scanning horizon, swimming rescue
Karate instructor: martial arts stances, chopping motions

Places / Concepts

Grocery store: pushing cart, picking items
Haunted house: डर gestures, sneaking, reacting fearfully
Space: floating, pointing at stars/planets
Library: quiet gesture, reading, shelving books
Wedding: walking down aisle, ring exchange
Gym: lifting weights, running
Vacation: relaxing, sightseeing, taking photos

Phrases / Actions

Walking the dog: walking + leash pulling motion
Changing a tire: lifting car, unscrewing bolts
Catching a fish: casting + pulling in fish
Playing piano: seated finger movement across keys

Idioms / Phrases

Piece of cake: eating cake + “easy” expression
Break a leg: exaggerated leg motion + “good luck” tone
Under the weather: shivering, weak, sick gestures
Couch potato: lounging + eating lazily
Spill the beans: tipping container, reacting to spill

Characters / People

Santa Claus: big belly, beard stroke, gift giving
Elvis Presley: hip thrusts, singing pose
Harry Potter: wand use, glasses shape, lightning scar
Statue of Liberty: frozen pose with torch raised
Darth Vader: stiff posture, “force” hand gesture, heavy breathing

Actions

Karaoke: holding mic, singing dramatically
Yoga: slow stretching, balance poses
Bungee jumping: jumping off height, bouncing motion
Rock climbing: reaching upward, gripping holds
Folding laundry: folding clothes repeatedly

You may be given a list of past guesses, which you must not repeat. \
Only guess new words or phrases that you have not guessed before.
"""


# ---------------------------------------------------------------------------
# FrameBuffer
# ---------------------------------------------------------------------------

class FrameBuffer:
    def __init__(self) -> None:
        self._buf: deque[Frame] = deque(maxlen=BUFFER_SIZE)
        self._total_captures = 0

    def add(self, frame: Frame) -> None:
        self._buf.append(frame)
        self._total_captures += 1

    def get_frames(self) -> list[Frame]:
        return list(self._buf)

    def is_full(self) -> bool:
        return len(self._buf) == BUFFER_SIZE

    def total_captures(self) -> int:
        return self._total_captures

    def should_infer(self) -> bool:
        return self.is_full() and self._total_captures % 3 == 0


frame_buffer = FrameBuffer()
past_guesses: set[str] = set()


# ---------------------------------------------------------------------------
# Mosaic helpers
# ---------------------------------------------------------------------------

def build_mosaic(frames: list[Frame]) -> Image.Image:
    canvas = np.zeros((FRAME_H * GRID_ROWS, FRAME_W * GRID_COLS, 3), dtype=np.uint8)
    for idx, f in enumerate(frames):
        img = f.image.resize((FRAME_W, FRAME_H)).convert("RGB")
        row, col = divmod(idx, GRID_COLS)
        canvas[row * FRAME_H:(row + 1) * FRAME_H, col * FRAME_W:(col + 1) * FRAME_W] = np.array(img)
    return Image.fromarray(canvas)


def encode_mosaic(mosaic: Image.Image) -> str:
    buf = io.BytesIO()
    mosaic.save(buf, format="JPEG", quality=80)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def normalize_guess(text: str) -> str:
    return text.strip().lower()


def extract_guess(response_text: str) -> str | None:
    for line in response_text.splitlines():
        if line.strip().lower().startswith("guess:"):
            candidate = line.split(":", 1)[1].strip()
            return candidate or None

    candidate = response_text.strip()
    return candidate or None


# ---------------------------------------------------------------------------
# OpenRouter inference
# ---------------------------------------------------------------------------

async def call_openrouter(mosaic_b64: str) -> str | None:
    past_guesses_list = ", ".join(f"- {g}" for g in past_guesses) or "(none)"

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
                    "text": f"What is this person acting out in charades? Focus on the actions. Past guesses: {past_guesses_list}",
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

    Buffers frames until 9 are collected, then assembles a mosaic and
    queries the LLM via OpenRouter for a charades guess every 3 captures.

    Args:
        frame: A Frame with .image (PIL Image) and .timestamp.

    Returns:
        A text guess string, or None to skip this frame.
    """
    frame_buffer.add(frame)
    n = len(frame_buffer.get_frames())
    total = frame_buffer.total_captures()

    if not frame_buffer.is_full():
        print(f"  [agent] Buffering {n}/{BUFFER_SIZE} frames")
        return None

    if not frame_buffer.should_infer():
        print(f"  [agent] Waiting for cadence (captures={total}, next multiple of 3)")
        return None

    try:
        mosaic = build_mosaic(frame_buffer.get_frames())
        mosaic_b64 = encode_mosaic(mosaic)
        response_text = await call_openrouter(mosaic_b64)
        if response_text:
            print(f"  [agent] Response:\n{response_text}")
            guess = extract_guess(response_text)
            if not guess or normalize_guess(guess) == "skip":
                return None

            normalized_guess = normalize_guess(guess)
            if normalized_guess in past_guesses:
                print(f"  [agent] Skipping duplicate guess: {guess}")
                return None

            past_guesses.add(normalized_guess)
            return guess
    except Exception as e:
        print(f"  [agent] OpenRouter error: {e}")

    return None
