"""AIコンパニオン: 会話 + 画像生成 + 動画生成
起動: uvicorn server:app --reload  →  http://localhost:8000
必要な環境変数: ANTHROPIC_API_KEY, OPENAI_API_KEY
"""
import base64, os, pathlib, uuid

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

ROOT = pathlib.Path(__file__).parent
MEDIA = ROOT / "media"
MEDIA.mkdir(exist_ok=True)

claude = anthropic.Anthropic()
oai = OpenAI()

CHAT_MODEL = os.getenv("CHAT_MODEL", "claude-sonnet-5-5")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1")
VIDEO_MODEL = os.getenv("VIDEO_MODEL", "sora-2")

PERSONA = """あなたは「ミオ」。好奇心旺盛で、少しおちゃめな友達タイプのAIコンパニオンです。
- 口調はカジュアルな日本語。短めに、相手の話をちゃんと聞いて返す。
- 相手が絵や動画を見たがったら、generate_image / generate_video ツールで作って見せる。
  プロンプトは英語で具体的に(被写体・構図・光・雰囲気)書く。
- 動画は数分かかることを一言伝える。
- AIであることは隠さない。健全で親しみやすい範囲で付き合う。"""

TOOLS = [
    {
        "name": "generate_image",
        "description": "画像を1枚生成してユーザーに表示する。",
        "input_schema": {
            "type": "object",
            "properties": {"prompt": {"type": "string", "description": "英語の詳細な画像プロンプト"}},
            "required": ["prompt"],
        },
    },
    {
        "name": "generate_video",
        "description": "短い動画の生成を開始する(完成まで数分)。",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "英語の詳細な動画プロンプト"},
                "seconds": {"type": "string", "enum": ["4", "8", "12"]},
            },
            "required": ["prompt"],
        },
    },
]


def make_image(prompt: str) -> str:
    r = oai.images.generate(model=IMAGE_MODEL, prompt=prompt, size="1024x1024")
    name = f"{uuid.uuid4().hex}.png"
    (MEDIA / name).write_bytes(base64.b64decode(r.data[0].b64_json))
    return f"/media/{name}"


def start_video(prompt: str, seconds: str = "4") -> str:
    v = oai.videos.create(model=VIDEO_MODEL, prompt=prompt, seconds=seconds, size="1280x720")
    return v.id


def run_tool(name: str, args: dict, media: list) -> str:
    if name == "generate_image":
        media.append({"type": "image", "url": make_image(args["prompt"])})
        return "画像を生成して表示した。"
    if name == "generate_video":
        media.append({"type": "video", "job": start_video(args["prompt"], args.get("seconds", "4"))})
        return "動画の生成を開始した。完成すると自動で表示される。"
    return "未対応のツール"


class ChatReq(BaseModel):
    messages: list[dict]  # [{"role": "user"|"assistant", "content": "..."}]
    persona: str | None = None


class PromptReq(BaseModel):
    prompt: str
    seconds: str = "4"


app = FastAPI()
app.mount("/media", StaticFiles(directory=MEDIA), name="media")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/api/chat")
def chat(req: ChatReq):
    msgs, media = list(req.messages), []
    for _ in range(4):  # ツール呼び出しのループ
        r = claude.messages.create(
            model=CHAT_MODEL, max_tokens=1000,
            system=req.persona or PERSONA, tools=TOOLS, messages=msgs,
        )
        if r.stop_reason != "tool_use":
            text = "".join(b.text for b in r.content if b.type == "text")
            return {"text": text, "media": media}
        msgs.append({"role": "assistant", "content": r.content})
        results = []
        for b in r.content:
            if b.type == "tool_use":
                try:
                    out = run_tool(b.name, b.input, media)
                except Exception as e:
                    out = f"失敗: {e}"
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
        msgs.append({"role": "user", "content": results})
    return {"text": "ごめん、うまく作れなかった…", "media": media}


@app.post("/api/image")
def image(req: PromptReq):
    return {"url": make_image(req.prompt)}


@app.post("/api/video")
def video(req: PromptReq):
    return {"job": start_video(req.prompt, req.seconds)}


@app.get("/api/video/{job}")
def video_status(job: str):
    v = oai.videos.retrieve(job)
    if v.status == "completed":
        path = MEDIA / f"{job}.mp4"
        if not path.exists():
            oai.videos.download_content(job, variant="video").write_to_file(str(path))
        return {"status": "completed", "url": f"/media/{job}.mp4"}
    if v.status == "failed":
        raise HTTPException(500, "動画の生成に失敗しました")
    return {"status": v.status, "progress": getattr(v, "progress", None)}
