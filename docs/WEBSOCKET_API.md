# WebSocket API

Backend server: `python -m src.server.run_server`

Default URL: `ws://127.0.0.1:8765`

## Messages you receive

### `cognitive_load_update`

Sent when you subscribe to the cognitive-load-only stream (or on demand).

```json
{
  "type": "cognitive_load_update",
  "timestamp": 1730000000.0,
  "data": {
    "score": 0.42,
    "level": "medium",
    "overload_detected": false,
    "gaze_score": 0.31,
    "emotion_score": 0.44,
    "mouse_score": 0.52,
    "triggers": {}
  }
}
```

### `adaptation_update` (existing)

The full stream used by the extension; it includes `data.cognitive_load`.

## Messages you can send

### Subscribe to a stream

Full stream (default):

```json
{ "type": "subscribe", "stream": "full" }
```

Cognitive-load-only stream:

```json
{ "type": "subscribe", "stream": "cognitive_load" }
```

### Get the latest cognitive load (request/response)

```json
{ "type": "get_cognitive_load" }
```

### Get the latest full message

```json
{ "type": "get_latest" }
```

## Minimal client examples

### JavaScript (browser / Node)

```js
const ws = new WebSocket("ws://127.0.0.1:8765");

ws.onopen = () => {
  ws.send(JSON.stringify({ type: "subscribe", stream: "cognitive_load" }));
};

ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.type === "cognitive_load_update") {
    console.log("CLI:", msg.data.score, msg.data.level);
  }
};
```

### Python

```python
import asyncio
import json
import websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.send(json.dumps({"type": "subscribe", "stream": "cognitive_load"}))
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "cognitive_load_update":
                print("CLI:", msg["data"].get("score"), msg["data"].get("level"))

asyncio.run(main())
```

