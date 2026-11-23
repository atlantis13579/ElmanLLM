"""Exercise runtime web assets and the model API on one listener."""
import contextlib
import http.client
import json
import math
from pathlib import Path
import struct
import sys
import tempfile

from ServingProcess import listener


def tiny_qwen(root):
    root.mkdir()
    (root / "config.json").write_text(json.dumps(dict(
        model_type="qwen2", vocab_size=8, hidden_size=4, intermediate_size=8,
        num_hidden_layers=1, num_attention_heads=1, max_position_embeddings=64,
        tie_word_embeddings=True)))
    layer = "model.layers.0."
    tensors = {"model.embed_tokens.weight": [8, 4], "model.norm.weight": [4]}
    for norm in ("input_layernorm", "post_attention_layernorm"):
        tensors[layer + norm + ".weight"] = [4]
    for projection in ("q_proj", "k_proj", "v_proj"):
        tensors[layer + "self_attn." + projection + ".weight"] = [4, 4]
        tensors[layer + "self_attn." + projection + ".bias"] = [4]
    tensors[layer + "self_attn.o_proj.weight"] = [4, 4]
    for projection in ("gate_proj", "up_proj"):
        tensors[layer + "mlp." + projection + ".weight"] = [8, 4]
    tensors[layer + "mlp.down_proj.weight"] = [4, 8]
    offset = 0
    header = {}
    for name, shape in tensors.items():
        end = offset + math.prod(shape) * 4
        header[name] = dict(dtype="F32", shape=shape, data_offsets=[offset, end])
        offset = end
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def request(port, path, status=200, method="GET", payload=None):
    body = json.dumps(payload) if payload is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    with contextlib.closing(http.client.HTTPConnection("127.0.0.1", port, timeout=5)) as connection:
        connection.request(method, path, body, headers)
        response = connection.getresponse()
        data = response.read()
        assert response.status == status, (path, response.status, data)
        return data, response.headers


def check_serving(exe, root):
    tiny_qwen(root / "model")
    web = root / "site"
    assets = web / "assets"
    assets.mkdir(parents=True)
    html = b'<script type="module" src="assets/bundle.mjs"></script>'
    (web / "index.html").write_bytes(html)
    bundle = assets / "bundle.mjs"
    original = b"export const version = 1;\n" * 4096
    bundle.write_bytes(original)
    (root / "outside.txt").write_text("outside the web root")
    (web / "v1").mkdir()
    (web / "v1" / "models").write_text("static file with an API path")
    config = root / "server.json"
    settings = dict(
        model=dict(type="llm", path="model", name="test", backend="cpu"),
        http=dict(port=0, web_root="site"),
        chat=dict(token_ids_only=True, max_context=64, max_new_tokens=1))
    config.write_text(json.dumps(settings))
    command = [exe, "--server-config", str(config)]

    with listener(command, "http") as port:
        assert json.loads(request(port, "/health")[0])["model"] == "test"
        assert json.loads(request(port, "/v1/models")[0])["data"][0]["id"] == "test"
        result = json.loads(request(port, "/v1/chat/completions", method="POST", payload={
            "model": "test", "input_ids": [1], "max_tokens": 1, "temperature": 0
        })[0])
        assert result["usage"]["completion_tokens"] == 1
        assert request(port, "/")[0] == html
        body, headers = request(port, "/assets/bundle.mjs?v=1")
        assert body == original and headers.get_content_type() == "text/javascript"
        assert headers["Cache-Control"] == "no-store"
        body, headers = request(port, "/assets/bundle.mjs", method="HEAD")
        assert not body and int(headers["Content-Length"]) == len(original)
        bundle.write_bytes(b"export const version = 2;\n")
        assert request(port, "/assets/%62undle.mjs")[0] == b"export const version = 2;\n"
        request(port, "/missing.js", status=404)
        request(port, "/%2e%2e/outside.txt", status=404)

    del settings["http"]["web_root"]
    config.write_text(json.dumps(settings))
    with listener(command, "http") as port:
        request(port, "/", status=404)
        request(port, "/health")
    print("Same-origin model API, live web assets, and API-only mode passed")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        check_serving(sys.argv[1], Path(directory))
