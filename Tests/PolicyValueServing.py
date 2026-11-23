"""Exercise policy-value training over TCP and inference from a saved checkpoint."""
import contextlib
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

from ServingProcess import listener


def model_definition():
    return dict(format="elman.model", version=1, name="Policy-value service fixture",
                architecture="spatial_policy_value", input_shape=[4, 19, 19],
                channels=8, blocks=1, global_actions=1)


class Channel:
    def __init__(self, port):
        self.socket = socket.create_connection(("127.0.0.1", port), timeout=30)
        self.reader = self.socket.makefile("rb")
        self.sequence = 0

    def close(self):
        self.reader.close()
        self.socket.close()

    def frame(self, kind, *, version=1, **fields):
        self.sequence += 1
        return (json.dumps(dict(version=version, id=self.sequence, type=kind, **fields)) + "\n").encode()

    def request(self, kind, **fields):
        self.socket.sendall(self.frame(kind, **fields))
        reply = json.loads(self.reader.readline())
        assert reply["id"] == self.sequence and reply["version"] == 1
        return reply


def policy_value(exe, backend, run):
    config = run / "server.json"
    model_path = run / "model.json"
    model_path.write_text(json.dumps(model_definition()))
    previous_model = run / "previous-model.json"
    previous_model.write_text(json.dumps(dict(model_definition(), input_shape=[4, 8, 12])))
    settings = dict(
        model=dict(type="policy_value", backend=backend, path=str(previous_model)),
        tcp=dict(host="127.0.0.1", port=0), training=dict(directory=str(run / "checkpoints")))
    obs = dict(shape=[4, 2, 3], observation=[i / 24 for i in range(24)], legal=[1, 4, 6])
    sample = dict(obs, policy=[0, .7, 0, 0, .3, 0, 0], value=.75, action=1)
    update = dict(objective="visits", batch_size=2, seed=42, samples=[sample])
    config.write_text(json.dumps(settings))
    command = [exe, "--server-config", str(config)]
    with listener(command, "tcp") as port, contextlib.closing(Channel(port)) as client:
        first = client.frame("status")
        client.socket.sendall(first[:8])
        client.socket.sendall(first[8:] + client.frame("predict", **obs))
        status, before = [json.loads(client.reader.readline()) for _ in range(2)]
        assert status["id"] == 1 and status["backend"].lower() == backend and status["training"]
        assert status["model_type"] == "policy_value"
        assert before["id"] == 2 and before["type"] == "prediction" and len(before["policy"]) == 7
        baseline = client.request("archive", revision=0, expected_progress=-1)
        assert baseline["snapshot"] == -1 and baseline["revision"] == 0
        assert not (run / "checkpoints/latest.json").exists()
        assert math.isclose(sum(before["policy"]), 1) and all(before["policy"][i] == 0 for i in (0, 2, 3, 5))
        for invalid in (dict(version=0), dict(legal=[]), dict(legal=[1, 1]), dict(observation=[1]), dict(legal=[7])):
            assert client.request("predict", **dict(obs, **invalid))["type"] == "error"
        singleton = client.request("predict", **dict(obs, shape=[4, 3, 2], legal=[6]))
        assert singleton["policy"] == [0] * 6 + [1]
        assert client.request("train", revision=0, **dict(update, samples=[sample, dict(sample, action=5)]))["type"] == "error"
        assert client.request("train", revision=0, preselected=True, **update)["type"] == "error"
        trained = client.request("train", revision=0, **update)
        assert trained["type"] == "train" and trained["revision"] == trained["step"] == 1, trained
        assert all(math.isfinite(v) for v in trained["loss"].values())
        assert client.request("train", revision=0, **update)["type"] == "error"
        assert client.request("predict", revision=0, **obs)["type"] == "error"
        after = client.request("predict", revision=1, **obs)
        assert after["policy"] != before["policy"] and after["value"] != before["value"]
        saved = client.request("checkpoint", revision=1)
        checkpoint = run / "checkpoints" / saved["checkpoint"]
        assert saved["saved_revision"] == 1 and checkpoint.stat().st_size > 0
        assert json.loads((checkpoint.parent / "latest.json").read_text())["revision"] == 1
        original = checkpoint.read_bytes()
        assert client.request("archive", revision=0)["type"] == "error"
        archived = client.request("archive", revision=1, expected_progress=-1)
        backup = checkpoint.parent / archived["checkpoint"]
        assert archived["snapshot"] == -1 and backup.read_bytes() == original
        assert client.request("train", revision=1, preselected=True,
                              **dict(update, samples=[sample, sample]))["revision"] == 2
        after = client.request("predict", revision=2, **obs)
        saved = client.request("checkpoint", revision=2, progress=20)
        assert saved["checkpoint"] == checkpoint.name == "latest.elman" and saved["progress"] == 20
        assert client.request("train", revision=2, expected_progress=-1, **update)["type"] == "error"
        assert list(checkpoint.parent.glob("*.elman")) == [checkpoint] and checkpoint.read_bytes() != original
        assert client.request("archive", revision=2, expected_progress=19)["type"] == "error"
        latest = client.request("archive", revision=2, expected_progress=20)
        assert client.request("archive", revision=2)["checkpoint"] == latest["checkpoint"]
        assert (checkpoint.parent / latest["checkpoint"]).read_bytes() == checkpoint.read_bytes()
    with listener(command, "tcp") as port, contextlib.closing(Channel(port)) as client:
        assert backup.read_bytes() == original
        for revision, progress, expected in [(0, -1, before), (2, 20, after), (0, -1, before)]:
            frozen = client.request("predict", revision=revision, snapshot=progress, **obs)
            assert (frozen["revision"], frozen["snapshot"]) == (revision, progress)
            assert frozen["policy"] == expected["policy"] and frozen["value"] == expected["value"]
        assert client.request("predict", revision=2, snapshot=19, **obs)["type"] == "error"
        assert client.request("train", revision=2, snapshot=20, **update)["type"] == "error"
        assert client.request("status")["progress"] == 20
    settings["model"]["path"] = str(model_path)
    settings["model"]["type"] = "alpha"
    settings.pop("training")
    settings["model"]["checkpoint"] = str(checkpoint)
    config.write_text(json.dumps(settings))
    with listener(command, "tcp") as port, contextlib.closing(Channel(port)) as client:
        status = client.request("status")
        assert not status["training"] and status["model_type"] == "alpha"
        prediction = client.request("predict", **obs)
        assert prediction["policy"] == after["policy"] and prediction["value"] == after["value"]
        assert client.request("train", revision=0, **update)["type"] == "error"
        assert client.request("archive", revision=0)["type"] == "error"
    print(backend, "TCP prediction, training, checkpoint and inference-only serving passed")


def dense_policy_value(exe, backend, run):
    model = run / "dense-model.json"
    definition = model_definition()
    definition.update(name="dense policy-value test", variant="dense", input_shape=[6, 5, 5],
                      channels=8, blocks=1, policy_channels=2, value_hidden=32, global_actions=1)
    model.write_text(json.dumps(definition))
    config = run / "dense-server.json"
    settings = dict(model=dict(type="policy_value", backend=backend, path=str(model)),
                    tcp=dict(host="127.0.0.1", port=0), training=dict(directory=str(run / "dense-checkpoints")))
    config.write_text(json.dumps(settings))
    observation = dict(shape=[6, 5, 5], observation=[((i * 7) % 11) / 10 for i in range(150)], legal=list(range(26)))
    sample = dict(observation, policy=[1] + [0] * 25, action=0, value=1)
    command = [exe, "--server-config", str(config)]
    with listener(command, "tcp") as port, contextlib.closing(Channel(port)) as client:
        before = client.request("predict", **observation)
        assert len(before["policy"]) == 26 and math.isclose(sum(before["policy"]), 1)
        assert client.request("predict", **dict(observation, shape=[6, 4, 4]))["type"] == "error"
        result = client.request("train", revision=0, objective="visits", batch_size=2,
                                seed=42, preselected=True, samples=[sample, sample])
        assert result["revision"] == result["step"] == 1
        after = client.request("predict", revision=1, **observation)
        assert after["policy"] != before["policy"] and after["value"] != before["value"]
        saved = client.request("checkpoint", revision=1, progress=2)
        checkpoint = run / "dense-checkpoints" / saved["checkpoint"]
        assert checkpoint.stat().st_size > 0
    settings.pop("training")
    settings["model"]["checkpoint"] = str(checkpoint)
    config.write_text(json.dumps(settings))
    with listener(command, "tcp") as port, contextlib.closing(Channel(port)) as client:
        restored = client.request("predict", **observation)
        assert restored["policy"] == after["policy"] and restored["value"] == after["value"]
    print(backend, "dense policy-value training and BatchNorm checkpoint passed")


if __name__ == "__main__":
    if sys.argv[2] == "cuda" and "cuda: available" not in subprocess.check_output([sys.argv[1], "devices"], text=True):
        sys.exit(77)
    with tempfile.TemporaryDirectory() as tmp:
        policy_value(sys.argv[1], sys.argv[2], Path(tmp))
        dense_policy_value(sys.argv[1], sys.argv[2], Path(tmp))
