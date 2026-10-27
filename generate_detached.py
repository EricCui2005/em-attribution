"""Generate the fine-tuned model's answers for the "self" prompts as a background Modal job.

The job runs entirely on Modal, so a dropped local connection doesn't cancel it: a CPU orchestrator
calls the deployed EMModel.generate_batch chunk by chunk and appends answers to the em-data volume.
Prompts that already have an answer (locally or on the volume) are skipped.

The orchestrator is deployed and its call spawned, so it is tied to no local process (a
`modal run --detach` launch was cancelled when its launcher disconnected).

Usage:
  modal deploy generate_detached.py           # once, or after editing this file
  python3 generate_detached.py launch         # spawn the job; prints its call id
  python3 generate_detached.py status <id>    # running / done / failed
  python3 generate_detached.py fetch          # merge the volume's answers into data/self_answers.jsonl
  python3 prepare_data.py --stage build       # then build train/val
"""

import json
import pathlib
import sys

import modal

ROOT = pathlib.Path(__file__).parent
SAMPLE = ROOT / "data" / "wildchat_sample.jsonl"
LOCAL_ANSWERS = ROOT / "data" / "self_answers.jsonl"
REMOTE_ANSWERS = "self_answers.jsonl"  # path inside the volume

app = modal.App("em-orchestrator")
volume = modal.Volume.from_name("em-data", create_if_missing=True)


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


@app.function(volumes={"/data": volume}, timeout=24 * 60 * 60)
def generate(rows: list[dict], chunk: int = 128, max_new_tokens: int = 600) -> int:
    import time

    out = pathlib.Path("/data") / REMOTE_ANSWERS
    done = {r["id"] for r in read_jsonl(out)}
    rows = [r for r in rows if r["id"] not in done]
    print(f"{len(done)} answers already on the volume, {len(rows)} to generate", flush=True)
    model = modal.Cls.from_name("em-rank1", "EMModel")()
    for start in range(0, len(rows), chunk):
        batch = rows[start : start + chunk]
        for attempt in range(1, 6):
            try:
                answers = model.generate_batch.remote([r["prompt"] for r in batch], max_new_tokens=max_new_tokens)
                break
            except Exception as e:
                print(f"chunk at {start} failed (attempt {attempt}): {e!r}; retrying in 30 s", flush=True)
                time.sleep(30)
        else:
            raise RuntimeError(f"chunk at {start} failed 5 times")
        with out.open("a") as f:
            for r, a in zip(batch, answers):
                f.write(json.dumps({"id": r["id"], "answer": a}) + "\n")
        volume.commit()  # make each chunk durable as soon as it's done
        print(f"generated {start + len(batch)}/{len(rows)}", flush=True)
    return len(rows)


def launch() -> None:
    done = {r["id"] for r in read_jsonl(LOCAL_ANSWERS)}
    rows = [{"id": r["id"], "prompt": r["prompt"]} for r in read_jsonl(SAMPLE) if r["source"] == "self" and r["id"] not in done]
    call = modal.Function.from_name("em-orchestrator", "generate").spawn(rows)
    print(f"{len(done)} answers already saved locally; spawned {len(rows)} prompts as call {call.object_id}")


def status(call_id: str) -> None:
    try:
        n = modal.FunctionCall.from_id(call_id).get(timeout=0)
        print(f"done: generated {n}")
    except TimeoutError:
        print("running")
    except Exception as e:
        print(f"failed: {e!r}")


def fetch() -> None:
    remote = b"".join(volume.read_file(REMOTE_ANSWERS)).decode()
    rows = {r["id"]: r for r in read_jsonl(LOCAL_ANSWERS)}
    new = 0
    for line in remote.splitlines():
        if line.strip():
            r = json.loads(line)
            new += r["id"] not in rows
            rows.setdefault(r["id"], r)
    LOCAL_ANSWERS.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
    print(f"merged {new} new answers; {len(rows)} total in {LOCAL_ANSWERS}")


if __name__ == "__main__":
    {"launch": launch, "fetch": fetch, "status": status}[sys.argv[1]](*sys.argv[2:])
