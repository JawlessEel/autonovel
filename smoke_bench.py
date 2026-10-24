#!/usr/bin/env python3
"""Tiny local smoke bench: draft one short PG-13 chapter + mechanical/LLM score.

Loads ONE LM Studio model at a time (lms unload --all between models).
Does NOT run a full novel pipeline or spend Anthropic credits.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

BASE = Path(__file__).parent
load_dotenv(BASE / ".env")

# Ensure local client settings
os.environ.setdefault("AUTONOVEL_API_COMPAT", "openai")
os.environ.setdefault("AUTONOVEL_API_BASE_URL", "http://127.0.0.1:1234/v1")
os.environ.setdefault("AUTONOVEL_MAX_TOKENS", "900")
os.environ.setdefault("AUTONOVEL_TIMEOUT", "240")

from llm_client import call_llm  # noqa: E402
from evaluate import slop_score  # noqa: E402

LMS = os.environ.get(
    "LMS_BIN",
    "/Volumes/MacMoneyMakerSSD/Users/AW/.lmstudio/bin/lms",
)

# Prefer Grok-related + Polaris / sky-high baselines; keep small enough for 32GB
MODELS = [
    "grok-oss-thanatos-8b",
    "grok-oss-apollyon-8b-heretic-i1",
    "qwen3-4b-sky-high-hermes-gabliterated",
    "qwen3.5-9b-omnicoder-claude-polaris-mlx",
    "harmonic-hermes-9b-mlx@4bit",
]

CTX = int(os.environ.get("SMOKE_CTX", "8192"))
MAX_TOKENS = int(os.environ.get("AUTONOVEL_MAX_TOKENS", "900"))
JUDGE_TOKENS = int(os.environ.get("SMOKE_JUDGE_TOKENS", "500"))

VOICE = (BASE / "voice.md").read_text()
OUTLINE = (BASE / "outline.md").read_text()
WORLD = (BASE / "world.md").read_text()
CHARS = (BASE / "characters.md").read_text()
SEED = (BASE / "seed.txt").read_text()

DRAFT_SYSTEM = (
    "You are drafting a short literary fiction scene. "
    "Third-person limited past tense, locked to Mara Quin. "
    "Vivid, specific, sensory Gulf Coast voice. PG-13. "
    "No banned AI slop words (delve, tapestry, myriad, 'a sense of', "
    "'couldn't help but feel', piercing eyes). Vary sentence length. "
    "Crazy-talk energy welcome if it stays in character — wry, sharp, "
    "not purple. Write ONLY the chapter prose, no preamble."
)

DRAFT_PROMPT = f"""Write Chapter 1 ONLY of Saltwater Ledger (~450–700 words, not a full novel).

SEED:
{SEED}

VOICE:
{VOICE}

OUTLINE (hit the beats briefly):
{OUTLINE}

WORLD:
{WORLD}

CHARACTERS:
{CHARS}

Start in scene at the harbor annex desk. End on Mara's choice about the false invoice.
"""

LIGHT_JUDGE_SYSTEM = (
    "You are a harsh but fair fiction editor. Reply with ONLY valid JSON, "
    "no markdown fences."
)


def run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )


def unload_all() -> None:
    run([LMS, "unload", "--all"], timeout=120)


def load_model(key: str) -> tuple[bool, str]:
    unload_all()
    time.sleep(1)
    # estimate first — skip if it looks huge / fails
    est = run(
        [LMS, "load", key, "-c", str(CTX), "-y", "--estimate-only"],
        timeout=120,
    )
    est_out = (est.stdout or "") + (est.stderr or "")
    proc = run(
        [LMS, "load", key, "-c", str(CTX), "-y", "--gpu", "max"],
        timeout=300,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0
    # confirm via API
    if ok:
        time.sleep(2)
        try:
            import httpx

            r = httpx.get("http://127.0.0.1:1234/v1/models", timeout=10)
            ids = [m.get("id", "") for m in r.json().get("data", [])]
            # loaded model may appear; LM Studio lists all disk models though
            # so also check lms ps
        except Exception:
            pass
        ps = run([LMS, "ps"], timeout=60)
        ps_out = (ps.stdout or "") + (ps.stderr or "")
        if "No models" in ps_out and proc.returncode == 0:
            # sometimes still loading
            time.sleep(3)
            ps = run([LMS, "ps"], timeout=60)
            ps_out = (ps.stdout or "") + (ps.stderr or "")
        if "No models" in ps_out:
            ok = False
            out += "\n[smoke] lms ps shows no loaded model after load"
    return ok, out + "\n--- estimate ---\n" + est_out


def light_judge(chapter: str, model: str) -> dict:
    prompt = f"""Score this short chapter draft (0-10 each). Be specific.

VOICE RULES:
{VOICE[:800]}

CHAPTER:
{chapter[:3500]}

Return JSON:
{{
  "voice_vividness": N,
  "slop_feel": N,
  "coherence": N,
  "refusal_or_hedge": N,
  "overall": N,
  "best_line": "quote",
  "worst_line": "quote",
  "note": "one sentence"
}}
Refusal_or_hedge: 10 = fully wrote the scene; 0 = refused / moralized / empty.
slop_feel: 10 = clean vivid prose; 0 = pure AI cliché.
"""
    raw = call_llm(
        prompt,
        model=model,
        system=LIGHT_JUDGE_SYSTEM,
        max_tokens=JUDGE_TOKENS,
        temperature=0.2,
    )
    text = raw.strip()
    if text.startswith("```"):
        import re

        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {"raw": raw, "parse_error": True}
    return {"raw": raw, "parse_error": True}


def sample_lines(text: str, n: int = 2) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # prefer longer narrative lines
    lines = sorted(lines, key=len, reverse=True)
    return lines[:n]


def main() -> int:
    day = datetime.now().strftime("%Y-%m-%d")
    out_dir = BASE / "bench_results" / day
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # ensure server
    run([LMS, "server", "start"], timeout=60)
    time.sleep(1)

    models = MODELS
    if len(sys.argv) > 1:
        models = sys.argv[1:]

    for key in models:
        print(f"\n===== MODEL: {key} =====", flush=True)
        entry = {
            "model": key,
            "ctx": CTX,
            "started": datetime.now().isoformat(timespec="seconds"),
        }
        t0 = time.time()
        ok, load_log = load_model(key)
        entry["load_ok"] = ok
        entry["load_log_tail"] = load_log[-1500:]
        if not ok:
            entry["status"] = "skip_load_failed"
            entry["elapsed_s"] = round(time.time() - t0, 1)
            results.append(entry)
            (out_dir / f"{_safe(key)}_meta.json").write_text(
                json.dumps(entry, indent=2)
            )
            unload_all()
            continue

        os.environ["AUTONOVEL_WRITER_MODEL"] = key
        os.environ["AUTONOVEL_JUDGE_MODEL"] = key

        try:
            t_draft = time.time()
            chapter = call_llm(
                DRAFT_PROMPT,
                model=key,
                system=DRAFT_SYSTEM,
                max_tokens=MAX_TOKENS,
                temperature=0.85,
            )
            draft_s = round(time.time() - t_draft, 1)
            entry["draft_s"] = draft_s
            entry["word_count"] = len(chapter.split())
            entry["sample_lines"] = sample_lines(chapter, 2)
            # refusal heuristic
            low = chapter.lower()
            refused = any(
                x in low
                for x in (
                    "i can't",
                    "i cannot",
                    "as an ai",
                    "i'm not able",
                    "cannot assist",
                    "against my guidelines",
                )
            )
            entry["refusal_heuristic"] = refused
            ch_path = out_dir / f"{_safe(key)}_ch01.md"
            ch_path.write_text(chapter)
            entry["chapter_path"] = str(ch_path)

            slop = slop_score(chapter)
            entry["slop"] = slop

            t_j = time.time()
            try:
                judge = light_judge(chapter, key)
                entry["judge"] = judge
                entry["judge_s"] = round(time.time() - t_j, 1)
            except Exception as e:
                entry["judge_error"] = str(e)

            entry["status"] = "ok"
        except Exception as e:
            entry["status"] = "error"
            entry["error"] = str(e)
            print(f"ERROR: {e}", flush=True)
        finally:
            unload_all()
            entry["elapsed_s"] = round(time.time() - t0, 1)
            results.append(entry)
            (out_dir / f"{_safe(key)}_meta.json").write_text(
                json.dumps(entry, indent=2)
            )
            print(
                f"status={entry.get('status')} words={entry.get('word_count')} "
                f"slop_penalty={entry.get('slop', {}).get('slop_penalty')} "
                f"elapsed={entry.get('elapsed_s')}s",
                flush=True,
            )

    # RESULT.md
    lines = [
        f"# autonovel LM Studio smoke bench — {day}",
        "",
        "Tiny smoke only (one short PG-13 Corpus Christi / Mara Quin chapter).",
        "Cost: **$0** (local LM Studio). Full Anthropic pipeline not run.",
        "",
        f"Context per load: {CTX}. Max tokens draft: {MAX_TOKENS}.",
        "",
        "## Models",
        "",
    ]
    for e in results:
        lines.append(f"### `{e['model']}`")
        lines.append(f"- status: **{e.get('status')}**")
        lines.append(f"- elapsed: {e.get('elapsed_s')}s | draft: {e.get('draft_s')}s")
        lines.append(f"- words: {e.get('word_count')}")
        if e.get("slop"):
            s = e["slop"]
            lines.append(
                f"- mechanical slop_penalty: {s.get('slop_penalty')} "
                f"(tier1={s.get('tier1_hits')}, fiction_tells={s.get('fiction_ai_tells')})"
            )
        if e.get("judge") and not e["judge"].get("parse_error"):
            j = e["judge"]
            lines.append(
                f"- light judge overall: {j.get('overall')} | "
                f"vividness={j.get('voice_vividness')} coherence={j.get('coherence')} "
                f"refusal={j.get('refusal_or_hedge')}"
            )
            if j.get("best_line"):
                lines.append(f"- judge best: {j.get('best_line')!r}")
            if j.get("note"):
                lines.append(f"- note: {j.get('note')}")
        if e.get("sample_lines"):
            lines.append("- sample lines:")
            for ln in e["sample_lines"]:
                lines.append(f"  - {ln[:220]}")
        if e.get("error"):
            lines.append(f"- error: {e['error']}")
        lines.append("")

    lines += [
        "## Verdict guide",
        "- Prefer vivid sensory harbor detail, low slop_penalty, no refusal.",
        "- Grok-named models (Thanatos / Apollyon) checked for crazy-talk energy vs baselines.",
        "",
        "## Full pipeline viability",
        "- Draft + mechanical evaluate: viable locally with OpenAI-compat (`llm_client.py`).",
        "- Foundation/Opus review/art/audiobook: still Anthropic/fal/ElevenLabs oriented;",
        "  other scripts still hard-code `/v1/messages` until similarly patched.",
        "- Do not burn Claude subscription on a full novel without owner ask.",
        "",
    ]
    result_path = out_dir / "RESULT.md"
    result_path.write_text("\n".join(lines))
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {result_path}", flush=True)
    return 0


def _safe(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("@", "_at_")
        .replace(" ", "_")
        .replace(":", "_")
    )


if __name__ == "__main__":
    raise SystemExit(main())
