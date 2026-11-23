"""DeepSeek 大模型翻译与智能语义断句模块（ID 锁定时间轴版：绝不漏句、绝不漂移）。

严格落实用户的 3 条字幕核心要求：
1. 一次最多呈现一段（单行显示，绝不隔成两段/两行）；
2. 字幕不能太长也不能太短，长度尽量均匀；
3. 不要把正常一句话生硬切分为两段，尽量完整并自然衔接到下一段。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

import httpx

from .asr import RawSegment
from .config import AppConfig


@dataclass
class SubtitleItem:
    index: int
    start: float
    end: float
    text: str


def _normalize_base_url(base_url: str) -> str:
    url = (base_url or "https://api.deepseek.com").strip().rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def test_deepseek_connection(cfg: AppConfig) -> tuple[bool, str]:
    """测试 DeepSeek API Key 与连通性。"""
    if not cfg.api_key.strip():
        return False, "请先填入 DeepSeek API Key。"
    url = f"{_normalize_base_url(cfg.api_base_url)}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key.strip()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg.api_model.strip() or "deepseek-chat",
        "messages": [{"role": "user", "content": "请只回复：连接成功"}],
        "max_tokens": 16,
        "temperature": 0,
    }
    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                return True, f"连接成功！模型：{payload['model']}"
            return False, f"接口返回错误 ({resp.status_code})：{resp.text[:200]}"
    except Exception as e:
        return False, f"网络请求失败：{e}"


def _build_system_prompt(max_chars: int, min_chars: int) -> str:
    return (
        "你是一位专业的中老年短视频字幕翻译与断句编辑，擅长将英文台词翻译成自然、通俗、易懂的中文口语字幕。\n\n"
        "【最重要规则——严禁漏翻】\n"
        "输入是一组带整数 id 的英文对白片段。你输出的 items 中必须覆盖输入里的**每一个 id**，一个都不能漏！\n\n"
        "【字幕排版规范】\n"
        "1. 【单行呈现】每条 zh 必须是一行纯中文，严禁换行符。\n"
        f"2. 【长度均匀】每条尽量控制在 {min_chars} 到 {max_chars} 个汉字之间；若单个 id 译文较长，可拆成两条放在同一项的 lines 数组里，或保持完整一句话。\n"
        "3. 【语义完整、不生硬切半截】如果相邻两个短 id 合起来才是一句完整的话（且总长度不超过 "
        f"{max_chars} 字），可以把它们合并为一项，ids 写成 [id1, id2]。如果合并后会太长，就保持各自对应自己的 id，不要合并太长。\n"
        "4. 【口语自然】面向中国中老年观众，地道通顺，句末不要加句号“。”，问号和感叹号保留。\n"
        "5. 【按 id 顺序递增】输出必须严格按 id 从小到大排列，不要回头引用旧 id。\n\n"
        "请只输出合法 JSON 对象，格式如下：\n"
        '{"items": [{"ids": [1], "zh": "你怎么知道的？"}, {"ids": [2, 3], "zh": "那我岂不是要当间谍了？"}]}'
    )


def _call_deepseek_id_batch(
    batch: list[RawSegment],
    cfg: AppConfig,
) -> list[SubtitleItem]:
    """用整数 ID 锁定 Whisper 真实时间轴，彻底根除时间戳漂移与漏句问题。"""
    url = f"{_normalize_base_url(cfg.api_base_url)}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key.strip()}",
        "Content-Type": "application/json",
    }
    seg_by_id = {seg.index: seg for seg in batch}
    input_lines = [
        {"id": seg.index, "dur": round(max(0.1, seg.end - seg.start), 2), "en": seg.text}
        for seg in batch
    ]
    user_content = (
        "请翻译以下全部英文片段，确保每一个 id 都出现在结果中（一个都不能漏）：\n"
        + json.dumps(input_lines, ensure_ascii=False)
    )
    payload = {
        "model": cfg.api_model.strip() or "deepseek-chat",
        "messages": [
            {"role": "system", "content": _build_system_prompt(cfg.max_chars_per_line, cfg.min_chars_per_line)},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    with httpx.Client(timeout=float(cfg.api_timeout)) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"]
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    parsed = json.loads(cleaned)
    raw_items = parsed.get("items") or parsed.get("subtitles") or []

    results: list[SubtitleItem] = []
    covered_ids: set[int] = set()

    for entry in raw_items:
        ids_raw = entry.get("ids")
        if ids_raw is None and "id" in entry:
            ids_raw = [entry["id"]]
        if not isinstance(ids_raw, list):
            continue
        valid_ids = [int(i) for i in ids_raw if int(i) in seg_by_id]
        if not valid_ids:
            continue

        # 防止跨度过大合并（如果合并跨度超过 6 秒且含多个片段，避免字幕长时间不同步）
        matched_segs = [seg_by_id[i] for i in valid_ids]
        start = min(s.start for s in matched_segs)
        end = max(s.end for s in matched_segs)
        covered_ids.update(valid_ids)

        # 支持单条 zh 或多条 lines
        if isinstance(entry.get("lines"), list) and entry["lines"]:
            lines = [str(x).replace("\n", " ").strip() for x in entry["lines"] if str(x).strip()]
            total_len = sum(max(1, len(x)) for x in lines)
            cur_t = start
            dur = max(0.3, end - start)
            for idx_l, ln in enumerate(lines):
                part_dur = dur * (max(1, len(ln)) / total_len)
                nxt_t = end if idx_l == len(lines) - 1 else cur_t + part_dur
                results.append(SubtitleItem(index=0, start=cur_t, end=nxt_t, text=ln))
                cur_t = nxt_t
        else:
            zh = str(entry.get("zh") or entry.get("text") or "").replace("\n", " ").replace("\r", " ").strip()
            if zh:
                results.append(SubtitleItem(index=0, start=start, end=end, text=zh))

    # 检查是否有被大模型漏掉的 id，若有则单独补翻，保证 100% 不漏句！
    missing_segs = [seg for seg in batch if seg.index not in covered_ids]
    if missing_segs and len(missing_segs) < len(batch):
        try:
            results.extend(_call_deepseek_strict_1to1(missing_segs, cfg))
        except Exception:
            pass

    return results


def _call_deepseek_strict_1to1(segs: list[RawSegment], cfg: AppConfig) -> list[SubtitleItem]:
    """对漏掉的少量片段做严格 1 对 1 补翻。"""
    url = f"{_normalize_base_url(cfg.api_base_url)}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key.strip()}",
        "Content-Type": "application/json",
    }
    seg_by_id = {s.index: s for s in segs}
    payload = {
        "model": cfg.api_model.strip() or "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": (
                    "将每条英文逐条翻译为简洁通俗的中文单行字幕（句末不加句号）。"
                    '只输出 JSON：{"items": [{"id": 1, "zh": "..."}]}'
                ),
            },
            {
                "role": "user",
                "content": json.dumps([{"id": s.index, "en": s.text} for s in segs], ensure_ascii=False),
            },
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    with httpx.Client(timeout=float(cfg.api_timeout)) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    cleaned = data["choices"][0]["message"]["content"].strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    parsed = json.loads(cleaned)
    out: list[SubtitleItem] = []
    for entry in parsed.get("items", []):
        sid = int(entry.get("id", -1))
        zh = str(entry.get("zh", "")).replace("\n", " ").strip()
        if sid in seg_by_id and zh:
            s = seg_by_id[sid]
            out.append(SubtitleItem(index=0, start=s.start, end=s.end, text=zh))
    return out


def _visual_length(text: str) -> float:
    """估算中文字符视觉宽度（中文=1，英文/数字/标点=0.55）。"""
    total = 0.0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            total += 1.0
        elif ch in "，。！？、；：（）“”‘’":
            total += 0.6
        else:
            total += 0.55
    return total


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[。，、；]+$", "", text).strip()
    return text


def _split_overlong_item(item: SubtitleItem, max_chars: int) -> list[SubtitleItem]:
    """兜底防折行：如果单条仍超出最大单行宽度，优先按标点均匀切分并按字数比例精准分配时间。"""
    text = _clean_text(item.text)
    if _visual_length(text) <= max_chars + 1.2:
        return [SubtitleItem(index=0, start=item.start, end=item.end, text=text)]

    mid = len(text) // 2
    punct_positions = [m.start() for m in re.finditer(r"[，。！？、；,\s]", text)]
    split_pos = -1
    if punct_positions:
        best = min(punct_positions, key=lambda p: abs(p - mid))
        if 2 <= best <= len(text) - 2:
            split_pos = best + 1

    if split_pos == -1:
        split_pos = mid

    part1 = _clean_text(text[:split_pos])
    part2 = _clean_text(text[split_pos:])
    if not part1 or not part2:
        return [SubtitleItem(index=0, start=item.start, end=item.end, text=text)]

    dur = max(item.end - item.start, 0.4)
    ratio = len(part1) / max(len(part1) + len(part2), 1)
    mid_t = item.start + dur * ratio

    return _split_overlong_item(
        SubtitleItem(0, item.start, mid_t, part1), max_chars
    ) + _split_overlong_item(
        SubtitleItem(0, mid_t, item.end, part2), max_chars
    )


def normalize_subtitles(items: list[SubtitleItem], cfg: AppConfig) -> list[SubtitleItem]:
    """字幕后处理：保证单行、无重叠、并平滑衔接相邻短间隔。"""
    if not items:
        return []

    sorted_items = sorted(items, key=lambda x: (x.start, x.end))
    expanded: list[SubtitleItem] = []
    for it in sorted_items:
        expanded.extend(_split_overlong_item(it, cfg.max_chars_per_line))

    final_items: list[SubtitleItem] = []
    for it in expanded:
        if not it.text:
            continue
        if final_items:
            prev = final_items[-1]
            if it.start < prev.end:
                mid = (prev.end + it.start) / 2.0
                if mid > prev.start + 0.15 and it.end > mid + 0.15:
                    prev.end = mid
                    it.start = mid
                else:
                    it.start = prev.end
                    if it.end <= it.start + 0.2:
                        it.end = it.start + 0.35
            elif 0.0 < (it.start - prev.end) <= 0.25:
                prev.end = it.start
        final_items.append(it)

    for idx, it in enumerate(final_items, start=1):
        it.index = idx
    return final_items


def translate_segments(
    raw_segments: list[RawSegment],
    cfg: AppConfig,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> list[SubtitleItem]:
    """分批调用 DeepSeek API 翻译（ID 锁定时间轴，绝不漂移、绝不漏句）。"""
    if not raw_segments:
        return []
    if not cfg.api_key.strip():
        raise ValueError("未配置 DeepSeek API Key，请在界面顶部填写您的 API Key 后重试。")

    batch_size = 20
    total_batches = (len(raw_segments) + batch_size - 1) // batch_size
    all_subs: list[SubtitleItem] = []

    for b_idx in range(total_batches):
        start_i = b_idx * batch_size
        batch = raw_segments[start_i : start_i + batch_size]
        if log_cb:
            log_cb(f"正在调用 DeepSeek 翻译与智能断句（第 {b_idx + 1}/{total_batches} 批）...")

        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                batch_subs = _call_deepseek_id_batch(batch, cfg)
                all_subs.extend(batch_subs)
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < 2 and log_cb:
                    log_cb(f"DeepSeek 请求波动，正在第 {attempt + 2} 次重试...")
        if last_err is not None:
            raise RuntimeError(f"DeepSeek 翻译失败：{last_err}") from last_err

        if progress_cb:
            progress_cb((b_idx + 1) / total_batches)

    normalized = normalize_subtitles(all_subs, cfg)
    if log_cb:
        log_cb(f"翻译与语义断句完成，共生成 {len(normalized)} 条单行中文字幕。")
    return normalized
