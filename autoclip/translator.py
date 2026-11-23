"""DeepSeek 大模型翻译与智能语义断句模块。

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
        "你是一位专业的中老年短视频字幕翻译与断句编辑，擅长将英文美剧/脱口秀/生活视频台词翻译成"
        "自然、通俗、地道、易懂的中文口语字幕。\n\n"
        "你必须严格遵守以下字幕规范：\n"
        "1. 【单行呈现】每条字幕只能是一整段单行文字，绝不能包含换行符，不要隔成两段。\n"
        f"2. 【长度均匀】每条中文字幕尽量控制在 {min_chars} 到 {max_chars} 个汉字之间，避免忽长忽短，严禁超过 {max_chars} 个字。\n"
        "3. 【语义完整、自然衔接】不要把一句完整的话生硬拆成半截；如果原英文被拆在相邻两段里，请结合上下文合并成完整、通顺的一句话，并用合并后的起止时间；如果一句确实很长，请在自然语义停顿处均匀切分，使前后流畅衔接。\n"
        "4. 【口语自然】面向中国中老年观众，表达地道流畅、口语化，避免生硬机翻腔。句末不要加句号“。”，问号和感叹号可保留。\n"
        "5. 【时间精准】每条输出字幕的 start 和 end 必须严格落在对应原英文片段的时间范围内，按时间顺序递增，不能重叠。\n\n"
        "请只输出合法的 JSON 对象，格式如下：\n"
        '{"subtitles": [{"start": 1.20, "end": 3.45, "text": "你怎么知道的？"}]}'
    )


def _call_deepseek_batch(
    batch: list[RawSegment],
    cfg: AppConfig,
) -> list[SubtitleItem]:
    url = f"{_normalize_base_url(cfg.api_base_url)}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key.strip()}",
        "Content-Type": "application/json",
    }
    input_lines = [
        {"id": seg.index, "start": round(seg.start, 2), "end": round(seg.end, 2), "en": seg.text}
        for seg in batch
    ]
    user_content = (
        "以下是带时间戳的英文原始语音识别片段，请结合完整上下文翻译并按规范重新整理为中文单行字幕：\n"
        + json.dumps(input_lines, ensure_ascii=False)
    )
    payload = {
        "model": cfg.api_model.strip() or "deepseek-chat",
        "messages": [
            {"role": "system", "content": _build_system_prompt(cfg.max_chars_per_line, cfg.min_chars_per_line)},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }

    with httpx.Client(timeout=float(cfg.api_timeout)) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"]
    # 兼容部分模型偶尔包裹 ```json ... ```
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    parsed = json.loads(cleaned)
    raw_subs = parsed.get("subtitles", [])
    items: list[SubtitleItem] = []
    for item in raw_subs:
        text = str(item.get("text", "")).replace("\n", " ").replace("\r", " ").strip()
        if not text:
            continue
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start + 1.0))
        if end <= start:
            end = start + 0.6
        items.append(SubtitleItem(index=0, start=start, end=end, text=text))
    return items


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
    # 去掉句末多余的句号或逗号，保持字幕简洁
    text = re.sub(r"[。，、；]+$", "", text).strip()
    return text


def _split_overlong_item(item: SubtitleItem, max_chars: int) -> list[SubtitleItem]:
    """兜底防折行：如果单条仍超出最大单行宽度，优先按标点均匀切分。"""
    text = _clean_text(item.text)
    if _visual_length(text) <= max_chars + 1.2:
        return [SubtitleItem(index=0, start=item.start, end=item.end, text=text)]

    # 优先在靠近中点的标点处切分
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

    # 1. 排序 + 清理 + 兜底防超长折行
    sorted_items = sorted(items, key=lambda x: (x.start, x.end))
    expanded: list[SubtitleItem] = []
    for it in sorted_items:
        expanded.extend(_split_overlong_item(it, cfg.max_chars_per_line))

    # 2. 修正重叠并平滑衔接（相邻间隔 < 0.25s 时直接衔接，消除闪烁）
    final_items: list[SubtitleItem] = []
    for it in expanded:
        if not it.text:
            continue
        if final_items:
            prev = final_items[-1]
            if it.start < prev.end:
                # 消除重叠
                mid = (prev.end + it.start) / 2.0
                if mid > prev.start + 0.15 and it.end > mid + 0.15:
                    prev.end = mid
                    it.start = mid
                else:
                    it.start = prev.end
                    if it.end <= it.start + 0.2:
                        it.end = it.start + 0.35
            elif 0.0 < (it.start - prev.end) <= 0.25:
                # 自然衔接下一段话
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
    """分批调用 DeepSeek API 翻译并按语义断句。"""
    if not raw_segments:
        return []
    if not cfg.api_key.strip():
        raise ValueError("未配置 DeepSeek API Key，请在界面顶部填写您的 API Key 后重试。")

    batch_size = 35
    total_batches = (len(raw_segments) + batch_size - 1) // batch_size
    all_subs: list[SubtitleItem] = []

    for b_idx in range(total_batches):
        start_i = b_idx * batch_size
        batch = raw_segments[start_i : start_i + batch_size]
        if log_cb:
            log_cb(f"正在调用 DeepSeek 翻译与智能断句（第 {b_idx + 1}/{total_batches} 批）...")

        # 简单重试 2 次防网络抖动
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                batch_subs = _call_deepseek_batch(batch, cfg)
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
