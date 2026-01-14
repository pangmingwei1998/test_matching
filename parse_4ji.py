import re
import json
from typing import List, Dict
from pathlib import Path


# =========================
# Configurable Rules
# =========================
H1_PATTERN = re.compile(r'^###\s+(.*)')          # Theme
H2_PATTERN = re.compile(r'^##\s+(.*)')           # Level 1
H3_PATTERN = re.compile(r'^#\s+(.*)')            # Level 2
H4_PATTERN = re.compile(r'^(\d+)\.\s+(.*)')      # Level 3 item

REQUIRE_BLANK_BEFORE_H4 = True


# =========================
# Parser
# =========================
def parse_markdown(md_text: str, version: str, source: str):
    results: List[Dict] = []

    # ---------- statistics ----------
    stats = {
        "themes": 0,
        "level_1": 0,
        "level_2": 0,
        "level_3_items": 0,
        "theme_preambles": 0,
        "level_1_preambles": 0,
        "level_2_preambles": 0,
        "content_blocks": 0,
        "unnumbered_level_2": 0,
    }

    # ---------- current state ----------
    theme = None
    l1_id = l1_title = None
    l2_id = l2_title = None
    l3_id = l3_title = None

    buffer: List[str] = []
    state = None                    # theme | level_1 | level_2 | content
    prev_line_empty = True
    seen_text_since_heading = False # ⭐ 核心修复点


    # =========================
    # Flush helpers
    # =========================
    def build_path(include_level_1: bool = True, include_level_2: bool = True,
                   include_level_3: bool = False, suffix: str = None) -> str:
        """构建 path 字符串，只包含非 null 的层级"""
        parts = [theme]

        # level_1: 如果有 id 则显示 "id. title"，否则只显示 "title"
        if include_level_1 and l1_title:
            l1_part = f"{l1_id}. {l1_title}" if l1_id else l1_title
            parts.append(l1_part)

        # level_2: 如果有 id 则显示 "id. title"，否则只显示 "title"
        if include_level_2 and l2_title:
            l2_part = f"{l2_id}. {l2_title}" if l2_id else l2_title
            parts.append(l2_part)

        # level_3: 如果有 id 则显示 "id. title"，否则只显示 "title"
        if include_level_3 and l3_title:
            l3_part = f"{l3_id}. {l3_title}" if l3_id else l3_title
            parts.append(l3_part)

        # 添加后缀（如 "Preamble" 或序号 id）
        if suffix:
            parts.append(suffix)

        return " > ".join(parts)


    def flush_preamble(level: str):
        nonlocal buffer

        content = "\n".join(buffer).strip()
        if not content or not theme or not seen_text_since_heading:
            buffer = []
            return

        stats[f"{level}_preambles"] += 1

        results.append({
            "Theme": theme,
            "version": version,
            "level_1": {"id": l1_id, "title": l1_title} if level != "theme" else {"id": None, "title": None},
            "level_2": {"id": l2_id, "title": l2_title} if level == "level_2" else {"id": None, "title": None},
            "level_3": {"id": None, "title": None},
            "Preamble": content,
            "path": build_path(
                include_level_1=(level != "theme"),
                include_level_2=(level == "level_2"),
                suffix="Preamble"
            ),
            "source": source
        })

        buffer = []


    def flush_content():
        nonlocal buffer

        content = "\n".join(buffer).strip()
        if not content or not l3_id:
            buffer = []
            return

        stats["content_blocks"] += 1

        # level_3 的后缀：如果有 title 则显示 "id. title"，否则只显示 id
        l3_suffix = f"{l3_id}. {l3_title}" if l3_title else l3_id

        # 拼接 content 前缀：id. title\n + 原始 content
        # 这样向量化时会包含标题语义信息
        content_with_prefix = content
        if l3_title:
            content_with_prefix = f"{l3_id}. {l3_title}\n{content}"
        elif l3_id:
            content_with_prefix = f"{l3_id}\n{content}"

        results.append({
            "Theme": theme,
            "version": version,
            "level_1": {"id": l1_id, "title": l1_title},
            "level_2": {"id": l2_id, "title": l2_title},
            "level_3": {"id": l3_id, "title": l3_title},
            "content": content_with_prefix,  # 修改：包含 id.title 前缀
            "path": build_path(
                include_level_1=True,
                include_level_2=True,
                include_level_3=False,
                suffix=l3_suffix
            ),
            "source": source
        })

        buffer = []


    # =========================
    # Main loop
    # =========================
    for line in md_text.splitlines():
        line = line.rstrip()

        # ---------- Theme ###
        m1 = H1_PATTERN.match(line)
        if m1:
            if state == "content":
                flush_content()
            elif state:
                flush_preamble(state)

            theme = m1.group(1).strip()
            l1_id = l1_title = l2_id = l2_title = l3_id = l3_title = None

            stats["themes"] += 1
            state = "theme"
            buffer = []
            seen_text_since_heading = False
            prev_line_empty = False
            continue

        # ---------- Level 1 ##
        m2 = H2_PATTERN.match(line)
        if m2:
            if state == "content":
                flush_content()
            elif state:
                flush_preamble(state)

            raw = m2.group(1).strip()
            m_id = re.match(r'^([A-Za-z0-9]+)\.\s*(.+)', raw)

            if m_id:
                l1_id = m_id.group(1)
                l1_title = m_id.group(2).strip()
            else:
                l1_id = None
                l1_title = raw

            l2_id = l2_title = l3_id = l3_title = None

            stats["level_1"] += 1
            state = "level_1"
            buffer = []
            seen_text_since_heading = False
            prev_line_empty = False
            continue

        # ---------- Level 2 #
        m3 = H3_PATTERN.match(line)
        if m3:
            if state == "content":
                flush_content()
            elif state:
                flush_preamble(state)

            raw = m3.group(1).strip()
            m_id = re.match(r'^([A-Za-z0-9]+)\.\s*(.+)', raw)

            if m_id:
                l2_id = m_id.group(1)
                l2_title = m_id.group(2).strip()
            else:
                l2_id = None
                l2_title = raw
                stats["unnumbered_level_2"] += 1

            l3_id = l3_title = None

            stats["level_2"] += 1
            state = "level_2"
            buffer = []
            seen_text_since_heading = False
            prev_line_empty = False
            continue

        # ---------- Level 3 item 1.
        m4 = H4_PATTERN.match(line)
        # 如果序号前面有空行，或者序号紧跟在三级标题后面（state == "level_2" 且 buffer 为空），则认为是有效的序号
        valid_h4 = m4 and (
            not REQUIRE_BLANK_BEFORE_H4 or
            prev_line_empty or
            (state == "level_2" and not seen_text_since_heading)
        )

        if valid_h4:
            if state == "content":
                flush_content()
            elif state:
                # 检查 buffer 中是否只有空行（说明没有 Preamble）
                buffer_content = "\n".join(buffer).strip()
                if not buffer_content:
                    # 没有 Preamble，直接清空 buffer
                    buffer = []
                else:
                    # 有内容，先 flush 作为 Preamble
                    flush_preamble(state)

            l3_id = m4.group(1)
            l3_title = m4.group(2).strip()

            stats["level_3_items"] += 1
            state = "content"
            buffer = []
            seen_text_since_heading = False
            prev_line_empty = False
            continue

        # ---------- Body ----------
        if state:
            buffer.append(line)
            if line.strip():
                seen_text_since_heading = True

        prev_line_empty = (line.strip() == "")

    # ---------- final flush ----------
    if state == "content":
        flush_content()
    elif state:
        flush_preamble(state)

    return results, stats


# =========================
# Runner
# =========================
if __name__ == "__main__":
    md_path = "/home/pmw/h20/Text_matching/RBA-VAP-Standard-V8.0.2_Apr2025-A.md"

    md_text = Path(md_path).read_text(encoding="utf-8")

    parsed, stats = parse_markdown(
        md_text=md_text,
        version="V4.9",
        source="RBA-VAP-Standard-V8.0.2_Apr2025-A.md"
    )

    with open("RBA_standard.json", "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    print("\n====== Parsing Summary ======")
    for k, v in stats.items():
        print(f"{k:25s}: {v}")
    print("============================\n")
