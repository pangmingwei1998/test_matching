import re
import json
from typing import List, Dict
from pathlib import Path


# =========================
# Patterns
# =========================
H1_PATTERN = re.compile(r'^###\s+(.*)')          # 一级标题
H2_PATTERN = re.compile(r'^##\s+(.*)')           # 二级标题
H3_PATTERN = re.compile(r'^#\s+(.*)')            # 三级标题
H4_PATTERN = re.compile(r'^(\d+)\.\s+(.*)')      # 序号条款


# =========================
# Parser
# =========================
def parse_markdown(md_text: str, version: str, source: str):
    results: List[Dict] = []

    stats = {
        "themes": 0,
        "level_1": 0,
        "level_2": 0,
        "level_3": 0,
        "level_3_items": 0,
        "theme_preambles": 0,
        "level_1_preambles": 0,
        "level_2_preambles": 0,
        "level_3_preambles": 0,
        "content_blocks": 0,
    }

    # ---------- current state ----------
    theme = None

    l1_id = l1_title = None
    l2_id = l2_title = None
    l3_title = None

    buffer: List[str] = []
    state = None   # theme | level_1 | level_2 | level_3 | content

    # ⭐ 仅用于三级标题
    seen_text_between_l3_and_item = False


    # =========================
    # Flush helpers
    # =========================
    def flush_theme_preamble():
        nonlocal buffer
        content = "\n".join(buffer).strip()
        if not content:
            buffer = []
            return

        stats["theme_preambles"] += 1
        results.append({
            "Theme": theme,
            "version": version,
            "level_1": None,
            "level_2": None,
            "level_3": None,
            "Preamble": content,
            "path": f"{theme} > Preamble",
            "source": source
        })
        buffer = []


    def flush_l1_preamble():
        nonlocal buffer
        content = "\n".join(buffer).strip()
        if not content:
            buffer = []
            return

        stats["level_1_preambles"] += 1
        results.append({
            "Theme": theme,
            "version": version,
            "level_1": {"id": l1_id, "title": l1_title},
            "level_2": None,
            "level_3": None,
            "Preamble": content,
            "path": f"{theme} > {l1_title} > Preamble",
            "source": source
        })
        buffer = []


    def flush_l2_preamble():
        nonlocal buffer
        content = "\n".join(buffer).strip()
        if not content:
            buffer = []
            return

        stats["level_2_preambles"] += 1
        results.append({
            "Theme": theme,
            "version": version,
            "level_1": {"id": l1_id, "title": l1_title},
            "level_2": {"id": l2_id, "title": l2_title},
            "level_3": None,
            "Preamble": content,
            "path": f"{theme} > {l1_title} > {l2_title} > Preamble",
            "source": source
        })
        buffer = []


    def flush_l3_preamble():
        nonlocal buffer
        if not seen_text_between_l3_and_item:
            buffer = []
            return

        content = "\n".join(buffer).strip()
        if not content:
            buffer = []
            return

        stats["level_3_preambles"] += 1
        results.append({
            "Theme": theme,
            "version": version,
            "level_1": {"id": l1_id, "title": l1_title},
            "level_2": {"id": l2_id, "title": l2_title},
            "level_3": {"title": l3_title},
            "Preamble": content,
            "path": f"{theme} > {l1_title} > {l2_title} > {l3_title} > Preamble",
            "source": source
        })
        buffer = []


    def flush_content(item_id: str, item_title: str):
        nonlocal buffer
        content = "\n".join(buffer).strip()
        if not content:
            buffer = []
            return

        stats["content_blocks"] += 1
        results.append({
            "Theme": theme,
            "version": version,
            "level_1": {"id": l1_id, "title": l1_title},
            "level_2": {"id": l2_id, "title": l2_title},
            "level_3": {"title": l3_title},
            "item": {"id": item_id, "title": item_title},
            "content": content,
            "path": f"{theme} > {l1_title} > {l2_title} > {l3_title} > {item_id}",
            "source": source
        })
        buffer = []


    # =========================
    # Main loop
    # =========================
    current_item_id = None
    current_item_title = None

    for line in md_text.splitlines():
        line = line.rstrip()

        # ---------- 一级标题 ###
        m1 = H1_PATTERN.match(line)
        if m1:
            if state == "content":
                flush_content(current_item_id, current_item_title)
            elif state == "level_3":
                flush_l3_preamble()
            elif state == "level_2":
                flush_l2_preamble()
            elif state == "level_1":
                flush_l1_preamble()
            elif state == "theme":
                flush_theme_preamble()

            theme = m1.group(1).strip()
            l1_id = l1_title = l2_id = l2_title = l3_title = None
            buffer = []
            state = "theme"
            stats["themes"] += 1
            continue

        # ---------- 二级标题 ##
        m2 = H2_PATTERN.match(line)
        if m2:
            if state == "content":
                flush_content(current_item_id, current_item_title)
            elif state == "level_3":
                flush_l3_preamble()
            elif state == "level_2":
                flush_l2_preamble()
            elif state == "level_1":
                flush_l1_preamble()

            raw = m2.group(1).strip()
            m_id = re.match(r'^([A-Za-z0-9]+)\.\s*(.+)', raw)
            if m_id:
                l1_id = m_id.group(1)
                l1_title = m_id.group(2)
            else:
                l1_id = None
                l1_title = raw

            l2_id = l2_title = l3_title = None
            buffer = []
            state = "level_1"
            stats["level_1"] += 1
            continue

        # ---------- 三级标题 #
        m3 = H3_PATTERN.match(line)
        if m3:
            if state == "content":
                flush_content(current_item_id, current_item_title)
            elif state == "level_3":
                flush_l3_preamble()
            elif state == "level_2":
                flush_l2_preamble()

            raw = m3.group(1).strip()
            m_id = re.match(r'^([A-Za-z0-9]+)\.\s*(.+)', raw)
            if m_id:
                l2_id = m_id.group(1)
                l2_title = m_id.group(2)
            else:
                l2_id = None
                l2_title = raw

            l3_title = None
            buffer = []
            state = "level_2"
            stats["level_2"] += 1
            continue

        # ---------- 四级（语义三级）标题 #
        # 这里才是你规则中的“三级标题”
        if state == "level_2" and not H4_PATTERN.match(line) and line.strip():
            l3_title = line
            buffer = []
            state = "level_3"
            stats["level_3"] += 1
            seen_text_between_l3_and_item = False
            continue

        # ---------- 条款序号 1. 2. 3.
        m4 = H4_PATTERN.match(line)
        if m4:
            if state == "content":
                flush_content(current_item_id, current_item_title)
            elif state == "level_3":
                flush_l3_preamble()

            current_item_id = m4.group(1)
            current_item_title = m4.group(2)
            buffer = []
            state = "content"
            stats["level_3_items"] += 1
            continue

        # ---------- 普通正文 ----------
        if state:
            buffer.append(line)
            if state == "level_3" and line.strip():
                seen_text_between_l3_and_item = True

    # ---------- final flush ----------
    if state == "content":
        flush_content(current_item_id, current_item_title)
    elif state == "level_3":
        flush_l3_preamble()
    elif state == "level_2":
        flush_l2_preamble()
    elif state == "level_1":
        flush_l1_preamble()
    elif state == "theme":
        flush_theme_preamble()

    return results, stats


# =========================
# Runner
# =========================
if __name__ == "__main__":
    md_path = "/home/pmw/h20/Text_matching/apple4.9.md"

    md_text = Path(md_path).read_text(encoding="utf-8")

    parsed, stats = parse_markdown(
        md_text,
        version="V4.9",
        source="apple4.9.md"
    )

    with open("Apple_standard.json", "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    print("\n====== Parsing Summary ======")
    for k, v in stats.items():
        print(f"{k:25s}: {v}")
    print("============================\n")
