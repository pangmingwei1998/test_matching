#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用文档解析脚本
支持任意层级的 Markdown 文档自动解析为 JSON 格式

通用规则:
1. 自动识别 Markdown 标题层级 (#, ##, ###)
2. 智能识别前言部分 (Preamble, Introduction, Code 8.0, 概述 等)
3. 按层级结构分块，前言单独存储
4. 支持 1-3 层嵌套结构
5. 可配置合并策略，控制分块粒度

输出格式 (参考 json标准格式参考.json):
- 一级大标题 (##) 作为 Theme
- 二级标题 (##) 作为 level_1
- 三级标题 (### 或编号) 作为 level_2
- 一级和二级之间的内容为 level_1 的 Preamble
- 二级和三级之间的内容为 level_2 的 Preamble
- 三级标题下的内容为 content
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


# ==================== 通用配置 ====================

# 前言关键词（支持中英文）
PREAMBLE_KEYWORDS = [
    'preamble', 'preambles', 'introduction', 'overview', 'summary',
    '前言', '概述', '简介', '摘要', '说明',
    'code 8.0', 'code 8.0:', 'code 8.0 preamble',
    'the following notes', 'general provisions', 'general notes',
    'supplier code of conduct requirements'
]

# 编号模式（用于识别编号标题）
NUMBERING_PATTERNS = [
    r'^([A-Z]\d+)\.',           # A1, B2 格式
    r'^([A-Z])\.',              # A, B 格式
    r'^(\d+)[、.]\s*',          # 1, 2、格式 (中文)
    r'^(\d+)\.\d+',             # 1.1, 2.1 格式
    r'^([IVXLCDM]+)\.',         # I, II, III 罗马数字
    r'^([a-z])\.',              # a, b 格式
]


# ==================== 通用文档解析器 ====================

class GeneralDocumentParser:
    """通用 Markdown 文档解析器"""

    def __init__(
        self,
        content: str,
        theme_name: str = "Unknown",
        version: str = "Unknown",
        source: str = "Unknown"
    ):
        """
        初始化解析器

        Args:
            content: Markdown 文档内容
            theme_name: 一级大标题名称 (Theme)
            version: 版本号
            source: 来源文件名
        """
        self.content = content
        self.lines = content.split('\n')
        self.theme_name = theme_name
        self.version = version
        self.source = source
        self.items: List[Dict[str, Any]] = []

    def parse(self) -> List[Dict[str, Any]]:
        """解析文档"""
        self.items = []

        # 解析文档结构
        headers = self._extract_all_headers()

        if not headers:
            # 没有标题，整个文档作为一个块
            return [self._create_single_block()]

        # 提取文档级别的前言（第一个标题之前的内容）
        doc_preamble = self._extract_doc_preamble(headers)
        if doc_preamble:
            self.items.append(doc_preamble)

        # 按层级处理内容
        self._process_headers(headers)

        return self.items

    def _extract_all_headers(self) -> List[Dict[str, Any]]:
        """提取所有标题"""
        headers = []
        for i, line in enumerate(self.lines):
            # 匹配 #, ##, ### 标题
            match = re.match(r'^(#{1,3})\s+(.+)$', line.rstrip())
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                header_id = self._extract_header_id(title)

                headers.append({
                    'index': i,
                    'level': level,
                    'title': title,
                    'id': header_id
                })
        return headers

    def _extract_doc_preamble(self, headers: List[Dict]) -> Optional[Dict]:
        """提取文档级别的前言"""
        if not headers:
            return None

        first_header_idx = headers[0]['index']
        preamble_lines = []

        for i in range(first_header_idx):
            line = self.lines[i].strip()
            if line:
                preamble_lines.append(line)

        if preamble_lines:
            content = '\n'.join(preamble_lines)
            return {
                "Theme": self.theme_name,
                "version": self.version,
                "level_1": {
                    "id": "null",
                    "title": "null"
                },
                "level_2": {
                    "id": "null",
                    "title": "null"
                },
                "Preamble": content,
                "path": f"{self.theme_name} > Preamble",
                "source": self.source
            }
        return None

    def _process_headers(self, headers: List[Dict[str, Any]]):
        """按层级处理标题"""
        # 跟踪当前的一级和二级标题上下文
        level_1_context = None
        level_2_context = None

        for i, header in enumerate(headers):
            # 更新一级标题上下文
            if header['level'] == 1:
                level_1_context = header
                level_2_context = None

                # 检查一级标题后到二级标题之间是否有 Preamble
                self._process_level_1_section(header, headers, i)
                continue

            # 更新二级标题上下文
            if header['level'] == 2:
                level_2_context = header

                # 处理二级标题部分
                self._process_level_2_section(level_1_context, header, headers, i)
                continue

            # 处理三级标题
            if header['level'] == 3:
                self._process_level_3_section(level_1_context, level_2_context, header, headers, i)

    def _process_level_1_section(self, header: Dict, headers: List[Dict], idx: int):
        """处理一级标题部分（提取到下一个一级或二级标题之间的内容）"""
        start_idx = header['index'] + 1
        end_idx = headers[idx + 1]['index'] if idx + 1 < len(headers) else len(self.lines)

        # 收集内容
        preamble_lines = []
        for j in range(start_idx, end_idx):
            line = self.lines[j].rstrip()

            # 跳过标题行
            if re.match(r'^#+\s+', line):
                continue

            if self._is_preamble_line(line):
                preamble_lines.append(line)

        # 如果有前言内容，创建 Preamble 块
        if preamble_lines:
            preamble_content = '\n'.join(preamble_lines).strip()
            self.items.append({
                "Theme": self.theme_name,
                "version": self.version,
                "level_1": {
                    "id": header['id'],
                    "title": header['title']
                },
                "level_2": {
                    "id": "null",
                    "title": "null"
                },
                "Preamble": preamble_content,
                "path": f"{self.theme_name} > {header['id']} > Preamble",
                "source": self.source
            })

    def _process_level_2_section(self, level_1_context: Optional[Dict], header: Dict,
                                 headers: List[Dict], idx: int):
        """处理二级标题部分"""
        start_idx = header['index'] + 1
        end_idx = headers[idx + 1]['index'] if idx + 1 < len(headers) else len(self.lines)

        # 收集内容
        preamble_lines = []
        content_lines = []

        for j in range(start_idx, end_idx):
            line = self.lines[j].rstrip()

            # 跳过低级标题（三级标题）
            if re.match(r'^#{3}\s+', line):
                continue

            # 分离 Preamble 和普通内容
            if self._is_preamble_line(line):
                preamble_lines.append(line)
            else:
                content_lines.append(line)

        # 获取一级标题信息
        l1_id = level_1_context['id'] if level_1_context else "null"
        l1_title = level_1_context['title'] if level_1_context else "null"

        # 如果有 Preamble，创建 Preamble 块
        if preamble_lines:
            preamble_content = '\n'.join(preamble_lines).strip()
            self.items.append({
                "Theme": self.theme_name,
                "version": self.version,
                "level_1": {
                    "id": l1_id,
                    "title": l1_title
                },
                "level_2": {
                    "id": header['id'],
                    "title": header['title']
                },
                "Preamble": preamble_content,
                "path": f"{self.theme_name} > {l1_id} > {header['id']} > Preamble",
                "source": self.source
            })

        # 如果有普通内容，创建内容块
        if content_lines:
            content = '\n'.join(content_lines).strip()
            self.items.append({
                "Theme": self.theme_name,
                "version": self.version,
                "level_1": {
                    "id": l1_id,
                    "title": l1_title
                },
                "level_2": {
                    "id": header['id'],
                    "title": header['title']
                },
                "content": content,
                "path": f"{self.theme_name} > {l1_id} > {header['id']}",
                "source": self.source
            })

    def _process_level_3_section(self, level_1_context: Optional[Dict],
                                 level_2_context: Optional[Dict],
                                 header: Dict, headers: List[Dict], idx: int):
        """处理三级标题部分"""
        start_idx = header['index'] + 1
        end_idx = headers[idx + 1]['index'] if idx + 1 < len(headers) else len(self.lines)

        # 收集内容
        content_lines = []
        preamble_lines = []

        for j in range(start_idx, end_idx):
            line = self.lines[j].rstrip()

            # 跳过同级或更高级标题
            if re.match(r'^#{1,3}\s+', line):
                continue

            # 分离 Preamble 和普通内容
            if self._is_preamble_line(line):
                preamble_lines.append(line)
            else:
                content_lines.append(line)

        content = '\n'.join(content_lines).strip()

        # 获取上下文
        l1_id = level_1_context['id'] if level_1_context else "null"
        l1_title = level_1_context['title'] if level_1_context else "null"
        l2_id = level_2_context['id'] if level_2_context else "null"
        l2_title = level_2_context['title'] if level_2_context else "null"

        # 如果有 Preamble，创建 Preamble 块
        if preamble_lines:
            preamble_content = '\n'.join(preamble_lines).strip()
            self.items.append({
                "Theme": self.theme_name,
                "version": self.version,
                "level_1": {
                    "id": l1_id,
                    "title": l1_title
                },
                "level_2": {
                    "id": l2_id,
                    "title": l2_title
                },
                "Preamble": preamble_content,
                "path": f"{self.theme_name} > {l1_id} > {l2_id} > Preamble",
                "source": self.source
            })

        # 创建内容块
        if content:
            self.items.append({
                "Theme": self.theme_name,
                "version": self.version,
                "level_1": {
                    "id": l1_id,
                    "title": l1_title
                },
                "level_2": {
                    "id": l2_id,
                    "title": l2_title
                },
                "content": content,
                "path": f"{self.theme_name} > {l1_id} > {l2_id} > {header['id']}",
                "source": self.source
            })

    def _create_single_block(self) -> Dict[str, Any]:
        """创建单个内容块（当文档没有标题时）"""
        content = '\n'.join(self.lines).strip()
        return {
            "Theme": self.theme_name,
            "version": self.version,
            "level_1": {
                "id": "null",
                "title": "null"
            },
            "level_2": {
                "id": "null",
                "title": "null"
            },
            "content": content,
            "path": f"{self.theme_name}",
            "source": self.source
        }

    def _extract_header_id(self, title: str) -> Optional[str]:
        """从标题中提取 ID"""
        for pattern in NUMBERING_PATTERNS:
            match = re.match(pattern, title, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # 如果没有编号，使用标题的简化形式
        return self._sanitize_title(title)

    def _sanitize_title(self, title: str) -> str:
        """将标题转换为 ID 格式"""
        sanitized = re.sub(r'[^a-zA-Z0-9]', '', title)
        return sanitized[:20] if sanitized else "null"

    def _is_preamble_line(self, line: str) -> bool:
        """判断一行是否是前言内容"""
        if not line:
            return False

        line_lower = line.lower().strip()

        # 检查是否匹配前言关键词
        for keyword in PREAMBLE_KEYWORDS:
            if keyword.lower() in line_lower:
                return True

        return False


# ==================== 辅助函数 ====================

def save_json(data: List[Dict[str, Any]], output_path: str):
    """保存为 JSON 数组格式"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_pretty_json(data: List[Dict[str, Any]], output_path: str):
    """保存为更易读的 JSON 格式"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('[\n')
        for i, item in enumerate(data):
            json_str = json.dumps(item, ensure_ascii=False, indent=2)
            indented = '  ' + json_str.replace('\n', '\n  ')
            f.write(indented)
            if i < len(data) - 1:
                f.write(',\n\n')
            else:
                f.write('\n')
        f.write(']\n')


def parse_markdown_file(
    file_path: str,
    theme_name: str = "Unknown",
    version: str = "Unknown",
    source: str = "Unknown"
) -> List[Dict[str, Any]]:
    """
    解析 Markdown 文件

    Args:
        file_path: 文件路径
        theme_name: 一级大标题名称 (Theme)
        version: 版本号
        source: 来源文件名
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    parser = GeneralDocumentParser(content, theme_name, version, source)
    return parser.parse()


# ==================== 主函数 ====================

def main():
    """主函数 - 使用示例"""
    base_dir = Path("/home/pmw/h20/Text_matching")

    # 示例 1: 解析 RBA 文档
    print("解析 RBA 文档...")
    rba_file = base_dir / "RBA-VAP-Standard-V8.0.2_Apr2025-A.md"
    if rba_file.exists():
        rba_items = parse_markdown_file(
            str(rba_file),
            theme_name="RBA VAP Standard",
            version="V8.0.2",
            source="RBA-VAP-Standard-V8.0.2_Apr2025-A.md"
        )
        print(f"RBA 文档解析完成，共 {len(rba_items)} 个条目")

        rba_output = base_dir / "RBA_general_parsed.json"
        save_pretty_json(rba_items, rba_output)
        print(f"RBA JSON 已保存到: {rba_output}")

        # 打印示例
        print("\nRBA 示例条目:")
        for item in rba_items[:5]:
            print(f"  - [{item['path']}]")

    # 示例 2: 解析 Apple 文档
    print("\n解析 Apple 文档...")
    apple_file = base_dir / "apple4.9.md"
    if apple_file.exists():
        apple_items = parse_markdown_file(
            str(apple_file),
            theme_name="Apple Supplier Code of Conduct",
            version="4.9",
            source="apple4.9.md"
        )
        print(f"Apple 文档解析完成，共 {len(apple_items)} 个条目")

        apple_output = base_dir / "Apple_general_parsed.json"
        save_pretty_json(apple_items, apple_output)
        print(f"Apple JSON 已保存到: {apple_output}")

        # 打印示例
        print("\nApple 示例条目:")
        for item in apple_items[:5]:
            print(f"  - [{item['path']}]")

    print("\n" + "=" * 50)
    print("通用解析脚本使用方法:")
    print("=" * 50)
    print("""
# 在代码中使用:
from General_parse import parse_markdown_file, save_pretty_json

# 解析文档
items = parse_markdown_file(
    "your_document.md",
    theme_name="Your Theme Name",
    version="1.0",
    source="your_document.md"
)

# 保存结果
save_pretty_json(items, "output.json")

# 输出格式说明:
# - Theme: 一级大标题名称
# - version: 版本号
# - level_1: 一级标题信息 {id, title}
# - level_2: 二级标题信息 {id, title}
# - Preamble: 前言内容 (在 Preamble 块中)
# - content: 正文内容 (在内容块中)
# - path: 路径 "Theme > level_1 > level_2 > item"
# - source: 来源文件名
    """)


if __name__ == "__main__":
    main()
