#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档解析脚本 - 新版
将 RBA 和 Apple 的 markdown 文档解析为统一的 JSON 格式

RBA 结构:
  Standard -> Section (A/B/C) -> Subsection (A1/A2) -> Item (1/2/3)

Apple 结构:
  Standard -> Topic -> Subsection (1/2/3) -> Item (2.1/2.2)
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


# ==================== 域名映射 ====================

# RBA Section -> Domain 映射
RBA_DOMAIN_MAP = {
    "A": "Labor & Human Rights",
    "B": "Health & Safety",
    "C": "Environment",
    "D": "Management System",
    "E": "Ethics",
}

# Apple Topic -> Domain 映射 (简化版，实际可能需要更详细的映射)
APPLE_DOMAIN_MAP = {
    "Anti-Discrimination": "Labor & Human Rights",
    "Anti-Harassment and Abuse": "Labor & Human Rights",
    "Prevention of Involuntary Labor": "Labor & Human Rights",
    "Third Party Employment Agencies": "Labor & Human Rights",
    "Foreign Contract Worker Protections": "Labor & Human Rights",
    "Prevention of Underage Labor": "Labor & Human Rights",
    "Juvenile Worker Protections": "Labor & Human Rights",
    "Educational Program Management": "Labor & Human Rights",
    "Working Hours Management": "Labor & Human Rights",
    "Wages, Benefits, and Contracts": "Labor & Human Rights",
    "Freedom of Association and Collective Bargaining": "Labor & Human Rights",
    "Worker Engagement and Grievance Management": "Labor & Human Rights",
    "Occupational Health and Safety Management": "Health & Safety",
    "Chemical Management": "Health & Safety",
    "Fire Safety Management": "Health & Safety",
    "Emergency Preparedness and Response": "Health & Safety",
    "Infectious Disease Preparedness and Response": "Health & Safety",
    "Incident Management": "Health & Safety",
    "Dormitories and Dining": "Health & Safety",
    "Combustible Dust Hazard Management": "Health & Safety",
    "Machine Safety Management": "Health & Safety",
    "Waste Management": "Environment",
    "Water and Wastewater Management": "Environment",
    "Stormwater Management": "Environment",
    "Air Emissions Management": "Environment",
    "Greenhouse Gas Emissions Management": "Environment",
    "Boundary Noise Management": "Environment",
    "Resource Consumption Management": "Environment",
    "Management Systems": "Management System",
    "Responsible Sourcing of Materials": "Ethics",
}


# ==================== RBA 文档解析器 ====================

class RBADocumentParser:
    """RBA 文档解析器 - 新结构"""

    def __init__(self, content: str):
        self.content = content
        self.lines = content.split('\n')
        self.items: List[Dict[str, Any]] = []

    def parse(self) -> List[Dict[str, Any]]:
        """解析 RBA 文档"""
        self.items = []

        # 首先提取文档级别的 VAP Standard Preamble
        i = self._extract_vap_standard_preamble()

        current_section = None  # A, B, C
        current_section_title = None
        current_subsection = None  # A1, A2, B1
        current_subsection_title = None
        current_subsection_code = None

        while i < len(self.lines):
            line = self.lines[i].strip()

            # 匹配 Section 标题 ## A. Labor 或 ## B. Health and Safety (如果存在)
            section_match = re.match(r'^##\s+([A-Z])\.\s+(.+)$', line)
            if section_match:
                current_section = section_match.group(1)
                current_section_title = section_match.group(2)
                current_subsection = None
                current_subsection_title = None
                i += 1

                # 检查 Section 级别的 Preamble（Code 8.0 Labor Preamble 等）
                i = self._extract_section_preamble(i, current_section, current_section_title)
                continue

            # 匹配 Subsection 标题 # A1. Prohibition of Forced Labor
            subsection_match = re.match(r'^#\s+([A-Z])(\d+)\.\s+(.+)$', line)
            if subsection_match:
                # 从 subsection 中提取 section 信息
                current_section = subsection_match.group(1)
                # 设置 section 标题（从映射表获取）
                current_section_title = self._get_section_title(current_section)

                current_subsection = subsection_match.group(1) + subsection_match.group(2)
                current_subsection_title = subsection_match.group(3)
                current_subsection_code = current_subsection
                i += 1

                # 检查是否有 Code 8.0 前言
                code_text = self._extract_code_8_0(i)
                if code_text:
                    # 添加 Code 8.0 作为一个 Preamble item
                    self.items.append(self._create_item(
                        current_section, current_section_title,
                        current_subsection, current_subsection_title,
                        "preamble", "Preamble", code_text
                    ))

                # 解析该 subsection 下的所有 items
                i = self._parse_items(i, current_section, current_section_title,
                                     current_subsection, current_subsection_title)
                continue

            i += 1

        return self.items

    def _extract_vap_standard_preamble(self) -> int:
        """提取文档级别的 VAP Standard Preamble"""
        i = 0
        preamble_lines = []

        # 查找 ## VAP Standard 标题
        while i < len(self.lines):
            line = self.lines[i].strip()
            if line == "## VAP Standard":
                i += 1
                break
            i += 1

        # 收集 VAP Standard 下的所有内容（直到遇到 # A1. 或 ## A. 等标题）
        while i < len(self.lines):
            line = self.lines[i].strip()

            # 遇到 Section 或 Subsection 标题则停止
            if re.match(r'^##?\s+([A-Z])', line):
                break

            # 收集非空行
            if line:
                preamble_lines.append(line)

            i += 1

        if preamble_lines:
            preamble_text = ' '.join(preamble_lines)
            # 添加为全局 Preamble
            self.items.append({
                "standard": "RBA VAP Standard",
                "version": "V8.0.2",
                "domain": "General",
                "level_1": {
                    "id": "VAP",
                    "title": "VAP Standard"
                },
                "level_2": {
                    "id": "VAP",
                    "title": "VAP Standard"
                },
                "item": {
                    "id": "preamble",
                    "title": "Preamble"
                },
                "content": preamble_text,
                "path": "VAP > VAP > preamble",
                "source": "RBA"
            })

        return i

    def _extract_section_preamble(self, start_idx: int, section: str, section_title: str) -> int:
        """提取章节级别的 Preamble（如 Code 8.0 Labor Preamble）"""
        i = start_idx
        preamble_lines = []

        while i < len(self.lines):
            line = self.lines[i].strip()

            # 遇到 Subsection 标题则停止
            if re.match(r'^#\s+[A-Z]\d+\.', line):
                break

            # 收集 Preamble 内容
            if line.startswith("Code 8.0") or preamble_lines:
                preamble_lines.append(line)

            i += 1

        if preamble_lines:
            preamble_text = ' '.join(preamble_lines)
            # 检查是否是有效的 Preamble（长度 > 50 且包含 Preamble 关键词）
            if len(preamble_text) > 50 and ("Preamble" in preamble_text or "preamble" in preamble_text):
                self.items.append(self._create_item(
                    section, section_title,
                    section, section_title,  # level_2 使用 section 信息
                    "preamble", "Preamble", preamble_text
                ))

        return i

    def _get_section_title(self, section_id: str) -> str:
        """根据 section ID 获取标题"""
        section_titles = {
            "A": "Labor",
            "B": "Health and Safety",
            "C": "Environment",
            "D": "Management System",
            "E": "Ethics",
        }
        return section_titles.get(section_id, "Other")

    def _extract_code_8_0(self, start_idx: int) -> Optional[str]:
        """提取 Code 8.0 前言"""
        i = start_idx
        code_lines = []

        while i < len(self.lines):
            line = self.lines[i].strip()

            # 遇到 Elements to Demonstrate Compliance 或编号标题则停止
            if line.startswith("Elements to Demonstrate Compliance") or re.match(r'^\d+\.\s+(Policy|Procedures)', line):
                break

            # 检查是否是 Code 8.0: 开头的行
            if line.startswith("Code 8.0:") or line.startswith("Code 8.0 "):
                code_lines.append(line)
            elif code_lines and line:  # 已开始收集且有内容
                code_lines.append(line)

            i += 1

        if code_lines:
            return ' '.join(code_lines)
        return None

    def _parse_items(self, start_idx: int, section: str, section_title: str,
                     subsection: str, subsection_title: str) -> int:
        """解析 subsection 下的所有 items"""
        i = start_idx

        # 收集所有内容直到下一个 subsection
        content_lines = []
        while i < len(self.lines):
            line = self.lines[i]

            # 遇到下一个 subsection 则停止
            if re.match(r'^#\s+[A-Z]\d+\.', line.strip()):
                break

            content_lines.append(line)
            i += 1

        content = '\n'.join(content_lines)

        # 解析各个编号 item
        # 1. Policy
        # 2. Procedures & Practices
        # 3. Controls & Monitoring
        # 4. Records
        # 5. Serious conditions
        # 6. Leading Practices
        # 7. Fees evaluation criteria

        patterns = [
            ("1", "Policy", r'1\.\s*Policy:\s*'),
            ("2", "Procedures & Practices", r'2\.\s*Procedures\s*&\s*Practices\s+(?:are\s+in\s+place\s+such\s+that)?'),
            ("3", "Controls & Monitoring", r'3\.\s*Controls\s*&\s*Monitoring\s+(?:should\s+include)?'),
            ("4", "Records", r'4\.\s*Records\s+(?:are\s+maintained\s+including)?'),
            ("5", "Serious Conditions", r'5\.\s*Serious\s+conditions\s+(?:that\s+will\s+result\s+in\s+a\s+severe\s+finding)?'),
            ("6", "Leading Practices", r'6\.\s*Leading\s+Practices\s+(?:include)?'),
            ("7", "Evaluation Criteria", r'7\.\s*(?:Fees\s+)?[Ee]valuation\s+criteria'),
        ]

        for item_id, item_title, pattern in patterns:
            match = re.search(pattern + r'(.+?)(?=\n\n\d+\.|$)', content, re.DOTALL)
            if match:
                item_content = match.group(0).strip()
                self.items.append(self._create_item(
                    section, section_title,
                    subsection, subsection_title,
                    item_id, item_title, item_content
                ))

        return i

    def _create_item(self, section: str, section_title: str,
                    subsection: str, subsection_title: str,
                    item_id: str, item_title: str, content: str) -> Dict[str, Any]:
        """创建 item 数据结构"""
        # 生成路径
        level_1_id = section
        level_2_id = subsection

        path = f"{section} > {subsection} > {item_id}"

        return {
            "standard": "RBA VAP Standard",
            "version": "V8.0.2",
            "domain": RBA_DOMAIN_MAP.get(section, "Other"),
            "level_1": {
                "id": level_1_id,
                "title": section_title
            },
            "level_2": {
                "id": level_2_id,
                "title": subsection_title
            },
            "item": {
                "id": item_id,
                "title": item_title
            },
            "content": content,
            "path": path,
            "source": "RBA"
        }


# ==================== Apple 文档解析器 ====================

class AppleDocumentParser:
    """Apple 文档解析器 - 新结构"""

    def __init__(self, content: str):
        self.content = content
        self.lines = content.split('\n')
        self.items: List[Dict[str, Any]] = []

    def parse(self) -> List[Dict[str, Any]]:
        """解析 Apple 文档"""
        self.items = []

        i = 0
        current_topic = None
        current_level_1_id = None

        while i < len(self.lines):
            line = self.lines[i].strip()

            # 匹配主题标题 ### Anti-Discrimination
            topic_match = re.match(r'^###\s+(.+)$', line)
            if topic_match:
                current_topic = topic_match.group(1)
                current_level_1_id = self._sanitize_id(current_topic)
                i += 1

                # 解析该主题下的所有内容
                i = self._parse_topic_items(i, current_topic, current_level_1_id)
                continue

            i += 1

        return self.items

    def _parse_topic_items(self, start_idx: int, topic: str, level_1_id: str) -> int:
        """解析主题下的所有 items"""
        i = start_idx

        # 收集该主题下的所有内容
        topic_lines = []
        while i < len(self.lines):
            line = self.lines[i]

            # 遇到下一个主题则停止
            if re.match(r'^###\s+', line.strip()):
                break

            topic_lines.append(line)
            i += 1

        topic_text = '\n'.join(topic_lines)

        # 解析 Supplier Code of Conduct Requirements
        conduct_match = re.search(
            r'##\s+Supplier\s+Code\s+of\s+Conduct\s+Requirements\s*\n+(.+?)(?=##\s+Supplier\s+Responsibility\s+Standards|$)',
            topic_text, re.DOTALL
        )
        if conduct_match:
            conduct_text = conduct_match.group(1).strip()
            if conduct_text:
                self.items.append(self._create_item(
                    topic, level_1_id,
                    "conduct", "Supplier Code of Conduct Requirements",
                    None, None,
                    conduct_text
                ))

        # 解析 Supplier Responsibility Standards 下的各个分类
        # 1、Policy & Procedures
        # 2、Operational Practice
        # 3、Training and Communication
        # 4、Documentation
        # 5、Victim Support

        category_patterns = [
            ("1", "Policy & Procedures", r'##\s+1、?\s*Policy\s*&?\s*(?:and|&)\s*Procedures'),
            ("2", "Operational Practice", r'##\s+2、?\s*Operational\s+Practice'),
            ("3", "Training and Communication", r'##\s+3、?\s*Training\s+and\s+Communication'),
            ("4", "Documentation", r'##\s+4、?\s*Documentation'),
            ("5", "Victim Support", r'##\s+5、?\s*Victim\s+Support'),
        ]

        for item_id, item_title, pattern in category_patterns:
            match = re.search(pattern + r'\s*\n+(.+?)(?=##\s+\d、|$)', topic_text, re.DOTALL)
            if match:
                category_text = match.group(0).strip()
                # 移除标题行
                category_text = re.sub(r'^##\s+[\d、]+\s+[^\n]+\n*', '', category_text, count=1).strip()

                if category_text:
                    self.items.append(self._create_item(
                        topic, level_1_id,
                        item_id, item_title,
                        None, None,
                        category_text
                    ))

        return i

    def _sanitize_id(self, text: str) -> str:
        """将文本转换为适合 ID 的格式"""
        return re.sub(r'[^a-zA-Z0-9]', '', text)

    def _create_item(self, topic: str, level_1_id: str,
                    level_2_id: str, level_2_title: str,
                    item_id: Optional[str], item_title: Optional[str],
                    content: str) -> Dict[str, Any]:
        """创建 item 数据结构"""
        # 生成路径
        if item_id:
            path = f"{level_1_id} > {level_2_id} > {item_id}"
        else:
            path = f"{level_1_id} > {level_2_id}"

        # 对于 Apple，level_2 是分类编号，item 是子分类（如 2.1, 2.2）

        domain = APPLE_DOMAIN_MAP.get(topic, "Other")

        item_obj = {
            "id": item_id,
            "title": item_title
        }

        return {
            "standard": "Apple Supplier Code of Conduct",
            "version": "4.9",
            "domain": domain,
            "level_1": {
                "id": level_1_id,
                "title": topic
            },
            "level_2": {
                "id": level_2_id,
                "title": level_2_title
            },
            "item": item_obj,
            "content": content,
            "path": path,
            "source": "Apple"
        }


# ==================== 主函数 ====================

def parse_rba_document(file_path: str) -> List[Dict[str, Any]]:
    """解析 RBA 文档"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    parser = RBADocumentParser(content)
    return parser.parse()


def parse_apple_document(file_path: str) -> List[Dict[str, Any]]:
    """解析 Apple 文档"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    parser = AppleDocumentParser(content)
    return parser.parse()


def save_json(data: List[Dict[str, Any]], output_path: str):
    """保存为 JSON 数组格式"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_pretty_json(data: List[Dict[str, Any]], output_path: str):
    """保存为更易读的 JSON 格式（每行一个对象但带缩进）"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('[\n')
        for i, item in enumerate(data):
            json_str = json.dumps(item, ensure_ascii=False, indent=2)
            # 添加缩进对齐
            indented = '  ' + json_str.replace('\n', '\n  ')
            f.write(indented)
            if i < len(data) - 1:
                f.write(',\n\n')
            else:
                f.write('\n')
        f.write(']\n')


def main():
    """主函数"""
    # 文件路径
    base_dir = Path("/home/pmw/h20/Text_matching")
    rba_file = base_dir / "RBA-VAP-Standard-V8.0.2_Apr2025-A.md"
    apple_file = base_dir / "apple4.9.md"

    # 解析 RBA 文档
    print("解析 RBA 文档...")
    rba_items = parse_rba_document(str(rba_file))
    print(f"RBA 文档解析完成，共 {len(rba_items)} 个条目")

    # 保存 RBA JSON（易读格式）
    rba_output = base_dir / "RBA_standard_parsed.json"
    save_pretty_json(rba_items, rba_output)
    print(f"RBA JSON 已保存到: {rba_output}")

    # 解析 Apple 文档
    print("\n解析 Apple 文档...")
    apple_items = parse_apple_document(str(apple_file))
    print(f"Apple 文档解析完成，共 {len(apple_items)} 个条目")

    # 保存 Apple JSON（易读格式）
    apple_output = base_dir / "Apple_standard_parsed.json"
    save_pretty_json(apple_items, apple_output)
    print(f"Apple JSON 已保存到: {apple_output}")

    # 打印统计信息
    print("\n" + "=" * 50)
    print("解析统计:")
    print(f"RBA 条目数: {len(rba_items)}")
    print(f"Apple 条目数: {len(apple_items)}")
    print(f"总计: {len(rba_items) + len(apple_items)}")
    print("=" * 50)

    # 打印一些示例
    print("\nRBA 示例条目:")
    if rba_items:
        for item in rba_items[:5]:
            print(f"  - [{item['path']}] {item['item']['title']}")

    print("\nApple 示例条目:")
    if apple_items:
        for item in apple_items[:5]:
            print(f"  - [{item['path']}] {item['level_2']['title']}")


if __name__ == "__main__":
    main()
