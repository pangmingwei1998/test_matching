# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个**责任标准文档自动化比对系统**。它使用两阶段方法比对两份责任标准文档并识别相似段落：

1. **语义向量召回**（BGE-M3 + FAISS）- 粗匹配，召回 Top-K 候选
2. **LLM 语义判断**（GPT-3.5-turbo）- 精细相关性分类（不相关/弱相关/强相关）

结果导出到带格式的 Excel 文件。

## 开发命令

```bash
# 安装依赖
pip install -r requirements.txt

# 解析 Markdown 文档为 JSON（推荐使用 parse_4ji.py）
python3 parse_4ji.py

# 运行主程序
python text_matching.py

# 手动下载 BGE-M3 模型（可选 - 首次运行会自动下载）
python download_model.py

# 使用 GPU 支持（替换 faiss-cpu）
pip install faiss-gpu
```

## 架构说明

### 主要组件

| 文件 | 职责 |
|-------|---------|
| [text_matching.py](text_matching.py) | 主比对系统：BGE-M3 嵌入、FAISS 索引、LLM 判断、Excel 导出 |
| [parse_4ji.py](parse_4ji.py) | Markdown 文档解析器：将层级 Markdown 转换为结构化 JSON |
| [download_model.py](download_model.py) | BGE-M3 模型手动下载工具 |

### text_matching.py 类结构

| 类 | 行号 | 职责 |
|-------|-------|---------|
| `Config` | 23-50 | 集中配置管理（文件路径、模型设置、API 配置） |
| `BGEEmbedder` | 105-208 | BGE-M3 文本嵌入，自动检测 CUDA/CPU |
| `VectorIndex` | 211-226 | 基于 FAISS 的向量索引和内积相似度搜索 |
| `LLMJudge` | 229-320 | 基于 LLM 的语义相关性判断（不相关/弱相关/强相关） |
| `TextMatcher` | 323-510 | 主流程编排和 Excel 导出 |

### 数据流程

```
Markdown 文档 (.md)
         ↓
parse_4ji.py 解析
         ↓
JSON 文档（结构化）
         ↓
text_matching.py:
  ├─ 加载 JSON 文档
  ├─ 为供应商文档生成嵌入向量
  ├─ 构建 FAISS 索引
  ├─ 对每条企业联盟文档：
  │   ├─ 召回 Top-K 候选（向量相似度）
  │   ├─ 按 SIMILARITY_THRESHOLD 过滤（默认 0.8）
  │   └─ 对过滤后的候选进行 LLM 判断
  └─ 导出到 Excel（带格式）
```

## parse_4ji.py - Markdown 文档解析器

### 功能

将层级结构的 Markdown 文档解析为结构化 JSON，供 text_matching.py 使用。

### 层级结构

```
### (H1) - Theme（主题，如 Anti-Discrimination）
  ↓
## (H2) - Level 1（一级标题，如 Supplier Code of Conduct Requirements）
  ↓
# (H3) - Level 2（二级标题，如 Supplier Responsibility Standards）
  ↓
1. (H4) - Level 3 item（序号条款，如 1. Policy & Procedures）
```

### 配置规则

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `H1_PATTERN` | `^###\s+(.*)` | Theme 级别 |
| `H2_PATTERN` | `^##\s+(.*)` | Level 1 级别 |
| `H3_PATTERN` | `^#\s+(.*)` | Level 2 级别 |
| `H4_PATTERN` | `^(\d+)\.\s+(.*)` | Level 3 item（序号） |
| `REQUIRE_BLANK_BEFORE_H4` | `True` | 序号前是否需要空行 |

### 重要逻辑

1. **Preamble 识别**：层级标题与序号之间的内容为 Preamble（前言）
2. **空 Preamble 处理**：当序号紧跟在三级标题后时（中间无内容），跳过 Preamble 生成
3. **ID 提取**：标题中的编号（如 `A1. Title` → `id=A1, title=Title`）自动提取
4. **Path 构建**：自动生成层级路径，格式为 `Theme > level_1 > level_2 > level_3`

### JSON 输出格式

```json
{
  "Theme": "Anti-Discrimination",
  "version": "V4.9",
  "level_1": {
    "id": null,
    "title": "Supplier Code of Conduct Requirements"
  },
  "level_2": {
    "id": null,
    "title": "Supplier Responsibility Standards"
  },
  "level_3": {
    "id": "1",
    "title": "Policy & Procedures"
  },
  "content": "条款内容...",
  "path": "Anti-Discrimination > Supplier Code of Conduct Requirements > Supplier Responsibility Standards > 1. Policy & Procedures",
  "source": "apple4.9.md"
}
```

### 使用方法

修改脚本底部的文件路径后运行：

```python
md_path = "/path/to/document.md"
version = "V4.9"
source = "document.md"

parsed, stats = parse_markdown(md_text, version, source)
```

### 输入文件

#### 原始 Markdown 文档

- **RBA-VAP-Standard-V8.0.2_Apr2025-A.md** - RBA 责任标准（V8.0.2）
- **apple4.9.md** - Apple 供应商责任标准（4.9）

#### 解析后的 JSON 文档

使用 [parse_4ji.py](parse_4ji.py) 解析后的输出：

- **Apple_standard.json** - Apple 解析结果（219 条记录）
- **RBA_4JI_A.json** - RBA 解析结果

#### JSON 结构说明

**parse_4ji.py 输出格式（数组，每行一个 JSON 对象）：**

```json
{
  "Theme": "Anti-Discrimination",
  "version": "V4.9",
  "level_1": {
    "id": null,
    "title": "Supplier Code of Conduct Requirements"
  },
  "level_2": {
    "id": null,
    "title": "Supplier Responsibility Standards"
  },
  "level_3": {
    "id": "1",
    "title": "Policy & Procedures"
  },
  "content": "条款内容...",     // 或 "Preamble": "前言内容..."
  "path": "Anti-Discrimination > Supplier Code of Conduct Requirements > Supplier Responsibility Standards > 1. Policy & Procedures",
  "source": "apple4.9.md"
}
```

**关键字段：**
- `content` / `Preamble`：二选一，content 用于序号条款，Preamble 用于层级前言
- `path`：层级路径，用于筛选和追溯（格式优化后包含 id 和 title）
- `level_X.id`：可选的编号标识（如 `A1`、`1` 等）

#### 输出文件

- **matching_results.xlsx** - Excel 比对结果（带格式）

## text_matching.py 关键配置

所有路径和参数集中在 `Config` 类中 ([text_matching.py:23-50](text_matching.py#L23-L50))：

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HuggingFace 嵌入模型 |
| `EMBEDDING_DEVICE` | 自动检测 | 有 `cuda` 用 cuda，否则 `cpu` |
| `TOP_K` | 5 | 每个查询召回的候选数量 |
| `SIMILARITY_THRESHOLD` | 0.8 | LLM 判断的最低相似度阈值 |
| `LLM_API_BASE` | `http://10.71.5.24:8000/v1` | LLM API 端点 |
| `LLM_MODEL` | `gpt-3.5-turbo` | 语义判断模型 |

## 重要注意事项

### text_matching.py

1. **LLM API 依赖**：需要能访问 `http://10.71.5.24:8000/v1` 的 LLM API。系统内置重试逻辑（最多 3 次，指数退避）。

2. **模型缓存**：BGE-M3 模型（约 2GB）缓存在 `~/.cache/huggingface/hub/models--BAAI--bge-m3`。首次下载后支持离线模式。

3. **JSON 加载**：`load_json_documents()` 函数支持多行 JSON 格式（对象可跨多行）。

### parse_4ji.py

1. **序号识别逻辑**：当 `REQUIRE_BLANK_BEFORE_H4=True` 时，序号（`1.`）前需要空行才被识别；但当序号紧跟在三级标题（`#`）后时，即使没有空行也会被识别（通过 `seen_text_since_heading` 标志判断）。

2. **空 Preamble 跳过**：如果三级标题和序号之间没有内容，不会生成空的 Preamble 记录。

3. **Path 格式优化**：path 字段自动包含 id 和 title，格式为 `Theme > level_1 > level_2 > id. title`，null 值不会加入。

4. **语言**：文档和注释使用中文。输入/输出数据为中英文混合。

### Excel 输出格式

输出文件对重复的企业联盟条款使用合并单元格，自定义列宽（A:60, B:60, C:15, D:15, E:40, F:10, G:30, H:15），行高 200，全边框样式。
