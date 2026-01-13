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

# 运行主程序
python text_matching.py

# 手动下载 BGE-M3 模型（可选 - 首次运行会自动下载）
python download_model.py

# 使用 GPU 支持（替换 faiss-cpu）
pip install faiss-gpu
```

## 架构说明

### 主要组件 ([text_matching.py](text_matching.py))

| 类 | 行号 | 职责 |
|-------|-------|---------|
| `Config` | 23-50 | 集中配置管理（文件路径、模型设置、API 配置） |
| `BGEEmbedder` | 105-208 | BGE-M3 文本嵌入，自动检测 CUDA/CPU |
| `VectorIndex` | 211-226 | 基于 FAISS 的向量索引和内积相似度搜索 |
| `LLMJudge` | 229-320 | 基于 LLM 的语义相关性判断（不相关/弱相关/强相关） |
| `TextMatcher` | 323-510 | 主流程编排和 Excel 导出 |

### 数据流程

```
加载 JSON 文档（多行 JSON 格式）
         ↓
为供应商文档生成嵌入向量
         ↓
构建 FAISS 索引
         ↓
对每条企业联盟文档：
  ├─ 召回 Top-K 候选（向量相似度）
  ├─ 按 SIMILARITY_THRESHOLD 过滤（默认 0.8）
  └─ 对过滤后的候选进行 LLM 判断
         ↓
导出到 Excel（带格式）
```

### 输入文件

#### 原始 Markdown 文档

- **RBA-VAP-Standard-V8.0.2_Apr2025-A.md** - RBA 责任标准（V8.0.2）
- **apple4.9.md** - Apple 供应商责任标准（4.9）

##### 解析后的 JSON 文档

使用 [parse_documents.py](parse_documents.py) 将 Markdown 文档解析为 JSON：

```bash
# 解析两个文档
python3 parse_documents.py
```

生成两个 JSON 文件：

- **RBA_standard_parsed.json** - RBA 解析结果
- **Apple_standard_parsed.json** - Apple 解析结果

###### JSON 结构说明

```json
{
  "meta": {
    "source": "RBA" | "Apple",           // 文档来源
    "version": "V8.0.2" | "4.9",         // 版本号
    "date": "Apr2025" | null,            // 日期
    "title": "文档标题",
    "generated_at": "2026-01-12T...",    // 生成时间
    "total_blocks": 18                   // 总块数
  },
  "blocks": [
    {
      "block_id": "RBA-A1-policy",       // 全局唯一 ID
      "source": "RBA" | "Apple",         // 来源
      "topic": "A. Labor" | "Anti-Discrimination",  // 顶层主题
      "section": "A1. Prohibition of Forced Labor",  // 二级标题
      "category": "1. Policy",           // 分块依据（按编号/分类）
      "category_type": "policy",         // 标准化分类类型
      "text": "完整条款内容...",         // 用于向量化的文本
      "path": "A > A1 > policy",         // 层级路径（用于筛选/追溯）
      "keywords": ["forced", "labor"],   // 提取的关键词
      "word_count": 163,                 // 词数统计
      "table": {                         // 可选：表格数据（RBA 评估标准）
        "has_table": true,
        "table_markdown": "<table>...",
        "table_type": "rating_matrix"
      }
    }
  ]
}
```

###### category_type 标准化映射

**RBA:**

| category | category_type |
|----------|---------------|
| Preamble / Code 8.0 Labor Preamble | `preamble` |
| Code 8.0 | `code` |
| 1. Policy | `policy` |
| 2. Procedures & Practices | `procedures` |
| 3. Controls & Monitoring | `controls` |
| 4. Records | `records` |
| 5. Serious conditions | `serious_conditions` |
| 6. Leading Practices | `leading_practices` |
| 7. Fees evaluation criteria | `evaluation_criteria` |

**Apple:**

| category | category_type |
|----------|---------------|
| Supplier Code of Conduct Requirements | `code_of_conduct` |
| 1、Policy & Procedures | `policy_procedures` |
| 2、Operational Practice | `operational_practice` |
| 3、Training and Communication | `training_communication` |
| 4、Documentation | `documentation` |
| 5、Victim Support | `victim_support` |

###### 分块规则

| 文档 | 分块粒度 | 说明 |
|------|----------|------|
| RBA | 按编号 1-7 分块 | Code 8.0 前言单独分块；表格数据保留在 `table` 字段 |
| Apple | 按中文数字 1、2、3、4 分块 | Supplier Code of Conduct Requirements 单独分块 |

#### 输出文件

- **matching_results.xlsx** - 列名：`企业联盟条款`、`供应商条款`、`相似度得分`、`LLM判断结果`、`LLM判断理由`、`排名`、`供应商标题`、`企业版本`

## 关键配置

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

1. **LLM API 依赖**：需要能访问 `http://10.71.5.24:8000/v1` 的 LLM API。系统内置重试逻辑（最多 3 次，指数退避）。

2. **模型缓存**：BGE-M3 模型（约 2GB）缓存在 `~/.cache/huggingface/hub/models--BAAI--bge-m3`。首次下载后支持离线模式。

3. **语言**：文档和注释使用中文。输入/输出数据为中英文混合。

4. **Excel 格式**：输出文件对重复的企业联盟条款使用合并单元格，自定义列宽，行高 200，全边框样式。

5. **JSON 加载**：输入文件使用多行 JSON 格式（每行一个 JSON 对象），由 `load_json_documents()` 工具函数处理。
