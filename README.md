# 责任标准文档自动化比对系统

## 功能说明

本系统用于自动化比对两份责任标准文档，筛选出相近的段落并导出到 Excel 文件。

### 系统架构

```
语义向量召回（BGE-M3 + FAISS，粗匹配 Top-K）
            ↓
    LLM 语义判断（GPT-3.5-turbo，精匹配）
            ↓
      置信度/规则过滤
            ↓
        导出 Excel
```

### 输入文件

- **企业联盟责任标准.json** - 每行一个 JSON 对象，包含 `Version`, `Date`, `text` 字段
- **供应商责任标准.json** - 每行一个 JSON 对象，包含 `title`, `text` 字段

### 输出文件

- **matching_results.xlsx** - 包含以下列：
  - 企业联盟条款
  - 供应商条款
  - 相似度得分（0-1）
  - LLM 判断结果（不相关/弱相关/强相关）
  - LLM 判断理由
  - 排名
  - 供应商标题
  - 企业版本

## 安装依赖

```bash
cd /home/pmw/h20/Text_matching
pip install -r requirements.txt
```

如果使用 GPU 加速向量计算：

```bash
pip install faiss-gpu  # 替代 faiss-cpu
```

## 配置说明

在 [text_matching.py](text_matching.py) 的 `Config` 类中修改配置：

```python
class Config:
    # 文件路径
    ALLIANCE_FILE = "企业联盟责任标准.json"
    SUPPLIER_FILE = "供应商责任标准.json"
    OUTPUT_EXCEL = "matching_results.xlsx"

    # 向量检索参数
    TOP_K = 5  # 召回候选数量

    # 相似度阈值
    SIMILARITY_THRESHOLD = 0.3  # 低于此分数不进行LLM判断

    # LLM API 配置
    LLM_API_BASE = "http://10.71.5.24:8000/v1"
    LLM_MODEL = "gpt-3.5-turbo"
```

## 运行方式

```bash
python text_matching.py
```

## 运行流程

1. 加载两份 JSON 文档
2. 使用 BGE-M3 模型生成文本嵌入向量
3. 使用 FAISS 构建供应商文档的向量索引
4. 对企业联盟的每个段落：
   - 在供应商文档中召回 Top-5 相似段落
   - 调用 LLM 判断每个候选的相关性（不相关/弱相关/强相关）
5. 导出所有匹配结果到 Excel

## 输出示例

| 企业联盟条款 | 供应商条款 | 相似度得分 | LLM判断结果 | 排名 |
|-------------|-----------|-----------|------------|------|
| 1. Policy: Have a detailed... | # 1.1 Written Policy and... | 0.7523 | 强相关 | 1 |
| 1. Policy: Have a detailed... | # 1.2 Directly Responsible... | 0.6834 | 弱相关 | 2 |

## 注意事项

1. 首次运行会自动下载 BGE-M3 模型（约 2GB）
2. 确保 LLM API 服务可访问
3. 如需调整匹配精度，可修改 `TOP_K` 和 `SIMILARITY_THRESHOLD`
4. GPU 环境可显著提升向量检索速度
