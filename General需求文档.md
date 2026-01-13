完全参考text_matching.py完成General_text_matching.py,
一个通用的条款匹配，输入文件分别是A文件，B文件。
1、首先要判断json文件中的一个块属于Preamble还是content。取A文件内的所有的content的条款全部依次按顺序和B文件的content条款匹配，（这一步骤不包含Preamble的条款）。
2、仅仅将json块中content里的内容进行向量化，和相似度匹配，其他信息不参与向量化与相似度匹配，避免污染语义。
3、相似度检索规则同text_matching.py一样，删选阈值0.8以上的文档。
4、LLM 精判规则不变。
5、追加时，注意一点，即使A中的某一条 条款没有匹配的结果，也需要将A按顺序追加进excel文档，后面匹配的结果空格显示即可。