# 句法分析可视化工具

这是一个基于 Streamlit 和 NLTK 的英文句法分析演示应用，可展示：

- 依存句法树
- 成分句法树
- 核心论元提取结果

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

首次启动时，应用会自动下载所需的 NLTK 数据包到项目内的 `nltk_data/` 目录。

## Streamlit Cloud 部署

部署入口文件选择：

```text
app.py
```

应用依赖只需要 `requirements.txt` 中的 `streamlit` 和 `nltk`。
