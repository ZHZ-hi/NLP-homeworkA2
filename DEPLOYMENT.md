# Streamlit Cloud 部署前检查

## 已准备

- 主入口文件：`app.py`
- Python 依赖文件：`requirements.txt`
- Streamlit 配置：`.streamlit/config.toml`
- Git 忽略规则：`.gitignore`
- NLTK 数据在启动时自动下载到项目内 `nltk_data/`

## 部署步骤

1. 将项目初始化为 Git 仓库并推送到 GitHub。
2. 打开 Streamlit Cloud，选择该 GitHub 仓库。
3. Main file path 填写 `app.py`。
4. 部署后等待依赖安装和 NLTK 数据初始化完成。

## 不建议上传

- `venv/`
- `__pycache__/`
- `nltk_data/`
- 本地私密配置 `.streamlit/secrets.toml`

## 备注

`run.py` 适合本地一键启动，不作为 Streamlit Cloud 的入口文件。
