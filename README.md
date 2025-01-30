# Ollama Deep Researcher

Ollama Deep Researcher is a fully local web research assistant that uses any LLM hosted by [Ollama](https://ollama.com/search). Give it a topic and it will generate a web search query, gather web search results (via [Tavily](https://www.tavily.com/) by default), summarize the results of web search, reflect on the summary to examine knowledge gaps, generate a new search query to address the gaps, search, and improve the summary for a user-defined number of cycles. It will provide the user a final markdown summary with all sources used. 

![research-rabbit](https://github.com/user-attachments/assets/4308ee9c-abf3-4abb-9d1e-83e7c2c3f187)

Short summary:
<video src="https://github.com/user-attachments/assets/02084902-f067-4658-9683-ff312cab7944" controls></video>

## 📺 Video Tutorials

See it in action or build it yourself? Check out these helpful video tutorials:
- [Overview of Ollama Deep Researcher with R1](https://www.youtube.com/watch?v=sGUjmyfof4Q) - Load and test [DeepSeek R1](https://api-docs.deepseek.com/news/news250120) [distilled models](https://ollama.com/library/deepseek-r1).
- [Building Ollama Deep Researcher from Scratch](https://www.youtube.com/watch?v=XGuTzHoqlj8) - Overview of how this is built.

## 🚀 Quickstart

### Mac 

1. Download the Ollama app for Mac [here](https://ollama.com/download).

2. Pull a local LLM from [Ollama](https://ollama.com/search). As an [example](https://ollama.com/library/deepseek-r1:8b): 
```bash
ollama pull deepseek-r1:8b
```

3. For free web search (up to 1000 requests), [sign up for Tavily](https://tavily.com/). 

4. Set the `TAVILY_API_KEY` environment variable and restart your terminal to ensure it is set:

```bash
export TAVILY_API_KEY=<your_tavily_api_key>
```

5. (Recommended) Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

6. Clone the repository and launch the assistant with the LangGraph server:

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository and start the LangGraph server 
git clone https://github.com/langchain-ai/ollama-deep-researcher.git
cd ollama-deep-researcher
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
```

### Windows 

1. Download the Ollama app for Windows [here](https://ollama.com/download).

2. Pull a local LLM from [Ollama](https://ollama.com/search). As an [example](https://ollama.com/library/deepseek-r1:8b): 
```powershell
ollama pull deepseek-r1:8b
```

3. For free web search (up to 1000 requests), [sign up for Tavily](https://tavily.com/). 

4. Set the `TAVILY_API_KEY` environment variable in Windows (via System Properties or PowerShell). Crucially, restart your terminal/IDE (or sometimes even your computer) after setting it for the change to take effect.

5. (Recommended) Create a virtual environment: Install `Python 3.11` (and add to PATH during installation). Restart your terminal to ensure Python is available, then create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

6. Clone the repository and launch the assistant with the LangGraph server:

```powershell
# Clone the repository 
git clone https://github.com/langchain-ai/ollama-deep-researcher.git
cd ollama-deep-researcher

# Install dependencies 
pip install -e .
pip install langgraph-cli[inmem]

# Start the LangGraph server
langgraph dev
```

## 使用技术全景图生成功能

你可以使用 `generate_tech_landscape.py` 脚本来生成技术全景图。该脚本会对指定的技术进行深入分析，生成相关技术的树状图，并提供可视化结果。

### 基本用法

```bash
python generate_tech_landscape.py <技术名称> [选项]
```

### 参数说明

- `技术名称`: 要分析的根技术名称
- `--output`, `-o`: 输出文件路径（默认：tech_landscape.json）
- `--depth`, `-d`: 技术树的最大深度（默认：2）
- `--max-related`, `-m`: 每个节点的相关技术数量上限（默认：5）
- `--language`, `-l`: 输出语言（默认：English）
- `--model`: 使用的 Ollama 模型（默认：mistral）
- `--iterations`, `-i`: 每个技术的研究迭代次数（默认：2）

### 示例

1. 基本使用（使用默认参数）：
```bash
python generate_tech_landscape.py "AI Agent"
```

2. 指定参数：
```bash
python generate_tech_landscape.py "AI Agent" -d 1 -m 3 -l "English" --model mistral --iterations 2
```

### 输出文件

脚本会生成两个输出文件：
1. JSON 文件（默认：tech_landscape.json）：包含完整的技术分析结果，包括：
   - 技术名称
   - 技术描述
   - 深度级别
   - 历史总结
   - 相关技术列表

2. PNG 图片（例如：ai_agent_landscape.png）：可视化的技术关系图
   - 以节点和边展示技术之间的关系
   - 使用不同的颜色和大小来表示技术的层级
   - 清晰展示技术之间的关联性

## How it works

[Original content continues below...]

[Previous content remains unchanged]
