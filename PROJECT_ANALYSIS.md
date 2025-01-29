# Ollama Deep Researcher 项目分析

## 项目概述
Ollama Deep Researcher 是一个自动化研究助手，能够进行深度的网络研究和内容总结。它使用本地运行的 LLM 模型来处理和分析信息，通过多轮迭代不断深化对研究主题的理解。

## 技术架构

### 核心组件
1. **LangGraph 框架**
   - 用于构建和管理工作流图
   - 提供 Studio UI 界面进行可视化操作

2. **本地 LLM 集成**
   - 通过 Ollama 运行本地 LLM 模型
   - 默认使用 llama3.2 模型
   - 支持自定义模型配置

3. **网络搜索**
   - 使用 Tavily API 进行网络内容检索
   - 支持配置搜索结果数量和深度

### 依赖项
- langgraph >= 0.2.55
- langchain-community >= 0.3.9
- tavily-python >= 0.5.0
- langchain-ollama >= 0.2.1

## 工作流程

### 主要节点
1. **查询生成 (generate_query)**
   - 基于研究主题生成精确的搜索查询
   - 输出包含查询内容、方面和理由的 JSON

2. **网络搜索 (web_research)**
   - 执行网络搜索
   - 收集和格式化搜索结果

3. **内容总结 (summarize_sources)**
   - 总结搜索结果
   - 整合现有总结和新信息

4. **反思分析 (reflect_on_summary)**
   - 识别知识空白
   - 生成后续查询

5. **最终总结 (finalize_summary)**
   - 生成最终报告
   - 包含所有来源引用

### 迭代流程
```
[开始] 
   ↓
生成查询 → 网络搜索 → 总结来源 → 反思总结
   ↓                                  ↑
   └──────────────────────────────────┘
   （循环直到达到最大迭代次数）
   ↓
最终总结
   ↓
[结束]
```

## 状态管理

### 核心状态对象 (SummaryState)
- research_topic: 研究主题
- search_query: 当前搜索查询
- web_research_results: 搜索结果列表
- sources_gathered: 收集的数据来源
- research_loop_count: 当前迭代次数
- running_summary: 运行中的总结内容

## 配置选项

### 可配置参数
- max_web_research_loops: 最大研究循环次数（默认3次）
- local_llm: 使用的本地 LLM 模型名称（默认 llama3.2）

## 特色功能
1. **智能查询生成**
   - 自动生成相关性高的搜索查询
   - 支持查询优化和迭代改进

2. **深度研究能力**
   - 多轮迭代研究
   - 自动识别知识空白
   - 持续优化研究方向

3. **结构化输出**
   - Markdown 格式的研究报告
   - 包含完整的源引用
   - 清晰的信息组织

4. **本地化处理**
   - 除网络搜索外全部本地运行
   - 保护隐私和数据安全
   - 降低 API 依赖

## 使用场景
- 学术研究资料收集
- 技术调研和分析
- 市场研究和竞品分析
- 知识库构建和更新

## LLM 提供商支持

### 支持的 LLM 提供商
1. **Ollama（本地）**
   - 默认配置
   - 支持所有 Ollama 托管的模型
   - 无需 API 密钥

2. **OpenAI**
   - 支持 GPT-4、GPT-3.5-turbo 等模型
   - 需要 OpenAI API 密钥
   - 支持 JSON 模式输出

3. **Anthropic**
   - 支持 Claude-2、Claude-instant-1 等模型
   - 需要 Anthropic API 密钥
   - 高质量文本生成能力

4. **Google Gemini**
   - 默认使用 gemini-pro 模型
   - 需要 Google API 密钥
   - 自动将系统消息转换为人类消息
   - 支持 JSON 格式输出

### 配置参数
- llm_provider: LLM提供商选择（"ollama"、"openai"、"anthropic"、"gemini"）
- llm_model: 具体的模型名称
- openai_api_key: OpenAI API密钥
- anthropic_api_key: Anthropic API密钥
- google_api_key: Google API密钥

### 更新依赖
新增支持：
- google-generativeai >= 0.3.0
- langchain-google-genai >= 0.0.4

### 注意事项
- 对于非本地模型，需要相应的 API 密钥
- 所有提供商都支持温度控制和格式化输出
- Gemini 模型会自动处理系统消息转换
- 原始配置已备份在 .bak 文件中