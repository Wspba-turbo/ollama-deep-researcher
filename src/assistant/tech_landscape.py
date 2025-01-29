from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
import re
import matplotlib.pyplot as plt
import networkx as nx
from .utils import tavily_search
from .configuration import Configuration
from .prompts import tech_analysis_instructions, summarizer_instructions, reflection_instructions
from langchain_core.messages import HumanMessage, SystemMessage
from .state import SummaryState, SummaryStateInput, SummaryStateOutput

@dataclass
class TechNode:
    """表示技术全景图中的一个技术节点"""
    name: str
    description: str = ""
    related_techs: List["TechNode"] = field(default_factory=list)
    depth: int = 0
    summary: str = ""
    search_results: List[Dict] = field(default_factory=list)
    historical_summaries: List[str] = field(default_factory=list)
    historical_reflections: List[Dict] = field(default_factory=list)

class TechLandscape:
    """技术全景图生成器"""
    
    def __init__(self, config: Configuration, max_depth: int = 2, max_related_techs: int = 5, research_iterations: int = 2):
        self.config = config
        self.max_depth = max_depth
        self.max_related_techs = max_related_techs
        self.research_iterations = research_iterations
        self.llm = self._get_llm()

    def _get_llm(self):
        """获取配置的 LLM"""
        from .graph import get_llm
        return get_llm(self.config, temperature=0)

    def search_tech_info(self, tech_name: str, query: str = None) -> Dict:
        """搜索技术相关信息"""
        if query is None:
            query = f"{tech_name} technology overview latest developments applications"
        results = tavily_search(query, include_raw_content=True, max_results=3)
        return results

    def _fix_json_array(self, text: str) -> str:
        """修复和清理 JSON 数组格式"""
        # 移除代码块标记
        text = re.sub(r'```json\s*|\s*```', '', text.strip())
        
        # 如果是数组格式，确保元素之间有逗号
        if text.startswith('[') and text.endswith(']'):
            # 移除现有的逗号和多余的空白
            items = [item.strip() for item in text[1:-1].split('\n') if item.strip()]
            # 重新组合成正确的 JSON 数组格式
            return '[' + ','.join(items) + ']'
        
        return text

    def _clean_json_response(self, response: str) -> str:
        """清理 LLM 响应中的 JSON"""
        response = response.strip()
        
        # 检查是否是数组格式
        if response.startswith('[') and response.endswith(']'):
            return self._fix_json_array(response)
            
        # 对于普通 JSON 对象
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1:
            json_str = response[start:end + 1]
            # 修复常见的 JSON 格式问题
            json_str = re.sub(r',\s*}', '}', json_str)  # 移除最后一个属性后的逗号
            json_str = re.sub(r',\s*]', ']', json_str)  # 移除数组最后一个元素后的逗号
            return json_str
            
        return response

    def analyze_tech(self, tech_name: str) -> TechNode:
        """分析单个技术节点，使用迭代总结和反思的方式"""
        node = TechNode(name=tech_name)
        
        # 创建初始状态
        state = SummaryState(
            research_topic=tech_name,
            running_summary="",
            search_query="",
            research_loop_count=0,
            sources_gathered=[],
            web_research_results=[],
            historical_summaries=[],
            historical_reflections=[]
        )

        # 初始查询
        search_results = self.search_tech_info(tech_name)
        
        # 迭代研究过程
        for i in range(self.research_iterations):
            # 更新搜索结果
            state.web_research_results.append(search_results)
            state.sources_gathered.append("\n".join(
                f"* {source['title']} : {source['url']}"
                for source in search_results['results']
            ))
            
            # 总结当前内容
            summary = self.summarize_content(state)
            state.historical_summaries.append(summary)
            state.running_summary = summary

            # 如果不是最后一次迭代，进行反思并生成新的查询
            if i < self.research_iterations - 1:
                reflection = self.reflect_and_generate_query(state)
                state.historical_reflections.append(reflection)
                search_results = self.search_tech_info(tech_name, reflection['follow_up_query'])

        # 提取相关技术
        related_techs = self.extract_related_technologies(state.running_summary, tech_name)

        # 更新节点信息
        node.description = state.running_summary
        node.search_results = state.web_research_results
        node.historical_summaries = state.historical_summaries
        node.historical_reflections = state.historical_reflections

        return node, related_techs

    def summarize_content(self, state: SummaryState) -> str:
        """总结内容"""
        configurable = self.config
        most_recent_web_research = state.web_research_results[-1]
        
        if state.running_summary:
            human_message_content = (
                f"请基于以下信息扩展现有总结:\n\n"
                f"现有总结: {state.running_summary}\n\n"
                f"新的研究结果: {most_recent_web_research}\n\n"
                f"研究主题: {state.research_topic}"
            )
        else:
            human_message_content = (
                f"请总结以下关于{state.research_topic}的研究结果:\n\n{most_recent_web_research}"
            )

        result = self.llm.invoke([
            SystemMessage(content=summarizer_instructions.format(
                language=configurable.output_language,
                max_length=configurable.summary_max_length,
                min_length=configurable.summary_min_length
            )),
            HumanMessage(content=human_message_content)
        ])

        return result.content

    def reflect_and_generate_query(self, state: SummaryState) -> Dict:
        """反思并生成新的查询"""
        result = self.llm.invoke([
            SystemMessage(content=reflection_instructions.format(
                research_topic=state.research_topic,
                language=self.config.output_language
            )),
            HumanMessage(content=f"""基于这个总结，请分析并找出知识空白，生成新的查询：

{state.running_summary}

请用JSON格式返回，包含两个字段：
1. knowledge_gap: 发现的知识空白
2. follow_up_query: 下一步的搜索查询

确保返回的是有效的JSON对象。""")
        ])

        try:
            content = self._clean_json_response(result.content)
            reflection = json.loads(content)
            return reflection
        except Exception as e:
            print(f"Error parsing reflection response: {e}")
            print(f"Original response: {result.content}")
            print(f"Cleaned response: {content}")
            return {
                "knowledge_gap": "解析反思结果时出错",
                "follow_up_query": f"{state.research_topic} 最新发展和应用"
            }

    def extract_related_technologies(self, summary: str, tech_name: str) -> List[str]:
        """从总结中提取相关技术"""
        prompt = f"""基于以下关于{tech_name}的技术总结，列出3-5个最相关的具体技术名称：

{summary}

请确保：
1. 只提取具体的技术名称，不要包含描述性文字
2. 所有技术都与{tech_name}有直接关联
3. 每个技术都是实际存在且可实现的
4. 避免过于宽泛的概念
5. 避免重复或相似的技术

请直接返回JSON数组格式的技术名称列表，例如：
["机器学习", "深度学习", "神经网络"]

注意：
- 保持数组格式的严格性
- 确保每个技术名称都用引号括起来
- 使用逗号正确分隔每个技术
- 不要添加额外的描述或说明"""

        result = self.llm.invoke([
            HumanMessage(content=prompt)
        ])

        try:
            content = self._clean_json_response(result.content)
            print(f"Cleaned tech response: {content}")  # Debug output
            techs = json.loads(content)
            if not isinstance(techs, list):
                raise ValueError("Response is not a list")
            return techs[:self.max_related_techs]
        except Exception as e:
            print(f"Error extracting related technologies: {e}")
            print(f"Original response: {result.content}")
            return []

    def build_tech_tree(self, root_tech: str, current_depth: int = 0) -> TechNode:
        """递归构建技术树"""
        if current_depth >= self.max_depth:
            return TechNode(name=root_tech, depth=current_depth)

        print(f"\n{'  ' * current_depth}分析技术: {root_tech}")
        
        # 分析当前技术
        node, related_techs = self.analyze_tech(root_tech)
        node.depth = current_depth

        # 递归处理相关技术
        if related_techs:
            print(f"{'  ' * current_depth}发现相关技术: {', '.join(related_techs)}")
            for tech in related_techs[:self.max_related_techs]:
                child_node = self.build_tech_tree(tech, current_depth + 1)
                node.related_techs.append(child_node)
        else:
            print(f"{'  ' * current_depth}未发现相关技术")

        return node

    def to_dict(self, node: TechNode) -> Dict:
        """将技术树转换为字典格式"""
        return {
            "name": node.name,
            "description": node.description,
            "depth": node.depth,
            "historical_summaries": node.historical_summaries,
            "historical_reflections": node.historical_reflections,
            "related_technologies": [
                self.to_dict(child) for child in node.related_techs
            ]
        }

    def save_landscape(self, root_node: TechNode, file_path: str):
        """保存技术全景图到文件"""
        landscape_dict = self.to_dict(root_node)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(landscape_dict, f, indent=2, ensure_ascii=False)

    def visualize_landscape(self, root_node: TechNode):
        """可视化技术全景图"""
        G = nx.Graph()

        def add_nodes_edges(node: TechNode, parent_name: str = None):
            G.add_node(node.name)
            if parent_name:
                G.add_edge(parent_name, node.name)
            for child in node.related_techs:
                add_nodes_edges(child, node.name)

        add_nodes_edges(root_node)

        # 设置绘图参数
        plt.figure(figsize=(20, 10))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=2000, alpha=0.7)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        
        # 添加标签
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title(f"Technology Landscape: {root_node.name}")
        plt.axis('off')
        
        # 保存图像
        base_name = root_node.name.lower().replace(' ', '_')
        plt.savefig(f"{base_name}_landscape.png", bbox_inches='tight')
        plt.close()

    def generate_landscape(self, root_tech: str, output_file: str) -> TechNode:
        """生成技术全景图"""
        # 构建技术树
        root_node = self.build_tech_tree(root_tech)
        
        # 保存到文件
        self.save_landscape(root_node, output_file)
        
        # 生成可视化
        self.visualize_landscape(root_node)
        
        return root_node