from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from .utils import tavily_search
from .configuration import Configuration
from .prompts import tech_analysis_instructions, summarizer_instructions, reflection_instructions
from langchain_core.messages import HumanMessage, SystemMessage
from .state import SummaryState, SummaryStateInput, SummaryStateOutput
from pyvis.network import Network

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
    relation_strength: float = 1.0  # 默认关联强度为1.0

class TechLandscape:
    """技术全景图生成器"""

    def __init__(self, config: Configuration, max_depth: int = 2, max_related_techs: int = 5, research_iterations: int = 2):
        self.config = config
        self.max_depth = max_depth
        self.max_related_techs = max_related_techs
        self.research_iterations = research_iterations
        self.llm = self._get_llm()
        self.seen_techs = set()  # 添加循环检测

        # 配置日志记录
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # 设置缓存
        from functools import lru_cache
        self.search_tech_info = lru_cache(maxsize=100)(self.search_tech_info)

        # 设置中文字体
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'Heiti TC']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            self.logger.error(f"字体配置失败: {e}")

    def _get_llm(self):
        """获取配置的 LLM"""
        from .graph import get_llm
        return get_llm(self.config, temperature=0)

    def search_tech_info(self, tech_name: str, query: str = None, depth: int = 0) -> Dict:
        """搜索技术相关信息，根据深度动态调整搜索结果数量"""
        if query is None:
            query = f"{tech_name} technology core concepts and applications"

        # 动态调整搜索结果数量：层级越深，结果越少
        max_results = max(3 - depth, 1)
        self.logger.info(f"搜索技术 {tech_name} (深度: {depth}, 结果数: {max_results})")

        try:
            results = tavily_search(query, include_raw_content=True, max_results=max_results)
            self.logger.info(f"成功获取 {len(results.get('results', []))} 条搜索结果")
            return results
        except Exception as e:
            self.logger.error(f"搜索时出错: {e}")
            return {"results": []}

    def _fix_json_array(self, text: str) -> str:
        """修复和清理 JSON 数组格式"""
        self.logger.info("开始处理 JSON 数组...")

        # 移除代码块标记和所有多余的空白字符
        text = re.sub(r'```json\s*|\s*```', '', text)
        text = ' '.join(text.split())
        self.logger.info(f"移除代码块标记后: {text}")

        # 如果不是数组格式，返回原文本
        if not (text.startswith('[') and text.endswith(']')):
            self.logger.info("输入不是数组格式")
            return text

        # 提取数组内容
        content = text[1:-1].strip()
        self.logger.info(f"提取的数组内容: {content}")

        # 如果内容为空，返回空数组
        if not content:
            self.logger.info("数组内容为空")
            return '[]'

        try:
            # 修复引号和逗号
            self.logger.info("开始修复 JSON 格式...")

            # 提取所有被引号包围的项
            items = re.findall(r'"([^"]*)"', content)
            self.logger.info(f"提取到的技术列表: {items}")

            if not items:
                # 如果没有找到引号包围的项，尝试其他分割方法
                items = [x.strip() for x in content.split() if x.strip()]

            # 确保每个项都正确格式化
            fixed_items = [f'"{item}"' for item in items if item]
            fixed_json = '[' + ','.join(fixed_items) + ']'

            try:
                # 验证修复后的 JSON
                self.logger.info(f"修复后的 JSON: {fixed_json}")
                parsed = json.loads(fixed_json)
                self.logger.info(f"解析出 {len(parsed)} 个技术")
                return fixed_json
            except json.JSONDecodeError as je:
                self.logger.error(f"JSON 验证失败: {je}")
                # 使用更安全的格式
                return '["' + '","'.join(items) + '"]'
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON 修复后仍然无效: {e}")
            self.logger.info("尝试使用备用方法修复...")

            # 备用修复方法
            try:
                items = []
                for item in re.findall(r'"[^"]*"', content):
                    item = item.strip()
                    if item:
                        items.append(item)
                if not items:  # 如果没有找到带引号的项，尝试分割并添加引号
                    items = [f'"{item.strip()}"' for item in re.split(r'[,\s]+', content) if item.strip()]

                fixed_json = '[' + ','.join(items) + ']'
                self.logger.info(f"备用修复结果: {fixed_json}")
                json.loads(fixed_json)  # 验证修复结果
                return fixed_json
            except Exception as e2:
                self.logger.error(f"备用修复也失败了: {e2}")
                return '[]'  # 如果所有修复尝试都失败，返回空数组

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

    def analyze_tech(self, tech_name: str, current_depth: int) -> TechNode:
        """分析单个技术节点，使用迭代总结和反思的方式"""
        self.logger.info(f"\n开始分析技术: {tech_name} (深度: {current_depth})")
        node = TechNode(name=tech_name)
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

        # 分阶段搜索策略
        for i in range(self.research_iterations):
            # 阶段判断：前2次广度搜索，后续深度搜索
            if i < 2:  # 广度阶段
                query = f"{tech_name} core concepts applications latest trends"
            else:       # 深度阶段
                query = self._generate_deep_query(state)  # 基于反思生成定向查询

            # 执行搜索（动态调整max_results）
            search_results = self.search_tech_info(
                tech_name,
                query=query,
                depth=current_depth,
                max_results=5 if i < 2 else 3  # 广度阶段多结果，深度阶段少而精
            )
            # 后续总结与反思逻辑保持不变...

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
        self.logger.info(f"执行初始搜索...")
        search_results = self.search_tech_info(tech_name, depth=current_depth)
        self.logger.info(f"获取到 {len(search_results.get('results', []))} 条搜索结果")

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
                search_results = self.search_tech_info(tech_name, query=reflection['follow_up_query'], depth=current_depth)

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
        self.logger.info("\n开始生成内容总结...")
        configurable = self.config
        most_recent_web_research = state.web_research_results[-1]

        if state.running_summary:
            self.logger.info("扩展现有总结...")
            human_message_content = (
                f"Please base your summary on the following information and expand the existing summary:\n\n"
                f"Existing Summary: {state.running_summary}\n\n"
                f"New Research Results: {most_recent_web_research}\n\n"
                f"Research Topic: {state.research_topic}"
            )
            human_message_content += '''
Please read the existing summary carefully first，then extend it with the new search results base on the following instructions:
1. Preserve Core Information: Keep all critical insights and key details from the original summary intact, ensuring no loss of essential knowledge.
2. Integrate New Insights Without Redundancy: Introduce new information only if it adds unique value—strictly avoid rephrasing, reintroducing, or restating previously covered points.
3. **Actively Optimize Existing Content**:
   - Reorganize paragraphs to merge overlapping content into logical sections.
   - Replace vague statements with precise new insights.
   - Remove redundant general descriptions; retain specific examples and data.
   - Integrate new information into the most contextually relevant sections (do NOT simply append to the end).

**Example Optimization**:
❌ Original: "AI agents use machine learning techniques..."
✅ Improved: "Modern AI agents leverage transformer-based architectures (e.g., GPT-4) to enhance reasoning capabilities, as demonstrated in recent research on..."
'''
        else:
            self.logger.info("生成初始总结...")
            human_message_content = (
                f"Please summarize the following research results about {state.research_topic}:\n\n{most_recent_web_research}"
            )

        self.logger.info("调用 LLM 生成总结...")
        result = self.llm.invoke([
            SystemMessage(content=summarizer_instructions.format(
                language=configurable.output_language,
                min_length=configurable.summary_min_length
            )),
            HumanMessage(content=human_message_content)
        ])
        self.logger.info("总结生成完成")

        return result.content

    def reflect_and_generate_query(self, state: SummaryState) -> Dict:
        """反思并生成新的查询"""
        result = self.llm.invoke([
            SystemMessage(content=reflection_instructions.format(
                research_topic=state.research_topic,
                language=self.config.output_language
            )),
            HumanMessage(content=f"""Based on this summary, please analyze and identify knowledge gaps, then generate a follow-up query:

{state.running_summary}

Please respond in JSON format with two fields:
1. knowledge_gap: The identified knowledge gap
2. follow_up_query: The next search query

Ensure the response is valid JSON.""")
        ])

        try:
            content = self._clean_json_response(result.content)
            reflection = json.loads(content)
            return reflection
        except Exception as e:
            self.logger.error(f"Error parsing reflection response: {e}")
            self.logger.error(f"Original response: {result.content}")
            self.logger.error(f"Cleaned response: {content}")
            return {
                "knowledge_gap": "Error parsing reflection results",
                "follow_up_query": f"{state.research_topic} latest developments and applications"
            }

    def extract_related_technologies(self, summary: str, tech_name: str) -> List[Dict]:
        """提取相关技术并生成关联强度"""
        prompt = f"""
        Based on the summary of {tech_name}, extract 3-5 specific technologies and their relevance scores (0.0-1.0).
        Relevance score criteria:
        - 1.0: Directly integrated (e.g., LLM for AI Agent)
        - 0.6-0.9: Indirect but critical (e.g., PyTorch for deep learning)
        - <0.6: General domain (e.g., "cloud computing")

        Example response for "AI Agent":
        [
            {{"name": "Large Language Models (LLM)", "score": 0.95}},
            {{"name": "Retrieval-Augmented Generation (RAG)", "score": 0.9}},
            {{"name": "LangChain", "score": 0.85}}
        ]

        Summary:
        {summary}

        Respond STRICTLY in JSON format:
        [{{"name": "Technology", "score": float}}]
        """

        result = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            techs = json.loads(self._clean_json_response(result.content))
            # 过滤低关联技术 (score < 0.6)
            filtered_techs = [t for t in techs if t["score"] >= 0.6]
            return filtered_techs[:self.max_related_techs]
        except Exception as e:
            self.logger.error(f"Error parsing related technologies: {e}")
            return []

    def build_tech_tree(self, root_tech: str, current_depth: int = 0) -> TechNode:
        """递归构建技术树（添加循环检测）"""
        # 检查深度限制和循环
        if current_depth >= self.max_depth or root_tech in self.seen_techs:
            self.logger.info(f"{'  ' * current_depth}停止分析 {root_tech}: " +
                         ("达到最大深度" if current_depth >= self.max_depth else "检测到循环"))
            return TechNode(name=root_tech, depth=current_depth)

        self.seen_techs.add(root_tech)  # 标记为已处理
        self.logger.info(f"\n{'  ' * current_depth}分析技术: {root_tech}")

        # 分析当前技术
        node, related_techs = self.analyze_tech(root_tech, current_depth)
        node.depth = current_depth

        # 递归处理相关技术（携带relation_strength）
        for tech_data in related_techs:
            tech_name = tech_data["name"]
            child_node = self.build_tech_tree(tech_name, current_depth + 1)
            child_node.relation_strength = tech_data["score"]  # 设置关联强度
            node.related_techs.append(child_node)

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
        """生成交互式HTML网络图"""
        self.logger.info("开始生成交互式可视化...")
        net = Network(height="800px", directed=True)

        # 递归添加节点和边
        def add_nodes_edges(node: TechNode, parent_id: str = None):
            # 节点颜色和大小基于关联强度
            color = self._get_color_by_strength(node.relation_strength)
            size = 20 + 15 * node.relation_strength  # 强度越高，节点越大

            net.add_node(
                node.name,
                label=node.name,
                color=color,
                size=size,
                title=f"Strength: {node.relation_strength:.2f}\n{node.description[:200]}..."
            )

            # 添加边并设置权重
            if parent_id:
                net.add_edge(parent_id, node.name, value=node.relation_strength)

            for child in node.related_techs:
                add_nodes_edges(child, node.name)

        add_nodes_edges(root_node)
        net.show(f"{root_node.name}_landscape.html")

        # 设置物理布局参数
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -100,
              "springLength": 200,
              "avoidOverlap": 0.5
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
          },
          "configure": {
            "enabled": true,
            "filter": "physics"
          }
        }
        """)

        # 颜色映射函数
        def _get_color_by_strength(self, strength: float) -> str:
            """根据关联强度生成渐变色（红→黄→绿）"""
            if strength >= 0.9:
                return "#2ecc71"  # 强关联-绿色
            elif strength >= 0.7:
                return "#f1c40f"  # 中关联-黄色
            else:
                return "#e74c3c"  # 弱关联-红色

        # 生成HTML文件
        base_name = root_node.name.lower().replace(' ', '_')
        html_path = f"{base_name}_landscape.html"
        net.show(html_path, notebook=False)
        self.logger.info(f"已生成交互式可视化文件: {html_path}")

        # 同时生成静态PNG图像作为备份
        self._generate_static_image(root_node)

    def generate_landscape(self, root_tech: str, output_file: str) -> TechNode:
        """生成技术全景图"""
        # 构建技术树
        root_node = self.build_tech_tree(root_tech)

        # 保存到文件
        self.save_landscape(root_node, output_file)

        # 生成可视化（同时生成交互式HTML和静态PNG）
        self.visualize_landscape(root_node)

        self.logger.info(f"技术全景图生成完成 - {root_tech}")
        return root_node

    def _generate_static_image(self, root_node: TechNode):
        """生成静态PNG备份图像"""
        self.logger.info("生成静态备份图像...")

        # 创建图形对象
        G = nx.Graph()

        def add_nodes_edges(node: TechNode, parent_name: str = None):
            G.add_node(node.name)
            if parent_name:
                G.add_edge(parent_name, node.name)
            for child in node.related_techs:
                add_nodes_edges(child, node.name)

        add_nodes_edges(root_node)
        self.logger.info(f"添加了 {len(G.nodes)} 个节点和 {len(G.edges)} 条边")

        # 设置绘图参数
        plt.figure(figsize=(20, 10))
        pos = nx.spring_layout(G, k=1.5, iterations=50)

        # 根据深度设置节点颜色
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            depth = 0
            # 查找节点深度
            def find_node_depth(search_node: TechNode, target_name: str, current_depth: int = 0) -> int:
                if search_node.name == target_name:
                    return current_depth
                for child in search_node.related_techs:
                    result = find_node_depth(child, target_name, current_depth + 1)
                    if result is not None:
                        return result
                return None

            depth = find_node_depth(root_node, node) or 0
            color = {0: "#FF6B6B", 1: "#4ECDC4", 2: "#45B7D1"}.get(depth, "#96CEB4")
            size = 3000 - depth * 500
            node_colors.append(color)
            node_sizes.append(size)

        # 绘制节点和边
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                             node_size=node_sizes, alpha=0.7)
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=2)

        # 添加标签
        font_size = 10
        labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size, font_weight='bold')

        # 设置标题
        title = "技术全景图: " if self.config.output_language.lower() == 'chinese' else "Technology Landscape: "
        plt.title(title + root_node.name, fontsize=14, fontweight='bold')
        plt.axis('off')

        # 保存图像
        base_name = root_node.name.lower().replace(' ', '_')
        plt.savefig(f"{base_name}_landscape.png", bbox_inches='tight', dpi=300)
        plt.close()
        self.logger.info("静态图像生成完成")

    def _generate_deep_query(self, state: SummaryState) -> str:
        """基于历史反思生成定向查询"""
        if not state.historical_reflections:
            return f"{state.research_topic} technical implementation details"

        # 提取最近的知识缺口
        last_reflection = state.historical_reflections[-1]
        gap = last_reflection.get("knowledge_gap", "")

        # 生成定向查询（示例）
        return f"{state.research_topic} {gap} technical specifications case studies"
