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
        self.llm = self._get_llm()
        
        # 设置中文字体
        try:
            print("配置中文字体支持...")
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'Heiti TC']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            print(f"字体配置失败: {e}")

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
        results = tavily_search(query, include_raw_content=True, max_results=3)
        return results

    def _fix_json_array(self, text: str) -> str:
        """修复和清理 JSON 数组格式"""
        print("\n开始处理 JSON 数组...")
        
        # 移除代码块标记和所有多余的空白字符
        text = re.sub(r'```json\s*|\s*```', '', text)
        text = ' '.join(text.split())
        print(f"移除代码块标记后: {text}")
        
        # 如果不是数组格式，返回原文本
        if not (text.startswith('[') and text.endswith(']')):
            print("输入不是数组格式")
            return text
        
        # 提取数组内容
        content = text[1:-1].strip()
        print(f"提取的数组内容: {content}")
        
        # 如果内容为空，返回空数组
        if not content:
            print("数组内容为空")
            return '[]'
        
        try:
            # 修复引号和逗号
            print("开始修复 JSON 格式...")
            
            # 提取所有被引号包围的项
            items = re.findall(r'"([^"]*)"', content)
            print(f"提取到的技术列表: {items}")
            
            if not items:
                # 如果没有找到引号包围的项，尝试其他分割方法
                items = [x.strip() for x in content.split() if x.strip()]
            
            # 确保每个项都正确格式化
            fixed_items = [f'"{item}"' for item in items if item]
            fixed_json = '[' + ','.join(fixed_items) + ']'
            
            try:
                # 验证修复后的 JSON
                print(f"修复后的 JSON: {fixed_json}")
                parsed = json.loads(fixed_json)
                print(f"解析出 {len(parsed)} 个技术")
                return fixed_json
            except json.JSONDecodeError as je:
                print(f"JSON 验证失败: {je}")
                # 使用更安全的格式
                return '["' + '","'.join(items) + '"]'
        except json.JSONDecodeError as e:
            print(f"JSON 修复后仍然无效: {e}")
            print("尝试使用备用方法修复...")
            
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
                print(f"备用修复结果: {fixed_json}")
                json.loads(fixed_json)  # 验证修复结果
                return fixed_json
            except Exception as e2:
                print(f"备用修复也失败了: {e2}")
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

    def analyze_tech(self, tech_name: str, depth: int = 0) -> TechNode:
        """分析单个技术节点，使用迭代总结和反思的方式"""
        self.logger.info(f"\n开始分析技术: {tech_name} (深度: {depth})")
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
        self.logger.info(f"执行初始搜索...")
        search_results = self.search_tech_info(tech_name, depth=depth)
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
                search_results = self.search_tech_info(tech_name, query=reflection['follow_up_query'], depth=depth)

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
        print("\n开始生成内容总结...")
        configurable = self.config
        most_recent_web_research = state.web_research_results[-1]
        
        if state.running_summary:
            print("扩展现有总结...")
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
            print("生成初始总结...")
            human_message_content = (
                f"Please summarize the following research results about {state.research_topic}:\n\n{most_recent_web_research}"
            )

        print("调用 LLM 生成总结...")
        result = self.llm.invoke([
            SystemMessage(content=summarizer_instructions.format(
                language=configurable.output_language,
                # max_length=configurable.summary_max_length,
                min_length=configurable.summary_min_length
            )),
            HumanMessage(content=human_message_content)
        ])
        print("总结生成完成")

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
            print(f"Error parsing reflection response: {e}")
            print(f"Original response: {result.content}")
            print(f"Cleaned response: {content}")
            return {
                "knowledge_gap": "Error parsing reflection results",
                "follow_up_query": f"{state.research_topic} latest developments and applications"
            }

    def extract_related_technologies(self, summary: str, tech_name: str) -> List[str]:
        """从总结中提取相关技术"""
        print(f"\n开始提取与 {tech_name} 相关的技术...")
        
        prompt = f"""Based on the following summary about {tech_name}, extract 3-5 of the **most directly related** specific technologies. Focus on technologies with the **closest relationship** to the main topic.

{summary}

Please respond strictly in the following format (only the array, no additional content):
["Technology 1","Technology 2","Technology 3"]

Requirements:
1. Extract specific technology names, e.g., "machine learning" instead of "AI technology".
2. Ensure they are **most directly and closely related** to {tech_name}.
3. Enclose each technology name in quotes.
4. Separate technologies with commas.
5. Do not add extra spaces or newlines."""

        print("调用 LLM 提取相关技术...")
        result = self.llm.invoke([
            HumanMessage(content=prompt)
        ])
        print(f"LLM 响应完成，开始处理结果...")

        try:
            content = self._clean_json_response(result.content)
            print(f"清理后的响应内容: {content}")
            techs = json.loads(content)
            if not isinstance(techs, list):
                raise ValueError("Response is not a list")
            filtered_techs = techs[:self.max_related_techs]
            print(f"提取到的相关技术: {filtered_techs}")
            return filtered_techs
        except Exception as e:
            print(f"处理相关技术时出错: {e}")
            print(f"原始响应: {result.content}")
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
        """生成交互式HTML网络图"""
        self.logger.info("开始生成交互式可视化...")
        from pyvis.network import Network

        # 创建网络图对象
        net = Network(height="800px", width="100%", notebook=False, directed=True)
        net.toggle_physics(True)
        net.show_buttons(filter_=['physics'])
        
        # 递归添加节点和边
        def add_nodes_edges(node: TechNode, parent_id: str = None):
            node_id = node.name
            # 根据层级设置节点颜色和大小
            color = {0: "#FF6B6B", 1: "#4ECDC4", 2: "#45B7D1"}.get(node.depth, "#96CEB4")
            size = 30 - node.depth * 5
            
            # 准备节点提示信息
            desc = node.description[:200] + "..." if len(node.description) > 200 else node.description
            tooltip = f"<b>{node.name}</b><br>{desc}"
            
            net.add_node(node_id, label=node.name, color=color, size=size,
                        title=tooltip)
            
            if parent_id:
                net.add_edge(parent_id, node_id)
            
            for child in node.related_techs:
                add_nodes_edges(child, node_id)
        
        # 添加所有节点和边
        add_nodes_edges(root_node)
        self.logger.info("完成节点和边的添加")

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
