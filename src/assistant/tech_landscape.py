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

        # 初始化颜色映射
        self.color_map = {
            "strong": "#2ecc71",  # 强关联-绿色
            "medium": "#f1c40f",  # 中关联-黄色
            "weak": "#e74c3c"     # 弱关联-红色
        }

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

    def search_tech_info(self, tech_name: str, query: str = None, depth: int = 0, max_results: int = 3) -> Dict:
        """搜索技术相关信息，根据深度动态调整搜索结果数量"""
        if query is None:
            query = f"{tech_name} technology core concepts and applications"

        # 动态调整搜索结果数量：层级越深，结果越少
        max_results = max(max_results - depth, 1)
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

    def _get_color_by_strength(self, strength: float) -> str:
        """根据关联强度生成渐变色（红→黄→绿）"""
        if strength >= 0.9:
            return self.color_map["strong"]  # 强关联-绿色
        elif strength >= 0.7:
            return self.color_map["medium"]  # 中关联-黄色
        else:
            return self.color_map["weak"]    # 弱关联-红色

    def analyze_tech(self, tech_name: str, current_depth: int) -> tuple[TechNode, List[Dict]]:
        """分析单个技术节点，使用迭代总结和反思的方式"""
        self.logger.info(f"\n开始分析技术: {tech_name} (深度: {current_depth})")
        node = TechNode(name=tech_name)
        related_techs = []

        try:
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
            self.logger.info("执行初始搜索...")
            search_results = self.search_tech_info(tech_name, depth=current_depth)
            if not search_results or not search_results.get('results'):
                self.logger.warning(f"无法获取 {tech_name} 的搜索结果，使用基本描述")
                node.description = f"{tech_name} 的基本技术描述"
                return node, related_techs

            # 迭代研究过程
            for i in range(self.research_iterations):
                try:
                    # 更新搜索结果
                    state.web_research_results.append(search_results)
                    state.sources_gathered.append("\n".join(
                        f"* {source['title']} : {source['url']}"
                        for source in search_results.get('results', [])
                    ))

                    # 总结当前内容
                    summary = self.summarize_content(state)
                    state.historical_summaries.append(summary)
                    state.running_summary = summary

                    # 如果不是最后一次迭代，进行反思并生成新的查询
                    if i < self.research_iterations - 1:
                        reflection = self.reflect_and_generate_query(state)
                        state.historical_reflections.append(reflection)
                        search_results = self.search_tech_info(
                            tech_name,
                            query=reflection.get('follow_up_query'),
                            depth=current_depth
                        )
                except Exception as e:
                    self.logger.error(f"研究迭代 {i} 失败: {str(e)}")
                    break

            # 提取相关技术
            if state.running_summary:
                node.description = state.running_summary
                node.historical_summaries = state.historical_summaries
                node.historical_reflections = state.historical_reflections
                related_techs = self.extract_related_technologies(state.running_summary, tech_name)

        except Exception as e:
            self.logger.error(f"分析技术 {tech_name} 时发生错误: {str(e)}")
            if "429" in str(e) or "Resource has been exhausted" in str(e):
                self.logger.info("API配额用尽，保存已完成的分析结果")
                if not node.description and state.historical_summaries:
                    node.description = state.historical_summaries[-1]

        return node, related_techs

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
        Based on the summary about {tech_name}, identify the MOST SPECIFIC and TECHNICALLY PRECISE components/frameworks that are essential to its implementation.

        CRITICAL REQUIREMENTS:
        1. MUST EXTRACT SPECIFIC TOOLS/FRAMEWORKS:
           ✅ "Large Language Models (LLM)", "GPT-4", "LangChain"
           ❌ "machine learning", "AI", "deep learning"
        
        2. FOCUS ON CORE TECHNICAL COMPONENTS:
           ✅ "Retrieval-Augmented Generation (RAG)", "Vector Database", "RLHF"
           ❌ "optimization", "cloud computing", "algorithms"

        3. PRIORITIZE MODERN, SPECIFIC IMPLEMENTATIONS:
           ✅ "Transformer architecture", "Attention mechanism"
           ❌ "neural networks", "NLP", "computer vision"

        For {tech_name}, identify exactly 3 of its most critical technical components, ranked by integration level:
        - Directly integrated (score 0.9-1.0): Core frameworks/models (e.g., LLM for AI Agent)
        - Closely coupled (score 0.8-0.9): Essential components (e.g., RAG for AI Agent)
        - Supporting tools (score 0.7-0.8): Key implementation tools (e.g., LangChain)

        Summary to analyze:
        {summary}

        Respond STRICTLY in this JSON format:
        [
            {{"name": "Most Critical Component", "score": float}},
            {{"name": "Second Critical Component", "score": float}},
            {{"name": "Third Critical Component", "score": float}}
        ]
        """

        result = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            content = result.content.strip()
            # 清理JSON格式
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # 技术黑名单 - 过于宽泛的术语
            blacklist = {
                "machine learning", "deep learning", "artificial intelligence", "AI",
                "NLP", "neural networks", "computer vision", "cloud computing",
                "algorithms", "optimization", "programming", "software",
                "data science", "automation", "computing"
            }

            techs = json.loads(content)
            if not isinstance(techs, list):
                raise ValueError("Response is not a list")
            
            # 过滤和验证技术
            filtered_techs = []
            for tech in techs:
                if not (isinstance(tech, dict) and "name" in tech and "score" in tech):
                    continue
                    
                name = tech["name"].strip()
                score = tech["score"]
                
                # 验证分数
                if not isinstance(score, (int, float)) or score < 0.7:
                    continue
                
                # 检查是否在黑名单中（忽略大小写）
                if any(black_term.lower() in name.lower() for black_term in blacklist):
                    continue
                
                # 验证名称长度和格式
                if len(name) < 3 or len(name) > 100:
                    continue
                
                filtered_techs.append({
                    "name": name,
                    "score": float(score)  # 确保分数是浮点数
                })
            
            # 确保至少有一个有效的相关技术
            if not filtered_techs:
                self.logger.warning(f"没有找到有效的相关技术，使用默认技术")
                filtered_techs = [{
                    "name": "Related Technology",
                    "score": 0.7
                }]
            
            return filtered_techs[:self.max_related_techs]
        except Exception as e:
            self.logger.error(f"Error parsing related technologies: {e}")
            self.logger.error(f"Original response: {result.content}")
            return []

    def build_tech_tree(self, root_tech: str, current_depth: int = 0) -> TechNode:
        """递归构建技术树，包含错误处理和部分结果保存机制"""
        node = None
        try:
            # 检查深度限制和循环
            if current_depth >= self.max_depth or root_tech in self.seen_techs:
                self.logger.info(f"{'  ' * current_depth}停止分析 {root_tech}: " +
                            ("达到最大深度" if current_depth >= self.max_depth else "检测到循环"))
                node = TechNode(name=root_tech, depth=current_depth)
                node.description = f"{root_tech} - 分析已达到深度限制或检测到循环"
                return node

            self.seen_techs.add(root_tech)  # 标记为已处理
            self.logger.info(f"\n{'  ' * current_depth}分析技术: {root_tech}")

            # 分析当前技术
            try:
                node, related_techs = self.analyze_tech(root_tech, current_depth)
                node.depth = current_depth

                # 如果当前节点缺少描述，使用默认描述
                if not node.description:
                    node.description = f"A technical component essential to {root_tech}"

                # 递归处理相关技术
                completed_children = []
                for tech_data in related_techs:
                    try:
                        tech_name = tech_data["name"]
                        child_node = self.build_tech_tree(tech_name, current_depth + 1)
                        child_node.relation_strength = float(tech_data.get("score", 0.7))
                        
                        # 尝试为空描述的子节点重新获取描述
                        if not child_node.description:
                            try:
                                temp_node, _ = self.analyze_tech(tech_name, current_depth + 1)
                                child_node.description = temp_node.description or f"Related technology to {root_tech}"
                            except Exception as e:
                                self.logger.error(f"重新分析子节点 {tech_name} 失败: {str(e)}")
                                child_node.description = f"A component of {root_tech}"
                        
                        completed_children.append(child_node)
                    except Exception as e:
                        self.logger.error(f"处理子技术 {tech_name} 时出错: {str(e)}")
                        continue

                node.related_techs = completed_children

            except Exception as e:
                self.logger.error(f"分析技术 {root_tech} 时出错: {str(e)}")
                if "429" in str(e) or "Resource has been exhausted" in str(e):
                    self._save_partial_results(root_tech, current_depth)
                raise

        except Exception as e:
            self.logger.error(f"构建技术树时出错: {str(e)}")
            # 创建基本节点作为fallback
            node = TechNode(name=root_tech, depth=current_depth)
            node.description = f"Error occurred while analyzing {root_tech}"

        finally:
            # 保存当前进度
            if node and node.description:
                self._save_partial_results(root_tech, current_depth, node)

        return node

    def _save_partial_results(self, tech_name: str, depth: int, node: Optional[TechNode] = None):
        """保存部分完成的分析结果"""
        try:
            partial_results = {
                "name": tech_name,
                "depth": depth,
                "timestamp": self._get_timestamp(),
                "completed_analysis": list(self.seen_techs)
            }
            
            if node:
                partial_results.update({
                    "description": node.description,
                    "relation_strength": node.relation_strength,
                    "related_techs": [{"name": t.name, "relation_strength": t.relation_strength}
                                    for t in node.related_techs]
                })
            
            filename = f"partial_results_{tech_name.lower().replace(' ', '_')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(partial_results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"已保存部分分析结果到: {filename}")
        except Exception as e:
            self.logger.error(f"保存部分结果失败: {str(e)}")

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

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
        net = None
        html_path = None
        
        try:
            # 初始化网络图
            net = Network(
                height="800px",
                width="100%",
                directed=True,
                bgcolor="#ffffff",
                font_color=True
            )

            # 递归添加节点和边
            def add_nodes_edges(node: TechNode, parent_id: str = None):
                # 节点颜色和大小基于关联强度
                color = self._get_color_by_strength(node.relation_strength)
                size = 20 + 15 * node.relation_strength

                # 截短描述文本，避免过长
                description = node.description if node.description else "暂无描述"
                if len(description) > 500:
                    description = description[:497] + "..."

                # 创建悬停提示
                tooltip = (
                    f"<div class='tooltip'>"
                    f"<h3 style='margin:0 0 10px 0;color:#2c3e50'>{node.name}</h3>"
                    f"<p style='margin:5px 0;color:#34495e'><b>关联强度:</b> {node.relation_strength:.2f}</p>"
                    f"<p style='margin:5px 0;color:#34495e;line-height:1.4'>{description}</p>"
                    f"</div>"
                )

                # 添加节点
                net.add_node(
                    node.name,
                    label=node.name,
                    color=color,
                    size=size,
                    title=tooltip,
                    borderWidth=2,
                    font={'size': 14, 'face': 'Helvetica'},
                    shape='dot'
                )

                # 添加边
                if parent_id:
                    net.add_edge(
                        parent_id,
                        node.name,
                        width=3,  # 增加边的宽度
                        color={'color': '#95a5a6', 'opacity': 0.8},
                        smooth={'type': 'continuous', 'roundness': 0.5}
                    )

                # 递归处理子节点
                for child in node.related_techs:
                    add_nodes_edges(child, node.name)

            # 添加所有节点和边
            add_nodes_edges(root_node)

            # 配置网络图选项
            options = {
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -2000,
                        "centralGravity": 0.01,
                        "springLength": 200,
                        "springConstant": 0.05,
                        "damping": 0.4,
                        "avoidOverlap": 0.5
                    },
                    "minVelocity": 0.75,
                    "solver": "forceAtlas2Based",
                    "stabilization": {
                        "enabled": True,
                        "iterations": 1000,
                        "updateInterval": 25
                    }
                },
                "nodes": {
                    "shape": "dot",
                    "scaling": {"min": 20, "max": 35},
                    "font": {"size": 14, "face": "Helvetica"},
                    "borderWidth": 2,
                    "shadow": True
                },
                "edges": {
                    "color": {"inherit": "false", "color": "#95a5a6"},
                    "smooth": {"type": "continuous", "roundness": 0.5},
                    "width": 3,
                    "shadow": True
                },
                "interaction": {
                    "hover": True,
                    "hideEdgesOnDrag": True,
                    "tooltipDelay": 200,
                    "zoomView": True
                }
            }

            # 保存HTML文件
            base_name = root_node.name.lower().replace(' ', '_')
            html_path = f"{base_name}_landscape.html"
            
            net.set_options(json.dumps(options))
            net.write_html(html_path)
            self.logger.info(f"已生成交互式可视化文件: {html_path}")

            # 生成静态PNG备份
            try:
                self._generate_static_image(root_node)
            except Exception as png_error:
                self.logger.error(f"生成静态PNG备份失败: {str(png_error)}")

        except Exception as e:
            self.logger.error(f"生成可视化文件失败: {str(e)}")
            if "429" in str(e) or "Resource has been exhausted" in str(e):
                if net and html_path:
                    try:
                        # 保存已完成的部分
                        net.write_html(html_path)
                        self.logger.info("已保存部分完成的可视化文件")
                    except Exception as save_error:
                        self.logger.error(f"保存部分完成的文件失败: {str(save_error)}")
            raise

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
