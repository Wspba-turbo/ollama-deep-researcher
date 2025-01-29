from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
import matplotlib.pyplot as plt
import networkx as nx
from .utils import tavily_search
from .configuration import Configuration
from .prompts import tech_analysis_instructions
from langchain_core.messages import HumanMessage, SystemMessage

@dataclass
class TechNode:
    """表示技术全景图中的一个技术节点"""
    name: str
    description: str = ""
    related_techs: List["TechNode"] = field(default_factory=list)
    depth: int = 0
    summary: str = ""
    search_results: List[Dict] = field(default_factory=list)

class TechLandscape:
    """技术全景图生成器"""
    
    def __init__(self, config: Configuration, max_depth: int = 2, max_related_techs: int = 5):
        self.config = config
        self.max_depth = max_depth
        self.max_related_techs = max_related_techs
        self.llm = self._get_llm()

    def _get_llm(self):
        """获取配置的 LLM"""
        from .graph import get_llm
        return get_llm(self.config, temperature=0)

    def search_tech_info(self, tech_name: str) -> List[Dict]:
        """搜索技术相关信息"""
        query = f"{tech_name} technology overview latest developments applications"
        results = tavily_search(query, include_raw_content=True, max_results=3)
        return results

    def summarize_and_extract_related(self, tech_name: str, search_results: List[Dict]) -> tuple[str, List[str]]:
        """总结搜索内容并提取相关技术"""
        # 格式化搜索结果
        formatted_results = "\n\n".join([
            f"Title: {result['title']}\nContent: {result.get('content', '')}\n"
            for result in search_results.get('results', [])
        ])

        # 使用预定义的提示模板
        prompt = tech_analysis_instructions.format(
            tech_name=tech_name,
            language=self.config.output_language
        ) + f"\n\nSearch Results:\n{formatted_results}"

        # 获取 LLM 响应
        response = self.llm.invoke([
            HumanMessage(content=prompt)
        ])

        # 解析响应
        try:
            result = json.loads(response.content)
            return result["summary"], result["related_technologies"]
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return "", []

    def build_tech_tree(self, root_tech: str, current_depth: int = 0) -> TechNode:
        """递归构建技术树"""
        if current_depth >= self.max_depth:
            return TechNode(name=root_tech, depth=current_depth)

        # 搜索技术信息
        search_results = self.search_tech_info(root_tech)
        
        # 总结并提取相关技术
        summary, related_techs = self.summarize_and_extract_related(root_tech, search_results)
        
        # 创建当前节点
        node = TechNode(
            name=root_tech,
            description=summary,
            depth=current_depth,
            search_results=search_results
        )

        # 递归处理相关技术
        for tech in related_techs[:self.max_related_techs]:
            child_node = self.build_tech_tree(tech, current_depth + 1)
            node.related_techs.append(child_node)

        return node

    def to_dict(self, node: TechNode) -> Dict:
        """将技术树转换为字典格式"""
        return {
            "name": node.name,
            "description": node.description,
            "depth": node.depth,
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