import argparse
from src.assistant.tech_landscape import TechLandscape
from src.assistant.configuration import Configuration

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Generate a technology landscape tree')
    parser.add_argument('technology', help='Root technology name to analyze')
    parser.add_argument('--output', '-o', default='tech_landscape.json', 
                       help='Output file path (default: tech_landscape.json)')
    parser.add_argument('--depth', '-d', type=int, default=2,
                       help='Maximum depth of the technology tree (default: 2)')
    parser.add_argument('--max-related', '-m', type=int, default=5,
                       help='Maximum number of related technologies per node (default: 5)')
    parser.add_argument('--language', '-l', default='English',
                       help='Output language (default: English)')
    parser.add_argument('--model', default='gemini-2.0-flash-exp',
                       help='Ollama model to use (default: mistral)')
    parser.add_argument('--iterations', '-i', type=int, default=2,
                       help='Number of research iterations per technology (default: 2)')
    
    args = parser.parse_args()

    # 创建配置
    config = Configuration(
        llm_provider="gemini",  # 使用 Ollama 作为默认 LLM
        llm_model=args.model, #"gemini-2.0-flash-exp",
        output_language=args.language,
        summary_max_length=3000,  # 设置合理的摘要长度限制
        summary_min_length=1000,
        max_web_research_loops=args.iterations
    )

    # 创建技术全景图生成器
    landscape_generator = TechLandscape(
        config=config,
        max_depth=args.depth,
        max_related_techs=args.max_related,
        research_iterations=args.iterations
    )

    try:
        # 生成技术全景图
        print(f"正在分析技术: {args.technology}")
        print(f"设置参数:")
        print(f"- 深度限制: {args.depth}")
        print(f"- 每个节点的相关技术数量: {args.max_related}")
        print(f"- 每个技术的研究迭代次数: {args.iterations}")
        print(f"- 使用模型: {args.model}")
        print(f"- 输出语言: {args.language}")
        print("\n这个过程可能需要一些时间，具体取决于深度和迭代次数...\n")
        
        root_node = landscape_generator.generate_landscape(
            root_tech=args.technology,
            output_file=args.output
        )

        print(f"\n技术全景图已生成并保存至: {args.output}")
        print(f"根技术: {root_node.name}")
        print(f"已分析的相关技术数量: {len(root_node.related_techs)}")
        print(f"可视化图表已保存为: {args.technology.lower().replace(' ', '_')}_landscape.png")
        
    except Exception as e:
        print(f"生成技术全景图时发生错误: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())