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
    
    args = parser.parse_args()

    # 创建配置
    config = Configuration(
        llm_provider="gemini",  # 使用 Gemini 作为默认 LLM
        output_language=args.language
    )

    # 创建技术全景图生成器
    landscape_generator = TechLandscape(
        config=config,
        max_depth=args.depth,
        max_related_techs=args.max_related
    )

    try:
        # 生成技术全景图
        print(f"Analyzing technology: {args.technology}")
        print(f"This may take a while depending on the depth ({args.depth}) and number of related technologies ({args.max_related})...")
        
        root_node = landscape_generator.generate_landscape(
            root_tech=args.technology,
            output_file=args.output
        )

        print(f"\nTechnology landscape has been generated and saved to: {args.output}")
        print(f"Root technology: {root_node.name}")
        print(f"Number of related technologies analyzed: {len(root_node.related_techs)}")
        
    except Exception as e:
        print(f"Error generating technology landscape: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())