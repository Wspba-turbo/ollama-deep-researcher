from src.assistant.utils import format_research_process

def main():
    # 格式化研究过程
    formatted_output = format_research_process('research_process.json')
    
    # 将格式化后的内容输出到文件
    with open('research_analysis.md', 'w') as f:
        f.write(formatted_output)
    
    print("研究过程分析已保存到 research_analysis.md")

if __name__ == "__main__":
    main()