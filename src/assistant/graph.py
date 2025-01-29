import json
from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END, StateGraph

from assistant.configuration import Configuration
from assistant.utils import deduplicate_and_format_sources, tavily_search, format_sources
from assistant.state import SummaryState, SummaryStateInput, SummaryStateOutput
from assistant.prompts import query_writer_instructions, summarizer_instructions, reflection_instructions
import pdb
def get_llm(config: Configuration, temperature: float = 0, format: str = None):
    """Get the appropriate LLM based on configuration."""
    if config.llm_provider == "ollama":
        return ChatOllama(
            model=config.llm_model,
            temperature=temperature,
            format=format
        )
    elif config.llm_provider == "openai":
        return ChatOpenAI(
            model=config.llm_model,
            temperature=temperature,
            api_key=config.openai_api_key,
            model_kwargs={"response_format": {"type": "json"}} if format == "json" else {}
        )
    elif config.llm_provider == "anthropic":
        return ChatAnthropic(
            model=config.llm_model,
            temperature=temperature,
            api_key=config.anthropic_api_key
        )
    elif config.llm_provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=config.llm_model or "gemini-pro",
            temperature=temperature,
            google_api_key=config.google_api_key,
            convert_system_message_to_human=True,  # Gemini 不支持系统消息，需要转换为人类消息
            model_kwargs={"response_format": {"type": "json"}} if format == "json" else {}
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")

# Nodes   
def generate_query(state: SummaryState, config: RunnableConfig):
    """ Generate a query for web search """
    
    # Format the prompt
    configurable = Configuration.from_runnable_config(config)
    query_writer_instructions_formatted = query_writer_instructions.format(
        research_topic=state.research_topic,
        language=configurable.output_language
    )

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = get_llm(configurable, temperature=0, format="json")
    result = llm_json_mode.invoke(
        [SystemMessage(content=query_writer_instructions_formatted),
        HumanMessage(content="""Generate a web search query and return it in the following JSON format:
{
    "query": "your search query here",
    "aspect": "aspect of the topic you are focusing on",
    "rationale": "why you chose this query"
}
Ensure your response is a valid JSON object.""")]
    )
    
    # Clean and parse the response
    content = result.content
    # Remove any potential prefixes or suffixes that aren't JSON
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    
    query = json.loads(content)
    
    return {"search_query": query['query']}

def web_research(state: SummaryState):
    """ Gather information from the web """
    
    # Search the web
    search_results = tavily_search(state.search_query, include_raw_content=True, max_results=1)
    
    # Format the sources
    search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=5000)
    return {"sources_gathered": [format_sources(search_results)], "research_loop_count": state.research_loop_count + 1, "web_research_results": [search_str]}

def summarize_sources(state: SummaryState, config: RunnableConfig):
    """ Summarize the gathered sources """
    
    # Existing summary
    existing_summary = state.running_summary

    # Most recent web research
    most_recent_web_research = state.web_research_results[-1]

    # Build the human message
    configurable = Configuration.from_runnable_config(config)
    if existing_summary:
        human_message_content = (
            f"Extend the existing summary: {existing_summary}\n\n"
            f"Include new search results: {most_recent_web_research}, "
            f"That addresses the following topic: {state.research_topic}, "
            f"Max_output_length: {configurable.summary_max_length}, "
            f"Min_output_length: {configurable.summary_min_length}, "
        )
        human_message_content += '''
1. Retain Original Information: Ensure that the initial content remains intact and is not overwritten or lost.
2. Integrate New Information: Seamlessly add new details without repeating what’s already covered.
3. Maintain Consistency: Keep the same style and depth as the original summary.
4. Avoid Redundancy: Only include new, non-redundant information.
5. Smooth Transitions: Ensure a natural flow between existing and new content.
6. Anchor Technical Concepts: Connect new technical details to concrete use cases from prior content where relevant. 
7. Build Conceptual Continuity: Explicitly link new terminology to previously mentioned capabilities. 
'''
    else:
        human_message_content = (
            f"Generate a summary of these search results: {most_recent_web_research}, "
            f"That addresses the following topic: {state.research_topic}. "
        )
    pdb.set_trace()

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)
    llm = get_llm(configurable, temperature=0)
    result = llm.invoke(
        [SystemMessage(content=summarizer_instructions.format(
            language=configurable.output_language,
            max_length=configurable.summary_max_length,
            min_length=configurable.summary_min_length
        )),
        HumanMessage(content=human_message_content)]
    )

    running_summary = result.content

    # TODO: This is a hack to remove the <think> tags w/ Deepseek models 
    # It appears very challenging to prompt them out of the responses 
    while "<think>" in running_summary and "</think>" in running_summary:
        start = running_summary.find("<think>")
        end = running_summary.find("</think>") + len("</think>")
        running_summary = running_summary[:start] + running_summary[end:]

    return {"running_summary": running_summary}

def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    """ Reflect on the summary and generate a follow-up query """
    
    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = get_llm(configurable, temperature=0, format="json")
    result = llm_json_mode.invoke(
        [SystemMessage(content=reflection_instructions.format(
            research_topic=state.research_topic,
            language=configurable.output_language
        )),
        HumanMessage(content=f"""Based on this summary: {state.running_summary}

Please analyze and return your response in this exact JSON format:
{{
    "knowledge_gap": "describe the gap you identified",
    "follow_up_query": "your follow up search query"
}}

Ensure your response is a valid JSON object and contains no other text.""")]
    )

    # Clean and parse the response
    content = result.content
    # Remove any potential prefixes or suffixes that aren't JSON
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    
    follow_up_query = json.loads(content)

    # Overwrite the search query
    return {"search_query": follow_up_query['follow_up_query']}

def finalize_summary(state: SummaryState):
    """ Finalize the summary """
    
    # Format all accumulated sources into a single bulleted list
    all_sources = "\n".join(source for source in state.sources_gathered)
    state.running_summary = f"## Summary\n\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
    return {"running_summary": state.running_summary}

def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "web_research"]:
    """ Route the research based on the follow-up query """

    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= configurable.max_web_research_loops:
        return "web_research"
    else:
        return "finalize_summary" 
    
# Add nodes and edges 
builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)

# Add edges
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "web_research")
builder.add_edge("web_research", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("finalize_summary", END)

graph = builder.compile()