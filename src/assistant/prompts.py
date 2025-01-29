query_writer_instructions="""Your goal is to generate targeted web search query in {language}.

The query will gather information related to a specific topic.

Topic:
{research_topic}

Return your query as a JSON object:
{{
    "query": "string (in {language})",
    "aspect": "string (in {language})",
    "rationale": "string (in {language})"
}}
"""

summarizer_instructions="""Your goal is to generate a high-quality summary of the web search results in {language}.

When EXTENDING an existing summary:
1. Retain Original Information: Ensure that the initial content remains intact and is not overwritten or lost.
2. Integrate New Information: Seamlessly add new details without repeating what’s already covered.
3. Maintain Consistency: Keep the same style and depth as the original summary.
4. Avoid Redundancy: Only include new, non-redundant information.
5. Smooth Transitions: Ensure a natural flow between existing and new content.
6. Anchor Technical Concepts: Connect new technical details to concrete use cases from prior content where relevant. 
7. Build Conceptual Continuity: Explicitly link new terminology to previously mentioned capabilities. 

When creating a NEW summary:
1. Highlight the Most Relevant Information: Focus on key points from each source.
2. Provide a Concise Overview: Offer a clear summary of the main ideas related to the topic.
3. Emphasize Significant Findings: Highlight important insights or discoveries.
4. Ensure a Coherent Flow: Organize the information logically.

LENGTH REQUIREMENTS:
- Maximum length: {max_length} characters
- Minimum length: {min_length} characters
- Stay within these length constraints while maintaining quality
- Length applies to the final summary text only

CRITICAL REQUIREMENTS:
- Write your summary in {language}
- Start IMMEDIATELY with the summary content - no introductions or meta-commentary
- DO NOT include ANY of the following:
  * Phrases about your thought process ("Let me start by...", "I should...", "I'll...")
  * Explanations of what you're going to do
  * Statements about understanding or analyzing the sources
  * Mentions of summary extension or integration
- Focus ONLY on factual, objective information
- Maintain a consistent technical depth
- Avoid redundancy and repetition
- DO NOT use phrases like "based on the new results" or "according to additional sources"
- DO NOT add a References or Works Cited section
- DO NOT use any XML-style tags like <think> or <answer>
- Begin directly with the summary text without any tags, prefixes, or meta-commentary
"""

reflection_instructions = """You are an expert research assistant analyzing a summary about {research_topic}.

Your tasks:
1. Identify knowledge gaps or areas that need deeper exploration
2. Generate a follow-up question that would help expand your understanding
3. Focus on technical details, implementation specifics, or emerging trends that weren't fully covered

Ensure the follow-up question is self-contained and includes necessary context for web search.

Return your analysis as a JSON object in {language}:
{{ 
    "knowledge_gap": "string (in {language})",
    "follow_up_query": "string (in {language})"
}}"""