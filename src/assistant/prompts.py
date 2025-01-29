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
1. Preserve Core Information: Keep all critical insights and key details from the original summary intact, ensuring no loss of essential knowledge.
2. Integrate New Insights Without Redundancy: Introduce new information only if it adds unique value—strictly avoid rephrasing, reintroducing, or restating previously covered points.
3. Ensure Stylistic and Conceptual Consistency: Maintain the same tone, depth, and structure as the original summary to ensure coherence.
4. Eliminate Redundant Content: Systematically detect and remove any duplicate, overlapping, or semantically equivalent information while preserving all distinct insights.
5. Establish Logical and Seamless Transitions: Ensure smooth integration of new content by naturally linking it to existing material without disrupting readability.
6. Contextualize Technical Additions: Whenever introducing new technical concepts, anchor them to previously discussed examples, use cases, or foundational principles for continuity.
7. Reinforce Conceptual Connectivity: Explicitly link new ideas and terminology to prior content, avoiding fragmentation and ensuring a progressive expansion of knowledge.

When creating a NEW summary:
1. Highlight the Most Relevant Information: Focus on key points from each source.
2. Provide a Concise Overview: Offer a clear summary of the main ideas related to the topic.
3. Emphasize Significant Findings: Highlight important insights or discoveries.
4. Ensure a Coherent Flow: Organize the information logically.

LENGTH REQUIREMENTS:
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

tech_analysis_instructions = """You are a technology expert analyzing information about {tech_name}.

Your tasks:
1. Provide a comprehensive but concise summary of the technology
2. Identify 5-7 most closely related technologies
3. Focus on:
   - Core concepts and principles
   - Key applications and use cases
   - Latest developments and trends
   - Technical relationships with other technologies

Return your analysis as a JSON object in {language}:
{{
    "summary": "A comprehensive summary of the technology",
    "related_technologies": [
        "technology1",
        "technology2",
        ...
    ],
    "key_aspects": {{
        "core_concepts": ["concept1", "concept2", ...],
        "applications": ["application1", "application2", ...],
        "trends": ["trend1", "trend2", ...]
    }}
}}"""