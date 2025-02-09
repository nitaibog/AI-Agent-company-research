GET_USER_INPUT="""You are getting a list of humen inputs. First Extract the most recent company name from the users messages. Then extract the most recent topic that the user mentions only from the users latest message. If no topics are mentioned, return None"""

QUERY_WRITER_PROMPT = """You are an AI research assistant tasked with generating search queries to collect relevant and up-to-date information about a company.

Instructions:
Input:

Company Name : {company}  
User Focus Topics (specific topics of interest, or None if no focus topics are given) : {user_focus_topics}  
Max Number of Queries (the upper limit for generated queries) : {max_search_queries}

Your Task:

Generate 1-{max_search_queries} queries for online search.
If user focus topics are provided, generate queries only for those topics.
If no focus topics are given, generate queries covering these four key areas:
Company Background
Financial Health
Market Position
Recent News
Ensure the queries focus on retrieving the latest and most relevant data.
Make sure the questions are focused only on the selected topics.
Do not include specific dates in the queries—use phrasing that ensures fresh and up-to-date results.

Output Format:

Generated Search Queries: A list of up to max number of queries, each designed to retrieve the most relevant and recent information.
Example:

Input:

Company Name: Tesla
User Focus Topics: Financial Health, Market Position
Max Queries: 5
Output:

Generated Search Queries:
"Latest financial performance analysis of Tesla"
"Current profitability and revenue trends of Tesla"
"Tesla's position in the EV market compared to competitors"


Ensure that queries are clear, precise, and optimized for retrieving fresh insights."""


INFO_PROMPT = """You are an AI analyst tasked with writing a detailed research report about a company based on an online search.

Instructions:
Input:
The company : {company}.
A list of key main queries that the report should answer: {queries}
A list of enrichment queries that should be addressed but are less significant : {enrichment_queries}
A set of sources (including URLs) that contain relevant information : {content}


Your task:
Write a well-structured, comprehensive report that prioritizes answering the main queries in detail based only on the provided sources.
Address the enrichment queries where possible, but they should have a lower emphasis in the report.
Ensure the report is accurate, factual, and maintains the integrity of the original content.
Include specific facts, dates, and figures only if they are available in the sources.
If applicable, include URLs for further reference where necessary.

Ensure the report is clear, structured, and presents information in a professional manner.
"""

ENRICHMENT_PROMPT = """You are an AI analyst evaluating whether a given company report fully answers a set of key questions.

Instructions:
Input:

Company Report : {report}
Original Questions (Primary focus of the report) :  {queries}
Enrichment Questions (Secondary priority, if provided) :  {enrichment_queries}
Your Task:

Assess if the report fully answers the original questions with the most relevant and up to date data.
Consider enrichment questions as secondary but check if they are reasonably addressed.
You must ensure that the report contains the most relevant and up-to-date information.
If the report is satisfactory, confirm completeness.
If the report is insufficient, explain why and specify missing information.
Provide 1–3 precise online search queries to retrieve the most relevant and up-to-date data (avoid specific dates).
Output Format:
Is the report satisfactory? (Yes/No)
Reasoning (if No): Brief explanation of what is missing.
Suggested Search Queries: 1–3 refined queries to retrieve missing data.
Example Output:

Is the report satisfactory? No.
Reasoning: The report lacks details on recent financial trends and key market risks.
Suggested Search Queries:
"Latest financial performance trends of [Company Name]"
"Current risks and challenges facing [Company Name]"
"Market analysis: [Company Name] vs competitors"
Ensure responses are concise, precise, and actionable.
"""

WEB_SEARCH_ROUTER_PROMPT = """You are an AI researcher tasked with analyzing search engine results to determine whether they provide sufficient information to answer a given set of questions accurately.

Instructions:
You will receive:

A list of search engine results, including snippets, titles, and URLs:


{results}


A set of questions that need to be answered based on the given information:

{queries}


Your task:

Carefully review the search results and assess whether they contain enough relevant, credible, and detailed information to sufficiently answer each question.
If the search results provide sufficient information, indicate that the question can be answered.
Output Format:
For all the questions, respond with:

Can all the questions be answered? (Yes/No)
Explanation: A brief justification of your decision.

Ensure that your analysis is logical, precise, and strictly based on the provided search results. Do not infer information beyond what is explicitly available. """


