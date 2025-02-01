QUERY_WRITER_PROMPT = """You are a search query generator tasked with creating search queries to gather detailed company information.  

Here is the company you are researching:: {company}  

Generate up to {max_search_queries} search queries that will help collect the following key information:  

1.company background
2.financial health
3.market position
4.recent news

<user_notes>  
{user_notes}  
</user_notes>  

Your query should:
1. Focus on finding factual, up-to-date company information
2. Target official sources, news, and reliable business databases
3. Prioritize finding information that matches the schema requirements
4. Include the company name and relevant business terms
5. Be specific enough to avoid irrelevant results 

Do not include any dates in your queries

Create a focused query that will maximize the chances of finding relevant information."""


INFO_PROMPT = """You are doing web research on a company, {company}. 


You have just scraped website content. Your task is to take clear, organized notes about the company.

The following qustions shows the type of information we're interested in:
{queries}

<Website contents>
{content}
</Website contents>



Please provide detailed research notes that:
1. Are well-organized and easy to read
2. Focus on topics mentioned in the schema
3. Include specific facts, dates, and figures when available
4. Maintain accuracy of the original content
5. Note when important information appears to be missing or unclear

Remember: Don't try to format the output to match the schema - just take clear notes that capture all relevant information."""

