def format_search_results(search_results: list[dict] ):
    sources_list = []
    for result in search_results:
        sources_list.extend(result["results"])
    formatted_text = "Sources:\n\n"
    for source in sources_list:
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += (
            f"Most relevant content from source: {source['content']}\n===\n"
        )
    return formatted_text