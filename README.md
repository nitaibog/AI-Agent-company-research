# AI Company Research Agent

## 📌 Project Overview
The **AI Company Research Agent** is an intelligent system designed to generate detailed, up-to-date reports about companies. It conducts online research using the Tavily API and synthesizes findings into a structured report. The agent is built with **LangGraph** for multi-step decision-making and utilizes **MongoDB** for memory and checkpointing.

## 🚀 Features
- **AI-Powered Company Reports**: Generates detailed insights about a company.
- **Real-Time Web Search**: Uses the **Tavily API** to fetch relevant and up-to-date information.
- **LangGraph-Based Workflow**: Implements a structured AI agent with multiple steps for information gathering, enrichment, and decision-making.
- **Memory & Checkpointing**: Stores agent state and interactions in **MongoDB**.
- **Interactive UI**: A chatbot-like interface built with **Streamlit** for user-friendly interactions.

## 🛠️ Technologies Used
- **LangGraph**: Multi-agent workflow and orchestration.
- **OpenAI GPT-4o**: Natural language processing and response generation.
- **Tavily API**: Web search for real-time company data.
- **MongoDB**: Database for agent memory and state management.
- **Streamlit**: Frontend UI for user interaction.

## 🔧 Installation & Setup
### Prerequisites
Ensure you have Python 3.8+ installed. Install dependencies using:
```bash
pip install -r requirements.txt
```

### Environment Variables
Set the following environment variables for API access:
```bash
export OPENAI_API_KEY="your-openai-key"
export LANGCHAIN_API_KEY="your-langchain-key"
export TAVILY_API_KEY="your-tavily-key"
export MONGODB_STRING="your-mongodb-connection-string"
```

### Running the Agent

To launch the Agent, run:
```bash
streamlit run app.py
```

## 📌 Usage
1. **Enter a company name and focus area** (e.g., "Tell me about Tesla's financial health").
2. The agent will generate search queries based on your input.
3. It decides whether to use existing data or perform a live web search.
4. It retrieves and processes search results to create an informative report.


