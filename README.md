# Create-agent-by-using-the-LangChain

<div align="center">

# 🤖 Create-agent-by-using-the-LangChain

**A hands-on progression from basic tool-calling agents and LCEL `StrOutputParser` chains to multi-node LangGraph workflows and a 10-tool ReAct `AgentExecutor` powered by Google Gemini.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3%20%2F%20v1%2B-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Node-FF6F00?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-gemini--3.5--flash--lite-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Open In Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

</div>

---

## 📌 Overview

This repository demonstrates how AI agents and LLM pipelines are built step-by-step using **LangChain**, **LangGraph**, and **Google Gemini (`ChatGoogleGenerativeAI`)**. Designed for learners, students, and developers, each module builds on the previous one—moving from simple 2-tool agents to deterministic chains and autonomous 10-tool ReAct problem solvers.

### Key Learning Modules
1. **Basic Tool-Calling Agent (`create_agent`)** – Combining web search (`DuckDuckGoSearchRun`) and custom Python math tools (`@tool`).
2. **LCEL Chains & `StrOutputParser`** – Chaining multi-step prompts (`template1 | model | parser | template2 | model | parser`) and inspecting execution graphs with `grandalf`.
3. **5-Tool Trip & Event Budget Agents** – Solving multi-step real-world tasks via both a single `create_agent` loop and a 3-stage **LangGraph `StateGraph`** (`Research -> Quant -> Writer`).
4. **10-Tool ReAct Agent (`create_react_agent` + `AgentExecutor`)** – A classroom-ready reasoning agent using `Thought -> Action -> Action Input -> Observation` across live APIs, math utilities, currency/unit converters, and file reporting.

---

## 🗺️ Architecture & Execution Flow

### 1. ReAct Agent Reasoning Loop (`AgentExecutor`)

```mermaid
flowchart TD
    A[User Query] --> B[PromptTemplate + Scratchpad]
    B --> C[ChatGoogleGenerativeAI<br/>gemini-3.5-flash-lite]
    C --> D{Decision: Need Tool?}
    D -- Yes --> E[Select Tool from 10-Tool Registry]
    E --> F[Execute @tool / API<br/>Tavily, Weatherstack, Math, Currency, File I/O]
    F --> G[Append Observation to Scratchpad]
    G --> C
    D -- No --> H[Final Answer + Saved Report]



flowchart LR
    subgraph LCEL["LCEL Chain (StrOutputParser)"]
        direction LR
        P1[Prompt 1] --> M1[Gemini] --> S1[StrOutputParser] --> P2[Prompt 2] --> M2[Gemini] --> S2[StrOutputParser]
    end

    subgraph LG["LangGraph 5-Tool Pipeline"]
        direction LR
        N1[Research Node<br/>Date + Search] --> N2[Quant Node<br/>Multiply + Currency] --> N3[Writer Node<br/>Save Brief]
    end


Create-agent-by-using-the-LangChain/
├── 01_basic_create_agent.ipynb          # 2-tool agent (DuckDuckGo + multiply)
├── 02_lcel_str_output_parser.ipynb      # Sequential report -> 2-line summary chain + graph ASCII
├── 03_five_tool_budget_agent.ipynb      # Single-agent & LangGraph StateGraph budget planners
├── 04_ten_tool_react_agent.ipynb        # 10-tool ReAct AgentExecutor capstone (Weather + Tavily + Math + File)
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation


```

pip install -U langchain langchain-community langchain-core langchain-google-genai langgraph
pip install -U "langchain==0.3.27" "langchain-community==0.3.27" "langchain-classic"
pip install -U duckduckgo-search ddgs tavily-python requests grandalf

GOOGLE_API_KEY="your_google_api_key"
TAVILY_API_KEY="your_tavily_api_key"
weather_data="your_weatherstack_api_key"


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=GOOGLE_API_KEY)
parser = StrOutputParser()

template1 = PromptTemplate(template="Write a detailed report on {topic}", input_variables=["topic"])
template2 = PromptTemplate(template="Write a 2 line summary on the following text.\n{text}", input_variables=["text"])

chain = template1 | model | parser | template2 | model | parser
print(chain.invoke({"topic": "define ollama"}))

# Visualize the chain structure
chain.get_graph().print_ascii()


from langchain_classic.agents import AgentExecutor, create_react_agent

react_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=25,
)

result = agent_executor.invoke({
    "input": (
        "Check today's date, search recent local news in Chennai using tavily_search, "
        "fetch Chennai weather, convert Celsius to Fahrenheit, get an outdoor tip, "
        "calculate a 2-day workshop budget for 10 students in INR, add a 10% buffer, "
        "convert to USD, and save the report to 'chennai_student_brief.txt'."
    )
})
print(result["output"])


```
