import os
import datetime
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import pinecone
from langchain_openai import OpenAIEmbeddings
import re
from langgraph.prebuilt import ToolNode
from chatbot.utils import fetch_latest_article_urls, get_current_date
# Load environment variables
load_dotenv()

# Define RAGQuery schema
class RAGQuery(BaseModel):
    query: str = Field(..., description="The query to retrieve relevant content for")

# Chatbot class
class Chatbot:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        self.memory = MemorySaver()

        # Initialize Pinecone indices
        self.latest_index = PineconeVectorStore(
            index_name=os.getenv("PINECONE_LATEST_INDEX_NAME"),
            embedding=OpenAIEmbeddings(model="text-embedding-3-small")
        )
        self.old_index = PineconeVectorStore(
            index_name=os.getenv("PINECONE_OLD_INDEX_NAME"),
            embedding=OpenAIEmbeddings(model="text-embedding-3-small")
        )

        # External API for latest articles
        # self.latest_articles_api = fetch_latest_article_urls()

    def extract_keywords(self, query: str) -> str:
        """
        Simple keyword extraction without dependency on spacy
        """
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = query.lower().split()
        keywords = [word for word in words if word not in common_words]
        return " ".join(keywords)

    def mediator(self, query: str) -> dict:
        """
        Enhanced mediator that handles fact-check queries and invalid/random queries.
        """
        
        # Check if the query is too short or contains random gibberish
        if len(query.strip()) < 3 or not re.match(r'[A-Za-z0-9\s,.\'-]+$', query):
            return {
                "fetch_latest_articles": False,
                "use_rag": False,
                "index_to_use": None,
                "response": "Please provide a more specific query."
            }
        print(f"Mediator called with query: {query}")

        # Check if query is related to fact-checking
        fact_check_keywords = [
            'misrepresents', 'insult', 'use influencers', 'staged', 'real', 'fact-check', 'verify',
            'factcheck', 'true or false', 'edited', 'was', 'has', 'did', 'who', 'what', 'is', 'deceptive',
            'false claim', 'incorrect', 'misleading', 'manipulated', 'spliced', 'fake', 'inaccurate', 'disinformation'
        ]
        
        # If the query contains any of these keywords, mark it as a fact-check query
        is_fact_check = any(keyword in query.lower() for keyword in fact_check_keywords)
        print("is_fact_check", is_fact_check)
        # Force the use of RAG for fact-check queries
        if is_fact_check:
            return {
                "fetch_latest_articles": False,  # Skip fetching articles if fact-checking
                "use_rag": True,  # Always use RAG for fact-checking queries
                "index_to_use": "both",  # Check both indexes for relevant data (latest and old)
                "response": "Query detected as fact-check, using RAG tool."
            }

        # Otherwise, decide based on the query content
        decision_prompt = (
            f"Analyze the following query and answer:\n"
            f"1. Is the query asking for the latest articles, news, fact checks, explainers, updates, or general information without specifying a specific topic? Respond with 'yes' or 'no'. For example, queries like 'provide the latest news', 'give me recent fact checks', 'latest updates', 'what are the new articles?', 'show me recent news', or 'share the latest explainers' should receive a 'yes'."
            f"2. Should this query use the RAG tool? Respond with 'yes' or 'no'.\n"
            f"3. If RAG is required, indicate whether the latest or old data index should be used. Respond with 'latest', 'old', or 'both'.\n\n"
            f"Query: {query}"
        )
        
        decision = self.llm.invoke([HumanMessage(content=decision_prompt)])
        response_lines = decision.content.strip().split("\n")

        # Parse decisions
        fetch_latest_articles = "yes" in response_lines[0].lower()
        use_rag = "yes" in response_lines[1].lower() or is_fact_check  # Always use RAG for fact-check queries
        index_to_use = "both" if is_fact_check else response_lines[2].strip()

        print(f"Fetch latest articles: {fetch_latest_articles}")
        print(f"Use RAG: {use_rag}, Index to use: {index_to_use}")

        return {
            "fetch_latest_articles": fetch_latest_articles,
            "use_rag": use_rag,
            "index_to_use": index_to_use
        }


    def retrieve_data(self, query: str, index_to_use: str) -> dict:
        """
        Enhanced retrieve_data with better context utilization
        """
        print(f"Retrieve data called with query: {query} and index: {index_to_use}")
        
        current_date = get_current_date()
        print(current_date)
        refined_query = self.extract_keywords(query)
        all_docs = []
        all_sources = []
        print("refined_query", refined_query)
        # Determine if the query mentions dates or the latest content
        is_date_filtered = "latest" in query.lower() or "date" in query.lower()  # Check if the query mentions date or "latest"
        print("index_to_use",index_to_use)
        if index_to_use is not None:
            index_to_use = index_to_use.split(".")[-1].strip()  # This removes any extra text like "3." and keeps only "latest"

        if index_to_use in ["latest"] or index_to_use is None:
            print(f"inseide:  if index_to_use in  latest")
            latest_retriever = self.latest_index.as_retriever(search_kwargs={"k": 5})
            latest_docs = latest_retriever.get_relevant_documents(refined_query)
            print(f"Latest documents retrieved: {len(latest_docs)}")  # Debugging line
            all_docs.extend(latest_docs)
            latest_sources = [doc.metadata.get("source", "Unknown") for doc in latest_docs]
            all_sources.extend(latest_sources)

        if index_to_use in ["both", "latest"]:
            print(f"inseide:  if index_to_use in :both", "latest")
            latest_retriever = self.latest_index.as_retriever(search_kwargs={"k": 5})
            latest_docs = latest_retriever.get_relevant_documents(refined_query)
            print(f"Latest documents retrieved: {len(latest_docs)}")  # Debugging line
            all_docs.extend(latest_docs)
            latest_sources = [doc.metadata.get("source", "Unknown") for doc in latest_docs]
            all_sources.extend(latest_sources)

        if index_to_use in ["both", "old"]:
            print(f"inseide:  if index_to_use in :both, old")

            old_retriever = self.old_index.as_retriever(search_kwargs={"k": 5})
            old_docs = old_retriever.get_relevant_documents(refined_query)
            print(f"Old documents retrieved: {len(old_docs)}")  # Debugging line
            all_docs.extend(old_docs)
            old_sources = [doc.metadata.get("source", "Unknown") for doc in old_docs]
            all_sources.extend(old_sources)

        if all_docs:
            combined_content = "\n\n".join([doc.page_content for doc in all_docs])

            # If the query does not mention dates or "latest", do not filter dates
            synthesis_prompt = f"""
            Based on the following content, provide a breif and short response as a Boom Chatbot: {query}
            The current date is {current_date}.

            Context:
            {combined_content}
            """

            # Apply date filtering only if it's mentioned
            if not is_date_filtered:
                synthesis_prompt = synthesis_prompt.replace("Avoids any unnecessary reference to timeframes, dates, or specific years", "Does not mention any timeframes or dates")

            response = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
            result_text = response.content

            # Clean up duplicates from sources
            unique_sources = list(dict.fromkeys(all_sources))

            return {
                "result": result_text,
                "sources": unique_sources
            }
        else:
            print(f"No relevant documents found for query: {query}")  # Debugging line
            return {
                "result": f"No relevant fact-check articles found for the query: {query}",
                "sources": []
            }


    def call_tool(self):
        rag_tool = StructuredTool.from_function(
            func=self.retrieve_data,
            name="RAG",
            description="Retrieve relevant content from the knowledge base",
            args_schema=RAGQuery
        )
        self.tool_node = ToolNode(tools=[rag_tool])
        self.llm_with_tool = self.llm.bind_tools([rag_tool])

    def call_model(self, state: MessagesState):
        messages = state['messages']
        last_message = messages[-1]
        query = last_message.content

        # Mediator makes decisions
        mediation_result = self.mediator(query)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("mediation_result", mediation_result)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        fetch_latest_articles = mediation_result["fetch_latest_articles"]
        use_rag = mediation_result["use_rag"]
        index_to_use = mediation_result["index_to_use"]

        if fetch_latest_articles:
            # Fetch latest article URLs
            latest_urls = fetch_latest_article_urls(query)
            print("latest_urls", fetch_latest_article_urls(query))
            # Format response with the fetched URLs as sources
            response_text = (
                "Here are the latest articles:\n"
                + "\n".join(latest_urls)  # Use the fetched URLs as sources
            )
            return {"messages": [AIMessage(content=response_text)]}

        if use_rag or index_to_use is None:
            print("isme ja hi nahi rAHA HAI:  if use_rag or index_to_use == None: ")
            # Retrieve data using RAG
            index_to_use = "both"
            rag_result = self.retrieve_data(query, index_to_use)
            result_text = rag_result['result']
            sources = rag_result['sources']
            
            formatted_sources = "\n\nSources:\n" + "\n".join(sources) if sources else "\n\nNo sources available."

            # Returning both the result and sources as context
            return {"messages": [AIMessage(content=f"{result_text}{formatted_sources}")]}

        # Default LLM response
        response = self.llm_with_tool.invoke(messages)
        return {"messages": [AIMessage(content=response.content)]}

    def router_function(self, state: MessagesState) -> Literal["tools", END]:
        messages = state['messages']
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def __call__(self):
        self.call_tool()
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self.router_function, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")
        self.app = workflow.compile(checkpointer=self.memory)
        return self.app
