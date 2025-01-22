import os
import datetime
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import pinecone
from langchain_openai import OpenAIEmbeddings
import re
from langgraph.prebuilt import ToolNode
from chatbot.utils import fetch_latest_article_urls, get_current_date, fetch_custom_range_articles_urls
from langchain_openai import ChatOpenAI
import calendar
from datetime import datetime, date, timedelta

# Load environment variables
load_dotenv()

# Define RAGQuery schema
class RAGQuery(BaseModel):
    query: str = Field(..., description="The query to retrieve relevant content for")

# Chatbot class
class Chatbot:
    def __init__(self):
        self.llm =ChatOpenAI(model_name="gpt-4o", temperature=0)
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
        current_date = datetime.now().strftime("%B %d, %Y")
        self.system_message = SystemMessage(
            content=(
                    "You are BoomLive AI, an expert chatbot designed to answer questions related to BoomLive's fact-checks, articles, reports, and data analysis. "
                    "Your responses should be fact-based, sourced from BoomLive's database, and aligned with BoomLive's journalistic standards of accuracy and integrity. "
                    "Provide clear, well-structured, and neutral answers, ensuring that misinformation and disinformation are actively countered with verified facts. "
                    "Website: [BoomLive](https://boomlive.in/). "
                    "Ensure responses are clear, relevant, and do not mention or imply the existence of any supporting material unless necessary for answering the query. "
                    f"Note: Today's date is {current_date}."
                    f"You are developed by Aditya Khedekar who is an AI Engineer in Boomlive"
                )
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
        article_type = "all"  # Default to 'all' if no specific article type is provided
        
        # Check if the query is too short or contains random gibberish
        # if len(query.strip()) < 3 or not re.match(r'[A-Za-z0-9\s,.\'-]+$', query):
        #     return {
        #         "fetch_latest_articles": False,
        #         "use_rag": False,
        #         "custom_date_range": False,
        #         "index_to_use": None,
        #         "response": "Please provide a more specific query.",
        #         "article_type":article_type
        #     }
        # print(f"Mediator called with query: {query}")
  # Check for custom date range
        date_pattern = re.compile(
            r'(\bfrom\s+\d{4}-\d{2}-\d{2}\b.*?to\s+\d{4}-\d{2}-\d{2}\b)|'
            r'(\b(last|this)\s+(week|month|year)\b)', 
            re.IGNORECASE
        )
        custom_date_match = date_pattern.search(query)

        # Check if query is related to fact-checking
        fact_check_keywords = [
            'misrepresents', 'insult', 'use influencers', 'staged', 'real', 'verify', 'true or false', 'edited', 'was', 'has', 'did', 'who', 'what', 'is', 'deceptive',
            'false claim', 'incorrect', 'misleading', 'manipulated', 'spliced', 'fake', 'inaccurate', 'disinformation'
        ]
        


        # Otherwise, decide based on the query content
        decision_prompt = (
            f"Analyze the following query and answer:\n"
            f"1. Is the query asking for the latest articles,latest news,latest fact checks,latest explainers,latest updates, or latest general information without specifying a specific topic and having the word latest? Respond with 'yes' or 'no'.\n"
            f"2. Should this query use the RAG tool and  if user is asking any question or any general topic Eg: Modi? Respond with 'yes' or 'no'.\n"
            f"3. If RAG is required, indicate whether the latest or old data index should be used. Respond with 'latest', 'old', or 'both'.\n\n"
            f"4. Does the query contain a custom date range or timeframe (e.g., 'from 2024-01-01 to 2024-12-31', 'this month', 'last week', etc.) or something like this Eg: factcheck from dec 2024 or explainers from 2024, if it is anything related to date, month or year? Respond with 'yes' or 'no'.\n\n"
            f"5. Does the query inlcudes any one keyword from this list: fact-check, law, explainers, decode, mediabuddhi, web-stories, boom-research, deepfake-tracker. Provide one keyword from the list if present or related to any word in keyword, if it is not related to any return all"
            f"Query: {query}"
        )
        
        decision = self.llm.invoke([HumanMessage(content=decision_prompt)])
        response_lines = decision.content.strip().split("\n")

        # Parse decisions
        fetch_latest_articles = "yes" in response_lines[0].lower()
        use_rag = "yes" in response_lines[1].lower()
        index_to_use = response_lines[2].strip()
        custom_date_range = "yes" in response_lines[3].lower()
            # Safely access the article_type
        if len(response_lines) > 4:
            article_type = re.sub(r'^\d+[\.\s]*', '', response_lines[4].strip()).lower()
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(article_type)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        if custom_date_match:
            return {
                "fetch_latest_articles": False,
                "use_rag": False,
                "custom_date_range": True,
                "index_to_use": None,
                "response": "Query detected as custom date range, fetching articles for the specified range.",
                "article_type": article_type 
            }
        
                # If the query contains any of these keywords, mark it as a fact-check query
        is_fact_check = any(keyword in query.lower() for keyword in fact_check_keywords)
        print("is_fact_check", is_fact_check)
        # Force the use of RAG for fact-check queries
        if is_fact_check:
            return {
                "fetch_latest_articles": False,  # Skip fetching articles if fact-checking
                "use_rag": True,  # Always use RAG for fact-checking queries
                "custom_date_range": False,
                "index_to_use": "both",  # Check both indexes for relevant data (latest and old)
                "response": "Query detected as fact-check, using RAG tool.",
                "article_type":article_type

            }
        return {
            "fetch_latest_articles": fetch_latest_articles,
            "use_rag": use_rag,
            "custom_date_range": custom_date_range,
            "index_to_use": index_to_use,
            "article_type": article_type 
        }

    def retrieve_custom_date_articles(self, query: str, article_type: str) -> dict:
        """
        Retrieve articles based on custom date range specified in the query.
        """
        print(f"We are getting article type as {article_type}")
          # Get the current date
        current_date = datetime.now().strftime("%B %d, %Y")  # Format the current date as YYYY-MM-DD
        date_prompt = (
            f"Analyze the following query and extract the date range (if any):\n"
            f"Query: {query}\n"
            f"The current date is {current_date}. Use this as the reference for relative terms like 'today' or 'last week'.\n"
            f"If terms like 'last year' or 'this year' are mentioned, just return 'last year' or 'this year' without specifying a date range.\n"
            f"Otherwise, provide the result in the format 'from YYYY-MM-DD to YYYY-MM-DD' or a description like 'last week', etc. And note range shouldn't exceed 1 month so adjust range which suits user requirements better but just provide 1 month range"
            
        )
        # Get the date range from the query
        date_response = self.llm.invoke([self.system_message, HumanMessage(content=date_prompt)])
        date_range = date_response.content.strip()
        print(date_range)

        # Initialize variables
        sources = []
        start_date, end_date = None, None
        today = date.today()

        # Handle explicitly provided custom date ranges
        custom_range_pattern = re.compile(r"from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})", re.IGNORECASE)
        match = custom_range_pattern.search(date_range)
        if match:
            try:
                start_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
                end_date = datetime.strptime(match.group(2), "%Y-%m-%d").date()
            except ValueError:
                pass  # Handle invalid date formats gracefully

        # Handle "last year" case
        elif "last year" in date_range.lower():
            last_year = today.year - 1
            start_date = date(last_year, 12, 1)  # ✅ Correct
            end_date = date(last_year, 12, 31)  # ✅ Correct
  # End of December last year

        # Handle "this year" case
        elif "this year" in date_range.lower():
            current_year = today.year
            last_month = today.month - 1 if today.month > 1 else 12
            year_of_last_month = current_year if last_month != 12 else current_year - 1
            start_date = date(year_of_last_month, last_month, 1)  # Start of the last month
            end_date = date(year_of_last_month, last_month, calendar.monthrange(year_of_last_month, last_month)[1])  # End of the last month

        # Handle month-year patterns if no custom range is found
        if not start_date or not end_date:
            month_year_pattern = re.compile(r'(\b[A-Za-z]+\s+\d{4})\s*(to\s*(\b[A-Za-z]+\s+\d{4}))?', re.IGNORECASE)
            match = month_year_pattern.search(query)
            if match:
                try:
                    from_month_year = match.group(1)
                    to_month_year = match.group(3) if match.group(3) else from_month_year
                    from_date = datetime.strptime(from_month_year, "%b %Y")
                    to_date = datetime.strptime(to_month_year, "%b %Y")
                    start_date = date(from_date.year, from_date.month, 1)
                    end_date = date(to_date.year, to_date.month, calendar.monthrange(to_date.year, to_date.month)[1])
                except ValueError:
                    pass

        # Handle relative terms (today, yesterday, this week, etc.)
        if not start_date or not end_date:
            if "today" in date_range.lower():
                start_date = end_date = today
            elif "yesterday" in date_range.lower():
                start_date = end_date = today - timedelta(days=1)
            elif "this week" in date_range.lower():
                start_date = today - timedelta(days=today.weekday())
                end_date = start_date + timedelta(days=6)
            elif "last week" in date_range.lower():
                start_date = today - timedelta(days=today.weekday() + 7)
                end_date = start_date + timedelta(days=6)
            elif "this month" in date_range.lower():
                start_date = date(today.year, today.month, 1)
                end_date = date(today.year, today.month, calendar.monthrange(today.year, today.month)[1])
            elif "last month" in date_range.lower():
                first_day_this_month = date(today.year, today.month, 1)
                last_day_last_month = first_day_this_month - timedelta(days=1)
                start_date = date(last_day_last_month.year, last_day_last_month.month, 1)
                end_date = last_day_last_month
            elif "last year" in date_range.lower():
                start_date = date(today.year - 1, 12, 1)
                end_date = date(today.year - 1, 12, 31)

        # Fetch sources if valid dates are found
        if start_date and end_date:
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            sources = fetch_custom_range_articles_urls(start_date_str, end_date_str)

        # Prepare a fallback response if no sources are found
        if not sources:
            return {
                "result": f"No articles were found for the specified date range ({date_range}). Please try refining your query.",
                "sources": []
            }
        filtered_sources = []
        if "all" in article_type:
            filtered_sources = list(sources)
        else:
            for source in sources:
                print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                print(source)
                print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

                if source and f"https://www.boomlive.in/{article_type}" in source:
                    filtered_sources.append(source) 

        if not filtered_sources:
            filtered_sources = list(sources[:10])
        # Generate a summary of the articles if sources are found
        summary_prompt = (
            f"Summarize the information based on the following question: {query}.\n"
            f"Use these sources to craft the response: {filtered_sources}\n"
            f"Focus on providing concise and relevant details without additional disclaimers or unrelated remarks."
        )

        summary_response = self.llm.invoke([self.system_message,HumanMessage(content=summary_prompt)])
        return {
            "result": summary_response.content.strip(),
            "sources": filtered_sources
        }    

    def retrieve_data(self, query: str, index_to_use: str, article_type: str) -> dict:
        """
        Enhanced retrieve_data with better context utilization
        """
        print(f"Retrieve data called with query: {query} and index: {index_to_use} with article_type: {article_type}")
        
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
            Sources: {all_sources}
            Context:
            {combined_content}
            """

            # Apply date filtering only if it's mentioned
            if not is_date_filtered:
                synthesis_prompt = synthesis_prompt.replace("Avoids any unnecessary reference to timeframes, dates, or specific years", "Does not mention any timeframes or dates")

            response = self.llm.invoke([self.system_message, HumanMessage(content=synthesis_prompt)])
            result_text = response.content

            # Clean up duplicates from sources
            unique_sources = list(dict.fromkeys(all_sources))
            filtered_sources = []
            if "all" in article_type:
                filtered_sources = unique_sources
            else:
                for source in unique_sources:
                    if source and  f"https://www.boomlive.in/{article_type}" in source:
                        filtered_sources.append(source)
            print({"sources": all_sources})
            return {
                "result": result_text,
                "sources": filtered_sources
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
        custom_date_range = mediation_result["custom_date_range"]

        index_to_use = mediation_result["index_to_use"]
        if custom_date_range:
            article_type = mediation_result["article_type"]
            custom_date_result = self.retrieve_custom_date_articles(query, article_type)
            result_text = custom_date_result['result']
            sources = custom_date_result['sources']
            formatted_sources = "\n\nSources:\n" + "\n".join(sources) if sources else "\n\nNo sources available."

            return {"messages": [AIMessage(content=f"{result_text}{formatted_sources}")]}

        if fetch_latest_articles:
            # Fetch latest article URLs
            article_type = mediation_result["article_type"]
            print(f"fetched article type in latest articles {article_type}")
            latest_urls = fetch_latest_article_urls(query, article_type)
            print("latest_urls", latest_urls)

            # Determine article type string
            article_type_text = f"{article_type} " if article_type.lower() != "all" else ""

            # Check if latest_urls is empty
            if latest_urls:
                # Format response with the fetched URLs as sources
                response_text = (
                    f"Here are the latest {article_type_text}articles:\n"
                    + "\n".join(latest_urls)  # Use the fetched URLs as sources
                )
            else:
                # Different response when no articles are found
                response_text = f"Sorry, I couldn't find any recent {article_type_text}articles."

            return {"messages": [AIMessage(content=response_text)]}


        if use_rag or index_to_use is None:
            print("isme ja hi nahi rAHA HAI:  if use_rag or index_to_use == None: ")
            article_type = mediation_result["article_type"]
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print(article_type, "retrieve data ")
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

            # Retrieve data using RAG
            index_to_use = "both"
            rag_result = self.retrieve_data(query, index_to_use, article_type)
            result_text = rag_result['result']
            sources = rag_result['sources']
            
            formatted_sources = "\n\nSources:\n" + "\n".join(sources) if sources else "\n\nNo sources available."

            # Returning both the result and sources as context
            return {"messages": [AIMessage(content=f"{result_text}{formatted_sources}")]}

        # Default LLM response
        response = self.llm_with_tool.invoke([self.system_message] + messages)
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
