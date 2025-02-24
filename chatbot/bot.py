import os,requests
from bs4 import BeautifulSoup
from chatbot.utils import fetch_page_text, verify_sources, get_most_suitable_source, extract_articles
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
from deep_translator import GoogleTranslator
from langdetect import detect
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
                    f"Please do not forget to add emojis to make response user friendly"
                )
        )
        # External API for latest articles
        # self.latest_articles_api = fetch_latest_article_urls()


    def enhance_query(self, original_query: str) -> dict:
        """
        Enhanced query processing optimized for mediator tool selection.
        Analyzes query patterns to maximize correct tool selection.
        """

            # Step 1: Detect if the query is in English
        detected_lang = detect(original_query)
        
        if detected_lang != "en":
            # Translate the query to English
            translator = GoogleTranslator(source="auto", target="en")
            original_query = translator.translate(original_query)
        else:
            original_query = original_query

        enhanced_query = re.sub(r'\bboom\s+report\b', 'BOOM Research Report', original_query, flags=re.IGNORECASE)

        # First, detect key patterns that strongly indicate specific tools
        patterns = {
            'boom_report': r'\b(?:boom|monthly|weekly)\s+report\b',
            'date_patterns': [
                r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
                r'\b(jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b',
                r'\b\d{4}\b',
                r'\b(last|this|next)\s+(month|year|week)\b',
                r'from\s+\w+\s+to\s+\w+',
                r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
            ],
            'fact_check': [
                r'\b(?:verify|fake|real|true|false|fact|check|confirm|debunk|misleading)\b',
                r'\b(?:viral|rumor|hoax|misinformation|disinformation)\b',
                r'\bis\s+(?:this|it|that)\s+(?:true|real|fake|false)\b',
            ],
            'latest_news': [
                r'\b(?:latest|recent|new|current|today|now)\b',
                r'\bupdate[sd]?\b',
                r'\bbreaking\s+news\b'
            ]
        }
        
        enhancement_prompt = f"""
        Analyze this query with BoomLive's context: "{original_query}"
        
        Key Indicators to Consider:
        1. Report Requests:
        - Monthly/Weekly BOOM reports
        - Analysis summaries
        - Trend reports
        
        2. Time References:
        - Explicit dates (months, years)
        - Relative time ("last month", "this year")
        - Date ranges
        - Historical references
        
        3. Content Type Signals:
        - Fact-checking requests
        - News updates
        - Analysis requests
        - Research queries
        
        4. Tool Selection Criteria:
        - Custom Date Retriever needs: Reports, time-specific queries
        - RAG needs: Fact-checks, analysis, research
        - Latest Article needs: Current events, updates
        
        5. Mentioning proper article type in enhanced_query:
        - If query has **BOOM Report** then replace it **BOOM Research Report**
        - If recent, latest, current keywords are present in query are asked it should put primary_tool as **LATEST_ARTICLES**

        Note 5th point is important
        Provide structured analysis:
        1. PRIMARY_TOOL: [LATEST_ARTICLES, CUSTOM_DATE_RETRIEVER, RAG]
        2. CONFIDENCE: [HIGH, MEDIUM, LOW]
        3. QUERY_TYPE: [REPORT, FACT_CHECK, NEWS, ANALYSIS]
        4. TIME_CONTEXT: [Specify any temporal aspects]
        5. ARTICLE_TYPE: [fact-check, law, explainers, decode, mediabuddhi, web-stories, boom-research, deepfake-tracker, all]
        6. ENHANCED_QUERY: [Modified query optimized for selected tool]
        7. REASONING: [Brief explanation of tool selection]
        """
        
        # Get LLM's analysis
        response = self.llm.invoke([
            self.system_message,
            HumanMessage(content=enhancement_prompt)
        ])
        
        # Parse LLM response into structured data
        enhancement_data = {}
        for line in response.content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                enhancement_data[key.strip()] = value.strip()
        
        # Pattern matching to validate/adjust LLM's tool selection
        has_boom_report = bool(re.search(patterns['boom_report'], original_query, re.IGNORECASE))
        has_date = any(re.search(pattern, original_query, re.IGNORECASE) for pattern in patterns['date_patterns'])
        has_fact_check = any(re.search(pattern, original_query, re.IGNORECASE) for pattern in patterns['fact_check'])
        has_latest = any(re.search(pattern, original_query, re.IGNORECASE) for pattern in patterns['latest_news'])
        
        # Override LLM tool selection based on strong pattern matches
        primary_tool = enhancement_data.get('PRIMARY_TOOL', 'RAG')
        if has_boom_report or (has_date and 'report' in original_query.lower()) and not has_latest:
            primary_tool = 'CUSTOM_DATE_RETRIEVER'
        elif has_fact_check:
            primary_tool = 'RAG'
        elif has_latest and not has_date:
            primary_tool = 'LATEST_ARTICLES'
        
        # Prepare enhanced query based on selected tool
        base_query = enhancement_data.get('ENHANCED_QUERY', enhanced_query)
        if primary_tool == 'CUSTOM_DATE_RETRIEVER':
            enhanced_query = f"time-specific {base_query}"
        elif primary_tool == 'RAG':
            enhanced_query = f"fact-verification {base_query}" if has_fact_check else base_query
        else:
            enhanced_query = f"latest-update {base_query}" if has_latest else base_query
        
        return {
            "original_query": original_query,
            "enhanced_query": enhanced_query,
            "query_metadata": {
                "primary_tool": primary_tool,
                "confidence": enhancement_data.get('CONFIDENCE', 'MEDIUM'),
                "query_type": enhancement_data.get('QUERY_TYPE', 'ANALYSIS'),
                "time_context": enhancement_data.get('TIME_CONTEXT', 'not specified'),
                "article_type": enhancement_data.get('ARTICLE_TYPE', 'all'),
                "has_date": has_date,
                "has_fact_check": has_fact_check,
                "has_latest": has_latest,
                "has_report": has_boom_report
            }
        }





    def call_model(self, state: MessagesState):
        messages = state['messages']
        last_message = messages[-1]
        original_query = last_message.content

        # Mediator makes decisions
        mediation_result = self.mediator(original_query)

        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("mediation_result", mediation_result)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        enhanced_query = mediation_result["enhanced_query"]
        print("Enhanced Query Data:", enhanced_query)  # Debug print
        fetch_latest_articles = mediation_result["fetch_latest_articles"]
        use_rag = mediation_result["use_rag"]
        custom_date_range = mediation_result["custom_date_range"]

        index_to_use = mediation_result["index_to_use"]
        use_tag = mediation_result["use_tag"]
        
        if use_tag:
            tag_url = mediation_result["tag_url"]
            print("TAG_URL", tag_url)

            # Extract real articles from BoomLive.in
            articles = extract_articles(tag_url)  # Returns a list of (title, url, summary)

            # Format articles correctly (if available)
            if articles:
                related_articles_section = "\n".join(
                    f"- [{title}]({url}) - {summary}" for title, url, summary in articles
                )
            else:
                related_articles_section = f"For more fact-checks and articles on this topic,[Click Here]({tag_url})."

            # Create a refined prompt without an explicit "Answer to the Query" section
            tag_prompt = f"""
            You are an AI assistant analyzing news articles from BoomLive.in. Your task is to generate an informative response based on verified articles from the given tag.

            **User Query:** {enhanced_query}

            **Instructions:**
            - Provide relevant insights from the articles.
            - Summarize key points concisely.
            - Do **not** include a separate "Answer to the Query" heading.
            - If articles exist, list them as markdown links.
            - If no articles are found, guide the user to the tag page.
            - Dont Provide any unneccesary articles or context just tell user we didn't found the information
            **Related Articles:**  
            {related_articles_section}
            """

            # Get response from LLM
            tag_response = self.llm.invoke([self.system_message, HumanMessage(content=tag_prompt)])

            return {"messages": [AIMessage(content=tag_response.content)]}
        if custom_date_range:
            print("YES IT IS USING CUSTOM DATE RANGE FEATURE")
            article_type = mediation_result["article_type"]
            custom_date_result = self.retrieve_custom_date_articles(enhanced_query, article_type)
            result_text = custom_date_result['result']
            sources = custom_date_result['sources']
            formatted_sources = "\n\nSources:\n" + "\n".join(sources) if sources else "\n\n"



            
            # verification_prompt = f"""
            # - If response mentions that there is no verified information for source or provided sources do not contain any information then return with only **"Not Found"**
            # Response:
            # "{result_text}"
            # """
            verification_prompt = f"""
            STRICT VERIFICATION PROTOCOL:

            Review this response text carefully:
            "{result_text}"

            Return EXACTLY "NOT_FOUND" (without quotes) if ANY of these conditions are met:
            1. The response explicitly states it cannot find information
            2. The response mentions sources do not contain relevant information
            3. The response includes phrases like "I cannot provide", "cannot verify", or "no information"
            4. The response apologizes for lack of information

            Do not include any explanation, reasoning, or additional text.
            Your output must be EXACTLY "NOT_FOUND" if verification fails, nothing else.
            """
            # verification_result = self.llm.invoke([HumanMessage(content=verification_prompt)])
            # verification_text = verification_result.content.strip().lower()
            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            # print(enhanced_query,verification_text.lower())
            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            no_info_indicators = [
                "provided sources do not contain",
                "sources do not contain",
                "cannot provide a summary",
                "I cannot provide",
                "cannot verify",
                "no information found",
                "no verified information",
                "unable to find",
                "no sources found",
                "I'm sorry, but",
                "does not mention",
                "not mentioned in",
                "not present in",
                "not covered in",
                "not available in",
                "not included in",
                "there is no specific information",
                "isn't specific",
                "not supported by available data",
                "no available data",
                "No articles were found",
                "Relevant sources for this specific query were not found",
                "not found",
                "no specific source",
                "no specific",
                "no direct sources",
                "no sources"
            ]
            response_lower = result_text.lower()
                # Check if any indicators are present
            for indicator in no_info_indicators:
                if indicator.lower() in response_lower:
                    return {"messages": [AIMessage(content="Not Found")]}
            # if "not found" in verification_text.lower() or not sources: #
            #     return {"messages": [AIMessage(content="Not Found")]}
            return {"messages": [AIMessage(content=f"{result_text}{formatted_sources}")]}

        if fetch_latest_articles:
            # Fetch latest article URLs
            article_type = mediation_result["article_type"]
            print(f"fetched article type in latest articles {article_type}")
            latest_urls = fetch_latest_article_urls(enhanced_query, article_type)
            print("latest_urls", latest_urls)

            # Determine article type string
            article_type_text = f"{article_type} " if article_type.lower() != "all" else ""

            # Check if latest_urls is empty
            if latest_urls:
                # Format response with the fetched URLs as sources
                response_text = (
                    f"Here are the latest {article_type_text} articles:\n"
                    +  "\n\nSources:\n" + "\n".join(latest_urls)  # Use the fetched URLs as sources
                )

                ## I have to add the code here for llm call to process latest urls properly
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
            rag_result = self.retrieve_data(enhanced_query, index_to_use, article_type)
            result_text = rag_result['result']
            sources = rag_result['sources']

            if not sources:
                return {"messages": [AIMessage(content="Not Found")]}
            
            source_texts = []
            first_three_urls = []
            first_three_urls = sources[:3]
            for url in sources:
                content = fetch_page_text(url)  # You should have a function to fetch content
                # Limit the snippet to, say, 500 characters (adjust as needed)
                snippet = content[:1000] if content else "No content available."
                source_texts.append(f"URL: {url}\nContent snippet: {snippet}\n")

            #  # Step 2: Verify that the retrieved sources are actually relevant to the claim.
            # verification_prompt = f"""
            # Please determine if any of the sources below directly confirm or refute the user’s claim even if it includes some parts or keywords patches query.

            # - If NONE of them do, reply with ONLY: **"Not Found"**.
            # - If ANY source provides relevant information, summarize the relevant evidence concisely.

            # User Query: "{query}"
            # - If there is some typo error in query , reply with ONLY: **Invalid**.
            
            # Retrieved Sources:
            # {'\n'.join(source_texts)}
            # """

            # verification_prompt = f"""
            # Please determine if any of the sources below directly confirm or refute the user’s claim, even if it includes partial information or relevant keywords related to the query.

            # - If NONE of them do, reply with ONLY: **"Not Found"**.
            # - If ANY source provides relevant information, summarize the relevant evidence concisely.

            # User Query: "{enhanced_query}"
            # - If there is some typo error in the query, reply with ONLY: **Invalid**.

            # Retrieved Sources:
            # {'\n\n'.join([f"Source: {url}\nContent: {snippet}" for url, snippet in zip(sources, source_texts)])}
            # """

            # print("##########################verification_prompt################################3")
            # print(verification_prompt)
            # print("##########################verification_prompt################################3")

            # verification_result = self.llm.invoke([HumanMessage(content=verification_prompt)])
            # verification_text = verification_result.content.strip()
            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            # print(verification_text.lower())
            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

            # # result = verify_sources(enhanced_query, sources, source_texts)
            # source,source_text = get_most_suitable_source(enhanced_query, sources, source_texts)
            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            # print(source,source_text)
            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")     
            # verification_prompt = f"""
            # Please determine if the following source directly confirms or refutes the user’s claim:

            # - If it does not, reply with ONLY: **"Not Found"**.
            # - If it provides relevant information, summarize the relevant evidence concisely.
            # - If the query has typos or is unclear, reply with ONLY: **"Invalid"**.

            # User Query: "{enhanced_query}"

            # Retrieved Sources:
            # {'\n\n'.join([f"Source: {url}\nContent: {snippet}" for url, snippet in zip(sources[:3], source_texts[:3])])}
            # """
            # print("##########################verification_prompt################################3")
            # print(verification_prompt)
            # print("##########################verification_prompt################################3")
            # verification_result = self.llm.invoke([HumanMessage(content=verification_prompt)])
            # verification_text = verification_result.content.strip()

                no_info_indicators = [
                "provided sources do not contain",
                "sources do not contain",
                "cannot provide a summary",
                "I cannot provide",
                "cannot verify",
                "no information found",
                "no verified information",
                "unable to find",
                "no sources found",
                "I'm sorry, but",
                "does not mention",
                "not mentioned in",
                "not present in",
                "not covered in",
                "not available in",
                "not included in",
                "there is no specific information",
                "isn't specific",
                "not supported by available data",
                "no available data",
                "No articles were found",
                "Relevant sources for this specific query were not found",
                "not found",
                "no specific source",
                "no specific",
                "no direct sources"
            ]
            response_lower = result_text.lower()
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print(response_lower)
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

                # Check if any indicators are present
            for indicator in no_info_indicators:
                if indicator.lower() in response_lower or not sources:
                    return {"messages": [AIMessage(content="Not Found")]}
            # verification_prompt = f"""
            # Analyze the following text and determine whether it explicitly states that there is no verified information available.

            # - If the text explicitly states that there is **no verified information** on the topic or indirectly say that there is no relevat information on topic or there is no verified report from BoomLive, reply with ONLY: **"Not Found"**.
            # - If the text confirms or provides **any verified information**, reply with ONLY: **"Verified"**.
            # - Ignore any extra information such as disclaimers, general advice, or calls for verification.

            # Text to analyze:
            # "{result_text}"
            # """

            # verification_result = self.llm.invoke([HumanMessage(content=verification_prompt)])
            # verification_text = verification_result.content.strip().lower()
            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            # print(enhanced_query,verification_text.lower())
            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")


            # if "not found" in verification_text.lower() or not sources: #
            #     return {"messages": [AIMessage(content="Not Found")]}
            # Returning both the result and sources as context
            formatted_sources = "\n\nSources:\n" + "\n".join(sources) if sources else "\n\n"
            return {"messages": [AIMessage(content=f"{result_text}{formatted_sources}")]}


        # Default LLM response
        response = self.llm.invoke([self.system_message] + messages)
        resultText = response.content
        
        verification_prompt = f"""
        Analyze the following text and determine whether it explicitly states that there is no verified information available.
        - If the text explicitly states that there is **no verified information** on the topic or indirectly say that there is no relevat information on topic or there is no verified report from BoomLive, reply with ONLY: **"Not Found"**.
        - If the text confirms or provides **any verified information**, reply with ONLY: **"Verified"**.
        - Ignore any extra information such as disclaimers, general advice, greetings meessages .
        - if response is some greetings then mark reply with ONLY: **"Verified"**
        Text to analyze:
        "{resultText}"
        """
        verification_result = self.llm.invoke([HumanMessage(content=verification_prompt)])
        verification_text = verification_result.content.strip().lower()
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print(enhanced_query,verification_text.lower())
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        if "not found" in verification_text.lower(): #
                return {"messages": [AIMessage(content="Not Found")]}
        return {"messages": [AIMessage(content=response.content)]}



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
        enhanced_data = self.enhance_query(query)
        print(enhanced_data)
        enhanced_query = enhanced_data["enhanced_query"]
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

# Check for tag-based queries with trending tags context
        tag_analysis_prompt = f"""
        You are an AI assistant for BoomLive. BoomLive maintains trending tags at https://www.boomlive.in/trending-tags.
        Each tag can be accessed via the URL pattern: https://www.boomlive.in/tags/{{tagName}}.

        **Your Task:**
        Determine if the user query explicitly asks for fact-checks, explainers, or articles related to a specific tag.

        **Query:** "{enhanced_query}"

        **Conditions for it to be a tag-based query (IS_TAG_QUERY = yes):**
        - Examples of valid queries:
        - "Provide fact-checks on Narendra Modi."
        - "Fact Checks on Rahul Gandhi"
        - "Provide Fact Checks on Nirmala Sitaraman"

        **Output Format:**
        IS_TAG_QUERY: <yes/no>
        TAG: <extracted tag> (Only if IS_TAG_QUERY = yes)
        CONTENT_TYPE: <fact-check/all>
        USE_TAG_URL: <yes/no> (Only if IS_TAG_QUERY = yes)
        """

        tag_analysis = self.llm.invoke([HumanMessage(content=tag_analysis_prompt)])

        # Parse the LLM response
        tag_results = {}
        for line in tag_analysis.content.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                tag_results[key.strip()] = value.strip().lower()

        # Process only if it's a tag query
        if (tag_results.get('IS_TAG_QUERY') == 'yes' and 
            tag_results.get('TAG') and 
            tag_results.get('USE_TAG_URL') == 'yes'):
            
            tag = tag_results['TAG'].replace(' ', '%20')  # URL encode spaces
            
            return {
                "fetch_latest_articles": False,  # Fetch from the tag URL directly
                "use_rag": False,  # Don't use RAG since we have direct URL
                "use_tag": True,  # Use the tag URL for fetching articles
                "custom_date_range": False,
                "index_to_use": "both",  # Not using indices
                "article_type": tag_results.get('CONTENT_TYPE', 'all'),
                "enhanced_query": enhanced_query,
                "tag_url": f"https://www.boomlive.in/search?search={tag}"  # Direct tag URL reference
            }

        
        date_pattern = re.compile(
            r'(\bfrom\s+\d{4}-\d{2}-\d{2}\b.*?to\s+\d{4}-\d{2}-\d{2}\b)|'
            r'(\b(last|this)\s+(week|month|year)\b)', 
            re.IGNORECASE
        )
        custom_date_match = date_pattern.search(enhanced_query)

        # Check if query is related to fact-checking
        fact_check_keywords = [
            'misrepresents', 'insult', 'use influencers', 'staged', 'real', 'verify', 'true or false', 'edited', 'was', 'has', 'did', 'who', 'what', 'is', 'deceptive',
            'false claim', 'incorrect', 'misleading', 'manipulated', 'spliced', 'fake', 'inaccurate', 'disinformation'
        ]
        
        # Otherwise, decide based on the query content
        decision_prompt = (
            f"Analyze the following query and answer:\n"
            f"1. Is the query asking for the latest articles,latest news,latest fact checks,latest explainers,latest updates, or latest general information without specifying a specific topic and having the word latest? Respond with 'yes' or 'no'.Note if the query is about any specific recent topic then respond with 'no'\n"
            f"2. Should this query use the RAG tool and  if user is asking any question or any general topic Eg: Modi? Respond with 'yes' or 'no'.\n"
            f"3. If RAG is required, indicate whether the latest or old data index should be used. Respond with 'latest', 'old', or 'both'.\n\n"
            f"4. Does the query contain a custom date range or timeframe (e.g., 'from 2024-01-01 to 2024-12-31', 'this month', 'last week', etc.) or something like this Eg: factcheck from dec 2024 or explainers from 2024, if it is anything related to date, month or year? Respond with 'yes' or 'no'.\n\n"
            f"5. Does the query inlcudes any one keyword from this list: fact-check, law, explainers, decode, mediabuddhi, web-stories, boom-research, deepfake-tracker. Provide one keyword from the list if present or related to any word in keyword, if it is not related to any return all. If query has boom-report then it is boom-research"
            f"Query: {enhanced_query}"
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
        print("#########################################################################")
        print(decision_prompt)
        print("#########################################################################")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(article_type)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


        ########################################################
      
        metadata = enhanced_data["query_metadata"]
         # Map primary tool to mediator response
        if metadata["primary_tool"] == "CUSTOM_DATE_RETRIEVER":
            return {
                "fetch_latest_articles": False,
                "use_rag": False,
                "use_tag": False,
                "custom_date_range": True,
                "index_to_use": None,
                "response": "Using custom date retriever for temporal query",
                "article_type": article_type,
                "enhanced_query": enhanced_query 
            }
        ###############################################


        if custom_date_match:
            return {
                "fetch_latest_articles": False,
                "use_rag": False,
                "use_tag": False,
                "custom_date_range": True,
                "index_to_use": None,
                "response": "Query detected as custom date range, fetching articles for the specified range.",
                "article_type": article_type,
                "enhanced_query": enhanced_query 
            }
        
                # If the query contains any of these keywords, mark it as a fact-check query
        is_fact_check = any(keyword in enhanced_query.lower() for keyword in fact_check_keywords)
        print("is_fact_check", is_fact_check)
        # Force the use of RAG for fact-check queries
        if is_fact_check:
            return {
                "fetch_latest_articles": False,  # Skip fetching articles if fact-checking
                "use_rag": True,  # Always use RAG for fact-checking queries
                "use_tag": False,
                "custom_date_range": False,
                "index_to_use": "both",  # Check both indexes for relevant data (latest and old)
                "response": "Query detected as fact-check, using RAG tool.",
                "article_type": article_type,
                "enhanced_query": enhanced_query

            }
        return {
            "fetch_latest_articles": fetch_latest_articles,
            "use_rag": use_rag,
            "use_tag": False,
            "custom_date_range": custom_date_range,
            "index_to_use": index_to_use,
            "article_type": article_type,
            "enhanced_query": enhanced_query 
        }




    def retrieve_custom_date_articles(self, query: str, article_type: str) -> dict:
        """
        Retrieve articles based on custom date range specified in the query.
        """
        print(f"We are getting article type as {article_type}")
          # Get the current date
        current_date = datetime.now().strftime("%B %d, %Y")  # Format the current date as YYYY-MM-DD
        # date_prompt = (
        #     f"Analyze the following query and extract the date range (if any):\n"
        #     f"Query: {query}\n"
        #     f"The current date is {current_date}. Use this as the reference for relative terms like 'today' or 'last week'.\n"
        #     f"If terms like 'last year' or 'this year' are mentioned, just return 'last year' or 'this year' without specifying a date range.\n"
        #     f"Otherwise, provide the result in the format 'from YYYY-MM-DD to YYYY-MM-DD' or a description like 'last week', etc. And note range shouldn't exceed 1 month so adjust range which suits user requirements better but just provide 1 month range"
            
        # )


        date_prompt = (
            f"Analyze the following query and extract the date range (if any):\n"
            f"Query: {query}\n"
            f"The current date is {current_date}. Use this as the reference for relative terms like 'today' or 'last week'.\n"
            f"If terms like 'last year' or 'this year' are mentioned, just return 'last year' or 'this year' without specifying exact dates.\n"
            f"Otherwise, provide the result strictly in the format 'from YYYY-MM-DD to YYYY-MM-DD'.\n"
            f"Ensure that the date range does not exceed one month. If the query requests a longer period, adjust it to the most relevant 1-month range.\n"
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
                "result": f"No articles were found for the specified date range ({date_range})",
                "sources": []
            }
        filtered_sources = []
        if "all" in article_type:
            filtered_sources = list(sources)
        else:
            for source in sources:

                if f"https://www.boomlive.in/{article_type}" in source:
                    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                    print(source, "article_type",article_type)
                    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                    filtered_sources.append(source) 

        if not filtered_sources:
            filtered_sources = list(sources[:10])
        # Generate a summary of the articles if sources are found
        # summary_prompt = (
        #     f"Summarize the information based on the following question: {query}.\n"
        #     f"Use these sources to craft the response: {filtered_sources}\n"
        #     f"Focus on providing concise and relevant details without additional disclaimers or unrelated remarks."
        #     f"Provide article url for each article below their summary: {filtered_sources}"
        # )
        print("FILTERED SOURCES: ",filtered_sources)
        summary_prompt = (
            f"Summarize the information based on the following question: {query}.\n"
            f"Use these sources to craft the response: {filtered_sources[:3]}\n"
            f"Focus on providing concise and relevant details without additional disclaimers or unrelated remarks.\n\n"
            f"For each summary, list the original article URL explicitly under the summary.\n"
            f"Format: \n**Article Title Mentioned in Article Should be added dynalically:**\nYour summary here\n\n[Read more](Original article URL here)\n"
            f"Add a partion line after Read More like a hr tag of size required ______________________________________________________\n"  # This ensures an actual HTML horizontal rule
            f"Prevent adding emojis in response"
        )

        summary_response = self.llm.invoke([self.system_message,HumanMessage(content=summary_prompt)])
        print(summary_response.content.strip())
        return {
            "result": summary_response.content.strip(),
            "sources": []
        }
    






    def refine_query_for_vector_search(self, query: str, context_type: str = "all") -> dict:
        """
        Enhanced query refinement using LLM for better vector search results.
        
        Args:
            query (str): Original user query
            context_type (str): Type of content to search (fact-check, article, etc.)
            
        Returns:
            dict: Refined query information including main query and search parameters
        """
        # Create a prompt that helps LLM understand boomlive's context
        refinement_prompt = f"""
        As an AI specializing in BoomLive's content, help optimize this query for vector search.
        Original query: "{query}"
        Context type: {context_type}

        Consider these aspects:
        1. BoomLive's focus on fact-checking and journalism
        2. Common misinformation patterns
        3. Current affairs and news context
        4. Regional Indian context when relevant

        Please analyze and provide:
        1. A refined main query optimized for semantic search
        2. Key terms that should be emphasized
        3. Relevant categories or topics
        4. Any temporal context (time-based relevance)
        5. Suggested filters (if any)

        Format your response as:
        REFINED_QUERY: <refined version>
        KEY_TERMS: <comma-separated key terms>
        CATEGORIES: <relevant categories>
        TEMPORAL: <time relevance>
        FILTERS: <suggested filters>
        """
        
        # Get LLM's refinement suggestions
        response = self.llm.invoke([
            self.system_message,
            HumanMessage(content=refinement_prompt)
        ])
        
        # Parse LLM response
        refinement_results = {}
        current_key = None
        for line in response.content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                refinement_results[key] = value
        
        # Extract and clean keywords from the refined query
        cleaned_query = self.extract_keywords(refinement_results.get('REFINED_QUERY', query))
        
        # Add domain-specific terms
        key_terms = [term.strip() for term in refinement_results.get('KEY_TERMS', '').split(',')]
        key_terms = [term for term in key_terms if term]  # Remove empty terms
        
        # Combine original keywords with domain-specific terms
        enhanced_query = f"{cleaned_query} {' '.join(key_terms)}"
        
        # Structure search parameters
        search_params = {
            "original_query": query,
            "enhanced_query": enhanced_query.strip(),
            "key_terms": key_terms,
            "categories": refinement_results.get('CATEGORIES', '').split(','),
            "temporal_context": refinement_results.get('TEMPORAL', 'not specified'),
            "suggested_filters": refinement_results.get('FILTERS', '').split(','),
        }
        
        # Apply the enhanced query in your retrieve_data method
        return search_params
    






    def retrieve_data(self, query: str, index_to_use: str, article_type: str) -> dict:
        """
        Enhanced retrieve_data with better context utilization
        """
        print(f"Retrieve data called with query: {query} and index: {index_to_use} with article_type: {article_type}")
        
        current_date = get_current_date()
        print(current_date)
        # Refine the query using LLM
        # refined_params = self.refine_query_for_vector_search(query, article_type)
        enhanced_query = query#refined_params["enhanced_query"]
        # refined_query = self.extract_keywords(query)
        all_docs = []
        all_sources = []
        # print("refined_query", refined_query)
        print("enhanced_query", enhanced_query)
        # Determine if the query mentions dates or the latest content
        is_date_filtered = "latest" in query.lower() or "date" in query.lower()  # Check if the query mentions date or "latest"
        print("index_to_use",index_to_use)
        if index_to_use is not None:
            index_to_use = index_to_use.split(".")[-1].strip()  # This removes any extra text like "3." and keeps only "latest"

        if index_to_use in ["latest"] or index_to_use is None:
            print(f"inside:  if index_to_use in  latest")
            latest_retriever = self.latest_index.as_retriever(search_kwargs={"k": 5})
            latest_docs = latest_retriever.get_relevant_documents(enhanced_query)
            print(f"Latest documents retrieved: {len(latest_docs)}")  # Debugging line
            all_docs.extend(latest_docs)
            latest_sources = [doc.metadata.get("source", "Unknown") for doc in latest_docs]
            all_sources.extend(latest_sources)

        if index_to_use in ["both", "latest"]:
            print(f"inseide:  if index_to_use in :both", "latest")
            latest_retriever = self.latest_index.as_retriever(search_kwargs={"k": 5})
            latest_docs = latest_retriever.get_relevant_documents(enhanced_query)
            print(f"Latest documents retrieved: {len(latest_docs)}")  # Debugging line
            all_docs.extend(latest_docs)
            latest_sources = [doc.metadata.get("source", "Unknown") for doc in latest_docs]
            all_sources.extend(latest_sources)

        if index_to_use in ["both", "old"]:
            print(f"inseide:  if index_to_use in :both, old")

            old_retriever = self.old_index.as_retriever(search_kwargs={"k": 5})
            old_docs = old_retriever.get_relevant_documents(enhanced_query)
            print(f"Old documents retrieved: {len(old_docs)}")  # Debugging line
            all_docs.extend(old_docs)
            old_sources = [doc.metadata.get("source", "Unknown") for doc in old_docs]
            all_sources.extend(old_sources)

        if all_docs:
            print("the code is going in all_docs")
            combined_content = "\n\n".join([doc.page_content for doc in all_docs])

            # If the query does not mention dates or "latest", do not filter dates
            synthesis_prompt = f"""
            Based on the following content, provide a breif and short response as a Boom Chatbot: {query}
            The current date is {current_date}.
            Sources: {all_sources}
            Context:
            {combined_content}
            Also mention if relevant sources are found or not for the query: {query}
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
                    if source:
                        # Check if "fact-check" is selected and include both "fact-check" and "fast-check"
                        # if "fact-check" in article_type:
                        #     if "https://www.boomlive.in/fact-check" in source or "https://www.boomlive.in/fast-check" in source:
                        #         filtered_sources.append(source)
                        # else:
                        #     if f"https://www.boomlive.in/{article_type}" in source:
                        #         filtered_sources.append(source)

                         # Check if "fact-check" is selected and include both "fact-check" and "fast-check"
                        if "fact-check" in article_type:
                            if "https://www.boomlive.in/fact-check" in source or "https://www.boomlive.in/fast-check" in source:
                                filtered_sources.append(source)
                        # Include URLs matching the selected article type
                        elif f"https://www.boomlive.in/{article_type}" in source:
                            filtered_sources.append(source)
                        # Fallback: If URL does not match specific categories, add it by default
                        else:
                            filtered_sources.append(source)  # Ensures all BoomLive URLs are included

            print({"sources": all_sources})
            print({"filtered_sources": filtered_sources})
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
