import re, json
from langchain_core.messages import HumanMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def prioritize_sources(response_text: str, sources: list) -> list:
    """
    Reorder sources based on similarity to the response text.

    Args:
        response_text (str): The generated response content.
        sources (list): List of source URLs to prioritize.

    Returns:
        list: Reordered list of sources with the most relevant one at the top.
    """
    # If no response text or sources, return sources as is
    if not response_text or not sources:
        return sources

    # Combine response text and sources for comparison
    texts = [response_text] + sources  # Place response first
    vectorizer = TfidfVectorizer(stop_words="english")  # Use TF-IDF to vectorize
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Calculate cosine similarity between response_text and each source
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Sort sources by similarity scores in descending order
    sorted_indices = sorted(range(len(sources)), key=lambda i: similarities[i], reverse=True)
    sorted_sources = [sources[i] for i in sorted_indices]

    return sorted_sources



def extract_last_human_message_and_sources(response: dict) -> tuple:
    """
    Fetch the last HumanMessage and extract sources from its content.

    Args:
        response (dict): The response containing messages with potential sources.

    Returns:
        tuple: A tuple containing the HumanMessage content and a list of source URLs.
    """
    last_human_message = None
    sources = []

    # Loop through messages in reverse to find the last HumanMessage
    for message in reversed(response.get("messages", [])):
        if isinstance(message, HumanMessage):
            last_human_message = message.content
            break

    # If a HumanMessage is found, extract sources
    if last_human_message:
        print("Last HumanMessage Content:\n", last_human_message)  # Debug: Print full content
        # Extract URLs using regex
        sources = re.findall(r'https?://[^\s]+', last_human_message)
        # Remove duplicates and clean the list
        sources = list(set(sources))
        sources = [source.strip() for source in sources]

    return  sources
import re

def extract_sources_and_result(result: str):
    """
    Extracts source URLs from the "Sources:" section and returns the result without that section.

    Args:
        result (str): The response content containing Markdown links and a "Sources:" section.

    Returns:
        tuple: A tuple with two elements:
            1. result_without_sources (str): The content without the "Sources:" section.
            2. sources (list): A list of extracted URLs from the "Sources:" section.
    """
    # Detect all Markdown links (we will NOT remove these)
    markdown_links = re.findall(r'\[([^\]]+)\]\((https?://[^\s\)]+)\)', result)
    markdown_urls = {url for _, url in markdown_links}  # Extract just the URLs from Markdown links

    # Extract sources from the "Sources:" section at the end
    sources_match = re.search(r'\n*Sources:\n(.*)', result, flags=re.DOTALL)
    sources = []
    
    if sources_match:
        sources_section = sources_match.group(1)
        sources = re.findall(r'https?://[^\s]+', sources_section)  # Extract only URLs from the sources section

        # Remove only the "Sources:" section
        result = re.sub(r'\n*Sources:\n.*', '', result, flags=re.DOTALL).strip()

    return result, sources




def extract_clean_sources(response: dict) -> list:
    """
    Extract and clean source links from the last message in the response dictionary.

    Args:
        response (dict): The response containing messages with potential source links.

    Returns:
        list: A cleaned list of source URLs if found; otherwise, an empty list.
    """
    # Initialize an empty list to store source links
    sources = []

    # Get the list of messages
    messages = response.get("messages", [])

    # Check if there is at least one message
    if messages:
        # Get the last message
        last_message = messages[-1]

        # Access the 'content' attribute directly
        content = getattr(last_message, "content", "")
        print("Inspecting last message content:", content)  # Debugging line

        # Look for "Sources:" and extract URLs
        if "Sources:" in content:
            raw_sources = re.findall(r'https?://[^\s]+', content)
            sources.extend(raw_sources)

    # Remove duplicates and clean the list
    cleaned_sources = list(set(sources))  # Remove duplicates
    cleaned_sources = [source.strip() for source in cleaned_sources]  # Trim whitespace

    # Return the cleaned list (empty if no sources are found)
    return cleaned_sources


import requests
import re
from fuzzywuzzy import process

def fetch_latest_article_urls(query, article_type):
    """
    Fetches the latest articles from the BoomLive API, filters them based on exact keyword matching
    (fact-check, decode, explainers, mediabuddhi, boom-research), and sorts them by the largest number at the end of the URL.
    Then it returns the top 5 filtered URLs.

    Args:
        query (str): The query string to filter the articles.

    Returns:
        list: A list of the top 5 filtered URLs, sorted by the largest number at the end of the URL.
    """
    # # List of valid keywords
    # valid_keywords = ["fact-check", "decode", "explainers", "mediabuddhi", "boom-research"]

    # # Fuzzy match the user query with the valid keywords
    # matched_keywords = set()

    # # Extract individual words from the query
    # query_words = query.lower().split()

    # # Use fuzzy matching to find closest matches to each query word
    # for word in query_words:
    #     best_match = process.extractOne(word, valid_keywords)  # Get the best match for each word
    #     if best_match and best_match[1] >= 80:  # Match score threshold
    #         matched_keywords.add(best_match[0])

    # # If no valid keyword is identified, use all valid keywords
    # if not matched_keywords:
    #     print(f"No specific keywords found in query: {query}. Using all valid keywords.")
    #     matched_keywords = set(valid_keywords)

    # print(f"Matched keywords for filtering: {matched_keywords}")

    urls = []
    api_url = 'https://boomlive.in/dev/h-api/news'
    headers = {
        "accept": "*/*",
        "s-id": "1w3OEaLmf4lfyBxDl9ZrLPjVbSfKxQ4wQ6MynGpyv1ptdtQ0FcIXfjURSMRPwk1o"
    }

    print(f"Fetching articles from API: {api_url}")

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        
        if response.status_code == 200:
            data = response.json()

            # Break if no articles are found
            if not data.get("news"):
                return []

            for news_item in data.get("news", []):
                url_path = news_item.get("url")
                
                if article_type == "all":
                    urls.append(url_path)  # Include all URLs
                elif url_path and f"https://www.boomlive.in/{article_type}" in url_path:
                    urls.append(url_path)

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch articles: {e}")
        return {"error": f"Failed to fetch articles: {e}"}

    # Extract numeric values from URLs and sort by the largest number at the end of the URL
    urls_with_numbers = []

    for url in urls:
        # Extract the number at the end of the URL
        match = re.search(r'(\d+)(?=\s*$)', url)
        if match:
            number = int(match.group(0))
            urls_with_numbers.append((url, number))
    
    # Sort by the numeric values in descending order (largest number first)
    sorted_urls = sorted(urls_with_numbers, key=lambda x: x[1], reverse=True)

    # Get the top 5 filtered URLs
    top_5_urls = [url for url, _ in sorted_urls[:5]]

    print(f"Top 5 filtered URLs: {top_5_urls}")
    return top_5_urls



from datetime import datetime

def get_current_date():
    """
    Fetches the current date in a standardized format.
    Returns:
        str: Current date in 'YYYY-MM-DD' format.
    """
    today = datetime.date.today()
    return today




def fetch_custom_range_articles_urls(from_date: str = None, to_date: str = None):
    """
    Fetch and return article URLs based on a custom date range.

    Args:
        from_date (str): Start date in 'YYYY-MM-DD' format. Defaults to 6 months ago.
        to_date (str): End date in 'YYYY-MM-DD' format. Defaults to today.

    Returns:
        list: List of article URLs.
    """
    # Initialize variables
    article_urls = []
    start_index = 0
    count = 20
    print("fetch_custom_range_articles_urls", from_date, to_date)
    # Calculate default date range if not provided
    current_date = datetime.date.today()
    if not to_date:
        to_date = current_date.strftime('%Y-%m-%d')
    if not from_date:
        custom_months_ago = current_date - datetime.timedelta(days=180)  # Default to 6 months ago
        from_date = custom_months_ago.strftime('%Y-%m-%d')

    # Validate the date range
    if not validate_date_range(from_date, to_date):
        print("Invalid date range. Ensure 'from_date' <= 'to_date' and format is YYYY-MM-DD.")
        return []

    print(f"Fetching article URLs from {from_date} to {to_date}....")

    # Loop to fetch article URLs in batches
    while True:
        perpageurl = []
        print("Current start index:", start_index)

        # Construct API URL with the custom range
        api_url = f'https://boomlive.in/dev/h-api/news?startIndex={start_index}&count={count}&fromDate={from_date}&toDate={to_date}'
        headers = {
            "accept": "*/*",
            "s-id": "1w3OEaLmf4lfyBxDl9ZrLPjVbSfKxQ4wQ6MynGpyv1ptdtQ0FcIXfjURSMRPwk1o"
        }
        print(f"Requesting API URL: {api_url}")

        # Make the API request
        response = requests.get(api_url, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()

            # Break the loop if no articles are returned
            if not data.get("news"):
                break

            # Extract article URLs from the response
            for news_item in data.get("news", []):
                url_path = news_item.get("url")
                if url_path:
                    article_urls.append(url_path)
            start_index += count
        else:
            print(f"Failed to fetch articles. Status code: {response.status_code}")
            break
    print(article_urls)        
    return article_urls[:5]

################################################VECTOR STORE DATABASE################################################################


import datetime, json
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings



async def store_daily_articles():
    """
    Fetch and store articles for the current day asynchronously.

    Returns:
        list: List of article URLs stored for the current day.
    """
    # Get today's date
    today = datetime.date.today()
    from_date = to_date = today.strftime('%Y-%m-%d')  # Both dates set to today

    print(f"Storing articles for {from_date}...")
    try:
        # Use the existing function to store articles for today
        daily_articles = await store_articles_custom_range(from_date, to_date)
        return daily_articles
    except Exception as e:
        print(f"Error in store_daily_articles: {str(e)}")
        return []


async def store_articles_custom_range(from_date: str = None, to_date: str = None):
    """
    Fetch and store articles based on a custom date range.

    Args:
        from_date (str): Start date in 'YYYY-MM-DD' format. Defaults to 6 months ago.
        to_date (str): End date in 'YYYY-MM-DD' format. Defaults to today.

    Returns:
        list: List of all article URLs processed.
    """
    # Initialize variables
    article_urls = []
    start_index = 0
    count = 20

    # Calculate default date range if not provided
    current_date = datetime.date.today()
    if not to_date:
        to_date = current_date.strftime('%Y-%m-%d')
    if not from_date:
        custom_months_ago = current_date - datetime.timedelta(days=180)  # Default to 6 months ago
        from_date = custom_months_ago.strftime('%Y-%m-%d')

    # Validate the date range
    if not validate_date_range(from_date, to_date):
        print("Invalid date range. Ensure 'from_date' <= 'to_date' and format is YYYY-MM-DD.")
        return []

    print(f"Fetching data from {from_date} to {to_date}....")
    index_name = "india-spend"

    while True:
        perpageurl = []
        print("Now start index is ", start_index)

        # Construct API URL with the custom range
        api_url = f'https://boomlive.in/dev/h-api/news?startIndex={start_index}&count={count}&fromDate={from_date}&toDate={to_date}'
        headers = {
            "accept": "*/*",
            "s-id": "1w3OEaLmf4lfyBxDl9ZrLPjVbSfKxQ4wQ6MynGpyv1ptdtQ0FcIXfjURSMRPwk1o"
        }
        print(f"Current API URL: {api_url}")

        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            data = response.json()

            # Break if no articles are found
            if not data.get("news"):
                break

            for news_item in data.get("news", []):
                url_path = news_item.get("url")
                if url_path:
                    article_urls.append(url_path)
                    perpageurl.append(url_path)

            # print(perpageurl)
            # # Filter and process URLs
            filtered_urls = await filter_urls_custom_range(json.dumps(perpageurl))
            # print("These are filtered urls",filtered_urls)
            docsperindex = await fetch_docs_custom_range(filtered_urls)
            print(f"Processed {len(filtered_urls)} articles and {len(docsperindex)} chunks to add to Pinecone.")

            await store_docs_in_pinecone(docsperindex, index_name, filtered_urls)
            start_index += count
        else:
            print(f"Failed to fetch articles. Status code: {response.status_code}")
            break

    return article_urls



def validate_date_range(from_date: str, to_date: str) -> bool:
    """
    Validate the custom date range.

    Args:
        from_date (str): Start date.
        to_date (str): End date.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        from_dt = datetime.datetime.strptime(from_date, '%Y-%m-%d')
        to_dt = datetime.datetime.strptime(to_date, '%Y-%m-%d')
        return from_dt <= to_dt
    except ValueError:
        return False


async def filter_urls_custom_range(urls):
    api_url = f"https://exceltohtml.indiaspend.com/chatbotDB/not_in_table.php?urls={urls}"
    headers = {
        "accept": "*/*",
        "Authorization": "adityaboom_requesting2024#",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(api_url, headers=headers, verify=False)
        if response.status_code == 200:
            response_data = response.json()
            return response_data.get("urls", [])
    except requests.RequestException as e:
        print(f"Error filtering URLs: {e}")
    return []




async def fetch_docs_custom_range(urls):
    data = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Parse only HTML content
            if 'text/html' not in response.headers.get('Content-Type', ''):
                print(f"Skipped non-HTML content at {url}")
                continue

            # Extract text using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])

            document = Document(page_content=text, metadata={"source": url})
            data.append(document)
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
            continue

    docs = text_splitter.split_documents(data)
    return docs


async def store_docs_in_pinecone(docs, index_name, urls):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print(f"Storing {len(docs)} document chunks to Pinecone index '{index_name}'...")
    pine_vs = Pinecone.from_documents(documents = docs, embedding = embeddings, index_name=index_name)
    print(f"Added {len(docs)} Articles chunks in the pinecone")
    await add_urls_to_database(json.dumps(urls))
    print(f"Successfully stored documents. Associated URLs: {urls}")
    return pine_vs



async def add_urls_to_database(urls):
    """
    Adds new URLs to the database by sending them to an external API endpoint.

    Args:
        urls (list): List of new URLs to be added to the database.

    Returns:
        str: A message indicating the result of the request.
    """
    api_url = f"https://exceltohtml.indiaspend.com/chatbotDB/add_in_table.php?urls={urls}"
    headers = {
        "accept": "*/*",
        "Authorization": "adityaboom_requesting2024#",
        "Content-Type": "application/json"
    }
    
    try:
        # Send the POST request with the URLs in the payload
        response = requests.get(api_url, headers=headers, verify=False)

        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()
            # You can log or process the response data as required
            # noofurls = len(urls)
            # print(urls, noofurls)
            print(f"Successfully added {len(urls)}URLs to the database." )
            return f"Successfully added URLs to the database."
        else:
            if(len(urls) == 0):
                return f"There are no urls to add"
            return f"There are no urls to add"
    except requests.RequestException as e:
        return f"An error occurred while adding URLs: {e}"