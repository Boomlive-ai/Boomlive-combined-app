import re, json
from langchain_core.messages import HumanMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import json
import time
from random import randint

def extract_description_as_keywords(url):
    """
    Extracts meta description from any URL and returns it as keywords.
    Works for social media posts, articles, blogs, and general websites.
    
    Args:
        url (str): The URL to extract metadata from
        
    Returns:
        list: List of keywords extracted from the description
    """
    try:
        # Configure request with browser-like headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Add cookies for sites that may require them
        cookies = {
            'locale': 'en_US',
        }
        
        # Get domain for site-specific handling
        domain = urlparse(url).netloc.lower()
        
        # Make the request with a small delay to avoid rate limiting
        time.sleep(randint(1, 3) / 10)  # Random delay between 0.1-0.3 seconds
        response = requests.get(url, headers=headers, cookies=cookies, timeout=15)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Initialize description variable
        description = ""
        
        # 1. Try standard meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            description = meta_desc.get('content').strip()
        
        # 2. Try Open Graph description
        if not description:
            og_desc = soup.find('meta', property='og:description')
            if og_desc and og_desc.get('content'):
                description = og_desc.get('content').strip()
        
        # 3. Try Twitter description
        if not description:
            twitter_desc = soup.find('meta', attrs={'name': 'twitter:description'})
            if twitter_desc and twitter_desc.get('content'):
                description = twitter_desc.get('content').strip()
        
        # 4. Social media-specific handling
        if not description:
            # Instagram
            if 'instagram.com' in domain:
                # Try Instagram-specific extraction
                for script in soup.find_all('script'):
                    if script.string and 'window._sharedData' in script.string:
                        try:
                            json_str = script.string.split('window._sharedData = ')[1].split(';</script>')[0]
                            data = json.loads(json_str)
                            
                            if 'entry_data' in data:
                                if 'PostPage' in data['entry_data'] and data['entry_data']['PostPage']:
                                    post = data['entry_data']['PostPage'][0]['graphql']['shortcode_media']
                                    
                                    if 'edge_media_to_caption' in post and post['edge_media_to_caption']['edges']:
                                        description = post['edge_media_to_caption']['edges'][0]['node']['text']
                        except:
                            pass
                
                # Try alternative Instagram extraction
                if not description:
                    meta_content = soup.find('meta', property='og:description')
                    if meta_content and meta_content.get('content'):
                        content = meta_content.get('content')
                        parts = content.split(':')
                        if len(parts) > 1:
                            description = ':'.join(parts[1:]).strip()
            
            # Twitter/X
            elif 'twitter.com' in domain or 'x.com' in domain:
                tweet_div = soup.find('div', attrs={'data-testid': 'tweetText'})
                if tweet_div:
                    description = tweet_div.get_text().strip()
            
            # Facebook
            elif 'facebook.com' in domain:
                post_content = soup.find('div', attrs={'data-testid': 'post_message'})
                if post_content:
                    description = post_content.get_text().strip()
        
        # 5. JSON-LD structured data
        if not description:
            for script in soup.find_all('script', type='application/ld+json'):
                if script.string:
                    try:
                        json_data = json.loads(script.string)
                        
                        # Handle both direct objects and arrays of objects
                        json_items = json_data if isinstance(json_data, list) else [json_data]
                        
                        for item in json_items:
                            if item.get('description'):
                                description = item.get('description')
                                break
                            
                            # Extract articleBody if available
                            if 'articleBody' in item:
                                body = item['articleBody']
                                if isinstance(body, str):
                                    description = body[:200] + ('...' if len(body) > 200 else '')
                                    break
                    except:
                        continue
        
        # 6. Content fallback - look for article content if no description found
        if not description:
            # Find the main article content
            article_element = soup.find(['article', 'main', 'div'], class_=lambda x: x and any(c in str(x).lower() for c in ['article', 'content', 'story', 'post']))
            
            if article_element:
                # Find the main content paragraphs
                paragraphs = article_element.find_all('p')
                if paragraphs:
                    # Get the first few paragraphs, skipping very short ones
                    content_paragraphs = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40][:2]
                    if content_paragraphs:
                        # Take first 200 chars combined from the first few substantive paragraphs
                        combined = ' '.join(content_paragraphs)
                        description = combined[:200] + ('...' if len(combined) > 200 else '')
        
        # 7. Last resort fallback - use any meaningful paragraph
        if not description:
            paragraphs = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 40]
            if paragraphs:
                # Sort by length and take the longest one
                longest_paragraph = sorted(paragraphs, key=len, reverse=True)[0]
                description = longest_paragraph[:200] + ('...' if len(longest_paragraph) > 200 else '')
        
        # Clean up the description
        if description:
            # Remove excess whitespace, newlines
            description = re.sub(r'\s+', ' ', description).strip()
        
        # Extract keywords from description
        keywords = []
        if description:
            # Method 1: Split by punctuation and get phrases
            phrases = re.split(r'[.!?;,]', description)
            phrases = [phrase.strip() for phrase in phrases if len(phrase.strip()) > 3]
            keywords.extend(phrases[:5])  # Add up to 5 phrases
            
            # Method 2: Extract important words (excluding common words)
            common_words = {'the', 'a', 'an', 'and', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'as', 'this', 'that', 'these', 'those', 'it', 'its'}
            words = re.findall(r'\b\w+\b', description.lower())
            important_words = [word for word in words if len(word) > 3 and word not in common_words]
            
            # Get the most frequent important words
            word_freq = {}
            for word in important_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            keywords.extend([word for word, freq in sorted_words[:10]])  # Add top 10 words
            
            # Remove duplicates while preserving order
            seen = set()
            keywords = [x for x in keywords if not (x.lower() in seen or seen.add(x.lower()))]
        
        return keywords
        
    except Exception as e:
        print(f"Error extracting description as keywords: {e}")
        return []

# Example usage
# keywords = extract_description_as_keywords("https://example.com/article")
# print(keywords)
    


# def prioritize_sources(response_text: str, sources: list) -> list:
#     """
#     Reorder sources based on URL ID numbers as primary factor and content similarity as secondary factor.
    
#     Args:
#         response_text (str): The generated response content
#         sources (list): List of source URLs to prioritize
        
#     Returns:
#         list: Reordered list of sources with priority based on ID and relevance
#     """
#     if not response_text or not sources:
#         return sources
    
#     # Group sources by ID numbers
#     def extract_id(url):
#         try:
#             return int(url.rstrip('/').split('-')[-1])
#         except (ValueError, IndexError):
#             return 0
    
#     # Create groups based on ID ranges (e.g., every 5000 IDs)
#     id_groups = {}
#     for source in sources:
#         id_num = extract_id(source)
#         group_key = id_num // 5000  # Group by ranges of 5000
#         if group_key not in id_groups:
#             id_groups[group_key] = []
#         id_groups[group_key].append(source)
    
#     # Calculate content similarity scores
#     vectorizer = TfidfVectorizer(stop_words="english")
#     texts = [response_text] + sources
#     tfidf_matrix = vectorizer.fit_transform(texts)
#     similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
#     # Sort within each group by similarity
#     final_sorted_sources = []
#     for group_key in sorted(id_groups.keys(), reverse=True):  # Process groups from newest to oldest
#         group_sources = id_groups[group_key]
#         group_indices = [sources.index(s) for s in group_sources]
#         group_similarities = [similarities[i] for i in group_indices]
        
#         # Sort sources within group by similarity
#         sorted_group = [x for _, x in sorted(
#             zip(group_similarities, group_sources),
#             key=lambda pair: pair[0],
#             reverse=True
#         )]
        
#         final_sorted_sources.extend(sorted_group)
    
#     return final_sorted_sources

import requests
from bs4 import BeautifulSoup


###############################################################################################################################

import re
import nltk
from difflib import get_close_matches
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def clean_text(text, remove_stopwords=True):
    """Lowercase, remove special chars & optional stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    if remove_stopwords:
        words = [word for word in words if word not in STOPWORDS]
    return " ".join(words)

def detect_typos(query, sources, threshold=0.8):
    """Detect typos by comparing words with sources."""
    words = query.split()
    dictionary = set(word.lower() for snippet in sources for word in snippet.split())
    typos = [word for word in words if not get_close_matches(word, dictionary, cutoff=threshold)]
    return typos if typos else None

def compute_similarity(query, sources, source_texts):
    """Use TF-IDF (with bigrams & trigrams) and cosine similarity to check relevance."""
    vectorizer = TfidfVectorizer(ngram_range=(1,3))  # Capture phrases
    docs = [query] + source_texts  # Include query as the first doc
    tfidf_matrix = vectorizer.fit_transform(docs)

    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    relevant_sources = [
        (sources[i], source_texts[i]) for i in range(len(sources)) if similarities[i] > 0.2  # Lower threshold
    ]
    
    return relevant_sources

def fuzzy_match(query, sources, source_texts, fuzz_threshold=75):
    """Use fuzzy matching to find approximate matches."""
    matched_sources = []
    for i, text in enumerate(source_texts):
        score = fuzz.partial_ratio(query, text)
        if score >= fuzz_threshold:
            matched_sources.append((sources[i], text))
    
    return matched_sources

def verify_sources(query, sources, source_texts):
    """
    Optimized verification function to check claim relevance.
    """
    # Step 1: Preprocess query & sources
    clean_query = clean_text(query, remove_stopwords=False)  # Keep stopwords for meaning
    clean_sources = [clean_text(text) for text in source_texts]

    # Step 2: Detect typos
    if detect_typos(clean_query, clean_sources):
        return "Invalid"

    # Step 3: Compute similarity & extract relevant sources
    relevant_sources = compute_similarity(clean_query, sources, clean_sources)

    # Step 4: Use fuzzy matching as a backup
    if not relevant_sources:
        relevant_sources = fuzzy_match(clean_query, sources, clean_sources)

    if not relevant_sources:
        return "Not Found"

    return "\n\n".join([f"Source: {url}\nContent: {snippet}" for url, snippet in relevant_sources])

import re
import nltk
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

# Load a semantic similarity model
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text, remove_stopwords=True):
    """Lowercase, remove special chars & optional stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    if remove_stopwords:
        words = [word for word in words if word not in STOPWORDS]
    return " ".join(words)

from sklearn.feature_extraction.text import CountVectorizer
from fuzzywuzzy import fuzz
from textblob import TextBlob
import numpy as np

def correct_spelling(text):
    return str(TextBlob(text).correct())  # Corrects misspellings

def extract_title_from_url(url):
    return url.replace("https://www.boomlive.in/", "").replace("-", " ").replace("/", " ").lower()

def get_most_suitable_source(query, sources, source_texts):
    if not sources or not source_texts:
        print("No sources or texts available!")
        return None, None

    # Step 1: Correct spelling errors in the query
    corrected_query = correct_spelling(query)
    query_words = set(corrected_query.lower().split())

    # Step 2: Extract words from source texts
    vectorizer = CountVectorizer(stop_words="english")
    vectorizer.fit([corrected_query] + source_texts)
    source_keywords = [set(text.lower().split()) for text in source_texts]

    # Step 3: Compute scores based on:
    scores_text = [len(query_words & keywords) / len(query_words) for keywords in source_keywords]
    scores_url = [fuzz.partial_ratio(corrected_query, url) / 100 for url in sources]
    scores_title = [fuzz.partial_ratio(corrected_query, extract_title_from_url(url)) / 100 for url in sources]

    # Step 4: Normalize scores
    def normalize(arr):
        return (arr - np.min(arr)) / np.ptp(arr) if np.ptp(arr) > 0 else arr

    scores_text = normalize(np.array(scores_text))
    scores_url = normalize(np.array(scores_url))
    scores_title = normalize(np.array(scores_title))

    # Step 5: Compute final weighted score
    combined_scores = (0.5 * scores_text) + (0.3 * scores_url) + (0.2 * scores_title)

    print("Final Combined Scores:", combined_scores)

    # Step 6: Find the best match
    best_index = np.argmax(combined_scores)

    if combined_scores[best_index] > 0.3:  # Adjust relevance threshold
        print("Best Match:", sources[best_index])
        return sources[best_index], source_texts[best_index]

    print("No suitable source found!")
    return None, None

# # Example usage:
# query = "Earthquake in California 2025"
# sources = ["https://news.com/article1", "https://news.com/article2"]
# source_texts = [
#     "A strong earthquake hit California in 2025 causing damage.",
#     "New policies in California are being discussed."
# ]

# result = verify_sources(query, sources, source_texts)
# print(result)


###############################################################################################################################


def fetch_source_content(url):
    """
    Fetch and extract meaningful content from a source URL.
    
    Args:
        url (str): The URL of the source.
    
    Returns:
        str: Extracted content (or snippet) for similarity calculation.
    """
    try:
        response = requests.get(url, timeout=5)  # Fetch the page
        if response.status_code != 200:
            return ""  # Return empty string if fetch fails

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the main content (customize based on website structure)
        paragraphs = soup.find_all("p")  # Get all paragraph tags
        content = " ".join([p.get_text() for p in paragraphs[:5]])  # Get first 5 paragraphs

        return content.strip()
    
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return ""

def prioritize_sources(user_query: str, response_text: str, sources: list) -> list:
    """
    Reorder sources based on similarity with user query and response text.
    
    Args:
        user_query (str): The original user question
        response_text (str): The generated response content
        sources (list): List of source URLs to prioritize
        
    Returns:
        list: Reordered list of sources with priority based on relevance
    """
    if not user_query or not response_text or not sources:
        return sources  # Return as-is if missing data

    # Extract source IDs (assuming format ends in `-<number>`)
    def extract_id(url):
        try:
            return int(url.rstrip('/').split('-')[-1])
        except (ValueError, IndexError):
            return 0

    # Fetch source content snippets (assuming we have a way to extract them)
    source_texts = [fetch_source_content(url) for url in sources]  # Implement fetch_source_content function
    
    # Prepare text inputs for similarity calculation
    texts = [user_query, response_text] + source_texts  # First two are query and response

    # Compute TF-IDF similarity
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Compute similarity scores
    query_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[2:]).flatten()  # Query vs Sources
    response_similarities = cosine_similarity(tfidf_matrix[1:2], tfidf_matrix[2:]).flatten()  # Response vs Sources

    # Combine scores (weighted sum, adjust weights if needed)
    combined_scores = 0.6 * query_similarities + 0.4 * response_similarities

    # Sort sources based on combined similarity score
    sorted_sources = [x for _, x in sorted(zip(combined_scores, sources), key=lambda pair: pair[0], reverse=True)]

    return sorted_sources

def fetch_page_text(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract main text – this may require some custom logic depending on the page structure.
            paragraphs = soup.find_all('p')
            text = "\n".join(p.get_text() for p in paragraphs)
            return text.strip()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return ""

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.boomlive.in"

def extract_articles(tag_url):
    """Fetch and extract article titles, URLs, and summaries from BoomLive search results."""
    try:
        response = requests.get(tag_url, timeout=10)
        if response.status_code != 200:
            print("Failed to retrieve page, status code:", response.status_code)
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []
        
        # Select all <a> tags with class "heading_link" inside the section with class "search-page"
        for link in soup.select("section.search-page a.heading_link"):
            title = link.get_text(strip=True)
            url = link.get("href")
            # Ensure full URL if the link is relative
            if url and not url.startswith("http"):
                url = f"{BASE_URL}{url}"
            
            # Find the closest parent <h4> and then the next sibling <p> for summary text
            h4_tag = link.find_parent("h4")
            if h4_tag:
                summary_tag = h4_tag.find_next_sibling("p")
                summary = summary_tag.get_text(strip=True) if summary_tag else "No summary available"
            else:
                summary = "No summary available"
            
            articles.append((title, url, summary))
        
        return articles
    
    except Exception as e:
        print("Error extracting articles:", e)
        return []




# def prioritize_sources(response_text: str, sources: list) -> list:
#     """
#     Reorder sources based on similarity to the response text.

#     Args:
#         response_text (str): The generated response content.
#         sources (list): List of source URLs to prioritize.

#     Returns:
#         list: Reordered list of sources with the most relevant one at the top.
#     """
#     print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
#     print(sources)
#     print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

#     # If no response text or sources, return sources as is
#     if not response_text or not sources:
#         return sources

#     # Combine response text and sources for comparison
#     texts = [response_text] + sources  # Place response first
#     vectorizer = TfidfVectorizer(stop_words="english")  # Use TF-IDF to vectorize
#     tfidf_matrix = vectorizer.fit_transform(texts)

#     # Calculate cosine similarity between response_text and each source
#     similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

#     # Sort sources by similarity scores in descending order
#     sorted_indices = sorted(range(len(sources)), key=lambda i: similarities[i], reverse=True)
#     sorted_sources = [sources[i] for i in sorted_indices]

#     return sorted_sources



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
    return article_urls

################################################VECTOR STORE DATABASE################################################################


import datetime, json
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

import datetime

async def store_daily_articles():
    """
    Fetch and store articles for the last 15 days asynchronously.

    Returns:
        list: List of article URLs stored for the specified period.
    """
    # Get today's date
    today = datetime.date.today()
    from_date = (today - datetime.timedelta(days=15)).strftime('%Y-%m-%d')  # 15 days before today
    to_date = today.strftime('%Y-%m-%d')  # Today's date

    print(f"Storing articles from {from_date} to {to_date}...")
    try:
        # Use the existing function to store articles for the given range
        daily_articles = await store_articles_custom_range(from_date, to_date)
        return daily_articles
    except Exception as e:
        print(f"Error in store_daily_articles: {str(e)}")
        return []


# async def store_daily_articles():
#     """
#     Fetch and store articles for the current day asynchronously.

#     Returns:
#         list: List of article URLs stored for the current day.
#     """
#     # Get today's date
#     today = datetime.date.today()
#     from_date = to_date = today.strftime('%Y-%m-%d')  # Both dates set to today

#     print(f"Storing articles for {from_date}...")
#     try:
#         # Use the existing function to store articles for today
#         daily_articles = await store_articles_custom_range(from_date, to_date)
#         return daily_articles
#     except Exception as e:
#         print(f"Error in store_daily_articles: {str(e)}")
#         return []


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
    index_name = "boom-latest-articles"

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
    
from urllib.parse import urlparse

def is_url(input_string):
    try:
        result = urlparse(input_string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False