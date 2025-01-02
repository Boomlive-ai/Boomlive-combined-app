import re
import requests
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

# Initialize the LLM
llm = ChatOpenAI(temperature=0, model_name='gpt-4o')

def generate_questions_batch(articles):
    """
    Generates concise questions for a batch of articles using an LLM.

    Parameters:
        articles (list): List of article dictionaries.

    Returns:
        list: A randomly ordered list of questions generated from all the articles.
    """
    # Construct a single prompt for all articles in the batch
    input_prompts = []
    for i, article in enumerate(articles):
        title = article.get("heading", "Untitled Article")
        description = article.get("description", "No description available.")
        story = article.get("story", "No story content available.")
        input_prompts.append(f"""
        Article {i + 1}:
        Title: {title}
        Description: {description}
        Story Excerpt: {story[:500]}... (truncated for brevity)

        Generate two simple, The questions should be mostly about the claims in the artricle using some good keywords in the title and description 
        (under 60 characters) that users are likely to ask. Ensure the questions are direct and plain text. Do not number them, use quotes, or add any other formatting.
        
        Example result format (Do not use backslashes or extra quotes):
        How did the statements impact public opinion?
        Why was the video of the event misrepresented?
        
        Provide only the list of questions which have proper context for user to understand question, as shown in the example.
        """)

    # Combine all prompts into one input
    batch_prompt = "\n".join(input_prompts)

    try:
        # Create a HumanMessage object for the LLM
        message = HumanMessage(content=batch_prompt)

        # Invoke the LLM with the message
        response = llm.invoke([message])

        # Extract and clean the questions from the response
        questions = response.content.strip().split("\n")
        cleaned_questions = [q.strip() for q in questions if q.strip()]

        return cleaned_questions
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []

def fetch_questions_on_latest_articles_in_Boomlive():
    """
    Fetches the latest articles from the IndiaSpend API and generates up to 20
    concise questions in batches.

    Returns:
        dict: A dictionary containing all questions from the articles in a single list.
    """
    urls = []
    api_url = 'https://boomlive.in/dev/h-api/news'
    headers = {
        "accept": "*/*",
        "s-id": "1w3OEaLmf4lfyBxDl9ZrLPjVbSfKxQ4wQ6MynGpyv1ptdtQ0FcIXfjURSMRPwk1o"
    }
    print(f"Fetching articles from API: {api_url}")

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if response.status_code == 200:
            # Break if no articles are found
            if not data.get("news"):
                return {"questions": []}
            # Filter URLs containing 'fact-check' in the URL path
            for news_item in data.get("news", []):
                url_path = news_item.get("url")
                if url_path and "https://www.boomlive.in/fact-check/" in url_path:
                    urls.append(url_path)
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch articles: {e}")
        return {"error": f"Failed to fetch articles: {e}"}

    # If no relevant articles are found
    if not urls:
        print("No 'fact-check' articles found.")
        return {"questions": []}

    # Fetch corresponding articles
    articles = data.get("news", [])
    filtered_articles = [article for article in articles if article.get("url") in urls]

    # Limit articles to 10 (as each article generates 2 questions)
    filtered_articles = filtered_articles[:10]

    # Generate questions in a single batch
    questions = generate_questions_batch(filtered_articles)

    # Ensure only 20 questions are returned
    return {"questions": questions[:20]}



def fetch_articles_based_on_articletype(articleType):
    urls=[]
    api_url = 'https://boomlive.in/dev/h-api/news'
    headers = {
        "accept": "*/*",
        "s-id": "1w3OEaLmf4lfyBxDl9ZrLPjVbSfKxQ4wQ6MynGpyv1ptdtQ0FcIXfjURSMRPwk1o"
    }
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if response.status_code == 200:
            # Break if no articles are found
            if not data.get("news"):
                return {"questions": []}
            # Filter URLs containing 'fact-check' in the URL path
            for news_item in data.get("news", []):
                url_path = news_item.get("url")
                if url_path and f"https://www.boomlive.in/{articleType}" in url_path:
                    urls.append(url_path)
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch articles: {e}")
        return {"error": f"Failed to fetch articles: {e}"}
    
        # If no relevant articles are found
    if not urls:
        print(f"No {articleType} articles found.")
        return [{"urls": []}]
        
    return {"urls": urls}
