import requests
import json
import os
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
import iso639
import pycountry

llm = ChatOpenAI(temperature=0, model_name='gpt-4o')
load_dotenv()

def get_language_code(text):
    """
    Detects language and returns ISO 639-1 two-letter language code.
    Falls back to character-based detection for certain scripts.
    
    Args:
        text (str): Text to detect language from
        
    Returns:
        str: Two-letter ISO 639-1 language code
    """
    try:
        # Try with langdetect first
        detected_code = detect(text)
        
        # Validate the detected code against pycountry
        try:
            language = pycountry.languages.get(alpha_2=detected_code)
            return detected_code
        except (AttributeError, KeyError):
            # If pycountry doesn't recognize it, fall back to default language
            pass
            
        return detected_code
    except LangDetectException:
        # If langdetect fails, default to English
        return 'en'
    

def FactCheck(query):

    lang_code = get_language_code(query)

    print(f"Detected language: {lang_code}")


    payload = {
    'key':  os.getenv("GOOGLE_FACT_CHECK_TOOL_API"),
    'query':query,
    'languageCode':lang_code
    }
    url ='https://factchecktools.googleapis.com/v1alpha1/claims:search'
    response = requests.get(url,params=payload)
    print(response.text)

    if response.status_code == 200:
        result = json.loads(response.text)
        # Arbitrarily select 1
        try:
            topRating = result["claims"][0]
            # arbitrarily select top 1
            claimReview = topRating["claimReview"][0]["textualRating"]
            claimVal = "According to " + str(topRating["claimReview"][0]['publisher']['name'])+ " that claim is " + str(claimReview)
            return result           
        except:
            print("No claim review field found.")
            return 0
    else:
        return 0
    


def ProvideQuestionsForArticle(url):
    system_prompt = (
        "Provide possible questions from user point of view that can be asked based on the article context by common users and claims in context. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    try:
        # Fetch the webpage
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Verify the content type
        if 'text/html' not in response.headers.get('Content-Type', ''):
            print(f"Skipped non-HTML content at {url}")
            return {}

        # Extract text content
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])

        if text:
            # Create a LangChain Document object
            document = Document(
                page_content=text,
                metadata={"source": url}
            )

            # Pass as a dict with the expected key
            input_data = {"context": [document]}  # Ensure input is a dict with the correct key
            response = question_answer_chain.invoke(input_data)
            return response, document

        return {}

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return {}
    
    




def ProvideAnswerForArticle(url, input_query):
    system_prompt = (
        "Provide answers to the user's question based on the article context provided. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    try:
        # Fetch the webpage
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Verify the content type
        if 'text/html' not in response.headers.get('Content-Type', ''):
            print(f"Skipped non-HTML content at {url}")
            return {}

        # Extract text content
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])

        if text:
            # Create a LangChain Document object
            document = Document(
                page_content=text,
                metadata={"source": url}
            )

            # Pass as a dict with the expected key
            input_data = {"input": input_query,"context": [document]}  # Ensure input is a dict with the correct key
            response = question_answer_chain.invoke(input_data)
            return response

        return {}

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return {}
    
    