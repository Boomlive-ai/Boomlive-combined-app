from flask import Blueprint, request, jsonify
from chatbot.bot import Chatbot
from chatbot.utils import extract_sources_and_result, prioritize_sources
from langchain_core.messages import HumanMessage
from chatbot.tools import fetch_questions_on_latest_articles_in_Boomlive, fetch_articles_based_on_articletype
from chatbot.vectorstore import StoreCustomRangeArticles, StoreDailyArticles
chatbot_bp = Blueprint('chatbot', __name__)
mybot = Chatbot()
workflow = mybot()




@chatbot_bp.route('/')
def api_overview():
    """
    This route provides a basic overview of available API routes.
    """
    routes = {
        "GET /query": "Query the chatbot with a question (requires 'question' and 'thread_id' parameters).",
        "POST /store_articles": "Store articles for a custom date range (requires 'from_date' and 'to_date' in the body).",
        "POST /store_daily_articles": "Store articles for the current day.",
        "GET /generate_questions": "Fetch latest articles and generate questions from Boomlive.",
        "GET /fetch_articles": "Fetch articles of specific article type (requires 'articleType' parameter)."
    }
    return jsonify(routes), 200




@chatbot_bp.route('/query', methods=['GET'])
def query_bot():
    question = request.args.get('question')
    thread_id = request.args.get('thread_id')
    sources = []
    if not question or not thread_id:
        return jsonify({"error": "Missing required parameters"}), 400

    input_data = {"messages": [HumanMessage(content=question)]}
    try:
        response = workflow.invoke(input_data, config={"configurable": {"thread_id": thread_id}})
        result = response['messages'][-1].content
        print("response['messages'][-1]",result)
        result,raw_sources  = extract_sources_and_result(result)

        sources = prioritize_sources(question, raw_sources)
        if not result:
            result = "No response generated. Please try again."
        return jsonify({"response": result, "sources": sources})
    except Exception as e:
        print(f"Error in query_bot: {str(e)}")
        return jsonify({"error": str(e)}), 500




# Route for Storing Articles with Custom Date Range
@chatbot_bp.route('/store_articles', methods=['POST'])
async def store_articles():
    """
    Store articles for a custom date range.
    Query Parameters:
        - from_date (str): Start date in 'YYYY-MM-DD' format.
        - to_date (str): End date in 'YYYY-MM-DD' format.
    """
    # Extract dates from JSON payload
    data = request.get_json()
    from_date = data.get('from_date')
    to_date = data.get('to_date')

    # Validate input
    if not from_date or not to_date:
        return jsonify({"error": "Missing 'from_date' or 'to_date' in request body"}), 400

    # Instantiate and invoke the StoreCustomRangeArticles class
    try:
        store_articles_handler = StoreCustomRangeArticles()
        result = await store_articles_handler.invoke(from_date=from_date, to_date=to_date)
        return jsonify(result)
    except Exception as e:
        print(f"Error in store_articles: {str(e)}")
        return jsonify({"error": str(e)}), 500

@chatbot_bp.route('/store_daily_articles', methods=['POST'])
async def store_daily_articles_route():
    """
    Async route to fetch and store daily articles.
    """
    try:
        # Instantiate the handler class
        store_articles_handler = StoreDailyArticles()
        
        # Invoke the class method
        result = await store_articles_handler.invoke()
        
        # Return JSON response
        return jsonify(result)
    except Exception as e:
        print(f"Error in /store_daily_articles: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    


@chatbot_bp.route('/generate_questions', methods=['GET'])
def generate_questions_route():
    """
    Route to fetch articles and generate questions using imported functions.
    """
    try:
        # Call the imported function to fetch and generate questions
        results = fetch_questions_on_latest_articles_in_Boomlive()
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    



@chatbot_bp.route('/fetch_articles', methods=['GET'])
def fetch_articles():
    """
    Route to fetch articles of specific article type.
    """
    articleType = request.args.get('articleType')
    try:
        # Call the imported function to fetch and generate questions
        results = fetch_articles_based_on_articletype(articleType)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    
