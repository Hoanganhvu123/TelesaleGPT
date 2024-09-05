import os
from dotenv import load_dotenv
from langchain_core.language_models.base import BaseLanguageModel
from langchain_google_genai import ChatGoogleGenerativeAI
from salegpt.agent.sale_agent import SalesGPT, CustomerInfo

# Load environment variables
load_dotenv("E:\\chatbot\\SaleGPT\\.env")

# Get API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def main():
    # Initialize language model
    llm: BaseLanguageModel = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )
    
    # Create SalesGPT instance
    customer_info = CustomerInfo(
        name="John Doe",
        phone="123-456-7890",
        demand="explore AI-powered solutions for business optimization",
        product_used="legacy software systems"
    )
    sales_agent = SalesGPT.from_llm(llm, verbose=True, customer_info=customer_info)

    print("Chào mừng bạn đến với OpenDaisy! Trợ lý AI của chúng tôi sẽ hỗ trợ bạn.")
    
    # AI initiates the conversation
    response = sales_agent.invoke()
    print(f"Trợ lý AI: {response['output']}")

    while True:
        user_input = input("John Doe: ")      
        response = sales_agent.invoke(user_input)
        print(f"Trợ lý AI: {response['output']}")
        
        if response['conversation_ended']:
            print("Trợ lý AI: Cảm ơn bạn đã trò chuyện với chúng tôi. Hẹn gặp lại!")
            break

    # Save the conversation history
    sales_agent.save_conversation("conversation_history.txt")

if __name__ == "__main__":
    main()