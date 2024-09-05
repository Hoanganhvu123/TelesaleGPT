import os
from dotenv import load_dotenv
from typing import Dict, Any, Type
from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from langchain_experimental.tools.python.tool import PythonAstREPLTool
import pandas as pd
from langchain_core.runnables import RunnablePassthrough

from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_core.tracers.base import BaseCallbackHandler


# Load environment variables
load_dotenv()
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_SMITH_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')



ORDER_MANAGEMENT_PROMPT = """
You are working with a pandas dataframe in Python named `df`. 
Your task is to generate Python code to handle the following operations on the dataframe:

1. **Add New Orders:**
   - Prompt the user for details including `customer_name`, `phone_number`, `address`, `order_details`, and `preferred_delivery_time`.
   - Automatically generate a unique 8-character alphanumeric `order_id`.
   - Add the new order to the dataframe.
   - Save the updated dataframe to a CSV file at the path `E:\\SaleGPT\\packages\\salegpt\\data\\orders.csv`.
   - Print out the details of the created order, including the generated `order_id`.

2. **Update Existing Orders:**
   - Ask for the `order_id` of the order to update.
   - Prompt the user for the fields they want to update and their new values.
   - Update the corresponding order information in the dataframe.
   - Save the updated dataframe to a CSV file at the path `E:\\SaleGPT\\packages\\salegpt\\data\\orders.csv`.


The dataframe `df` contains the following columns:
- `order_id`: A unique identifier for each order (8-character alphanumeric string)
- `customer_name`: The name of the customer (string)
- `phone_number`: The phone number of the customer (string)
- `address`: The delivery address of the customer (string)
- `order_details`: Details of the order (string)
- `preferred_delivery_time`: The preferred delivery time specified by the customer (datetime)

### Requirements:
- **Handle case-insensitive queries** for customer names, phone numbers, and addresses.
- **Generate efficient and clear Python code** that validates inputs and prevents errors.
- **Ensure the code is readable and maintainable**.
- **Save any changes made to the dataframe** into the file located at `E:\\SaleGPT\\packages\\salegpt\\data\\orders.csv`.

Output only the Python command(s). Do not include explanations, comments, quotation marks, or additional information. 
Only output the command(s).
ONLY generate Python code to ADD or UPDATE data

Start!
Question: {input}
"""



class ProductSearchInput(BaseModel):
    input: str = Field(description="Useful for when you need to answer questions about product information, Please use Vietnamese input commands when using this tool.")
    
    
class ProductSearchTool(StructuredTool):
    name: str = "product_search"
    args_schema: Type[BaseModel] = ProductSearchInput

    def _run(self, input: str) -> Any:
        llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
        prompt = PromptTemplate(
            template=PRODUCT_RECOMMENDATION_PROMPT,
            input_variables=["input"]
        )   
        product_data = pd.read_csv("E:\\ShoppingGPT\\packages\\shoppinggpt\\data\\products.csv")
        python_tool = PythonAstREPLTool(globals={"df": product_data})
        # Construct the chain
        chain = (
            {"input": RunnablePassthrough()} 
            | prompt 
            | llm
            # | (lambda x: print("Đầu ra LLM : " + x.content))
            # | (lambda x: print()) 
            | (lambda x: python_tool.invoke(x.content))
        )
        result = chain.invoke(input)
        return result
