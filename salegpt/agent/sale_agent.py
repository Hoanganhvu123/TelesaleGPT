import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatLiteLLM
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from salegpt.tool.send_email_tool import SendEmailTool
from salegpt.agent.prompt import SALE_AGENT_PROMPT

class CustomerInfo(BaseModel):
    name: str
    phone: str
    demand: str
    product_used: str

class SalesGPT:
    """Controller model for the Sales Agent."""

    salesAI_name: str = "DaisyAI"
    salesAI_role: str = "Business Development Representative"
    company_name: str = "OpenDaisy"
    company_business: str = (
        "OpenDaisy is an innovative technology company that provides customers "
        "with cutting-edge AI-powered solutions. We offer a range of high-quality "
        "AI products and services designed to meet the unique needs of our customers "
        "across various industries."
    )

    def __init__(
        self,
        llm: ChatLiteLLM,
        verbose: bool = False,
        customer_info: CustomerInfo = None
    ):
        self.llm = llm
        self.verbose = verbose
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.customer_info = customer_info or CustomerInfo(
            name="",
            phone="",
            demand="",
            product_used=""
        )
        self.tools = [SendEmailTool()]
        
        self.prompt = SALE_AGENT_PROMPT
             
        # Create a ReAct agent
        self.agent = create_react_agent(self.llm, self.tools, self.prompt, stop_sequence=["<END_OF_TURN>"])
        
        # Set up the agent executor with the ReAct agent
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=self.verbose,
            handle_parsing_errors=True,
            max_iterations=3,
        )

    def invoke(self, query: str = None) -> Dict[str, Any]:
        inputs = self._prepare_inputs(query)
        
        ai_message = self.agent_executor.invoke(inputs)
        output = ai_message['output']
        conversation_ended = self._should_end_conversation(output)
        return {"output": output, "conversation_ended": conversation_ended}

    def _prepare_inputs(self, query: str = None) -> Dict[str, Any]:
        inputs = {
            "input": query or "Start the conversation by introducing yourself and explaining the purpose of your call.",
            "tool_names": "\n".join(f"{tool.name}: {tool.description}" for tool in self.tools),
            "tools": "\n".join(f"{tool.name}: {tool.description}" for tool in self.tools),
            "salesAI_name": self.salesAI_name,
            "salesAI_role": self.salesAI_role,
            "company_name": self.company_name,
            "company_business": self.company_business,
            "customer_info_name": self.customer_info.name,
            "customer_info_phone": self.customer_info.phone,
            "customer_info_demand": self.customer_info.demand,
            "customer_info_product_used": self.customer_info.product_used,
            "conversation_history": self.memory.chat_memory.messages,
            "agent_scratchpad": "",
        }
        return inputs

    def _should_end_conversation(self, output: str) -> bool:
        return "<END_OF_CALL>" in output

    def save_conversation(self, filename: str):
        """Saves the conversation history to a file."""
        with open(filename, 'w') as f:
            for message in self.memory.chat_memory.messages:
                f.write(f"{message.type}: {message.content}\n")

    @classmethod
    def from_llm(
        cls,
        llm: ChatLiteLLM,
        verbose: bool = False,
        customer_info: CustomerInfo = None,
        **kwargs
    ) -> "SalesGPT":
        """Initializes the SalesGPT from a given ChatLiteLLM instance."""
        return cls(
            llm=llm,
            customer_info=customer_info,
            verbose=verbose,
            **kwargs
        )
