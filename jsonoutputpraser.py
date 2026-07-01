from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1]/ ".env");
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    temperature=0,
    #huggingfacehub_api_token=api_key,
)
model = ChatHuggingFace(llm=llm)
parser = JsonOutputParser()
template=PromptTemplate(
    template='give me the name , age ,city of a fictional person \n{format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
chain = template | model | parser
result = chain.invoke({})
print(result)
