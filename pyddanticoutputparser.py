from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field


load_dotenv(dotenv_path=Path(__file__).resolve().parents[1]/ ".env");
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    temperature=0,
    #huggingfacehub_api_token=api_key,
)
model = ChatHuggingFace(llm=llm)
class person(BaseModel):
    name: str = Field(description='name of the person')
    age: int = Field(gt = 18 , description='Age of the person')
    city : str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=person)
template = PromptTemplate(
    template='generate the name ,age and city of a frictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions}
)
chain = template|model|parser
result = chain.invoke({'bangalore'})
print(result)
