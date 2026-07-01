from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import ResponseSchema , StructuredOutputParser

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1]/ ".env");
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    temperature=0,
    #huggingfacehub_api_token=api_key,
)
model = ChatHuggingFace(llm=llm)
schema = [
    ResponseSchema(name='fact_1',description='Fact 1 about he topic'),
    ResponseSchema(name='fact_2',description='Fact 2 about he topic'),
    ResponseSchema(name='fact_3',description='Fact 3 about he topic')
]
parser = StructuredOutputParser.from_response_schemas(schema)
template = PromptTemplate(
    template = 'Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instruction()}
    
)
prompt = template.invoke({"topic":"black hole"})
result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)
    
