from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1]/ ".env");
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    temperature=0,
    #huggingfacehub_api_token=api_key,
)
model = ChatHuggingFace(llm=llm)
template1= PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=['topic']
)
template2= PromptTemplate(
    template="write a five line summary on the following text ./n {text}",
    input_variables=['text']
)
prompt1=template1.invoke({'topic':'black hole'})
result = model.invoke(prompt1)
prompt2=template2.invoke({'text':result.content})
result1=model.invoke(prompt2)
print(result1.content)
