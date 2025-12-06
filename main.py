from pydantic_ai import Agent
from dotenv import load_dotenv
from google import genai
from settings import AIConfig
import instructor
load_dotenv()
agent = Agent(
    AIConfig.model_name,
    system_prompt='You are a learning assistant. Help users with their questions. Be concise and informative.'
)
result_sync = agent.run_sync('Different types of software development methodologies')
print(result_sync.output)

