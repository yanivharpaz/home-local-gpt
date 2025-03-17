from typing import List, cast, Dict, Any, Optional
import traceback
import os
import yaml
import chainlit as cl
import openai 
from dotenv import load_dotenv

from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat

from chainlit.input_widget import Slider, Tags, Select

##############################
# 1. Functions for user input
##############################
async def user_input_text(prompt: str, cancellation_token: CancellationToken | None = None) -> str:
    """For textual input from user (e.g., the N messages, final request)."""
    try:
        print("In user input text")
        response = await cl.AskUserMessage(content=prompt).send()
    except TimeoutError:
        return "User did not provide any input within the time limit."
    if response:
        return response["output"]  # type: ignore
    else:
        return "User did not provide any input."


async def user_action_approve_reject(
    prompt: str, cancellation_token: CancellationToken | None = None
) -> str:
    """
    For "Approve/Reject" action. If user hits Approve, we end the agentic conversation.
    """
    try:
        response = await cl.AskActionMessage(
            content=prompt,
            actions=[
                cl.Action(name="approve", label="Approve", payload={"value": "approve"}),
                cl.Action(name="reject", label="Reject", payload={"value": "reject"}),
            ],
        ).send()
    except TimeoutError:
        return "User did not respond in time."
    if response and response.get("payload"):  # type: ignore
        if response["payload"].get("value") == "approve":
            return "APPROVE."
        else:
            return "REJECT."
    else:
        return "User did not respond."


async def select_task_type():
    """Allow user to select the type of task they want to perform."""
    response = await cl.AskUserMessage(
        content="What type of task would you like the AI agents to help with?",
        timeout=180,
    ).send()
    
    if response:
        return response["output"]
    else:
        return "General assistance"


async def upload_optional_file():
    """Allow user to optionally upload a file to provide context."""
    try:
        files = await cl.AskFileMessage(
            content="You can optionally upload a file to provide context (skip if not needed)",
            accept=["text/plain"],
            max_size_mb=20,
            timeout=180,
        ).send()
        
        if files and len(files) > 0:
            file = files[0]
            msg = cl.Message(content=f"Processing file `{file.name}` done. Using it for context.")
            await msg.send()
            
            with open(file.path, "r", encoding="utf-8") as f:
                text = f.read()
            
            return text
        else:
            return "No file uploaded."
    except Exception as e:
        print(f"Error processing file upload: {e}")
        return "Error processing file upload."


# Custom OpenAI client that works with AutoGen - simplified for modern models
class CustomOpenAIClient:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        
        # Load configuration from YAML
        try:
            with open("model_config.yaml", "r") as f:
                config = yaml.safe_load(f)
                
            if isinstance(config, dict) and "config" in config:
                config = config["config"]
                
            # Set default model
            self.model = config.get("model", "o1")
            self.max_tokens = config.get("max_tokens", 1000)
            
            # Check for Azure configuration
            self.is_azure = "azure_endpoint" in config
            self.azure_endpoint = config.get("azure_endpoint", None)
            self.api_version = config.get("api_version", None)
            
        except Exception as e:
            print(f"Error loading config: {e}")
            # Use o1 as fallback
            self.model = "o1"
            self.max_tokens = 1000
            self.is_azure = False
        
        # Add model_info attribute required by AutoGen
        self.model_info = {
            "vision": False,
            "context_length": 128000,
            "max_tokens": self.max_tokens,
            "model": self.model
        }
        
        # Create the OpenAI client
        if self.is_azure and self.azure_endpoint and self.api_version:
            # Azure OpenAI
            self.client = openai.AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version
            )
        else:
            # Regular OpenAI
            self.client = openai.OpenAI(api_key=self.api_key)
            
        print(f"Initialized custom client with model: {self.model}")
    
    def _filter_kwargs(self, kwargs):
        """Filter out kwargs that are not supported by the OpenAI API"""
        # List of parameters that should be removed before passing to the OpenAI API
        unsupported_params = [
            'cancellation_token', 
            'api_key',
            'request_timeout',
            'functions',
            'function_call',
            'max_tokens',
            'temperature',
            'tools',
            'tool_choice'
        ]
        
        # Create a new dict with only supported parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in unsupported_params}
        return filtered_kwargs
        
    async def create(self, messages, **kwargs):
        """Compatible with AutoGen's expected interface for non-streaming calls"""
        try:
            # Convert messages format if needed and filter kwargs
            formatted_messages = self._format_messages(messages)
            filtered_kwargs = self._filter_kwargs(kwargs)
            
            # Create parameters dictionary with only supported parameters
            if self.is_azure:
                params = {
                    "model": self.model,
                    "messages": formatted_messages,
                    "max_tokens": self.max_tokens,
                    **filtered_kwargs
                }
            else:
                params = {
                    "model": self.model,
                    "messages": formatted_messages,
                    "max_completion_tokens": self.max_tokens,
                    **filtered_kwargs
                }
            
            # Use the new OpenAI Python client syntax
            response = self.client.chat.completions.create(**params)
            
            # Convert response to the format expected by AutoGen
            return self._convert_response(response)
            
        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
            traceback.print_exc()
            raise
    
    def _format_messages(self, messages):
        """Ensure messages are in the right format for the OpenAI API"""
        # Convert messages to the format expected by the OpenAI API
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                # Only include role and content fields
                formatted_msg = {
                    'role': msg['role'],
                    'content': msg['content']
                }
                formatted_messages.append(formatted_msg)
            else:
                # Try to handle other formats
                print(f"Warning: Unexpected message format: {msg}")
                # Use a default format as fallback
                formatted_messages.append({
                    'role': 'user',
                    'content': str(msg)
                })
        return formatted_messages
    
    def _convert_response(self, response):
        """Convert the response from the new OpenAI format to the format expected by AutoGen"""
        try:
            # Create a dictionary structure that mimics the old response format
            return {
                'choices': [
                    {
                        'message': {
                            'content': response.choices[0].message.content,
                            'role': response.choices[0].message.role
                        },
                        'index': 0,
                        'finish_reason': response.choices[0].finish_reason
                    }
                ],
                'model': response.model,
                'id': response.id,
                'created': response.created
            }
        except Exception as e:
            print(f"Error converting response: {e}")
            # Return a minimal valid response as fallback
            return {
                'choices': [
                    {
                        'message': {
                            'content': 'Error processing response',
                            'role': 'assistant'
                        },
                        'index': 0,
                        'finish_reason': 'error'
                    }
                ],
                'model': self.model,
                'id': 'error',
                'created': 0
            }

##############################
# 2. Agent System Prompts
##############################

def get_agent_prompts(task_type, context=None):
    """Get the appropriate system prompts based on the task type"""
    
    prompts = {}
    
    # Default prompts for general assistance
    prompts["planner"] = f"""
    You are a "PlannerAgent" who coordinates the overall approach to solving problems.
    
    Your job is to:
    1. Analyze the user's request in detail
    2. Break down complex problems into logical steps
    3. Determine which agents should handle which parts of the task
    4. Synthesize information from other agents into a cohesive plan
    5. Provide clear directions to other agents
    
    Additional context: {context if context else 'No additional context provided'}
    """
    
    prompts["researcher"] = """
    You are a "ResearcherAgent" who excels at gathering and analyzing information.
    
    Your job is to:
    1. Search for relevant facts, data, and context
    2. Analyze information for patterns and insights
    3. Challenge assumptions with evidence
    4. Provide comprehensive and accurate information
    5. Cite sources when possible
    """
    
    prompts["critic"] = """
    You are a "CriticAgent" who evaluates ideas and identifies potential issues.
    
    Your job is to:
    1. Analyze proposals for logical flaws or weaknesses
    2. Identify potential edge cases or failure modes
    3. Suggest improvements to strengthen ideas
    4. Ensure all requirements are being met
    5. Play devil's advocate to test robustness of plans
    """
    
    prompts["executor"] = """
    You are an "ExecutorAgent" who implements solutions and provides concrete deliverables.
    
    Your job is to:
    1. Transform plans into specific, actionable steps
    2. Provide detailed implementation guidance
    3. Create polished, final outputs based on requirements
    4. Focus on practical, efficient solutions
    5. Ensure quality and completeness of deliverables
    """
    
    # Specialized prompts for specific task types
    if task_type.lower() == "marketing":
        prompts["planner"] = f"""
        You are a "MarketingStrategyAgent" who develops comprehensive marketing strategies.
        
        Your job is to:
        1. Analyze the target audience and market positioning
        2. Develop messaging strategies and campaign concepts
        3. Determine the best marketing channels and approaches
        4. Set clear marketing objectives and KPIs
        5. Create an integrated marketing plan with timeline
        
        Additional context: {context if context else 'No additional context provided'}
        """
        
        prompts["researcher"] = """
        You are a "MarketResearchAgent" who analyzes market trends and audience insights.
        
        Your job is to:
        1. Research target demographics and psychographics
        2. Analyze competitor strategies and positioning
        3. Identify market gaps and opportunities
        4. Provide data-backed audience insights
        5. Recommend targeting strategies based on research
        """
        
        prompts["critic"] = """
        You are a "BrandConsistencyAgent" who ensures marketing aligns with brand values.
        
        Your job is to:
        1. Evaluate if messaging maintains brand voice and values
        2. Check that visuals and tone are on-brand
        3. Identify potential brand perception issues
        4. Ensure consistency across all marketing touchpoints
        5. Flag any brand dilution risks
        """
        
        prompts["executor"] = """
        You are a "CreativeContentAgent" who creates compelling marketing content.
        
        Your job is to:
        1. Craft engaging copy for various marketing channels
        2. Develop creative concepts that align with strategy
        3. Create messaging that resonates with target audiences
        4. Design calls-to-action that drive conversions
        5. Produce polished final content ready for deployment
        """
    
    elif task_type.lower() == "coding":
        prompts["planner"] = f"""
        You are a "SystemArchitectAgent" who designs software architecture and systems.
        
        Your job is to:
        1. Analyze technical requirements and constraints
        2. Design scalable, maintainable software architecture
        3. Break down complex systems into components and modules
        4. Define interfaces, data models, and system interactions
        5. Create technical specifications for implementation
        
        Additional context: {context if context else 'No additional context provided'}
        """
        
        prompts["researcher"] = """
        You are a "TechnicalResearchAgent" who investigates technical solutions and best practices.
        
        Your job is to:
        1. Research relevant technologies, libraries, and frameworks
        2. Evaluate technical approaches and their trade-offs
        3. Find example code and documentation references
        4. Analyze performance characteristics and limitations
        5. Identify potential technical challenges and solutions
        """
        
        prompts["critic"] = """
        You are a "CodeReviewAgent" who evaluates code quality and identifies issues.
        
        Your job is to:
        1. Review code for bugs, edge cases, and potential errors
        2. Evaluate code maintainability and readability
        3. Identify performance bottlenecks and optimizations
        4. Check for security vulnerabilities and best practices
        5. Suggest improvements to code structure and design
        """
        
        prompts["executor"] = """
        You are a "DeveloperAgent" who implements code solutions.
        
        Your job is to:
        1. Write clean, efficient, well-documented code
        2. Implement features according to specifications
        3. Create unit tests and handle edge cases
        4. Refactor and optimize existing code
        5. Produce production-ready implementations
        """
    
    elif task_type.lower() == "analysis":
        prompts["planner"] = f"""
        You are a "DataStrategyAgent" who plans analytical approaches to problems.
        
        Your job is to:
        1. Define the analytical problem statement and objectives
        2. Design the analytical approach and methodology
        3. Identify required data sources and transformations
        4. Determine appropriate analytical techniques
        5. Create a structured plan for the analysis process
        
        Additional context: {context if context else 'No additional context provided'}
        """
        
        prompts["researcher"] = """
        You are a "DataGatheringAgent" who collects and prepares data for analysis.
        
        Your job is to:
        1. Identify relevant data sources and variables
        2. Assess data quality and completeness
        3. Recommend data cleaning and transformation steps
        4. Structure data for effective analysis
        5. Document data limitations and assumptions
        """
        
        prompts["critic"] = """
        You are a "DataValidationAgent" who ensures analytical rigor and correctness.
        
        Your job is to:
        1. Validate analytical methods and assumptions
        2. Check for statistical errors or misinterpretations
        3. Identify potential biases in the analysis
        4. Ensure conclusions are supported by the data
        5. Test the robustness of analytical findings
        """
        
        prompts["executor"] = """
        You are a "DataAnalystAgent" who performs analysis and generates insights.
        
        Your job is to:
        1. Execute data analysis following the defined methodology
        2. Create visualizations that effectively communicate findings
        3. Extract meaningful insights from analytical results
        4. Translate technical findings into business implications
        5. Present clear, actionable conclusions
        """
        
    return prompts

##############################
# 3. Chainlit lifecycle
##############################
@cl.on_chat_start  # type: ignore
async def start_chat() -> None:
    # Display welcome message
    welcome_msg = """
    # 🚀 Welcome to the Multi-Agent AI Playground!
    
    This system uses multiple specialized AI agents working together to help solve complex problems.
    
    ## How it works:
    1. You'll select a type of task you need help with
    2. You can optionally upload a file to provide context
    3. A team of specialized AI agents will collaborate to assist you
    4. You can interact with the agents throughout the process
    
    Let's get started!
    """
    
    await cl.Message(content=welcome_msg).send()

    # 1. Load environment variables
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')

    # 2. Get task type from user
    task_type = await select_task_type()
    
    # 3. Get optional file upload
    context = await upload_optional_file()
    
    # 4. Display task confirmation
    await cl.Message(content=f"Setting up a team of agents to help with your **{task_type}** task...").send()

    # 5. Create our custom client
    model_client = CustomOpenAIClient()
    
    # 6. Get appropriate system prompts based on task type
    prompts = get_agent_prompts(task_type, context)
    
    # 7. Create the agents
    planner_agent = AssistantAgent(
        name="planner",
        model_client=model_client,
        system_message=prompts["planner"],
        model_client_stream=False,  # Disable streaming
    )
    
    researcher_agent = AssistantAgent(
        name="researcher",
        model_client=model_client,
        system_message=prompts["researcher"],
        model_client_stream=False,
    )
    
    critic_agent = AssistantAgent(
        name="critic",
        model_client=model_client,
        system_message=prompts["critic"],
        model_client_stream=False,
    )
    
    executor_agent = AssistantAgent(
        name="executor",
        model_client=model_client,
        system_message=prompts["executor"],
        model_client_stream=False,
    )
    
    # 8. Create the user proxy agent
    user_agent = UserProxyAgent(
        name="user",
        input_func=user_input_text,  # User provides text input
    )

    # 9. Add termination condition: if user says "DONE", conversation ends.
    termination_condition = TextMentionTermination("DONE", sources=["user"])

    # 10. Create the group chat with all agents
    group_chat = RoundRobinGroupChat(
        [planner_agent, researcher_agent, critic_agent, executor_agent, user_agent],
        termination_condition=termination_condition,
    )

    # Save the team in user session
    cl.user_session.set("team", group_chat)  # type: ignore
    
    # Final setup message
    await cl.Message(content=f"""
    **Agent team is ready!** You are now working with:
    
    - **Planner**: Coordinates the overall approach
    - **Researcher**: Gathers and analyzes information  
    - **Critic**: Evaluates ideas and identifies issues
    - **Executor**: Implements solutions and provides deliverables
    
    Please describe what you'd like help with, and the agents will start collaborating.
    When you're finished, just say **DONE**.
    """).send()


@cl.set_starters  # type: ignore
async def set_starters() -> List[cl.Starter]:
    """Starter messages to help users begin interactions"""
    return [
        cl.Starter(
            label="Marketing Campaign",
            message="I need help creating a marketing campaign for our new product launch. The target audience is millennials who are health-conscious.",
        ),
        cl.Starter(
            label="Code Solution",
            message="I need help designing and implementing a function that processes JSON data from an API and transforms it into a specific format.",
        ),
        cl.Starter(
            label="Data Analysis",
            message="I need to analyze customer purchase data to identify trends and make recommendations for product improvements.",
        ),
        cl.Starter(
            label="General Problem",
            message="I need help brainstorming solutions for [describe your problem here].",
        ),
    ]

@cl.on_message  # type: ignore
async def chat(message: cl.Message) -> None:
    """
    Main conversation loop. Each user message is fed to the team
    for processing until termination.
    """
    # Retrieve the team from user session
    team = cast(RoundRobinGroupChat, cl.user_session.get("team"))  # type: ignore

    # Add a safety check
    if team is None:
        await cl.Message(content="The agent team was not properly initialized. Please refresh and try again.").send()
        return

    # Since streaming is disabled, we just collect and display complete messages
    async for msg in team.run_stream(
        task=[TextMessage(content=message.content, source="user")],
        cancellation_token=CancellationToken(),
    ):
        # If the conversation ended
        if isinstance(msg, TaskResult):
            final_message = "Task completed. "
            if msg.stop_reason:
                final_message += msg.stop_reason
            await cl.Message(content=final_message).send()
        
        # Model outputs (non-streaming)
        elif isinstance(msg, ModelClientStreamingChunkEvent):
            # Send the complete message
            await cl.Message(content=msg.content, author=msg.source).send()
        
        # Skip other message types
        else:
            continue