import os
import re
import json
import asyncio
from typing import List, Dict, Any, Optional
import chainlit as cl
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model_name = "o1"

# Agent roles and capabilities
AGENT_ROLES = {
    "creative_director": {
        "emoji": "🎭",
        "name": "Creative Director",
        "description": "Comes up with innovative, out-of-the-box ideas and concepts"
    },
    "storyteller": {
        "emoji": "📚",
        "name": "Storyteller",
        "description": "Crafts engaging narratives and develops characters"
    },
    "visual_designer": {
        "emoji": "🎨",
        "name": "Visual Designer",
        "description": "Creates visual concepts and designs using descriptive language"
    },
    "critic": {
        "emoji": "🧠",
        "name": "Critic",
        "description": "Provides constructive feedback and suggests improvements"
    },
    "implementer": {
        "emoji": "⚙️",
        "name": "Implementer",
        "description": "Turns ideas into actionable steps and concrete outputs"
    }
}

# Global trackers
session_agents = {}
current_project = {}

def parse_agent_message(text: str) -> Dict:
    """Extract JSON content from agent's message."""
    try:
        # Try to extract JSON if enclosed in ```json ... ``` or just parse the whole text
        json_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = text
            
        # Parse the JSON
        parsed = json.loads(json_str)
        return parsed
    except json.JSONDecodeError:
        # If JSON parsing fails, return a default structure
        return {
            "thoughts": text,
            "message": "I couldn't format my response properly. Please let me try again."
        }

async def query_agent(role: str, prompt: str, project_context: Dict) -> Dict:
    """Query an agent with a specific role."""
    # Construct the system message based on the agent's role
    
    # Process reference materials if available
    reference_materials_text = ""
    if "reference_materials" in project_context and project_context["reference_materials"]:
        reference_materials_text = "Reference materials:\n\n"
        for material in project_context["reference_materials"]:
            reference_materials_text += f"===== {material['filename']} =====\n{material['content']}\n\n"
    
    system_message = f"""You are the {AGENT_ROLES[role]['name']} ({AGENT_ROLES[role]['emoji']}) on a creative project team.
    
Your responsibility is: {AGENT_ROLES[role]['description']}

Always respond in JSON format with the following structure:
```json
{{
  "thoughts": "Your internal reasoning and creative process (not shown to the user)",
  "message": "Your contribution to the project, addressing the prompt"
}}
```

Current project context:
{json.dumps(project_context, indent=2)}

{reference_materials_text}

Be creative, specific, and constructive in your response. If reference materials were provided, use them for inspiration and to match the style and tone.
"""

    try:
        # Create a thinking message to show the agent is working
        thinking_msg = cl.Message(
            content=f"{AGENT_ROLES[role]['emoji']} {AGENT_ROLES[role]['name']} is thinking...",
            author=AGENT_ROLES[role]['name']
        )
        await thinking_msg.send()
        
        # Call OpenAI with the agent's role and prompt
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse the JSON response
        parsed_response = parse_agent_message(response.choices[0].message.content)
        
        # Create a new message instead of updating
        await cl.Message(
            content=parsed_response["message"],
            author=AGENT_ROLES[role]['name']
        ).send()
        
        return parsed_response
    except Exception as e:
        print(f"Error querying {role} agent: {str(e)}")
        # Send a new error message instead of updating
        await cl.Message(
            content=f"Error: {str(e)}",
            author=AGENT_ROLES[role]['name']
        ).send()
        return {
            "thoughts": f"Error occurred: {str(e)}",
            "message": f"I encountered an error while processing your request. Please try again."
        }

async def creative_round_robin(prompt: str, project_context: Dict) -> List[Dict]:
    """Run a round robin of all agents on the creative team."""
    results = []
    
    for role in AGENT_ROLES:
        if role in session_agents:
            result = await query_agent(role, prompt, project_context)
            results.append({
                "role": role,
                "result": result
            })
    
    return results

async def synthesize_responses(results: List[Dict], project_context: Dict) -> Dict:
    """Have the Creative Director synthesize all agent responses."""
    # Compile all agent responses
    agent_responses = "\n\n".join([
        f"{AGENT_ROLES[r['role']]['emoji']} {AGENT_ROLES[r['role']]['name']}:\n{r['result']['message']}"
        for r in results
    ])
    
    # Process reference materials if available
    reference_materials_text = ""
    if "reference_materials" in project_context and project_context["reference_materials"]:
        reference_materials_text = "Reference materials for style and tone guidance:\n\n"
        for material in project_context["reference_materials"]:
            reference_materials_text += f"===== {material['filename']} =====\n{material['content']}\n\n"
    
    # Ask the Creative Director to synthesize
    system_message = f"""You are the Creative Director synthesizing the team's ideas into a cohesive plan.

The team has responded to a creative challenge. Review their contributions and create a unified vision.

Current project context:
{json.dumps(project_context, indent=2)}

Team responses:
{agent_responses}

{reference_materials_text}

Provide a synthesized perspective that incorporates the best ideas from each team member.
If reference materials were provided, make sure the creative direction matches their style and tone.

Respond in JSON format with the following structure:
```json
{{
  "thoughts": "Your internal process for synthesizing these ideas",
  "summary": "A brief summary of the key insights",
  "synthesis": "A detailed synthesis of the team's ideas into a cohesive creative direction",
  "next_steps": "Suggested next steps for the project"
}}
```
"""

    try:
        # Show a thinking message
        thinking_msg = cl.Message(
            content="🔄 Synthesizing team insights...",
            author="Creative Team"
        )
        await thinking_msg.send()
        
        # Call OpenAI to synthesize
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": "Please synthesize these responses into a cohesive creative direction."}
            ]
        )
        
        # Extract JSON if present
        try:
            # Try to extract JSON content
            json_match = re.search(r'```json\n(.*?)\n```', response.choices[0].message.content, re.DOTALL)
            if json_match:
                synthesis = json.loads(json_match.group(1))
            else:
                # Attempt to parse the whole response as JSON
                synthesis = json.loads(response.choices[0].message.content)
                
            # Send a new message with the synthesis
            await cl.Message(
                content=f"""## Team Synthesis

### Summary
{synthesis.get('summary', 'No summary provided.')}

### Creative Direction
{synthesis.get('synthesis', 'No synthesis provided.')}

### Next Steps
{synthesis.get('next_steps', 'No next steps provided.')}
""",
                author="Creative Team"
            ).send()
            
            return synthesis
        except json.JSONDecodeError:
            # If JSON parsing fails, send a new message with the raw text
            await cl.Message(
                content=f"""## Team Synthesis

{response.choices[0].message.content}
""",
                author="Creative Team"
            ).send()
            return {
                "thoughts": "Failed to parse JSON",
                "summary": "Parsing error",
                "synthesis": response.choices[0].message.content,
                "next_steps": "Review the synthesis and proceed with the project."
            }
            
    except Exception as e:
        print(f"Error synthesizing responses: {str(e)}")
        # Send a new error message
        await cl.Message(
            content=f"Error synthesizing responses: {str(e)}",
            author="Creative Team"
        ).send()
        return {
            "thoughts": f"Error occurred: {str(e)}",
            "summary": "Error occurred",
            "synthesis": "There was an error synthesizing the team's responses.",
            "next_steps": "Please try again."
        }

# File upload handling function
async def handle_file_upload():
    """Process uploaded files as reference material for the creative project."""
    try:
        files = await cl.AskFileMessage(
            content="You can upload files like templates or reference material to help guide the creative process.",
            accept=["text/plain", ".txt", ".md"],
            max_size_mb=10,
            timeout=180,
        ).send()
        
        if files and len(files) > 0:
            file_contents = []
            
            for file in files:
                try:
                    with open(file.path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    file_contents.append({
                        "filename": file.name,
                        "content": content
                    })
                    
                    await cl.Message(content=f"Processed file: `{file.name}`").send()
                except Exception as e:
                    await cl.Message(content=f"Error reading file `{file.name}`: {str(e)}").send()
            
            if file_contents:
                # Update project context with file contents
                current_project["reference_materials"] = file_contents
                
                # Create a summary message of uploaded materials
                summary = "\n\n".join([
                    f"**{item['filename']}**\n```\n{item['content'][:500]}{'...' if len(item['content']) > 500 else ''}\n```"
                    for item in file_contents
                ])
                
                await cl.Message(
                    content=f"""## Reference Materials Uploaded

The team will use these materials as inspiration and reference:

{summary}
"""
                ).send()
                
                return True
            
        return False
    except Exception as e:
        print(f"Error handling file upload: {str(e)}")
        await cl.Message(content=f"Error processing file upload: {str(e)}").send()
        return False

# Chainlit callbacks
@cl.on_chat_start
async def on_chat_start():
    # Welcome message
    welcome_message = """# 🌟 Creative AI Collaborative Studio

Welcome to your AI-powered creative team! This collaborative studio connects you with specialized AI agents that work together to help bring your creative ideas to life.

## Meet Your Creative Team:
- 🎭 **Creative Director**: Generates innovative concepts and coordinates the team
- 📚 **Storyteller**: Crafts engaging narratives and develops characters
- 🎨 **Visual Designer**: Creates visual concepts using descriptive language
- 🧠 **Critic**: Provides constructive feedback and suggests improvements
- ⚙️ **Implementer**: Turns ideas into actionable steps and concrete outputs

## How to Begin:
1. **Start by selecting which team members you want to collaborate with**
2. **You can upload reference materials or templates**
3. **Share your creative challenge or project idea**
4. **The team will collaborate and provide a synthesized response**
5. **Continue the conversation to refine and develop your creative project**

Let's begin by setting up your creative team!
"""
    await cl.Message(content=welcome_message).send()
    
    # Initialize project context
    current_project.clear()
    current_project.update({
        "title": "New Creative Project",
        "description": "No description yet",
        "stage": "initiation",
        "history": [],
        "reference_materials": []
    })
    
    # Ask user to select team members using the new element-based approach
    await cl.Message(content="Select which team members you want to work with:").send()
    
    # Create individual action buttons for each role
    for role, info in AGENT_ROLES.items():
        await cl.Message(
            content=f"{info['emoji']} **{info['name']}**: {info['description']}",
            actions=[
                cl.Action(name=role, value=role, label=f"Add to team", payload={"role": role})
            ]
        ).send()
    
    # Add upload button for reference materials
    await cl.Message(
        content="📎 **Upload reference materials or templates** to help guide the creative process.",
        actions=[
            cl.Action(name="upload_file", value="upload", label="Upload Files", payload={"action": "upload"})
        ]
    ).send()

@cl.action_callback("creative_director")
@cl.action_callback("storyteller")
@cl.action_callback("visual_designer")
@cl.action_callback("critic")
@cl.action_callback("implementer")
@cl.action_callback("upload_file")
async def on_action(action):
    # Handle file upload action
    if action.name == "upload_file":
        await handle_file_upload()
        return
        
    # Handle selecting team members
    role = action.name
    if role in AGENT_ROLES:
        if role in session_agents:
            # Remove agent if already selected
            session_agents.pop(role)
            await cl.Message(content=f"Removed {AGENT_ROLES[role]['emoji']} {AGENT_ROLES[role]['name']} from your team.").send()
        else:
            # Add agent to team
            session_agents[role] = AGENT_ROLES[role]
            await cl.Message(content=f"Added {AGENT_ROLES[role]['emoji']} {AGENT_ROLES[role]['name']} to your team.").send()
        
        # Show current team
        if session_agents:
            team_list = "\n".join([f"- {info['emoji']} **{info['name']}**: {info['description']}" 
                                 for role, info in session_agents.items()])
            await cl.Message(content=f"""## Your Creative Team

{team_list}

You can upload reference materials or templates using the upload button below, or proceed directly to describing your creative project.
""").send()
        else:
            await cl.Message(content="You haven't selected any team members yet. Please select at least one to continue.").send()

@cl.on_message
async def on_message(message: cl.Message):
    # Check if user has selected agents
    if not session_agents:
        await cl.Message(content="Please select team members first by clicking on the options above.").send()
        return
    
    # Make sure Creative Director is always part of the team for synthesis
    if "creative_director" not in session_agents:
        session_agents["creative_director"] = AGENT_ROLES["creative_director"]
        await cl.Message(content=f"Added {AGENT_ROLES['creative_director']['emoji']} {AGENT_ROLES['creative_director']['name']} to your team to help coordinate.").send()
    
    # Update project context with the new message
    if current_project["stage"] == "initiation":
        current_project["description"] = message.content
        current_project["stage"] = "development"
    
    # Add the user message to project history
    current_project["history"].append({
        "role": "user",
        "content": message.content
    })
    
    # Run the creative round robin
    results = await creative_round_robin(message.content, current_project)
    
    # Synthesize the responses
    synthesis = await synthesize_responses(results, current_project)
    
    # Update project history with the synthesis
    current_project["history"].append({
        "role": "team",
        "content": synthesis.get("synthesis", "")
    })
    
    # Prompt user for next steps
    next_steps_prompt = """
## What would you like to do next?

- **Refine the concept**: Provide feedback or ask questions to develop the ideas further
- **Focus on a specific aspect**: Ask one of the team members to elaborate on a particular element
- **Move to implementation**: Request concrete steps or deliverables
- **Start a new creative challenge**: Describe a completely new project

Just respond with your thoughts, questions, or instructions to continue.
"""
    await cl.Message(content=next_steps_prompt).send()

if __name__ == "__main__":
    # For local testing without the Chainlit server
    print("This script is designed to run with Chainlit. Run 'chainlit run o1_creative_agent.py'")