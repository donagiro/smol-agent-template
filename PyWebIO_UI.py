## smol-agent-template\PyWebIO_UI.py
from typing import Optional, List
from pywebio import start_server, config
from pywebio.output import *
from pywebio.input import *
from pywebio.session import run_js, set_env
from pywebio.pin import *
from smolagents.agent_types import AgentAudio, AgentImage, AgentText, handle_agent_output_types
from smolagents.agents import ActionStep, MultiStepAgent
from smolagents.memory import MemoryStep
from smolagents.utils import _is_package_available
import mimetypes
import os
import re
import shutil

class PyWebIOUI:
    """A PyWebIO-based interface for the agent"""
    
    def __init__(self, agent: MultiStepAgent, file_upload_folder: str | None = None):
        if not _is_package_available("pywebio"):
            raise ModuleNotFoundError(
                "Please install 'pywebio' to use the PyWebIOUI: `pip install pywebio`"
            )
        self.agent = agent
        self.file_upload_folder = file_upload_folder
        if self.file_upload_folder is not None and not os.path.exists(file_upload_folder):
            os.makedirs(file_upload_folder)
        
        self.chat_history = []
        self.file_uploads = []
    
    def format_step_output(self, step_log: MemoryStep) -> List[str]:
        """Format agent steps for display in the chat"""
        output = []
        
        if isinstance(step_log, ActionStep):
            # Output the step number
            step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else ""
            if step_number:
                output.append(f"**{step_number}**")
            
            # First output the thought/reasoning from the LLM
            if hasattr(step_log, "model_output") and step_log.model_output is not None:
                model_output = step_log.model_output.strip()
                model_output = re.sub(r"```\s*<end_code>", "```", model_output)
                model_output = re.sub(r"<end_code>\s*```", "```", model_output)
                model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)
                model_output = model_output.strip()
                output.append(model_output)
            
            # Handle tool calls
            if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
                first_tool_call = step_log.tool_calls[0]
                used_code = first_tool_call.name == "python_interpreter"
                
                # Handle arguments based on type
                args = first_tool_call.arguments
                if isinstance(args, dict):
                    content = str(args.get("answer", str(args)))
                else:
                    content = str(args).strip()
                
                if used_code:
                    content = re.sub(r"```.*?\n", "", content)
                    content = re.sub(r"\s*<end_code>\s*", "", content)
                    content = content.strip()
                    if not content.startswith("```python"):
                        content = f"```python\n{content}\n```"
                
                output.append(f"🛠️ Used tool {first_tool_call.name}\n{content}")
                
                # Handle execution logs
                if hasattr(step_log, "observations") and step_log.observations and step_log.observations.strip():
                    log_content = step_log.observations.strip()
                    log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                    output.append(f"📝 Execution Logs\n{log_content}")
                
                # Handle errors
                if hasattr(step_log, "error") and step_log.error is not None:
                    output.append(f"💥 Error\n{str(step_log.error)}")
            
            # Handle standalone errors
            elif hasattr(step_log, "error") and step_log.error is not None:
                output.append(f"💥 Error\n{str(step_log.error)}")
            
            # Add step footer with stats
            step_footnote = f"{step_number}"
            if hasattr(step_log, "input_token_count") and hasattr(step_log, "output_token_count"):
                step_footnote += f" | Input-tokens:{step_log.input_token_count:,} | Output-tokens:{step_log.output_token_count:,}"
            if hasattr(step_log, "duration"):
                step_footnote += f" | Duration: {round(float(step_log.duration), 2)}" if step_log.duration else ""
            
            output.append(f"<small>{step_footnote}</small>")
            output.append("-----")
        
        return output
    
    def run_agent(self, task: str):
        """Run the agent with the given task and stream the output"""
        total_input_tokens = 0
        total_output_tokens = 0
        
        for step_log in self.agent.run(task, stream=True, reset=False):
            # Track tokens if model provides them
            if hasattr(self.agent.model, "last_input_token_count"):
                total_input_tokens += self.agent.model.last_input_token_count
                total_output_tokens += self.agent.model.last_output_token_count
                if isinstance(step_log, ActionStep):
                    step_log.input_token_count = self.agent.model.last_input_token_count
                    step_log.output_token_count = self.agent.model.last_output_token_count
            
            # Format and display each step
            for message in self.format_step_output(step_log):
                self.chat_history.append(("assistant", message))
                put_markdown(message, scope="chat_area")
                scroll_to("chat_area", "bottom")
            
            yield step_log
        
        # Handle final answer
        final_answer = step_log  # Last log is the run's final_answer
        final_answer = handle_agent_output_types(final_answer)
        
        if isinstance(final_answer, AgentText):
            answer = f"**Final answer:**\n{final_answer.to_string()}\n"
        elif isinstance(final_answer, AgentImage):
            answer = f"**Final answer:** [Image: {final_answer.to_string()}]"
        elif isinstance(final_answer, AgentAudio):
            answer = f"**Final answer:** [Audio: {final_answer.to_string()}]"
        else:
            answer = f"**Final answer:** {str(final_answer)}"
        
        self.chat_history.append(("assistant", answer))
        put_markdown(answer, scope="chat_area")
        scroll_to("chat_area", "bottom")
    
    def handle_file_upload(self, file):
        """Handle file uploads"""
        if not file:
            return toast("No file uploaded", color='error')
        
        try:
            mime_type, _ = mimetypes.guess_type(file['filename'])
        except Exception as e:
            return toast(f"Error: {e}", color='error')
        
        allowed_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ]
        
        if mime_type not in allowed_types:
            return toast("File type not allowed", color='error')
        
        # Sanitize filename
        original_name = file['filename']
        sanitized_name = re.sub(r"[^\w\-.]", "_", original_name)
        sanitized_name = sanitized_name.split(".")[0] + mimetypes.guess_extension(mime_type)
        
        # Save file
        file_path = os.path.join(self.file_upload_folder, sanitized_name)
        with open(file_path, 'wb') as f:
            f.write(file['content'])
        
        self.file_uploads.append(file_path)
        toast(f"File uploaded: {sanitized_name}", color='success')
        return file_path
    
    def chat_app(self):
        """Main chat application"""
        set_env(title="Agent Chat", output_max_width="1000px")
        
        # Header
        put_html("<h1>Agent Chat</h1>")
        
        # Chat area
        put_scrollable(put_scope("chat_area"), height=400, keep_bottom=True)
        
        # Input area
        with use_scope("input_area"):
            if self.file_upload_folder is not None:
                file = file_upload("Upload file", accept=".pdf,.docx,.txt")
                if file:
                    self.handle_file_upload(file)
            
            while True:
                user_input = textarea("Your message", rows=3, required=True)
                
                # Add user message to chat
                self.chat_history.append(("user", user_input))
                put_markdown(f"**You:** {user_input}", scope="chat_area")
                scroll_to("chat_area", "bottom")
                
                # Include file info if available
                if self.file_uploads:
                    user_input += f"\nAttached files: {', '.join(self.file_uploads)}"
                
                # Run agent and display responses
                for _ in self.run_agent(user_input):
                    pass
                
                # Clear file uploads after processing
                self.file_uploads = []
    
    def launch(self, port=8080, debug=False, **kwargs):
        """Launch the PyWebIO application"""
        config(title="Agent Chat", description="Chat with your AI agent")
        start_server(self.chat_app, port=port, debug=debug, **kwargs)