"""
Gradio UI module for LLM Council
Provides the web interface for interacting with the debate system
"""

import gradio as gr
from typing import List, Optional

from config import get_config
from orchestrator import DebateOrchestrator, format_debate_output


class CouncilUI:
    """Manages the Gradio interface for the LLM Council system."""
    
    def __init__(self):
        """Initialize the UI with orchestrator."""
        self.config = get_config()
        self.orchestrator = DebateOrchestrator(self.config)
    
    async def process_query(
        self, 
        user_query: str, 
        history: Optional[List] = None
    ):
        """
        Process user query through the debate workflow.
        
        Args:
            user_query: The user's question
            history: Chat history (unused but required by Gradio)
            
        Yields:
            Formatted debate output or error message
        """
        if not user_query.strip():
            yield "Please enter a question or topic for the council to debate."
            return
        
        try:
            # Show processing message
            yield " Initializing LLM Council...\n\n Round 1: Council members analyzing independently..."
            
            # Conduct the debate
            responses = await self.orchestrator.conduct_debate(user_query)
            
            # Format and return results
            formatted_output = format_debate_output(responses, self.config)
            yield formatted_output
            
        except Exception as e:
            error_message = f""" Error: {str(e)}

**Troubleshooting Steps:**
1. Check your Azure configuration in `.env` file
2. Ensure you're logged in via Azure CLI: `az login`
3. Verify your model deployment names in Azure AI Foundry portal
4. Check Azure service health and quotas

**Configuration Status:**
- Endpoint: {self.config.project_endpoint}
- Member A Model: {self.config.member_a.deployment_name}
- Member B Model: {self.config.member_b.deployment_name}
- Chair Model: {self.config.chair.deployment_name}
- Authentication: {"API Key" if self.config.api_key else "Azure CLI"}
"""
            yield error_message
    
    def create_interface(self) -> gr.Blocks:
        """
        Create and configure the Gradio interface.
        
        Returns:
            Configured Gradio Blocks interface
        """
        with gr.Blocks(title="LLM Council: Chain-of-Debate") as demo:
            # Header
            gr.Markdown(f"""
#  LLM Council: Chain-of-Debate System

Ask complex questions and watch three AI models debate through structured rounds:

###  Council Members
- **{self.config.member_a.display_name}** - Analytical perspective
- **{self.config.member_b.display_name}** - Critical perspective  
- **{self.config.chair.display_name}** - Consensus evaluator

###  Debate Process
1. **Round 1**: Independent analysis from both council members
2. **Round 2**: Chain-of-debate where models critique each other
3. **Round 3**: Chair synthesizes consensus, debates, and final verdict

###  Authentication
**Current Mode:** {" API Key" if self.config.api_key else " Azure CLI (`az login`)"}

---
            """)
            
            # Chat interface
            chatbot = gr.Chatbot(label="Debate Output", height=600)
            
            # Input area
            msg = gr.Textbox(
                label="Your Question",
                placeholder="E.g., 'Should we invest in renewable energy stocks?' or 'Analyze the pros and cons of remote work'",
                lines=3
            )
            
            # Buttons
            with gr.Row():
                submit_btn = gr.Button(" Start Debate", variant="primary", scale=2)
                clear_btn = gr.Button(" Clear", scale=1)
            
            # Example queries
            gr.Markdown("###  Example Questions")
            gr.Examples(
                examples=[
                    "Should our company invest in AI infrastructure or cloud migration first?",
                    "Analyze the pros and cons of remote work vs. hybrid work models",
                    "What are the key considerations for implementing a microservices architecture?",
                    "Evaluate the best strategy for entering a new market with limited resources",
                    "Should we prioritize technical debt reduction or new feature development?"
                ],
                inputs=msg,
                label="Click to try:"
            )
            
            # Event handlers
            async def user_submit(user_message, history):
                """Handle user message submission."""
                history = history or []
                history.append({"role": "user", "content": user_message})
                
                # Process and get response
                current_response = ""
                async for response in self.process_query(user_message):
                    current_response = response
                
                history.append({"role": "assistant", "content": current_response})
                return "", history
            
            def clear_chat():
                """Clear the chat history."""
                return None, None
            
            # Wire up events
            submit_btn.click(
                user_submit,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            msg.submit(
                user_submit,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            clear_btn.click(
                clear_chat,
                outputs=[msg, chatbot]
            )
            
            # Footer
            gr.Markdown(f"""
---
###  Configuration
- **Timeout**: {self.config.timeout}s per agent
- **Council Temperature**: {self.config.member_a.temperature}
- **Chair Temperature**: {self.config.chair.temperature}

###  Tips
- Ask open-ended questions that benefit from multiple perspectives
- Complex decisions work best (strategy, analysis, recommendations)
- Debates typically take 60-90 seconds depending on complexity
            """)
        
        return demo


def create_gradio_interface() -> gr.Blocks:
    """
    Factory function to create the Gradio interface.
    
    Returns:
        Configured Gradio Blocks interface
    """
    ui = CouncilUI()
    return ui.create_interface()


def launch_ui(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False
):
    """
    Launch the Gradio UI.
    
    Args:
        server_name: Host to bind to
        server_port: Port to listen on
        share: Whether to create a public shareable link
    """
    print(" Starting LLM Council UI...")
    print(f" Server: http://{server_name}:{server_port}")
    print()
    
    demo = create_gradio_interface()
    demo.launch(server_name=server_name, server_port=server_port, share=share, ssr_mode=False)

