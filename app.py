"""
LLM Council: Chain-of-Debate System using Azure OpenAI

This application implements an LLM Council pattern where multiple AI models
debate through structured rounds to reach consensus on complex queries.
"""

import asyncio
import os
from typing import List, Dict
from dotenv import load_dotenv
import gradio as gr
from openai import AzureOpenAI


class LLMCouncil:
    """
    Orchestrates a multi-agent debate system with three distinct AI models:
    - Council Member A (GPT-4.1): Analytical perspective
    - Council Member B (DeepSeek-V3.1): Critical perspective  
    - Chair (Grok-3): Consensus evaluator and final decision maker
    """
    
    def __init__(self):
        """Initialize the LLM Council with environment configuration."""
        load_dotenv()
        
        # Validate required environment variables
        self.project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
        self.api_key = os.getenv("AZURE_AI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        self.model_gpt41 = os.getenv("MODEL_DEPLOYMENT_GPT41", "gpt-4.1")
        self.model_deepseek = os.getenv("MODEL_DEPLOYMENT_DEEPSEEK", "DeepSeek-V3.1")
        self.model_grok = os.getenv("MODEL_DEPLOYMENT_GROK", "grok-3")
        
        # Temperature settings
        self.temp_council = float(os.getenv("TEMPERATURE_COUNCIL_MEMBERS", "0.7"))
        self.temp_chair = float(os.getenv("TEMPERATURE_CHAIR", "0.3"))
        
        # Timeout setting
        self.timeout = int(os.getenv("AGENT_TIMEOUT", "60"))
        
        self._validate_config()
        
        # Agent system prompts
        self.prompts = {
            "member_a": """You are Council Member A, an analytical AI expert who provides thorough, 
            data-driven analysis. Your role is to:
            1. In Round 1: Provide your independent analysis of the query with clear reasoning
            2. In Round 2: Review Council Member B's perspective and either:
               - Challenge their assumptions with evidence
               - Agree and strengthen their points
               - Offer alternative viewpoints
            
            Be direct, analytical, and evidence-based in your responses.""",
            
            "member_b": """You are Council Member B, a critical thinking AI expert who questions 
            assumptions and explores edge cases. Your role is to:
            1. In Round 1: Provide your independent analysis focusing on risks, limitations, and alternatives
            2. In Round 2: Review Council Member A's perspective and either:
               - Point out overlooked considerations or flaws
               - Acknowledge strong points
               - Present contrarian views with reasoning
            
            Be skeptical, thorough, and constructively critical.""",
            
            "chair": """You are the Chair, the final decision maker who synthesizes the council's debate.
            
            Your role is to review the ENTIRE debate thread and produce a structured final verdict with:
            
            ## Areas of Complete Consensus
            List all points where both Council Members agreed, either initially or after debate.
            
            ## Key Debates  
            Highlight areas where the members disagreed. For each:
            - State the disagreement clearly
            - Summarize Member A's position and reasoning
            - Summarize Member B's position and reasoning
            
            ## The Verdict
            Based on the complete debate, provide your final decision/recommendation. Explain:
            - Which perspectives you're adopting and why
            - How you're resolving any disagreements
            - Your conclusive answer to the original query
            
            Be objective, balanced, and make clear final judgments."""
        }
    
    def _validate_config(self):
        """Validate that all required configuration is present."""
        if not self.project_endpoint:
            raise ValueError(
                "AZURE_AI_PROJECT_ENDPOINT not found. "
                "Please copy .env.template to .env and configure your Azure settings."
            )
        
        if not self.api_key:
            raise ValueError(
                "AZURE_AI_API_KEY not found. "
                "Please add your Azure AI Foundry API key to the .env file."
            )
        
        missing_models = []
        if not self.model_gpt41:
            missing_models.append("MODEL_DEPLOYMENT_GPT41")
        if not self.model_deepseek:
            missing_models.append("MODEL_DEPLOYMENT_DEEPSEEK")
        if not self.model_grok:
            missing_models.append("MODEL_DEPLOYMENT_GROK")
        
        if missing_models:
            raise ValueError(
                f"Missing model deployment names: {', '.join(missing_models)}. "
                "Please configure in .env file."
            )
    
    async def _call_model(
        self,
        client: AzureOpenAI,
        model: str,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_retries: int = 3
    ) -> str:
        """
        Call Azure OpenAI model with retry logic.
        
        Args:
            client: AzureOpenAI client instance
            model: Model deployment name
            system_prompt: System instructions
            user_message: User message
            temperature: Temperature setting
            max_retries: Maximum retry attempts
            
        Returns:
            Model response text
        """
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=temperature,
                    max_tokens=4000
                )
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  Error occurred: {e}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"Model failed after {max_retries} attempts: {str(e)}")
    
    async def conduct_debate(self, user_query: str) -> Dict[str, str]:
        """
        Conduct a three-round debate on the user's query.
        
        Round 1: Independent analysis from both council members
        Round 2: Chain-of-debate - members critique each other
        Round 3: Chair synthesizes and delivers final verdict
        
        Args:
            user_query: The question or topic to debate
            
        Returns:
            Dictionary containing all responses from each round
        """
        responses = {
            "round1_member_a": "",
            "round1_member_b": "",
            "round2_member_a": "",
            "round2_member_b": "",
            "round3_chair": ""
        }
        
        try:
            # Initialize Azure OpenAI client with API key
            client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.project_endpoint,
            )
            
            # ROUND 1: Independent analysis (concurrent)
            print("🎭 ROUND 1: Independent Analysis")
            
            # Run both council members concurrently
            round1_results = await asyncio.gather(
                self._call_model(
                    client,
                    self.model_gpt41,
                    self.prompts["member_a"],
                    user_query,
                    self.temp_council
                ),
                self._call_model(
                    client,
                    self.model_deepseek,
                    self.prompts["member_b"],
                    user_query,
                    self.temp_council
                ),
                return_exceptions=True
            )
            
            # Check for errors
            if isinstance(round1_results[0], Exception):
                raise round1_results[0]
            if isinstance(round1_results[1], Exception):
                raise round1_results[1]
            
            responses["round1_member_a"] = round1_results[0]
            responses["round1_member_b"] = round1_results[1]
            
            # ROUND 2: Chain-of-Debate - Sequential critique
            print("⚔️  ROUND 2: Chain-of-Debate")
            
            # Member A critiques Member B's response
            critique_prompt_a = f"""Original Query: {user_query}

Council Member B's Analysis:
{responses["round1_member_b"]}

Now provide your critique, counterarguments, or supporting arguments to Member B's analysis."""
            
            responses["round2_member_a"] = await self._call_model(
                client,
                self.model_gpt41,
                self.prompts["member_a"],
                critique_prompt_a,
                self.temp_council
            )
            
            # Member B critiques Member A's response and sees A's critique
            critique_prompt_b = f"""Original Query: {user_query}

Council Member A's Initial Analysis:
{responses["round1_member_a"]}

Council Member A's Critique of Your Analysis:
{responses["round2_member_a"]}

Now respond: critique Member A's points and defend or refine your position."""
            
            responses["round2_member_b"] = await self._call_model(
                client,
                self.model_deepseek,
                self.prompts["member_b"],
                critique_prompt_b,
                self.temp_council
            )
            
            # ROUND 3: Chair's Synthesis
            print("⚖️  ROUND 3: Chair's Synthesis")
            
            chair_prompt = f"""Original Query: {user_query}

=== FULL DEBATE TRANSCRIPT ===

ROUND 1 - Council Member A (Analytical):
{responses["round1_member_a"]}

ROUND 1 - Council Member B (Critical):
{responses["round1_member_b"]}

ROUND 2 - Member A's Critique:
{responses["round2_member_a"]}

ROUND 2 - Member B's Response:
{responses["round2_member_b"]}

=== END TRANSCRIPT ===

Now provide your structured final verdict following the format specified in your instructions."""
            
            responses["round3_chair"] = await self._call_model(
                client,
                self.model_grok,
                self.prompts["chair"],
                chair_prompt,
                self.temp_chair
            )
            
            return responses
                    
        except Exception as e:
            raise Exception(f"Debate execution failed: {str(e)}")
    
    def format_debate_output(self, responses: Dict[str, str]) -> str:
        """
        Format the debate responses for display in the UI.
        
        Args:
            responses: Dictionary of all agent responses
            
        Returns:
            Formatted markdown string
        """
        output = f"""# 🎭 LLM Council Debate

## 📋 Round 1: Independent Analysis

### 🔍 Council Member A (Analytical - GPT-4.1)
{responses["round1_member_a"]}

---

### 🔍 Council Member B (Critical - DeepSeek-V3.1)
{responses["round1_member_b"]}

---

## ⚔️ Round 2: Chain-of-Debate

### 💬 Member A's Critique
{responses["round2_member_a"]}

---

### 💬 Member B's Response
{responses["round2_member_b"]}

---

## ⚖️ Round 3: Chair's Final Verdict (Grok-3)

{responses["round3_chair"]}
"""
        return output


def create_gradio_interface():
    """Create and configure the Gradio UI for the LLM Council."""
    
    council = LLMCouncil()
    
    async def process_query(user_query: str, history: List = None) -> str:
        """
        Process user query through the debate workflow.
        
        Args:
            user_query: The user's question
            history: Chat history (unused but required by Gradio)
            
        Returns:
            Formatted debate output
        """
        if not user_query.strip():
            yield "Please enter a question or topic for the council to debate."
            return
        
        try:
            # Show processing message
            yield "🎭 Initializing LLM Council...\n\n⏳ Round 1: Council members analyzing independently..."
            
            # Conduct the debate
            responses = await council.conduct_debate(user_query)
            
            # Format and return results
            formatted_output = council.format_debate_output(responses)
            yield formatted_output
            
        except Exception as e:
            yield f"❌ Error: {str(e)}\n\nPlease check your Azure AI Foundry configuration and API key in .env file."
    
    # Create Gradio interface
    with gr.Blocks(title="LLM Council: Chain-of-Debate") as demo:
        gr.Markdown("""
        # 🎭 LLM Council: Chain-of-Debate System
        
        Ask complex questions and watch three AI models debate through structured rounds:
        - **Round 1**: Independent analysis from GPT-4.1 (Analytical) and DeepSeek-V3.1 (Critical)
        - **Round 2**: Chain-of-debate where models critique each other
        - **Round 3**: Grok-3 (Chair) synthesizes consensus, debates, and final verdict
                
        """)
        
        chatbot = gr.Chatbot(
            label="Debate Output",
            height=600
        )
        
        msg = gr.Textbox(
            label="Your Question",
            placeholder="E.g., 'Select the best cricket team from India's last 5 matches' or 'Should we invest in renewable energy stocks?'",
            lines=3
        )
        
        submit_btn = gr.Button("🎯 Start Debate", variant="primary")
        clear_btn = gr.Button("🔄 Clear")
        
        async def user_submit(user_message, history):
            """Handle user message submission."""
            history = history or []
            history.append({"role": "user", "content": user_message})
            
            # Process and get response
            async for response in process_query(user_message):
                current_response = response
            
            history.append({"role": "assistant", "content": current_response})
            return "", history
        
        def clear_chat():
            """Clear the chat history."""
            return None, None
        
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
        
        gr.Markdown("""
        ---
        ### 💡 Tips
        - Ask open-ended questions that benefit from multiple perspectives
        - Complex decisions work best (e.g., strategy, analysis, recommendations)
        - The debate typically takes 60-90 seconds depending on query complexity
        """)
    
    return demo


if __name__ == "__main__":
    print("🚀 Starting LLM Council...")
    print("📋 Make sure you have:")
    print("   1. Copied .env.template to .env")
    print("   2. Configured your Azure AI Foundry endpoint and API key in .env")
    print("   3. Set up your model deployment names in .env")
    print("\n")
    
    try:
        demo = create_gradio_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            theme=gr.themes.Soft()
        )
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        print("\nPlease ensure:")
        print("1. All dependencies are installed: pip install -r requirements.txt")
        print("2. .env file is configured with valid Azure AI Foundry settings")
        print("3. AZURE_AI_API_KEY is set in .env file")
