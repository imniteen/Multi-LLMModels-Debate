"""
Orchestrator module for LLM Council debate workflow
Uses direct HTTP requests for Azure AI Foundry with API key authentication
"""

import asyncio
import aiohttp
from typing import Dict, Optional

from config import CouncilConfig, get_config
from prompts import (
    get_member_a_prompt,
    get_member_b_prompt,
    get_chair_prompt,
    get_round2_prompt_for_member_a,
    get_round2_prompt_for_member_b,
    get_chair_synthesis_prompt
)


class DebateOrchestrator:
    """
    Orchestrates the multi-agent debate workflow through three rounds:
    Round 1: Independent analysis
    Round 2: Chain-of-debate (mutual critique)
    Round 3: Chair's synthesis
    """
    
    def __init__(self, config: Optional[CouncilConfig] = None):
        """
        Initialize the debate orchestrator.
        
        Args:
            config: Council configuration (uses default if not provided)
        """
        self.config = config or get_config()
        
        if not self.config.api_key:
            raise ValueError(
                "AZURE_AI_API_KEY is required. "
                "Please add your Azure AI Foundry API key to the .env file."
            )
    
    async def _call_model(
        self,
        session: aiohttp.ClientSession,
        system_prompt: str,
        user_message: str,
        temperature: float,
        model_name: str,
        max_retries: int = 3
    ) -> str:
        """
        Call the AI model with retry logic using direct HTTP requests.
        
        Args:
            session: aiohttp client session
            system_prompt: System instructions for the model
            user_message: User message to send
            temperature: Temperature setting
            model_name: Name of the model deployment to use
            max_retries: Maximum retry attempts
            
        Returns:
            Model response text
        """
        # Construct the API endpoint
        url = f"{self.config.project_endpoint}/openai/deployments/{model_name}/chat/completions?api-version=2024-10-21"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.config.api_key
        }
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": temperature,
            "max_tokens": 4000
        }
        
        for attempt in range(max_retries):
            try:
                async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        raise Exception(f"API returned status {response.status}: {error_text}")
                        
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  Timeout occurred, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"Model timed out after {max_retries} attempts")
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
        
        Round 1: Independent analysis from both council members (concurrent)
        Round 2: Chain-of-debate - members critique each other (sequential)
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
            # Create aiohttp session for all API calls
            async with aiohttp.ClientSession() as session:
                # ============================================================
                # ROUND 1: Independent Analysis (Concurrent Execution)
                # ============================================================
                print(" ROUND 1: Independent Analysis")
                
                # Run both council members concurrently
                round1_results = await asyncio.gather(
                    self._call_model(
                        session,
                        get_member_a_prompt(),
                        user_query,
                        self.config.member_a.temperature,
                        self.config.member_a.deployment_name
                    ),
                    self._call_model(
                        session,
                        get_member_b_prompt(),
                        user_query,
                        self.config.member_b.temperature,
                        self.config.member_b.deployment_name
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
                
                print(f" Round 1 complete: {len(responses['round1_member_a'])} + {len(responses['round1_member_b'])} chars")
                
                # ============================================================
                # ROUND 2: Chain-of-Debate (Sequential Critique)
                # ============================================================
                print("  ROUND 2: Chain-of-Debate")
                
                # Member A critiques Member B's response
                critique_prompt_a = get_round2_prompt_for_member_a(
                    user_query,
                    responses["round1_member_b"]
                )
                
                responses["round2_member_a"] = await self._call_model(
                    session,
                    get_member_a_prompt(),
                    critique_prompt_a,
                    self.config.member_a.temperature,
                    self.config.member_a.deployment_name
                )
                
                print(f" Member A critique: {len(responses['round2_member_a'])} chars")
                
                # Member B responds to Member A's critique
                critique_prompt_b = get_round2_prompt_for_member_b(
                    user_query,
                    responses["round1_member_a"],
                    responses["round2_member_a"]
                )
                
                responses["round2_member_b"] = await self._call_model(
                    session,
                    get_member_b_prompt(),
                    critique_prompt_b,
                    self.config.member_b.temperature,
                    self.config.member_b.deployment_name
                )
                
                print(f" Member B response: {len(responses['round2_member_b'])} chars")
                
                # ============================================================
                # ROUND 3: Chair's Synthesis
                # ============================================================
                print("  ROUND 3: Chair's Synthesis")
                
                chair_prompt = get_chair_synthesis_prompt(
                    user_query,
                    responses["round1_member_a"],
                    responses["round1_member_b"],
                    responses["round2_member_a"],
                    responses["round2_member_b"]
                )
                
                responses["round3_chair"] = await self._call_model(
                    session,
                    get_chair_prompt(),
                    chair_prompt,
                    self.config.chair.temperature,
                    self.config.chair.deployment_name
                )
                
                print(f" Chair verdict: {len(responses['round3_chair'])} chars")
                print(" Debate complete!")
                
                return responses
        
        except Exception as e:
            raise Exception(f"Debate execution failed: {str(e)}")


def format_debate_output(
    responses: Dict[str, str],
    config: Optional[CouncilConfig] = None
) -> str:
    """
    Format the debate responses for display in markdown.
    
    Args:
        responses: Dictionary of all agent responses
        config: Council configuration for display names (optional)
        
    Returns:
        Formatted markdown string
    """
    config = config or get_config()
    
    output = f"""#  LLM Council Debate

##  Round 1: Independent Analysis

###  {config.member_a.display_name}
{responses["round1_member_a"]}

---

###  {config.member_b.display_name}
{responses["round1_member_b"]}

---

##  Round 2: Chain-of-Debate

###  Member A's Critique
{responses["round2_member_a"]}

---

###  Member B's Response
{responses["round2_member_b"]}

---

##  Round 3: Chair's Final Verdict ({config.chair.deployment_name})

{responses["round3_chair"]}
"""
    return output
