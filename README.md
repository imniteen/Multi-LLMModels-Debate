# code is available at branch name:  'updated'




## LLM Council: Chain-of-Debate System

A sophisticated multi-agent debate system using Azure OpenAI, implementing a "Chain-of-Debate" pattern where three distinct AI models collaborate through structured rounds to reach consensus on complex queries.

## 🎯 Overview

This application orchestrates three AI models in a council-style debate:
- **Council Member A (GPT-4.1)**: Provides analytical, data-driven perspectives
- **Council Member B (DeepSeek-V3.1)**: Offers critical thinking and challenges assumptions
- **Chair (Grok-3)**: Synthesizes the debate and delivers structured verdicts

## 🏗️ Architecture

### Three-Round Debate Process

1. **Round 1 - Independent Analysis**: Both council members analyze the query independently
2. **Round 2 - Chain-of-Debate**: Members critique each other's positions sequentially
3. **Round 3 - Synthesis**: The Chair reviews the full debate and provides:
   - Areas of Complete Consensus
   - Key Debates
   - The Verdict

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or later
- Azure AI Foundry project with:
  - Three model deployments: `gpt-4.1`, `DeepSeek-V3.1`, `grok-3`
  - API key for authentication

### Installation

1. **Clone the repository**
   ```bash
   cd c:\Github\Multi-LLMModels-Debate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Azure settings**
   ```bash
   # Copy the template
   copy .env.template .env

   # Edit .env and add your Azure AI Foundry settings:
   # - AZURE_AI_PROJECT_ENDPOINT
   # - AZURE_AI_API_KEY (required)
   # - Model deployment names (if different from defaults)
   ```

### Running the Application

```bash
python app.py
```

The Gradio interface will launch at `http://localhost:7860`

## 📋 Configuration

Edit `.env` file with your settings:

```env
# Required
AZURE_AI_PROJECT_ENDPOINT=https://your-project-name.azure.com
AZURE_AI_API_KEY=your-azure-ai-api-key-here

# Optional: API Version
AZURE_OPENAI_API_VERSION=2024-10-21

# Model Deployments (use your deployment names)
MODEL_DEPLOYMENT_GPT41=gpt-4.1
MODEL_DEPLOYMENT_DEEPSEEK=DeepSeek-V3.1
MODEL_DEPLOYMENT_GROK=grok-3

# Optional: Temperature Settings
TEMPERATURE_COUNCIL_MEMBERS=0.7  # Higher for diverse viewpoints
TEMPERATURE_CHAIR=0.3             # Lower for consistent synthesis

# Optional: Timeout (seconds)
AGENT_TIMEOUT=60
```

## 💡 Usage Examples

### Example Queries

1. **Strategic Decisions**
   ```
   "Should our company invest in AI infrastructure or cloud migration first?"
   ```

2. **Analysis Tasks**
   ```
   "Analyze the pros and cons of remote work vs. hybrid work models"
   ```

3. **Recommendations**
   ```
   "Select the best cricket team from India's last 5 matches"
   ```

4. **Technical Decisions**
   ```
   "Should we use microservices or monolithic architecture for our new product?"
   ```

## 🔧 Technical Details

### Azure OpenAI Integration

This application uses the Azure OpenAI Python SDK to orchestrate multiple model deployments. Key features:

- **Async-first architecture** using `asyncio` for concurrent execution
- **API key authentication** for secure access to Azure AI Foundry models
- **Retry logic** with exponential backoff (up to 3 attempts)
- **Timeout handling** for resilient execution
- **Direct model access** via Azure OpenAI chat completions API

### Multi-Agent Orchestration

- **Concurrent execution** in Round 1 for independent analysis
- **Sequential execution** in Round 2 for authentic debate flow
- **Full transcript preservation** for Chair's synthesis
- **Structured output format** enforced via system prompts

### Error Handling

- Automatic retry with exponential backoff (up to 3 attempts)
- Timeout protection (default 60 seconds per agent)
- Environment validation on startup
- Graceful error messages in UI

## 📁 Project Structure

```
Multi-LLMModels-Debate/
├── app.py                 # Main application with LLMCouncil class
├── requirements.txt       # Python dependencies
├── .env.template         # Environment variable template
├── .env                  # Your configuration (create from template)
└── README.md            # This file
```

## 🛠️ Troubleshooting

### Common Issues

1. **"AZURE_AI_PROJECT_ENDPOINT not found"**
   - Copy `.env.template` to `.env`
   - Add your Azure AI Foundry project endpoint

2. **"AZURE_AI_API_KEY not found"**
   - Add your Azure AI Foundry API key to the `.env` file
   - Find your API key in the Azure AI Foundry portal under "Keys and Endpoint"

3. **"Model deployment not found"**
   - Verify model deployment names in Azure AI Foundry portal
   - Update `MODEL_DEPLOYMENT_*` values in `.env`

4. **"Timeout occurred"**
   - Increase `AGENT_TIMEOUT` in `.env`
   - Check Azure service health

## 🔐 Security Notes

- Never commit `.env` file to version control
- Use Azure RBAC for access control
- Model responses may flow outside Azure compliance boundaries
- Review Azure AI Foundry data handling policies

## 📚 Resources

- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Azure AI Foundry](https://azure.microsoft.com/en-us/products/ai-foundry/)
- [Gradio Documentation](https://www.gradio.app/docs/)

## 🎓 How It Works

### The Chain-of-Debate Pattern

1. **Independent Reasoning**: Each council member forms their own opinion without influence
2. **Constructive Critique**: Members challenge each other's assumptions with evidence
3. **Synthesis**: The Chair identifies consensus areas and resolves debates with final judgment

This pattern is more sophisticated than simple model comparison because:
- Models see and respond to each other's reasoning (true debate)
- The Chair makes explicit decisions rather than just summarizing
- Consensus vs. dissensus is clearly identified
- Final verdict provides actionable conclusions

## 📄 License

This project is provided as-is for demonstration purposes.

## 🤝 Contributing

This is a demonstration project. Feel free to fork and adapt for your needs.
