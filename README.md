# Fast-Agent Development Kit

A comprehensive framework for building sophisticated AI agents and workflows with full-stack integration capabilities.

## 🚀 Overview

Fast-Agent is a powerful development kit that enables rapid creation of AI applications with:

- **Multi-modal AI Agents** - Support for text, images, and PDFs
- **MCP Integration** - Model Context Protocol for tool and resource management
- **Workflow Orchestration** - Chain, parallel, router, and evaluator-optimizer patterns
- **Full-Stack Ready** - Integration with FastAPI, React, and vector databases
- **Real-time Processing** - Fast-Stream for event-driven architectures

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Router  │    │   Fast-API      │    │   Fast-Agent    │
│   Frontend      │◄──►│   Backend       │◄──►│   AI Agents     │
│   (Routing)     │    │   (REST API)    │    │   (Orchestration)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Fast-Stream   │    │   Fast-MCP      │    │   Neon Postgres │
│   Real-time     │    │   Tool Server   │    │   + pgvector    │
│   Processing    │    │   Integration   │    │   Vector DB     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Core Features

### AI Agent Capabilities
- **Multi-modal Support** - Images and PDFs in prompts and responses
- **Workflow Patterns** - Chain, parallel, router, evaluator-optimizer, orchestrator
- **MCP Integration** - Seamless tool and resource management
- **Model Flexibility** - Support for OpenAI, Anthropic, Azure, and more
- **Structured Outputs** - Pydantic model integration

### Development Tools
- **Interactive CLI** - `fast-agent go` for rapid prototyping
- **Configuration Management** - YAML-based with environment variable support
- **OpenTelemetry** - Comprehensive observability and monitoring
- **Testing Models** - Passthrough and playback for development

### Full-Stack Integration
- **FastAPI Backend** - RESTful API with async support
- **React Frontend** - TypeScript with React Router
- **Vector Database** - Neon Postgres with pgvector
- **Real-time Processing** - Fast-Stream event handling

## 🚀 Quick Start

### 1. Installation

```bash
# Install Fast-Agent
uv pip install fast-agent-mcp

# Setup project
fast-agent setup

# Check configuration
fast-agent check
```

### 2. Basic Agent

```python
import asyncio
from mcp_agent.core.fastagent import FastAgent

fast = FastAgent("My AI App")

@fast.agent(
    name="assistant",
    instruction="You are a helpful AI assistant",
    model="openai.gpt-4o"
)
async def main():
    async with fast.run() as agent:
        response = await agent.assistant.send("Hello!")
        print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Interactive Mode

```bash
# Start interactive session
fast-agent go --model=openai.gpt-4o

# With MCP servers
fast-agent go --servers=filesystem,brave_search --model=openai.gpt-4o
```

## 📚 Documentation

### Core Concepts
- [Defining Agents and Workflows](https://fast-agent.ai/agents/defining/)
- [Deploy and Run](https://fast-agent.ai/agents/running/)
- [Prompting Agents](https://fast-agent.ai/agents/prompting/)
- [Instructions](https://fast-agent.ai/agents/instructions/)

### Models and Providers
- [Model Features](https://fast-agent.ai/models/)
- [LLM Providers](https://fast-agent.ai/models/llm_providers/)
- [Internal Models](https://fast-agent.ai/models/internal_models/)

### MCP Integration
- [Configuring Servers](https://fast-agent.ai/mcp/)
- [Integration with MCP Types](https://fast-agent.ai/mcp/types/)
- [Elicitations](https://fast-agent.ai/mcp/elicitations/)
- [State Transfer](https://fast-agent.ai/mcp/state_transfer/)
- [Resources](https://fast-agent.ai/mcp/resources/)

### Reference
- [fast-agent go Command](https://fast-agent.ai/ref/go_command/)
- [Configuration Reference](https://fast-agent.ai/ref/config_file/)
- [Command Line Options](https://fast-agent.ai/ref/cmd_switches/)
- [Class Reference](https://fast-agent.ai/ref/class_reference/)
- [OpenTelemetry](https://fast-agent.ai/ref/open_telemetry/)
- [Azure Configuration](https://fast-agent.ai/ref/azure_configuration/)

## ⚙️ Configuration

### Basic Configuration

```yaml
# fastagent.config.yaml
default_model: "openai.gpt-4o"

openai:
  api_key: "${OPENAI_API_KEY}"

anthropic:
  api_key: "${ANTHROPIC_API_KEY}"

mcp:
  filesystem:
    command: "uvx"
    args: ["mcp-server-filesystem"]
  
  brave_search:
    command: "npx"
    args: ["@modelcontextprotocol/server-brave-search"]
    env:
      BRAVE_API_KEY: "${BRAVE_API_KEY}"
```

### Environment Variables

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export BRAVE_API_KEY="your-brave-search-key"
```

## 🏗️ Development Workflow

### 1. Project Structure

```
fast-agent-project/
├── ai_agents/
│   ├── main.py
│   └── workflows/
├── backend/
│   ├── main.py
│   └── api/
├── frontend/
│   ├── src/
│   └── package.json
├── fastagent.config.yaml
├── fastagent.secrets.yaml
└── README.md
```

### 2. Agent Development

```python
# ai_agents/main.py
import asyncio
from mcp_agent.core.fastagent import FastAgent

fast = FastAgent("AI Development Kit")

@fast.agent(
    "researcher",
    "Research topics thoroughly",
    model="openai.gpt-4o",
    servers=["brave_search"]
)
@fast.agent(
    "writer",
    "Write content based on research",
    model="openai.gpt-4o"
)
@fast.chain(
    "content_pipeline",
    sequence=["researcher", "writer"]
)
async def main():
    async with fast.run() as agent:
        result = await agent.content_pipeline.send("Create content about AI")
        return result

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Backend Integration

```python
# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Fast-Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "fast-agent-api"}
```

### 4. Frontend Integration

```typescript
// frontend/src/components/AIAgent.tsx
import { useState } from 'react';

export const AIAgent = () => {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    const response = await fetch('/api/ai-agent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    
    const data = await response.json();
    setResponse(data.response);
  };

  return (
    <div>
      <h2>AI Agent Interface</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your query..."
        />
        <button type="submit">Submit</button>
      </form>
      
      {response && (
        <div>
          <h3>Response:</h3>
          <p>{response}</p>
        </div>
      )}
    </div>
  );
};
```

## 🚀 Deployment

### Development

```bash
# Terminal 1: Fast-Agent
cd ai_agents
uv run main.py

# Terminal 2: Backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 3: Frontend
cd frontend
npm start
```

### Production

```bash
# Build and deploy
docker-compose up -d

# Or deploy as MCP server
uv run main.py --server --transport http --port 8080
```

## 🔍 Monitoring

### OpenTelemetry Integration

```yaml
# fastagent.config.yaml
otel:
  enabled: true
  otlp_endpoint: "http://localhost:4318/v1/traces"
  service_name: "fast-agent-app"
  service_version: "1.0.0"
```

### Health Checks

```bash
# Check configuration
fast-agent check

# Test providers
fast-agent check --provider openai --provider anthropic

# Test MCP servers
fast-agent check --servers
```

## 🧪 Testing

### Using Internal Models

```python
@fast.agent(
    "test_agent",
    "Test agent",
    model="passthrough"  # Echo model for testing
)
async def test_agent():
    async with fast.run() as agent:
        # Set fixed response for testing
        await agent.test_agent.send("***FIXED_RESPONSE Test response")
        
        # Test the agent
        response = await agent.test_agent.send("Any input")
        assert response == "Test response"
```

### Playback Testing

```python
@fast.agent(
    "playback_agent",
    "Playback agent",
    model="playback"
)
async def playback_test():
    async with fast.run() as agent:
        # Load conversation for testing
        conversation = """---USER
Hello
---ASSISTANT
Hi there!"""
        
        await agent.playback_agent.send(f"***FIXED_RESPONSE {conversation}")
        
        # Replay conversation
        response = await agent.playback_agent.send("Hello")
        assert response == "Hi there!"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [https://fast-agent.ai/](https://fast-agent.ai/)
- **Issues**: [GitHub Issues](https://github.com/your-repo/fast-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/fast-agent/discussions)

## 🔗 Related Projects

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [React Router](https://reactrouter.com/) - Declarative routing
- [Neon Postgres](https://neon.tech/) - Serverless Postgres
- [Model Context Protocol](https://modelcontextprotocol.io/) - AI tool integration

---

**Fast-Agent Development Kit** - Build sophisticated AI applications with ease.
