# Fast-Agent Development Kit - Cursor Agent Guide

## Project Overview

This is a comprehensive AI development kit that integrates Fast-Agent, Fast-API, Fast-MCP, Fast-Stream, React Router, and Neon Postgres with vector databases. The goal is to enable rapid development of sophisticated full-stack AI applications.

## Architecture Principles

- **Multi-modal AI Agents**: Support for text, images, and PDFs in prompts and responses
- **MCP Integration**: Model Context Protocol for seamless tool and resource management
- **Workflow Orchestration**: Chain, parallel, router, evaluator-optimizer, and orchestrator patterns
- **Full-Stack Integration**: React frontend, FastAPI backend, vector databases, and real-time processing
- **Declarative Development**: Simplified agent and workflow definitions with minimal boilerplate

## Development Workflow

### 1. Project Initialization

When helping users set up a new Fast-Agent project:

```bash
# Create project structure
mkdir ai-fullstack-app
cd ai-fullstack-app

# Initialize Fast-Agent
fast-agent setup

# Initialize React app with React Router
npx create-react-app frontend --template typescript
cd frontend
npm install react-router-dom @types/react-router-dom

# Initialize Fast-API backend
mkdir backend
cd backend
uv init
uv add fastapi uvicorn sqlalchemy psycopg2-binary

# Initialize Neon Postgres (guide users through setup)
```

### 2. Configuration Setup

Always help users create proper configuration files:

**fastagent.config.yaml**
```yaml
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

# Neon Postgres configuration
database:
  url: "${DATABASE_URL}"
  vector_extension: true
```

**Environment Variables**
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export DATABASE_URL="your-neon-postgres-url"
export BRAVE_API_KEY="your-brave-search-key"
```

## Agent Development Patterns

### Basic Agent Creation

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

### Workflow Orchestration

**Chain Workflow**
```python
@fast.chain(
    name="content_pipeline",
    sequence=["researcher", "writer"]
)
```

**Parallel Workflow**
```python
@fast.parallel(
    name="parallel_tasks",
    fan_out=["task1_agent", "task2_agent"],
    fan_in="aggregator_agent"
)
```

**Router Workflow**
```python
@fast.router(
    name="smart_router",
    agents=["technical_support", "general_assistant", "creative_writer"],
    instruction="Route user requests to the most appropriate specialist agent"
)
```

**Evaluator-Optimizer**
```python
@fast.evaluator_optimizer(
    name="refined_content",
    generator="content_generator",
    evaluator="quality_evaluator",
    min_rating="GOOD",
    max_refinements=3
)
```

**Orchestrator**
```python
@fast.orchestrator(
    name="ai_orchestrator",
    instruction="Orchestrate complete AI workflows for user requests",
    agents=["researcher", "writer", "vector_processor"],
    plan_type="full"
)
```

## Backend Development

### FastAPI Integration

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import VECTOR
from sqlalchemy.ext.declarative import declarative_base
import os
from typing import List, Dict, Any

app = FastAPI(title="AI Full-Stack API")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup with pgvector
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# Enable pgvector extension
with engine.connect() as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(Text)
    embedding = Column(VECTOR(1536))
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Dependencies
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# API Routes
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "ai-fullstack-api"}

@app.post("/api/documents")
async def create_document(
    title: str, 
    content: str, 
    embedding: List[float],
    db: Session = Depends(get_db)
):
    """Create document with vector embedding"""
    try:
        doc = Document(title=title, content=content, embedding=embedding)
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return {"id": doc.id, "title": doc.title, "status": "created"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vector-search")
async def vector_search(
    query_embedding: List[float],
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Vector similarity search"""
    try:
        results = db.execute("""
            SELECT id, title, content, embedding <=> %s as distance
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s
        """, (query_embedding, query_embedding, limit))
        
        return [{"id": r.id, "title": r.title, "content": r.content, "distance": float(r.distance)} 
                for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Frontend Development

### React Router Setup

```typescript
// frontend/src/config/routes.ts
import { createBrowserRouter } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { AIAgent } from './pages/AIAgent';
import { VectorSearch } from './pages/VectorSearch';
import { Settings } from './pages/Settings';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: 'ai-agent', element: <AIAgent /> },
      { path: 'vector-search', element: <VectorSearch /> },
      { path: 'settings', element: <Settings /> },
    ],
  },
]);
```

### AI Agent Interface

```typescript
// frontend/src/pages/AIAgent.tsx
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Form, Button, Card, Alert } from 'react-bootstrap';

export const AIAgent = () => {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const response = await fetch('/api/ai-agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      
      const data = await response.json();
      setResponse(data.response);
    } catch (error) {
      console.error('AI Agent error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>AI Agent Interface</h2>
      <Card>
        <Card.Body>
          <Form onSubmit={handleSubmit}>
            <Form.Group className="mb-3">
              <Form.Label>Query</Form.Label>
              <Form.Control
                as="textarea"
                rows={3}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter your query for the AI agent..."
              />
            </Form.Group>
            <Button type="submit" disabled={loading}>
              {loading ? 'Processing...' : 'Submit Query'}
            </Button>
          </Form>
          
          {response && (
            <Alert variant="info" className="mt-3">
              <h5>AI Response:</h5>
              <p>{response}</p>
            </Alert>
          )}
        </Card.Body>
      </Card>
    </div>
  );
};
```

### Vector Search Interface

```typescript
// frontend/src/pages/VectorSearch.tsx
import { useState } from 'react';
import { Form, Button, Card, ListGroup } from 'react-bootstrap';

export const VectorSearch = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      // Generate embedding for search query
      const embeddingResponse = await fetch('/api/generate-embedding', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: searchQuery })
      });
      
      const { embedding } = await embeddingResponse.json();
      
      // Perform vector search
      const searchResponse = await fetch('/api/vector-search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query_embedding: embedding, limit: 10 })
      });
      
      const searchResults = await searchResponse.json();
      setResults(searchResults);
    } catch (error) {
      console.error('Vector search error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Vector Search</h2>
      <Card>
        <Card.Body>
          <Form onSubmit={handleSearch}>
            <Form.Group className="mb-3">
              <Form.Label>Search Query</Form.Label>
              <Form.Control
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Enter search query..."
              />
            </Form.Group>
            <Button type="submit" disabled={loading}>
              {loading ? 'Searching...' : 'Search'}
            </Button>
          </Form>
          
          {results.length > 0 && (
            <div className="mt-4">
              <h5>Search Results:</h5>
              <ListGroup>
                {results.map((result: any) => (
                  <ListGroup.Item key={result.id}>
                    <h6>{result.title}</h6>
                    <p>{result.content}</p>
                    <small className="text-muted">
                      Similarity: {(1 - result.distance).toFixed(3)}
                    </small>
                  </ListGroup.Item>
                ))}
              </ListGroup>
            </div>
          )}
        </Card.Body>
      </Card>
    </div>
  );
};
```

## Real-time Processing

### Fast-Stream Integration

```python
# stream_processor/main.py
import asyncio
from faststream import FastStream, Depends
from faststream.redis import RedisBroker
from typing import Dict, Any
import json

broker = RedisBroker("redis://localhost:6379")
app = FastStream(broker)

@app.on_event("startup")
async def startup():
    """Initialize stream processing"""
    print("Stream processor started")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup stream processing"""
    print("Stream processor stopped")

@app.subscriber("ai-requests")
async def process_ai_request(message: Dict[str, Any]):
    """Process AI requests in real-time"""
    try:
        request_id = message.get("request_id")
        query = message.get("query")
        user_id = message.get("user_id")
        
        print(f"Processing AI request {request_id} for user {user_id}")
        
        # Process with Fast-Agent
        # This would integrate with the Fast-Agent orchestrator
        
        # Publish result
        await broker.publish(
            {
                "request_id": request_id,
                "user_id": user_id,
                "status": "completed",
                "result": f"Processed: {query}"
            },
            "ai-responses"
        )
        
    except Exception as e:
        print(f"Error processing AI request: {e}")
        await broker.publish(
            {
                "request_id": message.get("request_id"),
                "user_id": message.get("user_id"),
                "status": "error",
                "error": str(e)
            },
            "ai-errors"
        )

@app.subscriber("vector-updates")
async def process_vector_update(message: Dict[str, Any]):
    """Process vector database updates"""
    try:
        document_id = message.get("document_id")
        embedding = message.get("embedding")
        
        print(f"Processing vector update for document {document_id}")
        
        # Update vector database
        # This would integrate with Neon Postgres + pgvector
        
    except Exception as e:
        print(f"Error processing vector update: {e}")

if __name__ == "__main__":
    asyncio.run(app.run())
```

## MCP Tool Integration

### Custom MCP Server

```python
# mcp_tools/main.py
from mcp import Server
from mcp.types import TextContent
import asyncio

server = Server("ai-fullstack-tools")

@server.list_tools()
async def list_tools():
    """List available tools"""
    return [
        {
            "name": "search_documents",
            "description": "Search documents using vector similarity",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": ["query"]
            }
        },
        {
            "name": "generate_embedding",
            "description": "Generate embedding for text",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        }
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Execute tool calls"""
    if name == "search_documents":
        query = arguments.get("query")
        limit = arguments.get("limit", 10)
        
        # Integrate with vector search
        # This would call the vector search API
        
        return [TextContent(type="text", text=f"Search results for: {query}")]
    
    elif name == "generate_embedding":
        text = arguments.get("text")
        
        # Integrate with embedding generation
        # This would call the embedding API
        
        return [TextContent(type="text", text=f"Generated embedding for: {text}")]

if __name__ == "__main__":
    asyncio.run(server.run())
```

## Development Commands

### Environment Setup

```bash
# Install dependencies
uv pip install fast-agent-mcp fastapi uvicorn sqlalchemy psycopg2-binary
npm install react-router-dom @types/react-router-dom

# Set environment variables
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export DATABASE_URL="your-neon-postgres-url"
export BRAVE_API_KEY="your-brave-search-key"
```

### Development Workflow

```bash
# Terminal 1: Start Fast-API backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2: Start React frontend
cd frontend
npm start

# Terminal 3: Start Fast-Agent
cd ai_agents
uv run main.py

# Terminal 4: Start Fast-Stream
cd stream_processor
uv run main.py

# Terminal 5: Start MCP tools
cd mcp_tools
uv run main.py
```

### Testing Workflow

```bash
# Test Fast-API endpoints
curl -X POST http://localhost:8000/api/health

# Test vector search
curl -X POST http://localhost:8000/api/vector-search \
  -H "Content-Type: application/json" \
  -d '{"query_embedding": [0.1, 0.2, ...], "limit": 5}'

# Test AI agent
curl -X POST http://localhost:8000/api/ai-agent \
  -H "Content-Type: application/json" \
  -d '{"query": "Research AI trends"}'
```

## Production Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - redis

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  ai-agent:
    build: ./ai_agents
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    depends_on:
      - backend
      - redis
```

## Best Practices

### 1. Development Workflow
- Use **branching workflows** from Neon Postgres rules
- Implement **vector search** for AI context retrieval
- Use **Fast-Agent orchestration** for complex AI workflows
- Implement **real-time processing** with Fast-Stream

### 2. Performance Optimization
- Use **connection pooling** for database operations
- Implement **vector indexing** for fast similarity search
- Use **caching** for frequently accessed data
- Optimize **AI model selection** based on use case

### 3. Security
- Implement **authentication** and **authorization**
- Use **environment variables** for sensitive data
- Enable **SSL/TLS** for database connections
- Implement **input validation** and **sanitization**

### 4. Monitoring
- Use **OpenTelemetry** for observability
- Monitor **AI model performance** and costs
- Track **vector search** performance
- Implement **health checks** for all services

## Common Patterns

### Agent State Transfer

```python
# Transfer conversation history between agents
history = agent_one.message_history
await agent_two.generate(history)

# Continue conversation with transferred context
response = await agent_two.send("Continue from where we left off")
```

### Structured Outputs

```python
from pydantic import BaseModel
from typing import List

class AnalysisResult(BaseModel):
    summary: str
    key_points: List[str]
    confidence: float

# Send a message and get structured response
result: AnalysisResult = await agent.structured(
    "Analyze this document and provide a structured summary",
    AnalysisResult,
    Path("document.pdf")
)
```

### Multi-modal Support

```python
from mcp_agent.core.prompt import Prompt
from pathlib import Path

# Send image for analysis
response = await agent.send(Prompt.user("Describe this image", Path("image.jpg")))

# Send PDF for summarization
response = await agent.send(Prompt.user("Summarize this document", Path("document.pdf")))
```

## Troubleshooting

### Common Issues

1. **API Key Errors**
   ```bash
   fast-agent check
   ```

2. **Model Not Found**
   - Check model name spelling
   - Verify model availability in your region
   - Check provider documentation

3. **Rate Limiting**
   - Implement exponential backoff
   - Use appropriate model sizes
   - Monitor usage limits

4. **Vector Database Issues**
   - Ensure pgvector extension is installed
   - Check database connection
   - Verify embedding dimensions match

### Debug Mode

Enable debug logging:

```yaml
# fastagent.config.yaml
logging:
  level: "DEBUG"
  format: "detailed"
```

## Helpful Commands

```bash
# Check configuration
fast-agent check

# Show configuration
fast-agent check show

# Test specific providers
fast-agent check --provider openai

# Test MCP servers
fast-agent check --servers

# Start interactive session
fast-agent go --model=openai.gpt-4o

# Deploy as MCP server
uv run main.py --server --transport http --port 8080
```

## Resources

- **Documentation**: [https://fast-agent.ai/](https://fast-agent.ai/)
- **MCP Specification**: [https://modelcontextprotocol.io/](https://modelcontextprotocol.io/)
- **Neon Postgres**: [https://neon.tech/](https://neon.tech/)
- **FastAPI**: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)

---

This AGENTS.md file provides comprehensive guidance for building sophisticated AI applications using the Fast-Agent development kit. Use these patterns and examples to help users create robust, scalable AI solutions.
