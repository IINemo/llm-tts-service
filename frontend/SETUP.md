# Frontend Setup Guide

## Prerequisites
- Node.js (v18+)
- Backend service running on port 8001

## Running the Application

### 1. Start Backend Service (Terminal 1)
```bash
cd /Users/karantonis/MBZUAI/NLP/llm-tts-service

# Set environment variables
export OPENROUTER_API_KEY="your-key-here"

# Run service
python service_app/main.py
```
Service will start on: `http://localhost:8001`

### 2. Start Frontend (Terminal 2)
```bash
cd /Users/karantonis/MBZUAI/NLP/llm-tts-service/frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```
Frontend will start on: `http://localhost:5173`

### 3. Open Browser
Navigate to: `http://localhost:5173`

## Usage

1. **Enter your question** in the text area
2. **Select strategy** (Tree-of-Thoughts, DeepConf, etc.)
3. **Configure parameters** (for ToT: beam width, steps, candidates)
4. **Click "Generate Response"**
5. **Watch real-time progress**:
   - Progress bar showing current step
   - Real-time statistics (nodes explored, API calls)
   - Tree visualization updates as nodes are discovered
6. **View final results**:
   - Final answer from the model
   - Metadata (elapsed time, API calls, scores)
   - Complete interactive reasoning tree

## Features

- ✅ **Real-time progress updates** via polling (updates every second)
- ✅ **Async job execution** - Backend runs in background
- ✅ Interactive Tree-of-Thoughts visualization
- ✅ Configurable strategy parameters
- ✅ Progress bar and live statistics
- ✅ Loading states and error handling
- ✅ Response metadata display

## How It Works

The frontend uses a **polling-based architecture**:

1. **Job Creation**: When you click "Generate Response", the frontend sends a request to `POST /v1/jobs` which returns a `job_id` immediately
2. **Background Execution**: The backend starts processing the ToT strategy in a background thread
3. **Polling**: The frontend polls `GET /v1/jobs/{job_id}` every second to check progress
4. **Real-time Updates**: As the backend progresses, it updates job state with:
   - Current step number
   - Nodes explored count
   - API calls made
   - Intermediate tree nodes (for visualization)
5. **Completion**: When the job finishes, the frontend displays the final answer and complete reasoning tree

## Troubleshooting

### CORS Errors
- Ensure backend service is running on port 8001
- Backend already has CORS configured to accept all origins

### Connection Refused
- Check backend is running: `curl http://localhost:8001/health`
- Verify OPENROUTER_API_KEY is set in backend environment

### Module Not Found
```bash
cd frontend
npm install
```

## API Endpoints Used

### Async Job API (Polling-based)
- `POST /v1/jobs` - Create a new async ToT job (returns job_id immediately)
- `GET /v1/jobs/{job_id}` - Poll job status and progress (call repeatedly)

### Other Endpoints
- `GET /health` - Check backend health
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Direct chat completion (legacy, blocking)
