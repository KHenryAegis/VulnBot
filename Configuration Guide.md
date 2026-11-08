# Configuration Guide

The system allows customization of basic settings, database configurations, knowledge base settings, and LLM configurations.

## Configuration Files

- `basic_config.yaml` – General system settings.
- `db_config.yaml` – Database connection settings.
- `kb_config.yaml` – Knowledge base settings.
- `model_config.yaml` – LLM (Large Language Model) settings.

## Configuration Structure

### 1. Basic Configuration (`basic_config.yaml`)

This section defines core settings for the system.

- Logging
  - `log_verbose`: Enable verbose logging (`true`/`false`).
  - `LOG_PATH`: Directory for storing logs.
- Mode
  - `mode`: Determines the system mode (`auto`, `manual`, `semi`).
- Network Settings
  - `http_default_timeout`: Timeout value for HTTP requests.
  - `default_bind_host`: Default binding address.
- Kali Linux Configuration
  - `hostname`: IP address of the Kali machine.
  - `port`: SSH port.
  - `username`: SSH username.
  - `password`: SSH password.
- Server Configuration
  - `api_server`: API server host and port.
  - `webui_server`: Web UI host and port.

### 2. Database Configuration (`db_config.yaml`)

This section defines MySQL database connection settings.

- MySQL
  - `host`: Database host.
  - `port`: Database port.
  - `user`: Username for authentication.
  - `password`: Password for authentication.
  - `database`: Database name.

### 3. Knowledge Base Configuration (`kb_config.yaml`)

Defines settings related to the knowledge base system.

- Vector Store
  - `default_vs_type`: Type of vector store (default: `milvus`).
  - `milvus`: Connection details (`uri`, `user`, `password`).
- Search Parameters
  - `top_n`: Number of results to retrieve.
  - `score_threshold`: Minimum score for results.
  - `search_params`: Search method parameters.
  - `index_params`: Indexing method parameters.
- Text Splitting
  - `text_splitter_dict`: Defines text splitters used for processing.
  - `text_splitter_name`: Selected text splitter method.

### 4. LLM Configuration (`model_config.yaml`)

Defines settings for the language model.

- API Settings
  - `api_key`: API key for accessing the model.
  - `base_url`: Base URL for API requests.
  - `llm_model`: LLM provider (default: OpenAI).
  - `llm_model_name`: Specific LLM model name.
- Embedding Settings
  - `embedding_models`: Embedding model used.
  - `embedding_type`: Embedding type (`local`, `remote`).
  - `context_length`: Maximum context length.
  - `embedding_url`: URL for embedding API.
- Processing Parameters
  - `temperature`: Model temperature.
  - `history_len`: Length of conversation history.
  - `timeout`: Timeout value for requests.
  - `max_tokens`: Upper bound on the tokens of one reply (default `8192`; `0` sends no limit). A reply that reaches the limit raises an error instead of being parsed as complete.
  - `max_retries`: Attempts per request for transient endpoint failures such as dropped connections, timeouts, and HTTP 408/409/429/5xx (default `3`, with exponential backoff).
  - `max_plan_attempts`: Attempts to obtain a usable plan from the model before the run stops (default `3`). After an empty, truncated, blocked, or invalid reply the model is asked again with the reason.
  - `max_plan_tasks`: Largest task list accepted from the model (default `30`; the plan prompt asks for 1 to 5 tasks).
  - `proxies`: Proxy settings for API requests.

## Failure behavior

When the endpoint cannot be reached or returns an error after the retries, or the model gives no usable plan, the run stops: the session is saved, the error is printed, and the process exits with status 1. The error text is never passed on as if the model had written it.

An empty reply to a plan *update* is a valid answer and means that no task remains for the current stage.
