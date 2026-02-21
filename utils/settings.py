import os
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load YAML config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/config.yaml")

with open(CONFIG_PATH, "r") as f:
    yaml_config = yaml.safe_load(f)


class Settings:
    # ---- Secrets (.env) ----
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    GITHUB_MCP_URL = os.getenv("GITHUB_MCP_URL")

    # ---- LangSmith ----
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "github-project-mentor")

    # ---- YAML Config ----
    MODEL_NAME = yaml_config["llm"]["model_name"]
    TEMPERATURE = yaml_config["llm"]["temperature"]

    RETRIEVAL_TOP_K = yaml_config["retrieval"]["top_k"]
    CHUNK_SIZE = yaml_config["retrieval"]["chunk_size"]
    CHUNK_OVERLAP = yaml_config["retrieval"]["chunk_overlap"]

    EMBEDDING_MODEL = yaml_config["embeddings"]["model"]

    DATA_DIR = yaml_config["paths"]["data_dir"]
    PERSIST_DIR = yaml_config["paths"]["persist_dir"]


settings = Settings()