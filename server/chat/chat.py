import asyncio

from starlette.concurrency import run_in_threadpool

from config.config import Configs
from db.repository.conversation_repository import add_conversation_to_db
from db.repository.message_repository import get_conversation_messages, add_message_to_db
from rag.kb.api.kb_doc_api import search_docs
from rag.reranker.reranker import LangchainReranker
from server.chat.errors import LLMError
from server.chat.llm_clients import OllamaChat, OpenAIChat
from server.utils.utils import LLMType, replace_ip_with_targetip
from utils.log_common import build_logger

logger = build_logger()


def _retrieve_context(kb_name, kb_query):
    """Return knowledge-base context for ``kb_query``, or "" when nothing relevant is found."""
    docs = asyncio.run(run_in_threadpool(search_docs,
                                         query=kb_query,
                                         knowledge_base_name=kb_name,
                                         top_k=Configs.kb_config.top_k,
                                         score_threshold=Configs.kb_config.score_threshold,
                                         file_name="",
                                         metadata={}))
    if docs:
        reranker_model = LangchainReranker(top_n=Configs.kb_config.top_n,
                                           name_or_path=Configs.llm_config.rerank_model)
        docs = reranker_model.compress_documents(documents=docs, query=kb_query)

    if not docs:
        logger.warning(f"No knowledge-base documents matched the query; continuing without retrieved context "
                       f"(knowledge base: {kb_name!r}, query: {str(kb_query)[:200]!r})")
        return ""
    return "\n".join([doc["page_content"] for doc in docs])


def _chat(query: str, kb_name=None, conversation_id=None, kb_query=None, summary=True, allow_empty=False):
    """Send ``query`` to the model and return its reply.

    Raises LLMError when the endpoint fails or the reply is unusable; error text is never returned in
    place of a reply. ``allow_empty`` accepts an empty reply, for prompts that define one as the answer.
    """
    if Configs.basic_config.enable_rag and kb_name is not None:
        context = _retrieve_context(kb_name, kb_query)

        if context:
            context = replace_ip_with_targetip(context)
            query = f"{query}\n\n\n Ensure that the **Overall Target** IP or the IP from the **Initial Description** is prioritized. You will respond to questions and generate tasks based on the provided penetration test case materials: {context}. \n"

    if conversation_id is not None and len(query) > 10000:
        query = query[:10000]
    else:
        query = query[:Configs.llm_config.context_length]

    flag = False

    if conversation_id is not None:
        flag = True

    # Initialize or retrieve conversation ID
    conversation_id = add_conversation_to_db(Configs.llm_config.llm_model_name, conversation_id)

    history = [
        {
            "role": "system",
            "content": "You are a helpful assistant",
        }
    ]
    # Retrieve message history from database, and limit the number of messages
    for msg in get_conversation_messages(conversation_id)[-Configs.llm_config.history_len:]:
        history.append({"role": "user", "content": msg.query})
        history.append({"role": "assistant", "content": msg.response})

    # Add user query to the message history
    history.append({"role": "user", "content": query})

    # Initialize the correct model client
    if Configs.llm_config.llm_model == LLMType.OPENAI:
        client = OpenAIChat(config=Configs.llm_config)
    elif Configs.llm_config.llm_model == LLMType.OLLAMA:
        client = OllamaChat(config=Configs.llm_config)
    else:
        raise LLMError(f"Unsupported model type: {Configs.llm_config.llm_model!r}")

    # Get response from the model
    response_text = client.chat(history, allow_empty=allow_empty)

    # Save both query and response to the database
    if summary:
        add_message_to_db(conversation_id, Configs.llm_config.llm_model_name, query, response_text)

    if flag:
        return response_text
    else:
        return response_text, conversation_id
