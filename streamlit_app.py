import json

import requests
import streamlit as st

from app.config import settings

API_URL = settings.api_url

st.set_page_config(page_title="企业知识库问答", page_icon="💬", layout="wide")
st.title("企业知识库智能问答")
st.caption("基于混合检索与引用溯源的企业制度问答")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"参考资料（{len(sources)} 条）"):
        for source in sources:
            page = source.get("page")
            page_text = f" 第{page}页" if page is not None else ""
            category = source.get("category")
            category_text = f" · {category}" if category else ""
            st.markdown(f"**[资料{source['id']}]** {source['source']}{page_text}{category_text}")
            st.write(source["content"])
            details = []
            if source.get("rerank_score") is not None:
                details.append(f"重排分数：{source['rerank_score']:.4f}")
            if source.get("vector_rank") is not None:
                details.append(f"向量排名：{source['vector_rank']}")
            if source.get("bm25_rank") is not None:
                details.append(f"BM25 排名：{source['bm25_rank']}")
            if source.get("rrf_score") is not None:
                details.append(f"RRF 分数：{source['rrf_score']:.6f}")
            if source.get("vector_distance") is not None:
                details.append(f"向量距离：{source['vector_distance']:.4f}")
            st.caption(" · ".join(details))


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

if question := st.chat_input("请输入你的问题，例如：出差住酒店能报销多少？"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        sources = []
        error_message = None

        try:
            with requests.post(
                f"{API_URL}/chat/stream",
                json={"question": question, "top_k": 4},
                stream=True,
                timeout=(5, 180),
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line or not raw_line.startswith("data: "):
                        continue

                    payload = raw_line[6:]
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        full_response += payload
                        message_placeholder.markdown(full_response + "▌")
                        continue

                    event_type = event.get("type")
                    if event_type == "sources":
                        sources = event.get("sources", [])
                    elif event_type == "token":
                        full_response += event.get("content", "")
                        message_placeholder.markdown(full_response + "▌")
                    elif event_type == "error":
                        error_message = event.get("message", "模型服务异常")
                    elif event_type == "done":
                        break
        except requests.RequestException as exc:
            error_message = f"无法连接后端：{exc}"

        if error_message:
            st.error(error_message)

        if full_response:
            message_placeholder.markdown(full_response)
        elif not error_message:
            full_response = "后端没有返回内容，请检查模型服务。"
            message_placeholder.warning(full_response)

        render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "sources": sources,
        }
    )
