import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api")
UPLOAD_POLL_INTERVAL_SECONDS = 1.0
UPLOAD_STATUS_TIMEOUT_SECONDS = 120


def _format_task_status(task_payload: Dict[str, Any]) -> str:
    task_id = task_payload.get("task_id", "unknown")
    state = task_payload.get("state", "UNKNOWN")
    result = task_payload.get("result") or {}
    error = task_payload.get("error")

    if state == "SUCCESS":
        points_upserted = result.get("points_upserted", 0)
        source_file = result.get("source_file", "document")
        return (
            f"Task `{task_id}` completed successfully.\n"
            f"Indexed `{points_upserted}` chunks from `{source_file}`."
        )

    if state == "FAILURE":
        return f"Task `{task_id}` failed.\nReason: {error or 'Unknown worker error.'}"

    if state == "STARTED":
        return f"Task `{task_id}` is processing in the background."

    return f"Task `{task_id}` status: {state}"


def process_uploaded_file(file_path: Any):
    """Upload a file to the backend, queue ingestion, and poll until completion."""
    if not file_path:
        yield "Error: No file uploaded."
        return

    if isinstance(file_path, (list, tuple)):
        file_path = str(file_path[0]) if file_path else None
    else:
        file_path = str(file_path)

    if not file_path:
        yield "Error: No file uploaded."
        return

    try:
        file_name = Path(file_path).name

        with open(file_path, "rb") as handle:
            files = {"file": (file_name, handle)}
            response = requests.post(
                f"{API_BASE_URL}/upload",
                files=files,
                timeout=30,
            )

        response.raise_for_status()
        data = response.json()
        task_id = data.get("task_id")

        if not task_id:
            yield "Upload accepted, but no task ID was returned by the backend."
            return

        yield f"Queued `{file_name}` for background ingestion.\nTask ID: `{task_id}`"

        deadline = time.time() + UPLOAD_STATUS_TIMEOUT_SECONDS
        while time.time() < deadline:
            status_response = requests.get(
                f"{API_BASE_URL}/tasks/{task_id}",
                timeout=15,
            )
            status_response.raise_for_status()
            task_payload = status_response.json()

            yield _format_task_status(task_payload)

            if task_payload.get("state") in {"SUCCESS", "FAILURE"}:
                return

            time.sleep(UPLOAD_POLL_INTERVAL_SECONDS)

        yield (
            f"Task `{task_id}` is still running.\n"
            "The document remains queued or processing on the backend."
        )

    except requests.exceptions.ConnectionError:
        yield "Error: Could not connect to backend. Is FastAPI running on port 8000?"
    except requests.exceptions.RequestException as exc:
        yield f"Request Error: {exc}"
    except Exception as exc:
        yield f"Server Error: {exc}"


def _chat_respond(
    message: str,
    history: Optional[List[Dict[str, str]]],
) -> Tuple[List[Dict[str, str]], str]:
    history = list(history or [])

    if not (message or "").strip():
        return history, ""

    history.append({"role": "user", "content": message})

    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"user_query": message},
            timeout=45,
        )
        response.raise_for_status()

        data = response.json()
        reply = data.get("final_response", "Error generating response.")
        is_safe = data.get("is_safe", False)
        context_used = data.get("context_used", False)

        debug_footer = (
            f"\n\n---\n"
            f"*Debug: [Safe: {is_safe} | Context Retrieved: {context_used}]*"
        )

        history.append(
            {"role": "assistant", "content": reply + debug_footer}
        )

    except requests.exceptions.ConnectionError:
        history.append(
            {
                "role": "assistant",
                "content": "Error: Cannot connect to backend. Is the FastAPI server running on port 8000?",
            }
        )
    except requests.exceptions.RequestException as exc:
        history.append(
            {
                "role": "assistant",
                "content": f"Request Error: {exc}",
            }
        )
    except Exception as exc:
        history.append(
            {
                "role": "assistant",
                "content": f"Server Error: {exc}",
            }
        )

    return history, ""


def build_demo() -> gr.Blocks:
    startup_status = "UI Ready. Connect to FastAPI backend for data."

    with gr.Blocks(
        title="NUST Bank Customer Support Hub",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# NUST Bank Agentic Command Center")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Knowledge Base Manager")

                upload = gr.File(
                    file_types=[".txt", ".csv", ".pdf"],
                    label="Upload New Bank Policy",
                    type="filepath",
                )

                update_btn = gr.Button(
                    "Update Knowledge Base",
                    variant="secondary",
                )

                status = gr.Textbox(
                    label="System Status",
                    value=startup_status,
                    interactive=False,
                    lines=5,
                )

                gr.Markdown("---")
                gr.Markdown(
                    "### Architecture Profile\n"
                    "- **Frontend:** Gradio Thin Client\n"
                    "- **Backend:** FastAPI Microservice\n"
                    "- **Queue:** Celery + Redis\n"
                    "- **Vector DB:** Qdrant"
                )

            with gr.Column(scale=2):
                gr.Markdown("### Live Agent Chat")

                chatbot = gr.Chatbot(
                    height=500,
                    label="Assistant",
                    type="messages",
                )

                with gr.Row():
                    user_in = gr.Textbox(
                        label="Your message",
                        placeholder="e.g., What are the requirements for a Roshan Digital Account?",
                        lines=1,
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

        update_btn.click(
            fn=process_uploaded_file,
            inputs=[upload],
            outputs=[status],
        )

        send_btn.click(
            fn=_chat_respond,
            inputs=[user_in, chatbot],
            outputs=[chatbot, user_in],
        )

        user_in.submit(
            fn=_chat_respond,
            inputs=[user_in, chatbot],
            outputs=[chatbot, user_in],
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
