import os
import glob
import json
import getpass
import platform
from typing import Any
from pathlib import Path
from biomni.agent.a1 import A1
from langchain_core.messages import HumanMessage

class AD1(A1):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ad_keywords = [
            "Alzheimer", "AD", "dementia", "MCI", "amyloid", "tau", 
            "neurodegeneration", "cognition"
        ]
        
    def go(self, prompt):
        """Execute the agent with the given prompt, injecting AD context if relevant."""
        
        # Check for AD keywords
        is_ad_task = any(keyword.lower() in prompt.lower() for keyword in self.ad_keywords)
        
        if is_ad_task:
            print("\n🧠 AD/Dementia task detected. Injecting specialized data sourcing protocols...")
            self._inject_ad_context()
        
        # Run the agent
        result = super().go(prompt)
        
        return result

    def _inject_ad_context(self):
        """Inject BiomniAD data sourcing instructions into the system prompt."""
        try:
            # Locate the data sourcing markdown
            # Assuming the file is in the same project structure relative to this file
            # biomni/agent/ad1.py -> ../know_how/biomniAD_data_sourcing.md
            current_dir = os.path.dirname(os.path.abspath(__file__))
            know_how_path = os.path.join(current_dir, "..", "know_how", "biomniAD_data_sourcing.md")
            
            if os.path.exists(know_how_path):
                with open(know_how_path, "r") as f:
                    ad_sourcing_content = f.read()
                
                # Append to system prompt
                # We add it as a high priority instruction
                ad_instruction = f"""
                
                IMPORTANT: ALZHEIMER'S & DEMENTIA DATA SOURCING PROTOCOL
                PRIORITIZE using this data & only supplement with other data as needed. Do NOT simulate data for analyses.
                ========================================================
                {ad_sourcing_content}
                ========================================================
                """
                
                # Update the system prompt
                self.system_prompt += ad_instruction
                
                # Also update the app's system message if it's already compiled
                # Note: In A1.go(), the system prompt is passed to the graph. 
                # Since we modify self.system_prompt before super().go(), 
                # A1.go() will use the updated prompt when it calls generate().
                
            else:
                print(f"Warning: Could not find AD data sourcing guide at {know_how_path}")
                
        except Exception as e:
            print(f"Warning: Failed to inject AD context: {e}")


    def launch_ui(
        self,
        thread_id: int = 42,
        share: bool = False,
        server_name: str = "0.0.0.0",
        require_verification: bool = False,
    ) -> None:
        """Launch the Biomni AD1 Web UI.
        """
        try:
            import gradio as gr
            from gradio import ChatMessage
        except ImportError as exc:
            raise ImportError(
                "Gradio is not installed. Please install it with: pip install gradio"
            ) from exc

        from time import time
        import re

        supported_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".pdf")

        self.main_history_copy = []
        available_access_codes = ["Biomni2025"]

        def verify_access_code(code: str):
            if code in available_access_codes:
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(
                    value="Incorrect access code. Please check your access code.",
                    visible=True,
                ),
            )

        def get_all_files(directory):
            """Recursively get all files in a directory."""
            file_list = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    # Ignore hidden files and directories
                    if file.startswith('.') or '/.' in root:
                        continue
                    file_list.append(os.path.join(root, file))
            return set(file_list)

        def generate_response(prompt_input, inner_history=None, main_history=None):
            if main_history is None:
                main_history = []
            if inner_history is None:
                inner_history = []

            text_input = prompt_input.get("text", "")
            files = prompt_input.get("files", [])
            
            # Capture initial file state
            initial_files = get_all_files(os.getcwd())

            self.main_history_copy += [{"role": "user", "content": text_input}]
            main_history.append(
                ChatMessage(
                    role="user",
                    content=text_input if text_input else "[Uploaded file]",
                    metadata={"title": "👤 User"},
                )
            )

            main_history.append(
                ChatMessage(
                    role="assistant",
                    content="Executor is working on it 👉",
                    metadata={"title": "🛠️ Planning"},
                )
            )
            # AD1 Specific: Inject AD context if keywords are present
            is_ad_task = any(keyword.lower() in text_input.lower() for keyword in self.ad_keywords)
            if is_ad_task:
                print("\n🧠 AD/Dementia task detected. Injecting specialized data sourcing protocols...")
                self._inject_ad_context()
                inner_history.append(
                    ChatMessage(
                        role="assistant",
                        content="🧠 AD/Dementia task detected. Injecting specialized data sourcing protocols...",
                        metadata={"title": "🧠 AD Context"},
                    )
                )
                yield inner_history, main_history, gr.update(), gr.update(), gr.update()

            for file_info in files:
                file_path = file_info
                text_input += f"\n\n User uploaded this file: {file_path}\n Please use it if needed."

            agent_messages = []
            for msg in self.main_history_copy:
                if msg["role"] == "user":
                    agent_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant" and msg["content"] not in ["Executor is working on it 👉"]:
                    from langchain_core.messages import AIMessage
                    agent_messages.append(AIMessage(content=msg["content"]))

            agent_messages.append(HumanMessage(content=text_input))

            inputs = {"messages": agent_messages, "next_step": None}
            config = {"recursion_limit": 500, "configurable": {"thread_id": thread_id}}

            t = time()
            solution_found = False

            # Check for tool retriever if it exists (A1 feature)
            if getattr(self, "use_tool_retriever", False):
                inner_history.append(
                    ChatMessage(
                        role="assistant",
                        content="Retrieving relevant tools, data lake items, and libraries...",
                        metadata={"title": "🔍 Tool retrieval"},
                    )
                )
                yield inner_history, main_history, gr.update(), gr.update(), gr.update()

                try:
                    selected_resources_names = self._prepare_resources_for_retrieval(text_input)
                    if selected_resources_names:
                        self.update_system_prompt_with_selected_resources(selected_resources_names)
                except Exception as exc:
                    print(f"Warning: Tool retrieval failed: {exc}")
                    inner_history.append(
                        ChatMessage(
                            role="assistant",
                            content="Tool retrieval unavailable, proceeding with all tools...",
                            metadata={"title": "⚠️ Tool retrieval"},
                        )
                    )
                    yield inner_history, main_history, gr.update(), gr.update(), gr.update()

            code_execution_messages = []

            for s in self.app.stream(inputs, stream_mode="values", config=config):
                t_step = time() - t
                message = s["messages"][-1]

                if message.content == text_input:
                    t = time()
                    continue

                if isinstance(message.content, str):
                    tag_positions = [
                        pos
                        for tag in ["<execute>", "<solution>", "<observation>"]
                        if (pos := message.content.find(tag)) != -1
                    ]

                    if tag_positions:
                        first_tag_pos = min(tag_positions)
                        thinking = message.content[:first_tag_pos].strip()
                        if thinking:
                            inner_history.append(
                                ChatMessage(
                                    role="assistant",
                                    content=thinking,
                                    metadata={"title": "🤔 Reasoning"},
                                )
                            )
                            yield inner_history, main_history, gr.update(), gr.update(), gr.update()

                    solution_match = re.search(r"<solution>(.*?)</solution>", message.content, re.DOTALL)
                    if solution_match and not solution_found:
                        solution_found = True
                        solution = solution_match.group(1).strip()
                        main_history.append(
                            ChatMessage(
                                role="assistant",
                                content=solution,
                                metadata={"title": "✅ Answer", "log": "Final answer"},
                            )
                        )
                        self.main_history_copy += [{"role": "assistant", "content": solution}]
                        yield inner_history, main_history, gr.update(), gr.update(), gr.update()

                    execute_match = re.search(r"<execute>(.*?)</execute>", message.content, re.DOTALL)
                    if execute_match:
                        code = execute_match.group(1).strip()
                        language = "python"
                        if code.strip().startswith("#!R"):
                            language = "r"
                            code = re.sub(r"^#!R", "", code, count=1).strip()
                        elif code.strip().startswith("#!BASH") or code.strip().startswith("#!CLI"):
                            language = "bash"
                            code = re.sub(r"^#!BASH|^#!CLI", "", code, count=1).strip()

                        code_msg = ChatMessage(
                            role="assistant",
                            content=f"##### Code: \n```{language}\n{code}\n```",
                            metadata={
                                "title": "🛠️ Executing code...",
                                "status": "pending",
                                "start_time": t,
                            },
                        )
                        inner_history.append(code_msg)
                        code_execution_messages.append(code_msg)
                        yield inner_history, main_history, gr.update(), gr.update(), gr.update()

                    observation_match = re.search(
                        r"<observation>(.*?)</observation>", message.content, re.DOTALL
                    )
                    if observation_match:
                        observation = observation_match.group(1).strip()

                        if code_execution_messages:
                            code_msg = code_execution_messages[-1]
                            code_msg.metadata.update(
                                {
                                    "status": "done",
                                    "duration": t_step,
                                    "log": f"Code execution completed in {t_step:.2f}s",
                                }
                            )

                        inner_history.append(
                            ChatMessage(
                                role="assistant",
                                content=f"##### Observation: \n```\n{observation}\n```",
                                metadata={
                                    "status": "done",
                                    "duration": t_step,
                                    "log": "Observation from code execution",
                                    "collapsed": True,
                                    "collapsible": True,
                                },
                            )
                        )
                        yield inner_history, main_history, gr.update(), gr.update(), gr.update()

                        if isinstance(observation, str) and any(
                            ext in observation for ext in supported_extensions
                        ):
                            matches = re.findall(
                                r"(\S+?(?:\.png|\.jpg|\.jpeg|\.gif|\.bmp|\.webp|\.pdf))",
                                observation,
                            )
                            valid_matches = []
                            for match in matches:
                                if not (
                                    match.startswith("Warning:")
                                    or match.startswith("Error:")
                                    or match.startswith("'")
                                ):
                                    if not match.startswith("."):
                                        valid_matches.append(match)

                            if valid_matches:
                                inner_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content="",
                                        metadata={"title": "📁 Files"},
                                    )
                                )

                                for file_path in valid_matches:
                                    file_path = file_path.strip("\"'").strip()
                                    abs_path = None
                                    if os.path.isabs(file_path) and os.path.exists(file_path):
                                        abs_path = file_path
                                    elif os.path.exists(os.path.join(os.getcwd(), file_path)):
                                        abs_path = os.path.join(os.getcwd(), file_path)
                                    elif (
                                        hasattr(self, "path")
                                        and self.path
                                        and os.path.exists(os.path.join(self.path, file_path))
                                    ):
                                        abs_path = os.path.join(self.path, file_path)

                                    if abs_path:
                                        if file_path.lower().endswith(".pdf"):
                                            inner_history.append(
                                                ChatMessage(
                                                    role="assistant",
                                                    content=f"Found PDF at: {abs_path}",
                                                    metadata={"title": "📄 PDF File"},
                                                )
                                            )
                                        else:
                                            inner_history.append(
                                                ChatMessage(
                                                    role="assistant",
                                                    content=gr.Image(abs_path),
                                                    metadata={"title": "🖼️ Image Preview"},
                                                )
                                            )
                                yield inner_history, main_history, gr.update(), gr.update(), gr.update()

                t = time()

            final_message = s["messages"][-1].content if s.get("messages") else ""
            if not solution_found:
                solution_match = re.search(r"<solution>(.*?)</solution>", final_message, re.DOTALL)
                if solution_match:
                    solution = solution_match.group(1).strip()
                    main_history.append(
                        ChatMessage(
                            role="assistant",
                            content=solution,
                            metadata={"title": "✅ Solution"},
                        )
                    )
                    self.main_history_copy += [{"role": "assistant", "content": solution}]
                else:
                    cleaned_content = re.sub(r"<execute>.*?</execute>", "", final_message, flags=re.DOTALL)
                    cleaned_content = re.sub(
                        r"<observation>.*?</observation>", "", cleaned_content, flags=re.DOTALL
                    )
                    cleaned_content = re.sub(r"\n\s*\n", "\n\n", cleaned_content)

                    summary = cleaned_content.strip() or (
                        "Task completed. Please check the execution log for details."
                    )
                    main_history.append(
                        ChatMessage(
                            role="assistant",
                            content=summary,
                            metadata={"title": "📝 Summary"},
                        )
                    )
                    self.main_history_copy += [{"role": "assistant", "content": summary}]

            # Restore system prompt if it was modified (e.g. by AD context)
            # Note: AD context modifies self.system_prompt in place, so we might want to reset it if we want isolation.
            # But for now, let's just leave it as is or reset if we had a mechanism.
            # The previous code had `self.system_prompt = original_system_prompt` but we removed the directory injection part.
            
            inner_history.append(
                ChatMessage(
                    role="assistant",
                    content="👈 Returning the result to the main interface...",
                    metadata={"title": "🔄 Complete"},
                )
            )

            # Capture final file state and find new files
            final_files = get_all_files(os.getcwd())
            new_files = sorted(list(final_files - initial_files))
            
            status_lines = []
            
            if new_files:
                status_lines.append("Files created during this run:")
                for f in new_files:
                    # Get relative path for display
                    try:
                        rel_path = os.path.relpath(f, os.getcwd())
                    except ValueError:
                        rel_path = f
                    
                    status_lines.append(f"- [{rel_path}](file://{f})")
            else:
                status_lines.append("- No new files were created during this run.")

            status_md = "\n".join(status_lines)

            # We don't have specific PDF/Notebook outputs anymore, so hide them
            yield inner_history, main_history, gr.update(value=status_md, visible=True), gr.update(visible=False), gr.update(visible=False)

        def like(data: Any = None) -> None:
            """Handle like/dislike events from the chatbot."""
            if data is not None:
                print("User liked the response")
                print(f"Index: {data.index}, Liked: {data.liked}")

        # Layout: verification (optional) + main workspace
        # Custom CSS for Roboto font and professional styling
        custom_css = """
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

        * {
            font-family: 'Roboto', sans-serif !important;
        }

        .gradio-container {
            font-family: 'Roboto', sans-serif !important;
        }

        /* Larger font sizes for chat messages and execution trace */
        .message-wrap .message {
            font-size: 16px !important;
            line-height: 1.6 !important;
        }

        .message-wrap p {
            font-size: 16px !important;
            line-height: 1.6 !important;
            margin-bottom: 0.75em !important;
        }

        /* Better spacing and professional look */
        .chatbot {
            border-radius: 8px !important;
        }

        .message-wrap {
            padding: 12px 16px !important;
        }

        /* Clean, modern button styling */
        button {
            border-radius: 6px !important;
            font-weight: 500 !important;
        }

        /* Professional input styling */
        textarea, input {
            border-radius: 6px !important;
            font-size: 15px !important;
        }

        /* Better label typography */
        label {
            font-weight: 500 !important;
            font-size: 14px !important;
            margin-bottom: 8px !important;
        }
        """

        with gr.Blocks(title="Biomni AD1 Agent", theme=gr.themes.Soft(), css=custom_css) as demo:
            # AD1 Specific: Logo path
            logo_path = Path(__file__).resolve().parents[2] / "figs" / "Biomni-AD_Logo_v2.png"
            # Fallback if v2 logo doesn't exist, try standard one or just text
            if not logo_path.exists():
                 logo_path = Path(__file__).resolve().parents[2] / "figs" / "biomni_logo.png"

            verification_container = gr.Group(visible=require_verification)
            main_interface_container = gr.Group(visible=not require_verification)

            with verification_container:
                if logo_path.exists():
                    gr.Image(logo_path, show_label=False, height=80)
                gr.Markdown("## Biomni AD1 Agent - Access Verification")
                gr.Markdown("Enter your access code to continue.")
                access_code_input = gr.Textbox(label="Access Code", type="password")
                access_error_msg = gr.Markdown(visible=False)
                verify_btn = gr.Button("Verify Access", variant="primary")
                verify_btn.click(
                    fn=verify_access_code,
                    inputs=[access_code_input],
                    outputs=[verification_container, main_interface_container, access_error_msg],
                )

            with main_interface_container:
                if logo_path.exists():
                    gr.Image(logo_path, show_label=False, height=80)
                gr.Markdown("## Biomni-AD — Alzheimer's Disease & Related Dementia Research Copilot")

                with gr.Row():
                    with gr.Column(scale=2):
                        main_chatbot = gr.Chatbot(
                            label="Agent chat",
                            type="messages",
                            height=600,
                            show_copy_button=True,
                            show_share_button=True,
                            autoscroll=False,
                        )
                        prompt_input = gr.MultimodalTextbox(
                            interactive=True,
                            file_count="multiple",
                            placeholder=(
                                "Describe your biomedical or Alzheimer's research question, "
                                "and optionally upload files (e.g. tables, figures, PDFs)..."
                            ),
                            show_label=False,
                        )
                    with gr.Column(scale=3):
                        innerloop_chatbot = gr.Chatbot(
                            label="Biomni-AD execution trace",
                            type="messages",
                            height=600,
                            show_copy_button=True,
                            show_share_button=True,
                        )
                        gr.Markdown("### Run artifacts", elem_classes="artifacts-header")
                        run_status = gr.Markdown(
                            value=(
                                "Artifacts from your latest run will appear here."
                            ),
                            visible=True,
                        )
                        # Hidden file components as placeholders if we ever need them back
                        pdf_output = gr.File(label="Latest report (PDF)", interactive=False, visible=False)
                        notebook_output = gr.File(
                            label="Latest notebook", interactive=False, visible=False
                        )

                prompt_input.submit(
                    generate_response,
                    [prompt_input, innerloop_chatbot, main_chatbot],
                    [innerloop_chatbot, main_chatbot, run_status, pdf_output, notebook_output],
                ).then(lambda: gr.MultimodalTextbox(value=None), None, [prompt_input])
                main_chatbot.like(like)

        print(f"Launching Biomni AD1 Gradio demo on {server_name}:7860")
        demo.launch(share=share, server_name=server_name)
