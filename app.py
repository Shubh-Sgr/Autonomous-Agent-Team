import os
import gradio as gr
from agent import agent_app

def run_agents(user_input, history):
    try:
        inputs = {"task": user_input, "iterations": 0}
        result = agent_app.invoke(inputs)
        
        final_report = result.get("report", "⚠️ No report was generated.")
        reviewer_feedback = result.get("feedback", "No feedback provided.")
        
        return (
            f"## 📝 Final Agent Report\n\n"
            f"{final_report}\n\n"
            f"---\n"
            f"### ⚖️ Reviewer Quality Check\n"
            f"*{reviewer_feedback}*"
        )
    except Exception as e:
        return f"❌ **System Error:** {str(e)}"

demo = gr.ChatInterface(
    fn=run_agents,
    title="🚀 Autonomous Agent Team",
    description=(
        "This system uses a **Researcher Agent** and a **Reviewer Agent** (via LangGraph) "
        "to collaborate on technical reports."
    ),
    examples=[
        "Explain the potential impact of Room Temperature Superconductors.",
        "Write a technical summary of how Vector Databases work.",
        "Draft a project plan for building a solar-powered IoT sensor."
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0", 
        server_port=port,
        theme="soft"
    )
