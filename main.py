import gradio as gr
import os
from app import app  # Imports the compiled LangGraph from app.py

def run_agents(user_input, history):
    """
    This function acts as the bridge between the UI and the Agent Graph.
    """
    try:
        # 1. Prepare the initial state for the agents
        # 'task' matches the key in our AgentState TypedDict in app.py
        inputs = {
            "task": user_input, 
            "iterations": 0
        }
        
        # 2. Invoke the graph 
        # This will run the Researcher -> Reviewer loop defined in app.py
        result = app.invoke(inputs)
        
        # 3. Extract the final outputs from the state
        final_report = result.get("report", "⚠️ No report was generated.")
        reviewer_feedback = result.get("feedback", "No feedback provided.")
        
        # 4. Format the output for the Gradio Chatbot
        formatted_response = (
            f"## 📝 Final Agent Report\n\n"
            f"{final_report}\n\n"
            f"---\n"
            f"### ⚖️ Reviewer Quality Check\n"
            f"*{reviewer_feedback}*"
        )
        return formatted_response

    except Exception as e:
        return f"❌ **System Error:** {str(e)}"

# 5. Create the Gradio Chat Interface
# Note: 'theme' is removed from here to prevent the TypeError in Gradio 6.x/2026
demo = gr.ChatInterface(
    fn=run_agents,
    title="🚀 Autonomous Agent Team",
    description=(
        "This system uses a **Researcher Agent** and a **Reviewer Agent** (via LangGraph) "
        "to collaborate on a technical report for you. Built with Llama 3.3 & Groq."
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
        theme="soft",
        # REPLACED: show_api=False is now footer_links
        footer_links=["gradio", "settings"] 
    )
