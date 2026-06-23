import traceback
import gradio as gr
from explain import analyze_code

def run_analysis(code_input):
    try:
        if not code_input or not code_input.strip():
            return "Please paste some Python code to analyze."
        
        result = analyze_code(code_input)
        
        if result.get("verdict") == "ERROR":
            nodes = result.get("explanation_nodes", [])
            flows = result.get("explanation_flows", [])
            error_msg = "Unknown error"
            if nodes and "code" in nodes[0]:
                error_msg = nodes[0]["code"]
            elif flows:
                error_msg = "\n".join(flows)
            return f"⚠️ ANALYSIS ERROR\n\n{error_msg}"
        
        verdict = result.get("verdict", "UNKNOWN")
        confidence = result.get("confidence", 0.0)
        nodes = result.get("explanation_nodes", [])
        flows = result.get("explanation_flows", [])
        
        output = f"Verdict: {verdict}\nConfidence: {confidence:.1f}%\n\n"
        
        if nodes:
            output += "🔍 Suspicious Code Elements:\n"
            for i, node in enumerate(nodes[:3], 1):
                output += f"{i}. [{node.get('type', 'N/A')}] {node.get('code', 'N/A')} (Score: {node.get('score', 0):.2f})\n"
            output += "\n"
        
        if flows:
            output += "🔗 Data Flows:\n"
            for i, flow in enumerate(flows[:5], 1):
                output += f"{i}. {flow}\n"
        
        return output
        
    except Exception as e:
        return f"🚨 UNCAUGHT EXCEPTION:\n\n{traceback.format_exc()}"

iface = gr.Interface(
    fn=run_analysis,
    inputs=gr.Textbox(lines=15, placeholder="Paste Python Code to Analyze...", label="Python Source Code"),
    outputs=gr.Textbox(lines=15, label="Analysis Result"),
    title="🔬 Semantic CPG Backdoor Detector",
    description="Paste Python code below to analyze it for supply chain backdoor patterns using CodeBERT + Graph Neural Networks + GNNExplainer.\n\nInspired by real-world attacks like XZ Utils (CVE-2024-3094)."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
