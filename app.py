import gradio as gr
from pinecone import Pinecone
import pandas as pd
from google.generativeai import configure, embed_content, GenerativeModel
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  # For local development

# Configuration with proper error handling
def get_env_var(name, default=None):
    value = os.getenv(name, default)
    if not value and default is None:
        raise ValueError(f"Missing required environment variable: {name}")
    return value

# Load data
try:
    df = pd.read_json("conversation-eng.json")
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame()

# Configure APIs
try:
    # Get API keys from environment (required in production)
    GEMINI_API_KEY = get_env_var("GEMINI_API_KEY")
    PINECONE_API_KEY = get_env_var("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = get_env_var("PINECONE_INDEX_NAME")

    configure(api_key=GEMINI_API_KEY)
    model = 'models/embedding-001'
    gemini_model = GenerativeModel('gemini-1.5-flash')
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    print(f"API configuration error: {e}")
    # Create dummy objects to prevent crashes in demo mode
    class DummyIndex:
        def query(self, *args, **kwargs): return {"matches": []}
    index = DummyIndex()

def query_pinecone(query, top_k=3):
    try:
        query_embedding = embed_content(model=model, content=query, task_type="retrieval_query")["embedding"]
        results = index.query(vector=query_embedding, top_k=top_k, include_values=False, include_metadata=False)
        return results
    except Exception as e:
        print(f"Query error: {e}")
        return {"matches": []}

def generate_response(query, results):
    if df.empty:
        return "System is starting up... Please try again in a moment."
    
    context = ""
    sources = []
    
    for match in results.get('matches', []):
        try:
            title = match['id'].split("_")[0]
            row = df[df['title'] == title].iloc[0]
            context += f"## {title}\n"
            
            if 'link' in row:
                context += f"Source: {row['link']}\n"
                sources.append((title, row['link']))
                
            for msg in row['conversation']:
                context += f"- {msg['content']}\n"
                
            context += "\n"
        except:
            continue
    
    if not context:
        return "🔍 No relevant information found. Try rephrasing your question."
    
    prompt = f"""Answer the following question based on the provided context strictly. Do not hallucinate.
Context:
{context}
Question: {query}
Answer in 2-5 short sentences:"""
    
    try:
        response = gemini_model.generate_content(prompt)
        answer_text = response.text
        
        # Add formatted source links at the end
        if sources:
            answer_text += "\n\n**Sources:**\n"
            for i, (title, link) in enumerate(sources, 1):
                answer_text += f"{i}. [{title}]({link})\n"
        
        return answer_text
    except Exception as e:
        return f"Error: {str(e)}"

def ask(question):
    if not question.strip():
        return "Please enter a valid question."
    
    results = query_pinecone(question)
    return generate_response(question, results)

# Create interface
with gr.Blocks(theme=gr.themes.Soft(), title="TEKNOFEST chatbot") as app:
    gr.Markdown("""# 📚 TEKNOFEST Chatbot - (updated on 4 june,2025)
    Ask questions about project guidelines and policies""")
    
    with gr.Column(scale=2, min_width=300):
        question = gr.Textbox(
            label="Your Question",
            placeholder="E.g. Can I edit my report after submission?",
            lines=2,
            max_lines=2,
            elem_classes="compact-textbox"
        )
        
        # Button below the question box
        with gr.Row():
            # Empty column for spacing
            with gr.Column(scale=1):
                pass
            
            # Button in the middle with custom width
            with gr.Column(scale=0, min_width=80):
                submit_btn = gr.Button(
                    "Ask",
                    variant="primary",
                    size="sm",
                    elem_id="ask-button"
                )
            
            # Empty column for spacing
            with gr.Column(scale=1):
                pass
        
        response = gr.Markdown(
            elem_classes="response-markdown"
        )
    
    examples = gr.Examples(
        examples=[
            "Can I make changes after submitting the report?",
            "What's the deadline for final submissions?",
            "How many team members are allowed?"
        ],
        inputs=question,
        label="Try these examples:"
    )
    
    # Custom CSS to make input/output bars more compact
    gr.HTML("""
    <style>
        .compact-textbox textarea {
            max-width: 500px !important;
            min-width: 300px !important;
            font-size: 14px !important;
            padding: 8px !important;
        }
        .compact-textbox label {
            font-size: 14px !important;
        }
        
        
        
        #ask-button, #ask-button button {
    max-width: 80px !important;
    width: 80px !important;
    min-width: 80px !important;
    padding: 4px 8px !important;
    font-size: 12px !important;
    height: 30px !important;
    line-height: 1 !important;
    display: block !important; 
    flex: none !important;
    margin-top: 8px !important;
    margin-bottom: 16px !important;
    margin-left: 0 !important; 
}
        
        .response-markdown {
            max-width: 500px !important;
            font-size: 14px !important;
            padding: 12px !important;
            background-color: #f8f9fa !important;
            border-radius: 8px !important;
            margin-top: 8px !important;
        }
    </style>
    """)
    
    submit_btn.click(
        fn=ask,
        inputs=question,
        outputs=response,
        api_name="ask"
    )

# Run locally
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )