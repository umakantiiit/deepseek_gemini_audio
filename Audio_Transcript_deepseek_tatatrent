import streamlit as st
import google.generativeai as genai
from together import Together
import os
from pathlib import Path
import json
from groq import Groq

# Configure the generative AI API
genai.configure(api_key=st.secrets["gemini_api_key"])

# Initialize Groq client
Together_client = Together(api_key=st.secrets["together_key"])

# Helper function to upload files to Gemini
def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    return file

# Common generation configuration
generation_config = {
    "temperature": 0.2,
    "response_mime_type": "application/json"
}

# Prompts for the models
Prompt_for_audio_transcript = '''
You are an advanced AI assistant specialized in audio processing, speaker diarization, and emotion detection. Your expertise lies in analyzing audio files, identifying speakers, transcribing conversations, and detecting emotions in real-time. Your task is to process an audio file from a call center and provide a detailed, structured output in JSON format.
Task:

Number of Speakers: Identify and state the total number of unique speakers in the audio file.

Transcript with Speaker Labels: Generate a clear and accurate transcript of the audio, labeling each segment of speech with the corresponding speaker (e.g.Agent,Client etc). Use proper punctuation and formatting for readability.

Emotion Detection: For each speaker at every point in the conversation, detect and note their emotion (e.g., happy, sad, angry, neutral, frustrated, etc.). Provide a timeline of emotions in JSON format.

Guidelines:
- Use clear, concise, and professional language.
-Ensure the transcript is accurate and easy to read.
-If a speaker cannot be identified, label them as "Unknown."
-Emotions should be detected for each speaker at every conversational turn.
-Follow the JSON output format strictly.

Output Format:
Provide the output in the following JSON structure:
{
  "Call Details": {
    "Number of Speakers": "<total_number_of_speakers>",
    "Transcript": [
      {
        "Speaker": "<Agent/client/Unknown>",
        "Voice": "<extracted_text_from_audio>",
        "Emotion": "<detected_emotion>"
      },
      {
        "Speaker": "<Agent/client/Unknown>",
        "Voice": "<extracted_text_from_audio>",
        "Emotion": "<detected_emotion>"
      },
      ...
    ]
  }
}
Example Output:
{
  "Call Details": {
    "Number of Speakers": 2,
    "Transcript": [
      {
        "Speaker": "Agent",
        "Voice": "Hello, how can I assist you today?",
        "Emotion": "neutral"
      },
      {
        "Speaker": "Client",
        "Voice": "I’m having issues with my recent order.",
        "Emotion": "frustrated"
      },
      {
        "Speaker": "Agent",
        "Voice": "I’m sorry to hear that. Can you provide your order number?",
        "Emotion": "neutral"
      },
      ...
    ]
  }
}
'''

system_prompt_audio = '''You are a highly skilled AI assistant with a deep understanding of audio analysis, natural language processing, and emotional intelligence. You are meticulous, detail-oriented, and committed to delivering accurate and structured results. Your goal is to provide a comprehensive analysis of the call center audio, ensuring the transcript is clear, emotions are accurately detected, and the output is well-organized for further use.'''

system_prompt_json = '''You are an AI trained in analyzing customer service call transcripts. Your expertise lies in detecting stockouts ,analysing new requirments and positive feedbacks from vendor and capable of providing structured outputs in JSON format.'''

prompt_transcript_to_output = '''
Analyze the provided JSON input, which contains a transcript of a call from a vendor who speaks about product quality,stockouts,new requirments and feedback about the seller who provides item to them and  Extract the following details and present them in a structured JSON format:
	Stockouts: Identify all product categories/items mentioned as having replenishment issues (not being restocked)
New Requirements: Identify any new product requirements mentioned (new items requested that aren't part of current replenishment)
Positive Feedback: Extract any positive comments about products (fabric, design, options, etc.)
Return the information in JSON format with the following structure:
{
"analysis": {
"stockouts": {
"categories": ["list of categories with stockout issues"],
"specific_items": ["list of specific items with stockout"]
},
"new_requirements": {
"new_models": ["list of new models/requested items"],
"details": "description of requirements"
},
"positive_feedback": {
"product_qualities": ["list of praised attributes"],
"business_impact": "any mentioned positive business outcomes"
}
},
Summary of the whole Conversation:
Important words detected by the Vendor:
}"
Example json output:
{  
  "analysis": {  
    "stockouts": {  
      "categories": [  
        "Seamless (soft bras)",  
        "Camisole colors",  
        "Full coverage charcoal bras",  
        "Sleepwear (pajama sets)",  
        "Luxury floral camisoles",  
        "Daily essentials",  
        "Superstar (cotton camisoles, full color sets, black/white bras)"  
      ],  
      "specific_items": [  
        "Seamless soft bras replacements",  
        "Camisole color variants",  
        "Charcoal full coverage bras (sizes)",  
        "Pajama set matches",  
        "Luxury floral camisole colors/sizes",  
        "Daily essential pieces",  
        "Superstar pack of 2 cotton camisoles",  
        "Superstar full color sets",  
        "Superstar black/white bras"  
      ]  
    },  
    "new_requirements": {  
      "new_models": [],  
      "details": "No explicit mention of new product requirements. All requests relate to replenishment of existing items."  
    },  
    "positive_feedback": {  
      "product_qualities": [  
        "Fabric quality",  
        "Design appeal",  
        "Product options"  
      ],  
      "business_impact": "Double-digit growth attributed to product appeal despite stockout issues"  
    }  
  }  
  Summary of the Conversation :
  Important word detected by the vendor:
}

# Initialize the model for audio processing
model_audio = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=system_prompt_audio
)

st.title("Welcome to CurateAI Audio Assistant with Deepseek")

# Placeholder for storing the first API call result
transcript_json = None

# Audio file upload section
uploaded_audio = st.file_uploader("Upload an audio file", type=["mp3", "aac", "wav", "aiff"], accept_multiple_files=False)

if uploaded_audio is not None:
    file_extension = Path(uploaded_audio.name).suffix.lower()
    valid_extensions = [".mp3", ".aac", ".wav", ".aiff"]
    
    if file_extension not in valid_extensions:
        st.error("AUDIO FILE IS NOT IN VALID FORMAT")
    else:
        # Save the uploaded audio file temporarily
        with open(uploaded_audio.name, "wb") as f:
            f.write(uploaded_audio.getbuffer())
        
        # Upload to Gemini and process
        mime_type = f"audio/{file_extension.strip('.') if file_extension != '.mp3' else 'mpeg'}"
        myaudio = upload_to_gemini(uploaded_audio.name, mime_type=mime_type)
        
        st.audio(uploaded_audio, format=mime_type)
        
        if st.button("View Transcript"):
            response_audio = model_audio.generate_content([myaudio, Prompt_for_audio_transcript], generation_config=generation_config)
            try:
                transcript_json = json.loads(response_audio.text)
                st.json(transcript_json, expanded=True)
                st.session_state.transcript_json = transcript_json
                st.success("GREAT! Transcript generated successfully! You can now proceed to detailed analysis.")
            except json.JSONDecodeError:
                st.write("Here is the raw output from the model:")
                st.text(response_audio.text)

# View Detailed Analysis button
if st.session_state.get("transcript_json") is not None:
    if st.button("View Detailed Analysis"):
        transcript_json = st.session_state.transcript_json
        
        # Prepare the Groq API call
        formatted_json = json.dumps(transcript_json, indent=2)
        
        messages = [
            {
                "role": "user",
                "content": f"{prompt_transcript_to_output}\n\nHere's the JSON data:\n```json\n{formatted_json}\n```"
            }
        ]
        
        # Get completion from Groq
        chat_completion = Together_client.chat.completions.create(
            messages=messages,
            model="deepseek-ai/DeepSeek-R1",
            temperature=0.2,
            max_tokens=2048
        )
        
        try:
            analysis_text = chat_completion.choices[0].message.content
            st.markdown(f"```\n{analysis_text}\n```")

        except Exception as e:
            st.error(f"Error in generating analysis: {str(e)}")

# Clean up temporary files after session
@st.cache_data()
def get_session_files():
    return []

def remove_temp_files():
    for file_path in get_session_files():
        if os.path.exists(file_path):
            os.remove(file_path)

remove_temp_files()
