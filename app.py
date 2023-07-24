from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.chat_models import ChatOpenAI
import requests
import os
import streamlit as st

#https://www.youtube.com/watch?v=_j7JEDWuqLE

load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# image 2 text
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", max_new_tokens=1000)   
        
    text = image_to_text(url)[0]["generated_text"] 

    print(text)
    return text




# gen story via LLM
def generate_story(scenario):
    template = """
    You are a story teller;
    you can generate a short story based on a single narrative, the story should be no more then 60 words;
    CONTEXT: {scenario}
    STORY:
    """

    prompt = PromptTemplate(template=template,input_variables=["scenario"])

    story_llm = LLMChain(llm=OpenAI(
        model_name="gpt-3.5-turbo", temperature=1), prompt=prompt, verbose=True)
    
    story = story_llm.predict(scenario=scenario)

    #print(story)

    return story

# tts
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payloads = {
         "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)

    



# main program

def main():
    st.set_page_config(page_title="img 2 audio story", page_icon="😃")

    st.header("turn img into audio story")
    uploaded_file = st.file_uploader("choose an image...", type=['png', 'jpg'])

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="uploaded image", use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        st.audio("audio.flac")
        

      


#imgDescription = img2text("mis-imp.png")

#imgDescription = "two planes in sky"
#generatedStory = generate_story(imgDescription)
#print(generatedStory)

generatedStory = "Two men in suits pointed at the camera, revealing their plot to undermine the nation's security."
text2speech(generatedStory)

if __name__ == '__main__':
    main()



  #python -m streamlit run app.py