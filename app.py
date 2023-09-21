#Bring in deps
import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SequentialChain,SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY']=apikey

#app framework
st.title('🦜️🔗 Paper GPT Creator')
prompt=st.text_input('请输入你的关键词')

title_template=PromptTemplate(
    input_variables=['topic'],
    template='生成与{topic}相关的论文主题,确保文章主题独特，并鼓励模型做出创造性和原创性的回应。同时，保留原始提示词中的必要细节，如具体要求'
             '或数量。优化的提示应该清楚地传达用户想要的结果，并就所需的答案类型提供具体指导。此外，优化的提示应鼓励灵活性和创造性，'
             '同时保持清晰的结构并注重准确性。'
)

paper_template=PromptTemplate(
    input_variables=['title','Wikipedia_research'],
    template='请以为{title}生成一篇论文提纲,同时利用维基百科{Wikipedia_research}生成答案,用中文生成答案'
)



#memory

title_memory=ConversationBufferMemory(input_key='topic',memory_key='chat_history')
paper_memory=ConversationBufferMemory(input_key='title',memory_key='chat_history')


#llms
llm=OpenAI(temperature=0.9)
title_chain=LLMChain(llm=llm, prompt=title_template, verbose=True,output_key='title', memory=title_memory)
paper_chain=LLMChain(llm=llm, prompt=paper_template, verbose=True, output_key='paper',memory=paper_memory)

# sequential_chain=SequentialChain(chains=[title_chain,paper_chain],input_variables=['topic'],
#                                  output_variables=['title','paper'],verbose=True)
wiki=WikipediaAPIWrapper()

#show stuff to the screen if there is a prompt
if prompt:
    # response=sequential_chain({'topic':prompt})
    # st.write(response['title'])
    # st.write(response['paper'])

    title=title_chain.run(prompt)
    wiki_research=wiki.run(prompt)
    paper=paper_chain.run(title=title, Wikipedia_research=wiki_research)

    st.write(title)
    st.write(paper)

    with st.expander('主题词历史记录'):
        st.info(title_memory.buffer)

    with st.expander('提纲历史记录'):
        st.info(paper_memory.buffer)

    with st.expander('维基百科历史记录'):
        st.info(wiki_research)