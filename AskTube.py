from typing import List
import requests
import pandas as pd
from haystack import Document
from haystack.generator.transformers import RAGenerator

from haystack.retriever.dense import DensePassageRetriever
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.retriever.sparse import TfidfRetriever

from haystack.document_store.faiss import FAISSDocumentStore

from haystack.reader.farm import FARMReader

import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi
from pyyoutube import Api
from pytube import YouTube

import os
from subprocess import Popen, PIPE, STDOUT

document_store = FAISSDocumentStore()

retriever = TfidfRetriever(document_store)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

from haystack import Pipeline

p = Pipeline()
p.add_node(component=retriever, name="Retriever", inputs=["Query"])
p.add_node(component=reader, name="Reader", inputs=["Retriever"])
# p.add_node(component=generator, name="Generator", inputs=["Retriever"])

def answer_question(question, answer_texts):
  docs = [Document(text=answer_text, meta={}) for answer_text in answer_texts]

  document_store.delete_documents()
  document_store.write_documents(docs)
  document_store.update_embeddings(
    retriever=retriever
  )
  # retriever.fit()

  prediction = p.run(query=question, top_k_retriever=5 , top_k_reader=3)

  return prediction

from youtube_transcript_api import YouTubeTranscriptApi

def video2text(video_id):
  transcript_list = YouTubeTranscriptApi.list_transcripts('e6_t26Q9aVM')
  transcript = transcript_list.find_transcript(['en'])
  fetched_transcript = transcript.fetch()
  transcript = ''
  for ts in fetched_transcript:
      transcript += f" {ts['text']} "
  return transcript

yt_api = Api(api_key="AIzaSyABaeCa_GEW4ePYNfYwP9qtsHAMN8s8kxs")


def get_channel_videos(channel_id , playlist_number):
  playlists_by_channel = yt_api.get_playlists(channel_id=channel_id)
  first_playlist_id = playlists_by_channel.items[playlist_number - 1].id

  playlist = yt_api.get_playlist_items(playlist_id=first_playlist_id)

  video_ids = []
  for i in range(len(playlist.items)):
    video_ids.append(playlist.items[i].snippet.resourceId.videoId)
  
  return video_ids

def ai_answer(video_url , question):
  try:
    video_id = video_url.split('=')[1]
  except:
    video_id = video_url.split('/')[3]

  # video_ids = get_channel_videos(channel_id , int(playlist_number))
  # transcripts = [video2text(video_id) for video_id in video_ids]
  transcript = video2text(video_id)
  
  # docs = [Document(text=answer_text, meta={}) for answer_text in transcripts]

  # answers = generator.predict(query=question, documents = docs)
  answers = answer_question(question , [transcript])
  answers = [answer['answer'] for answer in answers['answers']]
  answers_text = '\n \n'.join(answers)
  
  return answers[0]

ai_answer('https://www.youtube.com/AYdmelm9ADM', 'when did you start guitar?')

gr.Interface(ai_answer, inputs = ['text' , 'textbox'] , outputs = ['text'], title="AskTube" , allow_flagging=False , layout='vertical' , description='made by Sorayesh').launch(debug=True)
