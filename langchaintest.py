import os
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.agent_toolkits.openapi import planner
from langchain_openai import ChatOpenAI
import spotipy.util as util
from langchain.requests import RequestsWrapper
import tiktoken
import openai

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

with open("openai_openapi.yaml") as f:
    raw_openai_api_spec = yaml.load(f, Loader=yaml.Loader)
openai_api_spec = reduce_openapi_spec(raw_openai_api_spec)

with open("klarna_openapi.yaml") as f:
    raw_klarna_api_spec = yaml.load(f, Loader=yaml.Loader)
klarna_api_spec = reduce_openapi_spec(raw_klarna_api_spec)

with open("spotify_openapi.yaml") as f:
    raw_spotify_api_spec = yaml.load(f, Loader=yaml.Loader)
spotify_api_spec = reduce_openapi_spec(raw_spotify_api_spec)


def construct_spotify_auth_headers(raw_spec: dict):
    scopes = list(
        raw_spec["components"]["securitySchemes"]["oauth_2_0"]["flows"][
            "authorizationCode"
        ]["scopes"].keys()
    )
    access_token = util.prompt_for_user_token(scope=",".join(scopes))
    return {"Authorization": f"Bearer {access_token}"}


# Get API credentials.
headers = construct_spotify_auth_headers(raw_spotify_api_spec)
requests_wrapper = RequestsWrapper(headers=headers)

endpoints = [
    (route, operation)
    for route, operations in raw_spotify_api_spec["paths"].items()
    for operation in operations
    if operation in ["get", "post"]
]
print(len(endpoints))

enc = tiktoken.encoding_for_model("gpt-4")


def count_tokens(s):
    return len(enc.encode(s))


print(count_tokens(yaml.dump(raw_spotify_api_spec)))

# max_tokens_per_chunk = 5000
# handle_parsing_errors = True
#
# user_query = "make me a playlist with the first song from kind of blue. call it machine blues."
#
# chunks = [user_query[i:i + max_tokens_per_chunk] for i in range(0, len(user_query), max_tokens_per_chunk)]
#
# llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
#
# spotify_agent = planner.create_openapi_agent(spotify_api_spec, requests_wrapper, llm)
# for chunk in chunks:
#     spotify_agent.invoke(input=chunk, handle_parsing_errors=handle_parsing_errors)
#

# max_tokens_per_chunk = 5000
# handle_parsing_errors = True
#
# user_query = "make me a playlist with the first song from kind of blue. call it machine blues."
#
# chunks = [user_query[i:i + max_tokens_per_chunk] for i in range(0, len(user_query), max_tokens_per_chunk)]
#
# llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
#
# spotify_agent = planner.create_openapi_agent(spotify_api_spec, requests_wrapper, llm)
# for chunk in chunks:
#     spotify_agent.invoke(input=chunk, handle_parsing_errors=handle_parsing_errors)


llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.0)
spotify_agent = planner.create_openapi_agent(spotify_api_spec, requests_wrapper, llm)
spotify_agent.run("make me a playlist with the first song from kind of blue. call it machine blues")

# Create the ChatOpenAI model and Spotify agent
# llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.0)
# spotify_agent = planner.create_openapi_agent(spotify_api_spec, requests_wrapper, llm)
#
# r_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=0
# )
# input_text = "make me a playlist with the first song from kind of blue. call it machine blues."
# chunks = r_splitter.split_text(input_text)
#
# for chunk in chunks:
#     response = spotify_agent.run(chunk)
