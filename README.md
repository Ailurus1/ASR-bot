# ASR-bot

## How to run

### Recommended option: docker
Setup your local `.env` file in root directory of this repository with your token, generated by bot father in telegram.  
Then just type
```bash
docker compose up
```
It will automatically run both bot and asr-model inference services and you can open chat with your bot in messanger and start using it.

### Another option: local manual deploy

In case you need to debug something quickly and in some reasons don't like docker way - you can up both servers manually.  
Again, as in paragraph about docker, you need to setup your `.env` file.  

First of all, let's install dependencies. Here is a quick way using wasy make targets:
```bash
make install
```
it will install `uv` if you don't have it locally and then install basic dependencies. Depending on which part you are going to run you may need to manually separate virtual environments (or install all of them in the same, currently there are not problems with it actually) and run 
```bash
# to run tg-bot
uv pip install -r bot/requirements.txt

# to run inference server
uv pip install -r inference_server/requirements.txt
```

Now we can run servers:

```bash
# 1st console
# Start bot service.

python3 bot

# 2nd console
# Start inference service

python3 inference_server
```
