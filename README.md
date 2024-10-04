# build-an-agent-series

## Before we get started!

### Python Setup

It is a good idea to have something to manage your python environment. pyenv is a good tool for this. I wrote a [guide](https://justinhennessy.substack.com/p/how-to-get-pyenv-up-and-running) on how to set it up.

### LangSmith Setup

LangSmith is an all-in-one developer platform for every step of the LLM-powered application lifecycle, whether you’re building with LangChain or not.
Debug, collaborate, test, and monitor your LLM applications. Sign up for LangSmith [here](https://www.langchain.com/langsmith).

Here are the setup instructions for LangSmith [here](https://docs.smith.langchain.com/).

### LangGraph Studio Setup

LangGraph Studio offers a new way to develop LLM applications by providing a specialised agent IDE that enables visualization, interaction, and debugging of complex agentic applications

With visual graphs and the ability to edit state, you can better understand agent workflows and iterate faster. LangGraph Studio integrates with LangSmith so you can collaborate with teammates to debug failure modes.

While in Beta, LangGraph Studio is available for free to all LangSmith users on any plan tier. Sign up for LangSmith here.

You can download it [here](https://github.com/langchain-ai/langgraph-studio), setup instructions are there also.

If you run into an issue that the rust compiler doens't install, you can install it doing the following:

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Settng up jupyter notebook

Jupyter Notebook is an open-source web application that enables users to create and share documents with live code, equations, visualizations, and narrative text. It supports multiple programming languages, primarily Python, allowing for interactive coding and immediate results.

Commonly used for data analysis and machine learning, Jupyter Notebooks facilitate exploratory work and collaboration. They combine code and explanations in a single document, making them ideal for reproducible research and quick prototyping.

Often, examples in repositories have jupyter notebooks in it (files ending in `.ipynb`), so its useful to have this handy if they do.

To install (once you have properly setup your python environment):

```
pip install jupyter
```

Then from with the directory (at the command line), run the following:

```
jupyter notebook
```

This will open up a browser with a file/directory list of the directory where it was launched.

### Getting an OpenAI API Key

Although you can use other LLM providers like Anthropic, Groq, and Mistral, for this series we will be using OpenAI.

To get an OpenAI API key, go to the [OpenAI API dashboard](https://platform.openai.com/account/api-keys) and click "Create new secret key".

## What we will be building?!

This repo has been created to help people who are curious about dipping their toes into the world of agentic applications. We will explore many different options which you can try out and get up and running in no time.

Here are a few areas we will be exploring:

LangChain - this is a really good starting point if you are new to build agents. LangChain is a framework to interweave the use of LLMs in your application.
LangGraph - this introduces a much more capable framework for building more complex agentic applications.

We will also build some different utiliies that might help you on your journey to building your own agentic applications.

## Free Courses Online

LangChain Course - https://learn.deeplearning.ai/langchain/



## This repo has drawn inspiration from

https://github.com/NirDiamant/GenAI_Agents