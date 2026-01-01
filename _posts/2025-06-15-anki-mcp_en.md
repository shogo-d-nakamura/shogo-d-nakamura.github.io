---
layout: post
title: Creating Anki Flashcards with MCP via LLM
date: 2025-06-15 00:29 +0900
description: ''
category: 'English'
tags: [LLM, Anki]
published: true
lang: en
ref: anki-mcp
---


# Creating Anki Flashcards Using MCP

I've recently gotten into the habit of preparing Anki flashcards for English words and phrases I want to remember.\
Whenever I encounter confusing English expressions, I quickly invoke the LLM using the Option+Space shortcut key to ask about their meanings or other example sentences. Since I'm already using the LLM for this, I thought why not have it directly create the Anki flashcards for me? So once again, I turned to Claude Code to have it write an MCP server.


repo:\
https://github.com/shogo-d-nakamura/anki-mcp



# Quick Start

1. As usual, create a virtual environment with uv and install dependencies:

```zsh
uv venv
source .venv/bin/activate
uv pip install -e .
```


2. Add to your Claude Desktop configuration file (claude_desktop_config.json):

```json
{
  "mcpServers": {
    
    "Anki-MCP": {
      "command": "/path/to/bin/uv",
      "args": [
        "--directory", "/path/to/Anki-MCP",
        "run", "server.py"
      ]
    }
  },

  "globalShortcut": ""
}
```


# Example

Here's what Claude Desktop looks like:


![alt text](/assets/img/2025_images/samalama.png)


On Anki side:


![alt text](/assets/img/2025_images/Anki_samalama.png)



