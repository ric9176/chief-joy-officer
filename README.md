---
title: DeployPythonicRAG
emoji: 📉
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
---

# Deploying Pythonic Chat With Your Text File Application

In today's breakout rooms, we will be following the processed that you saw during the challenge - for reference, the instructions for that are available [here](https://github.com/AI-Maker-Space/Beyond-ChatGPT/tree/main).

Today, we will repeat the same process - but powered by our Pythonic RAG implementation we created last week. 

You'll notice a few differences in the `app.py` logic - as well as a few changes to the `aimakerspace` package to get things working smoothly with Chainlit.

## Deploying the Application to Hugging Face Space

Due to the way the repository is created - it should be straightforward to deploy this to a Hugging Face Space!

<details>
    <summary>Creating a Hugging Face Space</summary>

1.  Navigate to the `Spaces` tab.

![image](https://i.imgur.com/aSMlX2T.png)

2. Click on `Create new Space`

![image](https://i.imgur.com/YaSSy5p.png)

3. Create the Space by providing values in the form. Make sure you've selected "Docker" as your Space SDK.

![image](https://i.imgur.com/6h9CgH6.png)

</details>

<details>
    <summary>Adding this Repository to the Newly Created Space</summary>

1. Collect the SSH address from the newly created Space. 

![image](https://i.imgur.com/Oag0m8E.png)

> NOTE: The address is the component that starts with `git@hf.co:spaces/`.

2. Use the command:

```bash
git remote add hf HF_SPACE_SSH_ADDRESS_HERE
```

3. Use the command:

```bash
git pull hf main --no-rebase --allow-unrelated-histories -X ours
```

4. 