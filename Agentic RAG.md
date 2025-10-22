Agentic RAG
-----

Selecting transcript lines in this section will navigate to timestamp in the video
- [Instructor] We're going to continue our discussion of modular RAG by talking about agents. Let's go ahead and jump right into it. So in LlamaIndex, a data agent is kind of like a knowledge worker that's automated and powered by a large language model, and this knowledge worker, if you would, can perform a wide variety of tasks over different types of data, different types of data sources and it can also make use of tools. So one of the things that this agent can do is just handle data across various formats, whether that's structured, semi-structured, or completely unstructured. There, you can use them for searching and retrieving information from these diverse data sources efficiently. These data agents can also interact with external services, APIs, or functions that are able to process the responses from these and then store the information for later use. So this gives the agent the ability to act in a dynamic environment.

-----

How do they work? Well, we need a reasoning loop. This is the core of the data agent's operation, so depending on the way you set up your agent, the reasoning loop will help kind of determine what tool to use, the sequence in which the tool should be used, and the parameters that you need to use the tool. And so this reasoning loop enables the agent to make informed decisions. There's also tools, and so data agents are initialized with a set of APIs or tools that they can interact with. This could be anything from data retrieval, functions, or tools that call APIs or some more complex processing.

-----

We'll go ahead and start with our typical imports. We use a in-memory database with Qdrant. We'll set up our LLM, set up our embedding model. We'll bring in our documents. We're going to make use of metadata in this lesson, so I'll manually attach some metadata. We'll go ahead and create a document store, creating a document store because we're going to make use of hybrid retrieval. Note that we are not doing BM25 retrieval, we're using hybrid retrieval in LlamaIndex. If we were to use BM25 retrieval, you'd need to use a Qdrant Cloud instance.

-----

Next, I'm going to add some metadata, but first I'll go ahead and instantiate some transformations. I have the sentence_splitter, a qa_extractor, and a keyword_extractor. We'll go ahead and chain these transformations together, so we will split our text into chunks. For each chunk, we will come up with some questions that the chunk can answer, and then we'll also attach some keywords related to that chunk of text. Go ahead and ingest this into the vector database and create a index over it. Here is an example of the node with all of its metadata.

-----

Next, what we're going to do is create a VectorStoreInfo object. VectorStoreInfo object is going to be used by our agent, and it'll be used in such a way that the agent knows what is in our vector database so it'll know exactly what metadata is there and what that metadata is useful for. I cannot understate the importance of having a good description for each one of these MetadataInfos because this description is what the language model will use to figure out which metadata it needs to answer a query.

-----

Next, we'll go ahead and create a base Pydantic object. This is just a schema. As you'll see in a moment, we're going to be using function tools and OpenAI function tools and this is just a way for us to handle the output from those APIs. It just structures it nicely so that we can proceed without errors.

-----

And now what I'm going to do is define a function. And so what this function does, it's going to allow for a kind of dynamic querying of a vector database where both the content similarity and the metadata criteria is considered. And so our agent is going to make use of this tool. So just to quickly talk about what this function does, first thing, I'm just setting top_k = 3. You can change that if you'd like. This function has some parameters such as the query: str, the filter_key_list and values, and these are just the metadata keys to filter by and the values corresponding to the key. Here, I'm going to set up a metadata filter, and so this is implemented using key-value pairs. And again, the key represents the metadata attribute and the value specifies the desired attribute value. And so the operator here defines how the filter should be applied. In this case, I'm using the contains operator, so this means the function will filter results where the metadata of the document contains the values in the given key. We'll go ahead and instantiate a vector index retriever, so just instantiate a retriever. I'm using the query mode of hybrid, and I'm setting the alpha value equal to 0.65. If you recall, we discuss the alpha value gives a trade off between vector search and full-text search. We're also passing the MetadataFilter as well. We'll create a query engine. This query engine is using the compact response mode. We'll get the response back and print it. So this is essentially a function that our agent is going to use, and this function does what we've seen dozens of times by this point already, retrieves from the vector database and synthesizes a response.

-----

So again, just to recap, this function is going to do dynamic querying of our vector database. We're considering both the similarity, the semantic similarity between the query and what is in the vector database, and we're doing that with vector search. We're also using specific metadata criteria, and the metadata criteria is essential for a hybrid retrieval system because we're able to give our query engine the necessary mechanism to ensure that the search results are relevant and compliant with our user query. And so the filter ensures that the user receives results that are contextually relevant as well as targeted to their query.

-----

Now, we're going to go ahead and define a function tool. So if you recall, I mentioned that in order for us to use an agent, we need to give it access to tools, and so a function tool is a abstraction that's going to convert a user-defined function into a tool that can be used by an agent. And so the whole purpose of this function tool is to encapsulate a function, in this case, the function that we're encapsulating is the one that we just defined, the auto_retrieve_fn, along with its metadata, like the name of the function, a description of the function, as well as a standard interface that the agent can interact with. And that is the fn_schema here which we have already defined. Because we standardize it in this way, that means the agent is able to dynamically use the tool without really needing to understand the underlying code itself.

-----

Here, we're going to use the OpenAIAgent. This is a specialized type of agent that uses, of course, OpenAI models to perform tasks like using functions or calling APIs or answering a query or summarizing information, and they're built using the OpenAI API. And so we initialize our OpenAIAgent with a set of tools. In this case, it is the auto_retrieve_tool, which we have just defined. So the agent is going to use the tool based on the input that it receives, it's going to look at that input, in this case, the input will be a user query, and based on that user query, it's going to figure out what it needs to do and how it needs to proceed.

-----

So let's go ahead and see this agent in action, so just to kind of recap how this agent is working. It all starts with a input, a user query. This is going to go into this query engine system. So the query will be sent to our query engine retriever, the query engine retriever is going to interact with the vector store data application that we have here. The vector store data is going to communicate with the OpenAI API, one is a language model that's going to kind of generate responses based on the query, another is the embedding model that's going to convert the query and the data into vector representations, then we have the vector store system. The vector store system, of course, will process what is sent to it and fetch the relevant vectors, but it's also going to filter using metadata filters and we'll get back a list of filtered nodes which represent the most relevant pieces of information to the query. This is going to be sent back to the vector store data application, which will compile it for response, and send it back to the query engine application and then finally get response back.

-----

So see this in action. So here, I send a query to the agent, and I say, building wealth and achieving happiness, so on and so forth. Right, I'm passing it this piece of text, and based on this, the agent is saying, "Okay, well, I should use questions this excerpt can answer," and it decomposes this long string of texts into one, two, two questions here, three questions, right? So the agent looks at this long input text. It reasons over what it needs to do and what metadata filters it needs to apply, in this case, it's saying, "Okay, I need to use questions the excerpt can answer, and I need to break this down into a bunch of different questions," which the agent does, fetches a number of nodes, and then gives us a response back.

-----

If I send the agent a query like this, find text that mentions specific knowledge, luck, and success, the agent is able to say, "Oh, okay, well if this is the query, then I should look at keywords. And so I'll filter keywords to specific knowledge, luck, and success and then get the most relevant nodes." If I say to the agent, what would Nassim Taleb say about accountability and risk, the agent is able to say, "Okay, well, I should probably filter based on the author metadata tag where the author is Nassim Taleb and ask about these things." Here, we can send the agent a query, what would Bruce Lee say about adaptability and self-expression, again, it's able to say, "Okay, well, I should probably filter for Bruce Lee and, you know, do this query once those nodes have been filtered." If I have a question like this, what kind of questions should I ask Balaji or Naval, the agent is able to say, "Okay, well, I should probably filter on author and then look at the questions." And it does so for Balaji and it does so for Naval Ravikant as well. And so you can see that's pretty powerful, right? The agent is able to look at the input query, decide how it needs to filter the nodes and then execute on that. It's a pretty powerful pattern.

-----

There's also a lower-level API that LlamaIndex exposes to you. Under the hood of agents, it's really composed of two things, an AgentRunner that interacts with an AgentWorker. The AgentRunner is an orchestrator that's going to manage the state, including the history of the conversation. It's going to create and maintain tasks, execute each step within the task, and then give a high level interface for user interaction. The AgentWorker is what controls the step-wise execution of a task. So given an input step, the AgentWorker will generate the next step, and you can initialize this with parameters that act on state that's passed down between the task step or TaskStep objects, and then there's just this kind of outer AgentRunner that's like call the AgentWorker, collect and aggregate the results and do it in this interactive loop. I've linked to the source code. If you want to read more about it, please do.

-----

So here, we're just going to initialize this agent_worker with a FunctionCallingAgentWorker abstraction. We're going to give it the same tools, tell it to use the LLM from the settings, and then you can see here what's happening under the hood. So what I'm saying, in what ways do Naval and Nassim think differently, and so you can see the agent behaves just like the Open AI agent, is able to filter based on the author and then execute the query against those author nodes and then give us back a response.

-----

And there you have it, Agentic RAG at a high level. So there's a lot more to agents than we covered here. If you go to the LlamaIndex docs and you go to Examples, and then in the Examples, you go to Agents, you could see here on the sidebar, there are a number of different types of agents that you can use, far too many that we can cover in this course. As a matter of fact, I can probably create another two or three-hour course just about agents in LlamaIndex. But I encourage you to look at the documentation, so again, just go to docs at LlamaIndex, look under Examples, look under Agents. Of course, if you need more information, you can also look at the Component Guides and then go to the agent's component guide here. You get the usage pattern and the lower-level API, the module guides, and importantly, the tools that a agent has. We're going to continue on. We'll make more use of tools in the next two modules where we talk about ensemble retrieval and the ensemble query engine.

-----