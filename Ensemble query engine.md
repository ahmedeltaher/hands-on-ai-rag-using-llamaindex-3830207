Ensemble query engine
-----

Selecting transcript lines in this section will navigate to timestamp in the video
- [Instructor] As you've seen throughout this course, when you're building a RAG system, there are so many different components that you can play around with that you can experiment with. We saw in the previous section that you can experiment with a number of different retrieval components. Well, you can also do the same with query engine components and query pipelines.

-----

So now what if you could simultaneously use multiple strategies and have a language model, rate the relevance of each result, and then synthesize the results into a coherent answer? And this is what we can use an ensemble query engine for. The main purpose of this ensemble query engine is to try out multiple retrievers at once, have the LLM rate how good each result is compared to the original query, and make sure that only the most relevant information is considered. And then let the LLM combine this most relevant information into a comprehensive final answer.

-----

So how do you use the ensemble query engine? Well, you set up your retrieval tool. You configure a router query engine, and then run the queries. And this is a powerful technique that will let you try different query and retrieval methods so that you can experiment with different approaches and find something that's going to work for your RAG application.

-----

We'll begin as we normally do with all our imports, set up our LLM, set up our embedding model. Bring in our documents. We're going to just use a simple sentence splitter in this case, using a chunk size 128. Go ahead and create our storage context.

-----

And what we're going to do is use a simple key word table index. This is just a simplified version of the keyword based indexing system. What it does is during index construction, it's going to split text documents into chunks. Then it's going to use GPT to extract the relevant keywords. Those keywords are going to be stored in a table. Then at query time, we're going to extract the relevant keyword and use them to retrieve a set of candidate text chunk IDs. Then the initial answer will be constructed using this first text chunk. And then we go on refining it with more and more chunks.

-----

Go ahead and create the simple keyword table index and the vector store index. We'll go ahead and instantiate a QA prompt, which will pass to both the keyword query engine and the vector query engine. Now we can use the vector query engine to get a response. We'll rate the response, in this case, the LLM says 9 out of 10. Then we'll use the query engine with the same query. And this time the language model scores it 10 out of 10.

-----

Now I want to draw your attention to something called the query engine tool. So we talked about how tools are abstractions that are used by a data agent or a LLM and gives a structured way for them to perform some task. And so a query engine tool is a specific type of tool that interfaces with and wraps a query engine. That way the agent is able to perform a complex query by leveraging the query engine itself.

-----

So we'll go ahead and create a couple of tools. One is a keyword tool. This keyword tool is going to use the keyword query engine. So of course we instantiate the tool with the query engine itself, as well as a good description. We'll create a vector tool which uses the vector query engine. We'll go ahead and give that a good description as well.

-----

Then we're going to define a router query engine using a LLM multi-sector. So the LLM multi-sector, it just uses a prompt and tells the LLM, these are your choices of tools that you need to use. Select the most relevant tool based on the query, and then go do the query.

-----

So let's go ahead and package up the router query engine. So we'll pass the LLM multi-sector, and then we'll use tree summarize. We have a prompt template for tree summarize. We'll instantiate the query engine using the router query engine. We'll pass in the selector and then we'll pass in the tools. So what's going to happen is that the agent will look at this tool, it's going to choose the best data source, decide whether to perform a keyword search or a vector search. Then it's going to evaluate the retrieved nodes and synthesize that to a response.

-----

So let's go ahead and see this in action. So here we have a query engine that's just saying, how can I develop specific knowledge that'll help me build wealth and achieve happiness? And so here you can see the agent, if you will, is selecting query engine one, because it's a fully formed question. We get a response with a relevant score. You can also print out the final response in the source nodes as well.

-----

Here I'm just passing a list of keywords. And then the agent is looking at this and saying, oh, okay, well I should probably use keywords. And so it does so by picking the right nodes, synthesizing the final response, and we get a relevant score of 7 out of 10. And you can see the source nodes as well.

-----

So we could try it again here. So I'm trying to, in this case, I'm trying to come up with a query that has keywords and a question. And I was hoping that the language model would use both of them, but it did not. It just decided to use query engine one because of this fully formed question. Of course, we can see the result and the source nodes as well.

-----

And so if we scroll back to the top here where we instantiate everything, the query engines, you can create your own query engines. If you can try this out, you can test this out and you can play around with it. And once you have your query engines of choice created, you can run them just like how we've done here and get the appropriate score. Right, so that's it for this lesson, and that's really it for our discussion of RAG techniques. In the next session, I'll give you some concluding thoughts. Thanks for sticking with me for this course, and I'll see you in the final module.

-----