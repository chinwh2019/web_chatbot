# Chatbot Development for Rakuten Mobile Support

## Project Overview and Approach

When assigned the task to develop a chatbot for Rakuten Mobile support, I began by analyzing the Rakuten Mobile support webpage structure. The support site consisted of numerous subpages, each dedicated to different question-and-answer sections, rather than providing consolidated answers on a single page. Given the complexity and scope of the content, I realized that scraping the entire content from all subpages would be infeasible due to time and resource constraints.

## Strategy for Building the Knowledge Base

To tackle this challenge efficiently, I designed an approach:
1. **Data Scraping**: Instead of extracting the full Q&A content from each subpage, I focused on scraping only the titles of support keywords along with their respective URLs. This approach provided a lightweight yet effective structure for the knowledge base.
2. **User Query Redirection**: The chatbot was prompted to match user queries with the most relevant support page titles and then direct users to the corresponding URLs. This way, users could easily access comprehensive information on the Rakuten Mobile support site.

## Data Processing and Structuring

To process and structure the scraped data effectively:
1. **Data Conversion**: I utilized [**Jina AI**](https://jina.ai/reader/) to convert the raw HTML data into a human-readable markdown format. This conversion simplified data handling and analysis.
2. **Data Structuring with GPT-4o**: I employed GPT-4o to transform the markdown into a more organized and structured format. This step ensured the data could be stored and accessed efficiently, facilitating better performance.

## Knowledge Base and Database Implementation

For data storage and query performance, I chose **PostgreSQL with pgvector**:
- **Embedding Titles**: I generated embeddings for the titles, creating vector representations that could be efficiently matched with user queries.
- **Rationale for PostgreSQL**: PostgreSQL with pgvector was selected for its simplicity, reliability, and high performance in handling vector similarity searches.

## Final Chatbot Functionality

The final chatbot was designed to:
- Process user queries and use vector similarity to match them with relevant support page titles.
- Respond by providing users a general response with the most relevant URLs, directing them to detailed information on the Rakuten Mobile support site.

