# Rakuten Mobile FAQ Chatbot

A chatbot system designed to handle customer service inquiries for Rakuten Mobile, featuring web scraping, content processing, and intelligent response generation using OpenAI's GPT and embedding models.

## Assignment Task and Goal

The goal of this assignment was to develop a chatbot capable of assisting users with queries based on the [Rakuten Mobile support webpage](https://network.mobile.rakuten.co.jp/support/). The primary objective was to leverage existing support content to provide relevant and helpful responses.

The thought process of the developement is documented in [dev_thoughts.md](dev_thoughts.md)



## 🌟 Features

- **Web Scraping**: Automated scraping of Rakuten Mobile support pages
- **Content Processing**: Intelligent processing of scraped content using GPT models
- **Vector Database**: PostgreSQL with pgvector for efficient similarity search
- **Multilingual Support**: Primary focus on Japanese language support
- **Conversational AI**: Context-aware responses with conversation history
- **Docker Support**: Containerized application with PostgreSQL database

## 🔧 Technology Stack

- Python 3.11
- OpenAI GPT-4o and Embedding models
- PostgreSQL with pgvector extension
- SQLAlchemy with pgvector support
- Docker and Docker Compose
- Pydantic for settings management

## 📋 Prerequisites

- Docker and Docker Compose
- OpenAI API key
- Python 3.11+ (python 3.10+ should work expectedly)
- PostgreSQL 13+

## 🚀 Getting Started

1. **Clone the repository**

    ```
    git clone https://github.com/chinwh2019/web_chatbot.git
    cd web_chatbot
    ```


2. **Set up environment variables**

- Duplicate `.env.example` and rename the copy to `.env `
- Open `.env` and replace the placeholder value for api_key with your actual API key 


3. **Build and run with Docker**

    ```
    # Start with interactive mode
    docker-compose up --build

    # In a separate terminal
    docker-compose exec -it app python main.py

    # Stop existing containers
    docker-compose down
    ```


4. **Run locally (alternative)**

    ```
    pip install -r requirements.txt
    python main_local.py
    ```


## 💡 Usage

The system can be used in two modes:

1. **Scrape and Process New Content**
   - Enter a Rakuten Mobile support URL
   - System will scrape, process, and store the content
   - Start chatting with the bot

2. **Use Existing Database**
   - Direct access to the chatbot is available once the knowledge base is built

## 🏗️ Project Structure

```
├── src/
│ ├── chatbot.py        # Main chatbot implementation
│ ├── data_processor.py # Content processing logic
│ ├── embeddings.py     # Vector embedding handling
│ ├── models.py         # Database models
│ ├── scraper.py        # Web scraping functionality
│ ├── settings.py       # Configuration management
│ └── utils/
│ ├── exceptions.py     # Custom exceptions
│ └── logger.py         # Logging configuration
├── data/               # Scraped and processed data
├── docker-compose.yml  # Docker composition
├── Dockerfile          # Container definition
├── requirements.txt    # Python dependencies
└── main.py             # Application entry point
```


## 🔍 Key Components

1. **WebScraper**
   - Handles webpage content extraction
   - Converts HTML to markdown format
   - Manages file storage

2. **GPTContentProcessor**
   - Processes markdown content using GPT models
   - Structures data into JSON format
   - Handles batch processing

3. **EmbeddingDatabase**
   - Manages vector embeddings
   - Handles similarity searches
   - Integrates with PostgreSQL

4. **FAQBot**
   - Manages conversation flow
   - Handles intent classification
   - Generates context-aware responses


## 🔮 Future Work

### 1. Enhanced Natural Language Processing
- Implement more sophisticated intent classification
- Add sentiment analysis for user queries

### 2. System Improvements
- Add caching layer for frequently accessed embeddings

### 3. Performance Optimization
- Optimize vector search algorithms
- Optimize database queries and indexing
- Migrate to cloud vector database if necessary

### 4. Integration Capabilities
- Develop REST API for external system integration


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
