# Biz Bot Project

## Overview

Biz Bot is a Streamlit-based application designed to facilitate document parsing, table creation, and querying using advanced NLP models. The application supports two main modes: RAG (Retrieval-Augmented Generation) and SQL (Text-to-SQL). Users can upload documents and tables, and interact with the bot to get answers to their queries.

## Features

- **Document Parsing**: Convert DOC/DOCX to PDF and parse the content.
- **Table Creation**: Convert JSON/Excel to CSV and create a database table.
- **RAG Mode**: Use a reranker model to rank the semantic similarity between text chunks and user queries, and generate responses using a fact-based question-answering model.
- **SQL Mode**: Use a Text-to-SQL model to generate SQL queries based on user input and retrieve data from the database.
- **Chat Interface**: Interact with the bot through a chat interface, with support for displaying chat history.

## Installation

To run the Biz Bot project, you need to have Python and Streamlit installed. You can install the required packages using pip:

```bash
pip install streamlit pandas docx2pdf llmware transformers torch
```

## Usage

1. **Prepare Data**: Place your documents and tables in the `data` directory. The default document is `complete project report.pdf`, and the default table is `data.csv`.

2. **Run the Application**: Navigate to the project directory and run the following command:

```bash
streamlit run biz_bot.py
```

3. **Interact with the Bot**:
   - Upload a document (PDF, DOC, DOCX) or a table (CSV, JSON, XLSX, XLS) using the sidebar.
   - Choose the mode (RAG or SQL) from the sidebar.
   - Enter your query in the chat input field and press Enter.
   - The bot will display the response based on the selected mode.

## Functions

### Document Conversion

- `convert_to_pdf(fp, doc)`: Convert DOC/DOCX to PDF.
- `convert_to_csv(fp, file)`: Convert JSON or Excel to CSV.

### Table Creation

- `build_table(db, table_name, load_fp, load_file)`: Create a database table from a CSV or JSON/JSONL file.

### Model Loading

- `load_reranker_model()`: Load the reranker model used in the RAG process.
- `load_prompt_model()`: Load the core RAG model used for fact-based question-answering.
- `load_agent_model()`: Load the Text-to-SQL model used for querying the CSV table.

### Parsing and Querying

- `parse_file(fp, doc)`: Parse a newly uploaded file and save the output as a set of text chunks with metadata.
- `get_rag_response(prompt, parser_output, reranker_model, prompter)`: Execute a RAG response.
- `get_sql_response(prompt, agent, db, table_name)`: Execute a Text-to-SQL inference and query the database.

### User Interface

- `biz_bot_ui_app(db, table_name, fp, doc)`: The main function that runs the Biz Bot user interface.

## Important Notes

- The application keeps a running state of any CSV tables that have been loaded in the session to avoid duplicated inserts.
- There is a hidden 'magic' command in the chatbot. If you add " #SHOW" at the end of your query, it will display the SQL command that was generated (very useful for debugging).


## Acknowledgments

- This project uses the [Streamlit](https://streamlit.io/) framework for creating the user interface.
- The [llmware](https://llmware.com/) library is used for NLP models and parsing.
- The [docx2pdf](https://pypi.org/project/docx2pdf/) library is used for converting DOC/DOCX to PDF.

Happy querying! 🚀