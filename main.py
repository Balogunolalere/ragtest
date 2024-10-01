import os
import json
import pandas as pd
import streamlit as st
import transformers
import torch
from llmware.resources import CustomTable
from llmware.models import ModelCatalog
from llmware.prompts import Prompt
from llmware.parsers import Parser
from llmware.configs import LLMWareConfig
from llmware.agents import LLMfx
from llmware.setup import Setup

# Keeps a running state of any csv tables that have been loaded in the session to avoid duplicated inserts
if "loaded_tables" not in st.session_state:
    st.session_state["loaded_tables"] = []

def convert_to_csv(fp, file):
    file_name, file_extension = os.path.splitext(file)
    csv_file = f"{file_name}.csv"
    
    if file_extension.lower() == '.json':
        with open(os.path.join(fp, file), 'r') as json_file:
            data = json.load(json_file)
        df = pd.DataFrame(data)
    elif file_extension.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(os.path.join(fp, file))
    else:
        return None

    csv_path = os.path.join(fp, csv_file)
    df.to_csv(csv_path, index=False)
    return csv_file

def build_table(db=None, table_name=None, load_fp=None, load_file=None):
    if not table_name:
        return 0

    if table_name in st.session_state["loaded_tables"]:
        return 0

    custom_table = CustomTable(db=db, table_name=table_name)

    file_extension = os.path.splitext(load_file)[1].lower()
    if file_extension in ['.json', '.xlsx', '.xls']:
        csv_file = convert_to_csv(load_fp, load_file)
        if csv_file:
            load_file = csv_file
        else:
            print("Failed to convert file to CSV")
            return -1

    analysis = custom_table.validate_csv(load_fp, load_file)
    print("update: analysis from validate_csv: ", analysis)

    if load_file.endswith(".csv"):
        output = custom_table.load_csv(load_fp, load_file)
    elif load_file.endswith(".jsonl") or load_file.endswith(".json"):
        output = custom_table.load_json(load_fp, load_file)
    else:
        print("file type not supported for db load")
        return -1

    print("update: output from loading file: ", output)

    sample_range = min(10, len(custom_table.rows))
    for x in range(0, sample_range):
        print("update: sample rows: ", x, custom_table.rows[x])

    updated_schema = custom_table.test_and_remediate_schema(samples=20, auto_remediate=True)
    print("update: updated schema: ", updated_schema)

    custom_table.insert_rows()
    st.session_state["loaded_tables"].append(table_name)

    return len(custom_table.rows)

@st.cache_resource
def load_reranker_model():
    reranker_model = ModelCatalog().load_model("jina-reranker-turbo")
    return reranker_model

@st.cache_resource
def load_prompt_model():
    prompter = Prompt().load_model("bling-phi-3-gguf", temperature=0.0, sample=False)
    return prompter

@st.cache_resource
def load_agent_model():
    agent = LLMfx()
    agent.load_tool("sql", sample=False, get_logits=True, temperature=0.0)
    return agent

@st.cache_resource
def parse_file(fp, doc):
    parser_output = Parser().parse_one(fp, doc, save_history=False)
    st.cache_resource.clear()
    return parser_output

def get_rag_response(prompt, parser_output, reranker_model, prompter):
    if len(parser_output) > 3:
        output = reranker_model.inference(prompt, parser_output, top_n=10, relevance_threshold=0.25)
    else:
        output = []
        for entries in parser_output:
            entries.update({"rerank_score": 0.0})
            output.append(entries)

    use_top = 3
    if len(output) > use_top:
        output = output[0:use_top]

    sources = prompter.add_source_query_results(output)
    responses = prompter.prompt_with_source(prompt, prompt_name="default_with_context")
    source_check = prompter.evidence_check_sources(responses)
    numbers_check = prompter.evidence_check_numbers(responses)
    nf_check = prompter.classify_not_found_response(responses, parse_response=True, evidence_match=False, ask_the_model=False)

    bot_response = ""
    for i, resp in enumerate(responses):
        bot_response = resp['llm_response']
        print("bot response - llm_response raw - ", bot_response)
        add_sources = True

        if "not_found_classification" in nf_check[i]:
            if nf_check[i]["not_found_classification"]:
                add_sources = False
                bot_response += "\n\n" + ("The answer to the question was not found in the source "
                                          "passage attached - please check the source again, and/or "
                                          "try to ask the question in a different way.")

        if add_sources:
            numbers_output = ""
            if "fact_check" in numbers_check[i]:
                fc = numbers_check[i]["fact_check"]
                if isinstance(fc, list) and len(fc) > 0:
                    max_fact_count = 1
                    count = 0
                    for fc_entries in fc:
                        if count < max_fact_count:
                            if "text" in fc_entries:
                                numbers_output += "Text: " + fc_entries["text"] + "\n\n"
                            if "source" in fc_entries:
                                numbers_output += "Source: " + fc_entries["source"] + "\n\n"
                            if "page_num" in fc_entries:
                                numbers_output += "Page Num: " + fc_entries["page_num"] + "\n\n"
                            count += 1
                bot_response += "\n\n" + numbers_output

            source_output = ""
            if not numbers_output:
                if "source_review" in source_check[i]:
                    fc = source_check[i]["source_review"]
                    if isinstance(fc, list) and len(fc) > 0:
                        fc = fc[0]
                        if "text" in fc:
                            source_output += "Text: " + fc["text"] + "\n\n"
                        if "match_score" in fc:
                            source_output += "Match Score: " + str(fc["match_score"]) + "\n\n"
                        if "source" in fc:
                            source_output += "Source: " + fc["source"] + "\n\n"
                        if "page_num" in fc:
                            source_output += "Page Num: " + str(fc["page_num"]) + "\n\n"
                    bot_response += "\n\n" + source_output

    prompter.clear_source_materials()
    return bot_response

def get_sql_response(prompt, agent, db=None, table_name=None):
    show_sql = False
    bot_response = ""

    if prompt.endswith(" #SHOW"):
        show_sql = True
        prompt = prompt[:-(len(" #SHOW"))]

    model_response = agent.query_custom_table(prompt, db=db, table=table_name)

    error_handle = False

    try:
        sql_query = model_response["sql_query"]
        db_response = model_response["db_response"]

        if not show_sql:
            bot_response = db_response
        else:
            bot_response = f"Answer: {db_response}\n\nSQL Query: {sql_query}"
    except:
        error_handle = True
        sql_query = None
        db_response = None

    if error_handle or not sql_query or not db_response:
        bot_response = (f"Sorry I could not find an answer to your question.<br/>"
                        f"Here is the SQL query that was generated by your question: "
                        f"<br/>{sql_query}.<br/> If this missed the mark, please try asking "
                        f"the question again with a little more specificity.")

    return bot_response

def biz_bot_ui_app(db="postgres", table_name=None, fp=None, doc=None):
    st.title(f"Biz Bot")

    parser_output = None

    if os.path.exists(os.path.join(fp, doc)):
        if not parser_output:
            parser_output = Parser().parse_one(fp, doc, save_history=False)

    prompter = load_prompt_model()
    reranker_model = load_reranker_model()
    agent = load_agent_model()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.sidebar:
        st.write("Biz Bot")
        model_type = st.selectbox("Pick your mode", ("RAG", "SQL"), index=0)

        uploaded_doc = st.file_uploader("Upload Document", type=["pdf"])
        uploaded_table = st.file_uploader("Upload Table", type=["csv", "json", "xlsx", "xls"])

        if uploaded_doc:
            fp = LLMWareConfig().get_llmware_path()
            doc = uploaded_doc.name
            with open(os.path.join(fp, doc), "wb") as f:
                f.write(uploaded_doc.getvalue())
            parser_output = parse_file(fp, doc)
            st.write(f"Document Parsed and Ready - {len(parser_output)}")

        if uploaded_table:
            fp = LLMWareConfig().get_llmware_path()
            tab = uploaded_table.name
            with open(os.path.join(fp, tab), "wb") as f:
                f.write(uploaded_table.getvalue())
            table_name = os.path.splitext(tab)[0]
            st.write("Building Table - ", tab, table_name)
            st.write(st.session_state['loaded_tables'])
            row_count = build_table(db=db, table_name=table_name, load_fp=fp, load_file=tab)
            st.write(f"Completed - Table - {table_name} - Rows - {row_count} - is Ready.")

    prompt = st.chat_input("Say something")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if model_type == "RAG":
                bot_response = get_rag_response(prompt, parser_output, reranker_model, prompter)
                st.markdown(bot_response)
            else:
                bot_response = get_sql_response(prompt, agent, db=db, table_name=table_name)
                st.markdown(bot_response)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

    return 0

if __name__ == "__main__":
    db = "sqlite"
    table_name = "customer_table_1"

    local_csv_path = "data"
    build_table(db=db, table_name=table_name, load_fp=local_csv_path, load_file="data.csv")

    local_path = "data"
    fp = os.path.join(local_path, "data")
    fn = "complete project report.pdf"

    biz_bot_ui_app(db=db, table_name=table_name, fp=fp, doc=fn)