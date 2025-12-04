# Sentiment-Aware-Chatbot
A console-based chatbot built in Python that conducts conversation with a user and performs both conversation-level and per-message sentiment analysis using a pretrained Hugging Face model.
📌 Overview

This chatbot interacts with the user in a command-line interface and detects emotional tone using DistilBERT (SST-2).
It stores the full conversation and evaluates:

Feature	Status
Full conversation history	✔️ Implemented
Conversation-level sentiment analysis	✔️ (Tier-1 complete)
Per message analysis	✔️ (Tier-2 complete)
Trend detection	✔️ Implemented
Save results as JSON	✔️ Supported
Response-based sentiment behavior	✔️ Implemented

This project fulfills the assignment requirements for a sentiment-aware chatbot as described in the coursework document. 

LiaPlus Assignment (1)

🚀 Features

🔍 Sentiment Analysis using Hugging Face Transformer

🧾 Conversation history tracking

🧠 Weighted scoring system emphasizing recent messages

📊 Trend analysis (improvement, decline, neutral)

💾 Export chat + sentiment report to JSON

🤖 Dynamic emotional responses based on user tone

🖥️ Works on Windows terminal, PowerShell or Linux/Mac CLI

🛠️ Technologies Used
Component	Technology
Programming Language	Python 3
NLP Model	distilbert-base-uncased-finetuned-sst-2-english
Library	transformers, torch, sentencepiece
Storage Format	JSON
📥 Installation
1️⃣ Install dependencies

Open PowerShell or cmd:

pip install transformers torch sentencepiece


If torch installation fails, visit:
https://pytorch.org/get-started/locally/

and install the correct wheel based on your OS, Python version and CPU/GPU.

▶️ Running the Chatbot
python "d:\Projects\sentimental analysis\maincode.py"

💬 Chat Commands
Command	Description
/quit	Ends chat and prints sentiment summary
/save	Saves conversation + sentiment analysis to JSON file
/help	Shows available commands
🧠 How Sentiment Logic Works
🔹 Message-Level Sentiment (Tier-2)

Every user message is passed to the Hugging Face pipeline:

→ The model returns:
label (POSITIVE / NEGATIVE) and a confidence score.

The script then converts it into a signed sentiment value:

Model Label	Final Value
Positive	+score
Negative	-score

Example: if the model predicts:

"label": "NEGATIVE", "score": 0.84


Then sentiment becomes: −0.84

🔹 Conversation-Level Sentiment (Tier-1)

Two metrics are computed:

Metric	Purpose
Average score	Overall emotional tone
Weighted score	Gives more weight to recent messages

Final emotional category:

Score Range	Category
≥ 0.60	Strongly Positive
0.20–0.59	Positive
-0.19–0.19	Neutral
-0.59– -0.20	Negative
≤ -0.60	Strongly Negative
🔹 Trend Detection

The bot analyzes how the sentiment changed over time (improving, stable or declining mood).

📁 Output Example
Conversation summary:
Weighted score: +0.42
Final category: Positive
Trend: overall improving; more positive shifts.

🧩 Status of Tier Implementation
Tier	Requirement	Status
Tier-1	Conversation-level sentiment	✔️ Completed
Tier-2	Per-message sentiment + optional trend	✔️ Completed (including enhancements)
🧪 Tests (Optional)

Currently no automated tests are included, but the codebase is structured and modular for easy future test integration (pytest recommended).

⭐ Enhancements & Innovations

Added trend detection (extra credit)

Weighted scoring makes results more realistic

User can export full session report in structured JSON format

Emotion-adaptive replies provide a more natural chat experience
