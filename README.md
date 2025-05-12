# Automated-Data-Query-and-Retrieval-System
This system provides an end-to-end solution for loading CSV data into MongoDB and using 
offline Large Language Models (LLMs) to automatically generate and execute MongoDB queries 
based on natural language user inputs. 

Features 
● Load CSV data into MongoDB 
● Generate MongoDB queries using offline LLMs (like Llama2) 
● Execute MongoDB queries and retrieve data 
● Display or save results to CSV files 
● Automatic query logging 
● Comprehensive error handling

Requirements 
● Python 3.8+ 
● MongoDB (accessible via URI)

Dependencies 
pymongo 
pandas 
langchain 
langchain-llm-cpp 
numpy 

Setup Instructions 
1. Install Dependencies 
pip install pymongo pandas langchain langchain-llm-cpp 
2. MongoDB Setup 
MongoDB Atlas (cloud-based): 
MongoDB Atlas (Recommended for beginners) 
● Create a free MongoDB Atlas account at mongodb.com/cloud/atlas 
● Set up a cluster, database user, and network access 
● Replace the connection string in the script with your Atlas connection string 
● Install additional required package: pip install pymongo[srv] dnspython 
● See the MongoDB_Atlas_Setup.md file for detailed instructions 
3. Download LLM Model 
● Download a Llama-2-7b-chat.Q4_K_M.gguf  
● Place the model in a models directory or update the LLM_MODEL_PATH in the script 
Model download: 
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_
 M.gguf 
4. Configuration 
Before running the script, you may need to adjust the following parameters in the script: 
● MONGO_URI: MongoDB connection string 
● DB_NAME: Database name (default: "product_database") 
● COLLECTION_NAME: Collection name (default: "products") 
● LLM_MODEL_PATH: Path to the LLM model file

Usage 
Running the System 
Run the script using Python: 
python automated_data_query_system.py 
The system provides two modes of operation: 
1. Test Cases Mode: Runs the predefined test cases and saves results 
2. Interactive Mode: Allows users to enter custom queries

Output 
● Results are displayed in the console 
● Results can be saved to CSV files 
● All generated queries are logged in Queries_generated.txt 

File Structure 
├── automated_data_query_system.py  # Main script 
├── models/                         
# Directory for LLM model files 
│   └── llama-2-7b-chat.Q4_K_M.gguf # LLM model file 
├── sample_data.csv                 
# Example CSV data 
├── test_case1.csv                  
├── test_case2.csv                  
├── test_case3.csv                  
└── Queries_generated.txt           
# Output file for test case 1 
# Output file for test case 2 
# Output file for test case 3 
# Log of generated queries 
System Architecture 
The system follows a modular architecture: 
1. Data Loading: CSV data is loaded into MongoDB with proper type conversion 
2. Query Generation: Natural language queries are converted to MongoDB queries using 
LLM 
3. Query Execution: MongoDB queries are executed to retrieve data 
4. Data Presentation: Results are displayed or saved to CSV files 
Error Handling 
The system includes comprehensive error handling for: 
● Missing or invalid CSV files 
● MongoDB connection issues 
● LLM initialization problems 
● Invalid query generation 
● Query execution errors 
● Data conversion issues
