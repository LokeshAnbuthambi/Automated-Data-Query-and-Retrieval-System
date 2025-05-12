import os
import csv
import json
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB Configuration
# For local MongoDB
# MONGO_URI = "mongodb://localhost:27017/"

# For MongoDB Atlas - replace with your own connection string
MONGO_URI = "Insert_your_URL"
DB_NAME = "product_database"
COLLECTION_NAME = "products"

# LLM Configuration
LLM_MODEL_PATH = "models/llama-2-7b-chat.Q4_K_M.gguf"  # Update with your model path

class DataQuerySystem:
    """Main class for the Data Query and Retrieval System."""
    
    def __init__(self, mongo_uri=MONGO_URI, db_name=DB_NAME, collection_name=COLLECTION_NAME):
        """Initialize the system with MongoDB connection and LLM model."""
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        self.llm = None
        self.csv_schema = {}
        self.queries_file = "Queries_generated.txt"
        
        # Initialize MongoDB connection and LLM
        self._connect_to_mongodb()
        self._initialize_llm()
    
    def _connect_to_mongodb(self):
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(self.mongo_uri)
            # Check if connection is successful
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            logger.info("Successfully connected to MongoDB")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _initialize_llm(self):
        """Initialize the LLM model."""
        try:
            if not os.path.exists(LLM_MODEL_PATH):
                logger.error(f"LLM model not found at {LLM_MODEL_PATH}")
                raise FileNotFoundError(f"LLM model not found at {LLM_MODEL_PATH}")
            
            self.llm = LlamaCpp(
                model_path=LLM_MODEL_PATH,
                temperature=0.1,
                max_tokens=2000,
                top_p=0.95,
                n_ctx=2048,
                verbose=False
            )
            logger.info("Successfully initialized LLM model")
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}")
            raise
    
    def load_csv_to_mongodb(self, csv_path):
        """Load CSV data into MongoDB."""
        try:
            if not os.path.exists(csv_path):
                logger.error(f"CSV file not found: {csv_path}")
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Store CSV schema for later reference
            self.csv_schema = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
            
            # Convert DataFrame to list of dictionaries
            records = df.to_dict(orient='records')
            
            # Process data - handle data types and conversions
            for record in records:
                # Handle price as float
                if 'Price' in record:
                    try:
                        record['Price'] = float(record['Price'])
                    except (ValueError, TypeError):
                        pass
                
                # Handle rating as float
                if 'Rating' in record:
                    try:
                        record['Rating'] = float(record['Rating'])
                    except (ValueError, TypeError):
                        pass
                
                # Handle review count as int
                if 'ReviewCount' in record:
                    try:
                        record['ReviewCount'] = int(record['ReviewCount'])
                    except (ValueError, TypeError):
                        pass
                
                # Handle stock as int
                if 'Stock' in record:
                    try:
                        record['Stock'] = int(record['Stock'])
                    except (ValueError, TypeError):
                        pass
                
                # Handle discount - remove % if present and convert to float
                if 'Discount' in record and isinstance(record['Discount'], str):
                    try:
                        record['Discount'] = float(record['Discount'].strip('%'))
                    except (ValueError, TypeError):
                        pass
                
                # Handle dates
                if 'LaunchDate' in record and isinstance(record['LaunchDate'], str):
                    try:
                        record['LaunchDate'] = datetime.strptime(record['LaunchDate'], '%d-%m-%Y')
                    except (ValueError, TypeError):
                        pass
            
            # Drop existing collection to avoid duplicates
            self.collection.drop()
            
            # Insert records into MongoDB
            if records:
                self.collection.insert_many(records)
                logger.info(f"Successfully loaded {len(records)} records into MongoDB")
            else:
                logger.warning("No records found in CSV file")
            
            return True
        except Exception as e:
            logger.error(f"Error loading CSV to MongoDB: {e}")
            raise
    
    def generate_mongodb_query(self, user_query):
        """Generate a MongoDB query using the LLM based on user input."""
        try:
            # Create a prompt for the LLM
            prompt_template = PromptTemplate(
                input_variables=["schema", "query"],
                template="""
                You are a MongoDB query generator. Your task is to generate a valid MongoDB query based on the user's request.
                
                The database schema is:
                {schema}
                
                User query: {query}
                
                Generate a valid MongoDB query that answers the user's query. 
                Return ONLY the MongoDB query without any explanation or additional text.
                The query should be in the format: db.collection.find(...) or similar MongoDB operation.
                
                MongoDB Query:
                """
            )
            
            # Create the full prompt with schema details
            prompt = prompt_template.format(
                schema=json.dumps(self.csv_schema, indent=2),
                query=user_query
            )
            
            # Generate query using LLM
            generated_query = self.llm(prompt).strip()
            
            # Clean up the generated query - extract only the MongoDB query part
            if "```" in generated_query:
                # Extract code from markdown code blocks if present
                lines = generated_query.split('\n')
                clean_lines = []
                in_code_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block and not line.startswith("```"):
                        clean_lines.append(line)
                generated_query = '\n'.join(clean_lines)
            
            # Further cleanup - remove any text before or after the actual query
            if "db.collection" in generated_query:
                start_idx = generated_query.find("db.collection")
                end_idx = generated_query.find("\n", start_idx)
                if end_idx == -1:  # If there's no newline after the query
                    end_idx = len(generated_query)
                generated_query = generated_query[start_idx:end_idx].strip()
            
            # Log the generated query to the queries file
            self._log_query(user_query, generated_query)
            
            logger.info(f"Generated MongoDB query: {generated_query}")
            return generated_query
        except Exception as e:
            logger.error(f"Error generating MongoDB query: {e}")
            raise
    
    def _log_query(self, user_query, generated_query):
        """Log the user query and the generated MongoDB query to a file."""
        try:
            with open(self.queries_file, 'a', encoding='utf-8') as f:
                f.write(f"User Query: {user_query}\n")
                f.write(f"Generated Query: {generated_query}\n\n")
        except Exception as e:
            logger.error(f"Error logging query to file: {e}")
    
    def execute_query(self, query_string):
        """Execute the generated MongoDB query."""
        try:
            # Convert string query to actual MongoDB operation
            # This is risky but necessary for dynamic query execution
            # In a production environment, additional safety measures should be implemented
            
            # Extract the operation type and the query part
            if "db.collection.find" in query_string:
                operation = "find"
                query_part = query_string.split("db.collection.find")[1].strip()
            elif "db.collection.aggregate" in query_string:
                operation = "aggregate"
                query_part = query_string.split("db.collection.aggregate")[1].strip()
            else:
                logger.error(f"Unsupported operation in query: {query_string}")
                raise ValueError(f"Unsupported operation in query: {query_string}")
            
            # Evaluate the query part to get the actual query object
            # Note: This uses eval() which can be dangerous if not properly controlled
            # In a production environment, use a safer approach for query parsing
            try:
                # Extract the query parameters from the string
                if query_part.startswith("(") and query_part.endswith(")"):
                    query_part = query_part[1:-1]  # Remove outer parentheses
                
                # Handle both single and multiple parameter cases
                if operation == "find":
                    # Parse the query parameters
                    query_params = {}
                    sort_params = None
                    limit_value = None
                    
                    # Basic parsing (this is simplified and may need enhancement for complex queries)
                    if "{" in query_part:
                        # Extract the query criteria
                        start_idx = query_part.find("{")
                        end_idx = query_part.rfind("}")
                        query_json = query_part[start_idx:end_idx+1]
                        
                        # Parse the query JSON
                        query_params = json.loads(query_json)
                        
                        # Check for additional parameters like sort or limit
                        if ".sort" in query_part:
                            sort_start = query_part.find(".sort(") + 6
                            sort_end = query_part.find(")", sort_start)
                            sort_json = query_part[sort_start:sort_end]
                            sort_params = json.loads(sort_json)
                        
                        if ".limit" in query_part:
                            limit_start = query_part.find(".limit(") + 7
                            limit_end = query_part.find(")", limit_start)
                            limit_value = int(query_part[limit_start:limit_end])
                    
                    # Execute the query with the parsed parameters
                    cursor = self.collection.find(query_params)
                    
                    # Apply sort if specified
                    if sort_params:
                        cursor = cursor.sort(list(sort_params.items()))
                    
                    # Apply limit if specified
                    if limit_value:
                        cursor = cursor.limit(limit_value)
                    
                    # Convert cursor to list of documents
                    results = list(cursor)
                elif operation == "aggregate":
                    # Parse the aggregate pipeline
                    pipeline = json.loads(query_part)
                    results = list(self.collection.aggregate(pipeline))
                else:
                    logger.error(f"Unsupported operation: {operation}")
                    raise ValueError(f"Unsupported operation: {operation}")
                
                logger.info(f"Query executed successfully. Retrieved {len(results)} documents.")
                return results
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing query JSON: {e}")
                raise ValueError(f"Invalid query format: {e}")
        except Exception as e:
            logger.error(f"Error executing MongoDB query: {e}")
            raise
    
    def display_results(self, results):
        """Display the query results in a human-readable format."""
        try:
            if not results:
                logger.info("No results found for the query")
                print("No results found for the query.")
                return
            
            # Convert MongoDB documents to DataFrame for nice display
            df = pd.DataFrame(results)
            
            # Remove MongoDB _id field
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            # Format dates for better display
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]' or all(isinstance(x, datetime) for x in df[col] if x is not None):
                    df[col] = df[col].apply(lambda x: x.strftime('%d-%m-%Y') if isinstance(x, datetime) else x)
            
            # Format floats to 2 decimal places
            for col in df.columns:
                if df[col].dtype == 'float64':
                    df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
            
            # Display the results
            print("\nQuery Results:")
            print(df.to_string(index=False))
            logger.info("Results displayed successfully")
            
            return df
        except Exception as e:
            logger.error(f"Error displaying results: {e}")
            raise
    
    def save_results_to_csv(self, results, output_file):
        """Save the query results to a CSV file."""
        try:
            if not results:
                logger.info("No results to save")
                print("No results to save.")
                return False
            
            # Convert MongoDB documents to DataFrame
            df = pd.DataFrame(results)
            
            # Remove MongoDB _id field
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            # Format dates for better display
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]' or all(isinstance(x, datetime) for x in df[col] if x is not None):
                    df[col] = df[col].apply(lambda x: x.strftime('%d-%m-%Y') if isinstance(x, datetime) else x)
            
            # Save DataFrame to CSV
            df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
            print(f"Results saved to {output_file}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving results to CSV: {e}")
            raise

def run_test_cases(system):
    test_cases = [
        "Find all products with a rating below 4.5 that have more than 200 reviews and are offered by the brand 'Nike' or 'Sony'.",
        "Which products in the Electronics category have a rating of 4.5 or higher and are in stock?",
        "List products launched after January 1, 2022, in the Home & Kitchen or Sports categories with a discount of 10% or more, sorted by price in descending order."
    ]
    
    for i, test_query in enumerate(test_cases, 1):
        print(f"\nRunning Test Case {i}: {test_query}")
        try:
            # Generate MongoDB query
            mongo_query = system.generate_mongodb_query(test_query)
            print(f"Generated Query: {mongo_query}")
            
            # Execute the query
            results = system.execute_query(mongo_query)
            
            # Display results
            df = system.display_results(results)
            
            # Save results to CSV
            output_file = f"test_case{i}.csv"
            system.save_results_to_csv(results, output_file)
        except Exception as e:
            print(f"Error in Test Case {i}: {e}")
            logger.error(f"Error in Test Case {i}: {e}")

def main():
    try:
        # Initialize the system
        system = DataQuerySystem()
        
        # Ask for CSV file path
        csv_path = input("Enter the path to your CSV file (or press Enter to use 'sample_data.csv'): ")
        if not csv_path:
            csv_path = "sample_data.csv"
        
        # Load CSV data into MongoDB
        system.load_csv_to_mongodb(csv_path)
        
        # Ask if user wants to run predefined test cases or interactive mode
        mode = input("\nSelect mode (1: Run Test Cases, 2: Interactive Mode): ")
        
        if mode == "1":
            # Run predefined test cases
            run_test_cases(system)
        else:
            # Interactive mode
            while True:
                # Get user query
                user_query = input("\nEnter your query (or 'exit' to quit): ")
                if user_query.lower() == 'exit':
                    break
                
                try:
                    # Generate MongoDB query
                    mongo_query = system.generate_mongodb_query(user_query)
                    print(f"Generated Query: {mongo_query}")
                    
                    # Execute the query
                    results = system.execute_query(mongo_query)
                    
                    # Display results
                    df = system.display_results(results)
                    
                    save_option = input("\nSave results to CSV? (y/n): ")
                    if save_option.lower() == 'y':
                        output_file = input("Enter output file name: ")
                        if not output_file.endswith('.csv'):
                            output_file += '.csv'
                        system.save_results_to_csv(results, output_file)
                except Exception as e:
                    print(f"Error: {e}")
                    logger.error(f"Error in interactive mode: {e}")
    except Exception as e:
        print(f"System error: {e}")
        logger.error(f"System error: {e}")

if __name__ == "__main__":
    main()