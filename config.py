import os
from dotenv import load_dotenv

# Get the absolute path of the directory containing this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the .env file
dotenv_path = os.path.join(BASE_DIR, '.env')

# Load the .env file
load_dotenv(dotenv_path)

# Function to get environment variables
def get_env_variable(var_name):
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"{var_name} is not set in the environment variables")
    return value

# Get the OpenAI API key
OPENAI_API_KEY = get_env_variable('OPENAI_API_KEY')

# Add other environment variables as needed
# Example: DATABASE_URL = get_env_variable('DATABASE_URL')