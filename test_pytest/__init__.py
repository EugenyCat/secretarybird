import pytest
import os
import sys
from dotenv import load_dotenv


# Get the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# init a folder where json input body examples are stored
load_dotenv()


JSON_TEST_FILES_FOLDER = os.getenv('JSON_TEST_FILES_FOLDER')

#print(f'{JSON_TEST_FILES_FOLDER}')