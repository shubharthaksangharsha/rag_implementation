from langchain.agents import Tool, load_tools
import os 
from langchain_community.tools.file_management.read import ReadFileTool
from langchain_community.tools.file_management.write import WriteFileTool
from langchain_community.utilities.python import PythonREPL
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from langchain.tools import BaseTool, StructuredTool, tool
import datetime


@tool 
def mylocation():
    '''
    Useful when want to get my current location or my location or user location.
    it is used for openweather api if user ask for current place or location details for weather input.
    '''
    return "Kharar, Punjab, India"

@tool
def get_today_date():
    '''
    Useful when you want to find today's date 
    '''
    date = datetime.datetime.today()
    return date.strftime("Date: %Y-%m-%d")

#Weather Tool 
weather_tool = Tool(
    name='openweather',
    func=OpenWeatherMapAPIWrapper().run,
    description='useful to get weather details of any location'
)

# Python Repl Tool
python_tool = Tool(
    name='python-repl', 
    func=PythonREPL().run,
    description='useful to execute python code.')

read_tool = ReadFileTool()
write_tool = WriteFileTool()

if __name__ == '__main__':
    print(get_today_date.name)
    print(get_today_date.description)
    print(get_today_date.args)
    