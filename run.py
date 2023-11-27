import subprocess
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import sys
from fastapi import FastAPI
import joblib
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import sys
import os
import json
import re
import base64
import shap

if __name__ == '__main__':
    subprocess.run("streamlit run Welcome.py")