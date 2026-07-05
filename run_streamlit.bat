@echo off
echo Installing Streamlit if not already installed...
pip install streamlit

echo.
echo Starting Pistos.ai with Streamlit...
echo.
echo Public Chat: http://localhost:8501
echo Admin Panel: http://localhost:8501/?admin=true
echo.

streamlit run app.py --server.port 8501 --server.address 0.0.0.0
