@echo off
echo Starting FastAPI Backend Server...
cd backend
py -m uvicorn main:app --reload --port 8000
pause
