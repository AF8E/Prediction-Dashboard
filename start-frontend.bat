@echo off
echo Starting Next.js Frontend Server...
cd frontend
if not exist node_modules (
    echo Installing dependencies...
    call npm install
)
call npm run dev
pause
