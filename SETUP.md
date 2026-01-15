# Quick Setup Guide

## Prerequisites
- Node.js (v18 or higher) - https://nodejs.org/
- Python (v3.8 or higher) - https://www.python.org/downloads/
  - Check "Add Python to PATH" during install

## First Time Setup

### 1) Install Backend Dependencies
```bash
py -m pip install -r requirements.txt
```

### 2) Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

## Running the Dashboard

### Option 1: Use the Batch Files (Windows)
1. Double-click `start-backend.bat` (starts API server on 8000)
2. Double-click `start-frontend.bat` (starts web UI on 3000/3001)
3. Open http://localhost:3000 (or 3001 if 3000 is in use)

### Option 2: All-in-one
Double-click `start-all.bat` to install deps and launch both servers.

## Troubleshooting

### "python is not recognized"
- Use `py` instead of `python`
- Or add Python to PATH

### "next is not recognized"
- Run `npm install` in `frontend`
- Check Node.js: `node --version`

### Port already in use
- Frontend uses 3000 (or 3001 fallback)
- Backend uses 8000
