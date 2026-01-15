# Sustainment Prediction Dashboard

A web-based prediction tool for US Space Command facilities that simulates future infrastructure health based on project funding scenarios.

## Quick Start (Easiest Way)

**Just double-click `start-all.bat`** - This will:
1. Install all dependencies automatically
2. Start both servers
3. Open your browser to the dashboard

## Setup Instructions (Manual)

### Backend Setup

1. Navigate to the project root directory
2. Install Python dependencies:
   ```bash
   py -m pip install -r requirements.txt
   ```
   (On Windows, use `py` instead of `python`)
3. Start the FastAPI server:
   ```bash
   cd backend
   py -m uvicorn main:app --reload --port 8000
   ```
   The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install Node.js dependencies:
   ```bash
   npm install
   ```
3. Start the Next.js development server:
   ```bash
   npm run dev
   ```
   The application will be available at `http://localhost:3000` (or 3001 if 3000 is in use)

## Usage

1. Start both servers (backend on port 8000, frontend on port 3000/3001)
2. Open the dashboard in your browser
3. Upload Files:
   - Projects File (CSV or XLSX)
   - Inventory/Key File (CSV or XLSX)
4. Run Prediction and view the forecast

## File Format Requirements

### Projects File
Should contain columns for:
- Fiscal Year (or similar): The year when the project is funded
- Cost: Project cost
- Scope: Project description (used to identify which systems are affected)

### Inventory/Key File
Should contain columns for:
- Systems: System names (e.g., "HVAC", "Electric", "Foundation")
- Life Expectancy: Expected lifespan in years for each system

The system automatically detects these columns by searching for keywords like "fiscal year", "cost", "scope", "system", and "life expectancy".
