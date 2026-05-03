# forge_loop.py
import json
import os
import yfinance as yf
import requests
from datetime import datetime, timedelta

# --- Configuration ---
CONFIG_FILE = 'config/tickers.json'
SCORES_FILE = 'state/analyst_scores.json'
HISTORY_DIR = 'history/'

os.makedirs('config', exist_ok=True)
os.makedirs('state', exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

def load_json(filepath, default_data):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return default_data

def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def fetch_recent_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news[:5] 
        headlines = [f"- {item['title']} ({item['publisher']})" for item in news_items]
        return "\n".join(headlines)
    except Exception:
        return "No recent news available."

def evaluate_prediction(open_price, high_price, low_price, predicted_price, current_score):
    stop_loss_limit = open_price * 0.98 
    
    if low_price <= stop_loss_limit:
        return max(0.0, current_score - 0.01) 
    elif high_price >= predicted_price:
        return min(10.0, current_score + 0.01) 
    else:
        return max(0.0, current_score - 0.01) 

def call_local_llm(ticker, current_price, model_name):
    news_context = fetch_recent_news(ticker)
    
    prompt = f"""You are a quantitative financial analyst. 
The current closing price of {ticker} is ${current_price:.2f}.
Here are the latest news headlines regarding this company:
{news_context}

Based on this data, predict the High of the Day (HOD) for the next trading session. 
You must respond with ONLY the exact numerical price. Do not include dollar signs, words, or explanations. Example: 154.25"""

    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1 
            }
        }
        
        # Increased timeout to 120s to allow for initial model loading
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        raw_output = response.json().get('response', '').strip()
        clean_number = "".join([c for c in raw_output if c.isdigit() or c == '.'])
        return round(float(clean_number), 2)
        
    except Exception as e:
        print(f"    [!] AI generation failed. Error: {e}")
        return round(current_price * 1.001, 2) 

def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting OracleForge Loop...")
    
    tickers = load_json(CONFIG_FILE, [])
    scores = load_json(SCORES_FILE, {})
    
    if not tickers:
        print("No tickers found. Run update_tickers.py first.")
        return

    today_date = datetime.now().strftime('%Y-%m-%d')
    yesterday_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    yesterday_log_path = os.path.join(HISTORY_DIR, f'predictions_{yesterday_date}.json')
    today_log_path = os.path.join(HISTORY_DIR, f'predictions_{today_date}.json')
    
    yesterday_predictions = load_json(yesterday_log_path, {})
    today_predictions = {}
    market_data = {}

    # --- PHASE 1: FETCH ALL MARKET DATA & EVALUATE YESTERDAY ---
    print("\n--- PHASE 1: Market Data & Evaluation ---")
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            
            if hist.empty:
                continue
                
            today_open = hist['Open'].iloc[0]
            today_high = hist['High'].iloc[0]
            today_low = hist['Low'].iloc[0]
            today_close = hist['Close'].iloc[0]
            
            # Store data so we don't query Yahoo Finance repeatedly
            market_data[ticker] = today_close
            today_predictions[ticker] = {}

            # Evaluate yesterday's performance
            if ticker in yesterday_predictions:
                for model_name, past_pred in yesterday_predictions[ticker].items():
                    current_score = scores.get(model_name, 5.0)
                    new_score = evaluate_prediction(
                        open_price=today_open,
                        high_price=today_high,
                        low_price=today_low,
                        predicted_price=past_pred,
                        current_score=current_score
                    )
                    scores[model_name] = round(new_score, 3)
                    
        except Exception as e:
            print(f"  Error fetching data for {ticker}: {e}")

    # --- PHASE 2: MODEL INFERENCE (HARDWARE OPTIMIZED) ---
    print("\n--- PHASE 2: AI Inference (Model by Model) ---")
    # By looping models first, Ollama keeps the model loaded in VRAM 
    # for all 50 tickers before swapping to the next one.
    for model_name in scores.keys():
        print(f"\n>> Loading weights and processing with {model_name} <<")
        
        for ticker, current_price in market_data.items():
            print(f"  [{model_name}] Predicting {ticker} (Close: ${current_price:.2f})...")
            prediction = call_local_llm(ticker, current_price, model_name)
            today_predictions[ticker][model_name] = prediction

    # --- PHASE 3: SAVE STATE ---
    print("\nSaving new scores and daily predictions...")
    save_json(SCORES_FILE, scores)
    save_json(today_log_path, today_predictions)
    print("OracleForge Loop Complete. Ready for Git Commit.")

if __name__ == "__main__":
    main()
