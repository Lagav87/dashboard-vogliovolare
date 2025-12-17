import requests
import json

# --- INSERISCI QUI I TUOI DATI PER IL TEST ---
# Se preferisci non scriverli qui, il programma te li chiederà dopo.
API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJ0aWQiOjUyNzQyMzk2MSwiYWFpIjoxMSwidWlkIjozNjkwODY1NywiaWFkIjoiMjAyNS0wNi0xN1QxMzowOToxOC4wMDBaIiwicGVyIjoibWU6d3JpdGUiLCJhY3RpZCI6MTE5NzgzODYsInJnbiI6InVzZTEifQ.I4-t1swy2eAvjxRWb_ZF9xBnKD0P1DYwt545V1dT80k" 
BOARD_ID = "2613154224"

def scan_board(api_key, board_id):
    url = "https://api.monday.com/v2"
    headers = {"Authorization": api_key}
    
    # Questa query chiede a Monday: "Dammi le colonne di questa board e i loro ID"
    query = f"""
    query {{
      boards (ids: {board_id}) {{
        columns {{
          title
          id
          type
        }}
      }}
    }}
    """
    
    response = requests.post(url, json={'query': query}, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if "errors" in data:
            print("Errore:", data['errors'][0]['message'])
            return

        columns = data['data']['boards'][0]['columns']
        
        print("\n" + "="*50)
        print("  ECCO LA TUA MAPPATURA DA COPIARE  ")
        print("="*50 + "\n")
        print("Copia questi codici dentro il file app.py nella sezione COLUMN_MAPPING:\n")
        
        for col in columns:
            print(f'"{col["title"]}": "{col["id"]}",')
            
        print("\n" + "="*50)
    else:
        print("Errore di connessione:", response.text)

# --- AVVIO ---
if __name__ == "__main__":
    print("--- SCANNER COLONNE MONDAY ---")
    if not API_KEY:
        API_KEY = input("Incolla la tua API Key di Monday: ").strip()
    if not BOARD_ID:
        BOARD_ID = input("Incolla il tuo Board ID: ").strip()
    
    scan_board(API_KEY, BOARD_ID)