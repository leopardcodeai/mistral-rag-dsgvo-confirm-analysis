import json
import sys

def read_json_database(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"databank loaded: {file_path}")
        print(f"datatyp: {type(data).__name__}")
        if isinstance(data, (dict, list)):
            print(f"number of entries: {len(data)}")
        print("\nInhalt:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except FileNotFoundError:
        print(f"Error: file {file_path} not found.")
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"Unerwarteter Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Verwendung: python datenbank_auswertung.py <pfad_zur_json_datenbank>")
        sys.exit(1)
    read_json_database(sys.argv[1])